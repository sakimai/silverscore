"""
Base modules shared across the model: PretrainedConfig, PreTrainedModel, LayerNorm, loss functions.
Adapted from the HuggingFace / CLIP4Clip codebase.
"""

import os
import copy
import json
import logging
import tarfile
import tempfile
import shutil
import math
from pathlib import Path
from typing import Union
from hashlib import sha256
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import requests

logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", Path.home() / ".pytorch_pretrained_bert")
)


# ---------------------------------------------------------------------------
# File / cache utilities (simplified from file_utils.py, no S3/boto3 dep)
# ---------------------------------------------------------------------------

def _http_get(url: str, temp_file) -> None:
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total, desc="Downloading")
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def cached_path(url_or_filename: Union[str, Path], cache_dir: Union[str, Path] = None) -> str:
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        os.makedirs(cache_dir, exist_ok=True)
        url_hash = sha256(url_or_filename.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, url_hash)
        if not os.path.exists(cache_path):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                _http_get(url_or_filename, temp_file)
                temp_file.flush()
                temp_file.seek(0)
            shutil.move(temp_file.name, cache_path)
        return cache_path
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == "":
        raise FileNotFoundError(f"file {url_or_filename} not found")
    else:
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": F.relu}


# ---------------------------------------------------------------------------
# LayerNorm (TF-style, epsilon inside sqrt)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# ---------------------------------------------------------------------------
# PretrainedConfig
# ---------------------------------------------------------------------------

class PretrainedConfig:
    pretrained_model_archive_map = {}
    config_name = ""
    weights_name = ""

    @classmethod
    def get_config(cls, pretrained_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=None):
        archive_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model_name)
        if not os.path.exists(archive_file):
            if pretrained_model_name in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name]
            else:
                archive_file = pretrained_model_name

        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error("Model name '%s' was not found.", pretrained_model_name)
            return None, None

        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir

        config_file = os.path.join(serialization_dir, cls.config_name)
        config = cls.from_json_file(config_file)
        config.type_vocab_size = type_vocab_size

        if state_dict is None:
            weights_path = os.path.join(serialization_dir, cls.weights_name)
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")

        if tempdir:
            shutil.rmtree(tempdir)

        return config, state_dict

    @classmethod
    def from_dict(cls, json_object):
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return json.dumps(copy.deepcopy(self.__dict__), indent=2, sort_keys=True)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


# ---------------------------------------------------------------------------
# PreTrainedModel
# ---------------------------------------------------------------------------

class PreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of PretrainedConfig."
            )
        self.config = config

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        not_load_visual = getattr(task_config, "not_load_visual", False) if task_config else False
        if not_load_visual:
            old_keys, new_keys = [], []
            for key in state_dict.keys():
                if "visual" in key:
                    old_keys.append(key)
                    new_keys.append(key.replace("visual", "no_load_visual"))
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = list(state_dict.keys())
            for key in old_keys:
                state_dict[prefix + key] = state_dict.pop(key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if prefix is None:
            if missing_keys:
                logger.info("Weights not initialized from pretrained: %s", missing_keys[:10])
            if unexpected_keys:
                logger.info("Weights from pretrained not used: %s", unexpected_keys[:10])

        return model

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module):
                return [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None, *inputs, **kwargs):
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)
        return model


# ---------------------------------------------------------------------------
# Loss functions (only CrossEn needed for scoring)
# ---------------------------------------------------------------------------

class CrossEn(nn.Module):
    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        return nce_loss.mean()
