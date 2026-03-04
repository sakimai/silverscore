"""
CLIP model components: VisualTransformer, FeatureTransformer, text/image encoders.
Adapted from OpenAI CLIP (https://github.com/openai/CLIP) and the CiCo/CLCL codebase.
"""

from collections import OrderedDict

import hashlib
import os
import urllib
import warnings

import torch
from torch import nn
from tqdm import tqdm

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME = {
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists but SHA256 mismatch; re-downloading.")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model downloaded but SHA256 checksum does not match.")

    return download_target


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """Handle fp16 by casting to fp32 for the norm."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ---------------------------------------------------------------------------
# Vision Transformer blocks with video-mask support
# ---------------------------------------------------------------------------

class VResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, video_mask):
        video_mask = video_mask.to(device=x.device) == 1 if video_mask is not None else None
        return self.attn(x, x, x, key_padding_mask=video_mask, need_weights=False)[0]

    def forward(self, x_tuple: tuple):
        x, video_mask = x_tuple
        x = x + self.attention(self.ln_1(x), video_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, video_mask)


class VTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[VResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_mask):
        return self.resblocks((x, video_mask))[0]


# ---------------------------------------------------------------------------
# Text Transformer blocks
# ---------------------------------------------------------------------------

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, "__call__"):
            attn_mask_ = self.attn_mask(x.size(0))
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple: tuple):
        x, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]


# ---------------------------------------------------------------------------
# FeatureTransformer: processes pre-extracted I3D features (1024-d)
# ---------------------------------------------------------------------------

class FeatureTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int,
                 heads: int, output_dim: int, feature_len: int, linear_patch: str = "2d"):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.feature_len = feature_len

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.conv2_trans = nn.Conv2d(in_channels=feature_len * 2, out_channels=feature_len, kernel_size=1, stride=1, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(feature_len + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = VTransformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        assert linear_patch in ["2d", "3d"]
        self.linear_patch = linear_patch

    def forward(self, x: torch.Tensor, video_mask, video_frame=-1):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        if x.shape[1] > self.feature_len:
            x = x.unsqueeze(3)
            x = self.conv2_trans(x)
            x = x.squeeze(3)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_mask=video_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x


# ---------------------------------------------------------------------------
# CLIP: full model with text + visual encoders
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int,
                 vision_width: int, vision_patch_size: int, context_length: int,
                 vocab_size: int, transformer_width: int, transformer_heads: int,
                 transformer_layers: int, feature_len: int, linear_patch: str = "2d"):
        super().__init__()
        self.context_length = context_length
        self.feature_len = feature_len

        vision_heads = vision_width // 64
        self.visual = FeatureTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            feature_len=feature_len,
            linear_patch=linear_patch,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME.get(pretrained_clip_name, "ViT-B-32.pt"))
        if not os.path.exists(model_path):
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available: {list(_MODELS.keys())}")
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        return state_dict

    def build_attention_mask(self, context_length):
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_mask=None, video_frame=-1):
        hidden = self.visual(image.type(self.dtype), video_mask=video_mask, video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj
        x = hidden[:, 0, :]
        if return_hidden:
            return video_mask, hidden, x
        return x

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)
        pos_emd = self.positional_embedding[: x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection
        text_length = text.argmax(dim=-1)
        text_mask = torch.ones((hidden.shape[0], hidden.shape[1]), device=hidden.device)
        for i in range(hidden.shape[0]):
            text_mask[i, text_length[i] + 1 :] = 0

        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]
        if return_hidden:
            return text_mask, hidden, x
        return x


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16."""
    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()
        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
