"""
SiLVERScore: Sign Language Video Embedding Representation Score.

Main user-facing API for computing embedding similarity between
sign language videos and text descriptions.
"""

import logging
import os
import pickle
from typing import List, Optional, Union

import numpy as np
import torch

from .config import get_task_config, VARIANT_CONFIGS
from .model import CLIP4Clip
from .tokenizer import SimpleTokenizer
from .base_modules import PYTORCH_PRETRAINED_BERT_CACHE

logger = logging.getLogger(__name__)


class SiLVERScore:
    """
    Compute SiLVERScore between sign language video and text.

    Args:
        variant: Sign language variant to use. One of
            "phoenix" (DGS), "how2sign" (ASL), "csl" (CSL-Daily).
        model_path: Path to a local checkpoint file. If None,
            downloads from HuggingFace Hub. Pass ``"none"`` to
            skip loading finetuned weights (uses CLIP base only).
        device: Device to run inference on ("cpu", "cuda", "mps").
        alpha: Feature blending weight (domain-agnostic vs domain-aware).
            If None, uses the variant default.
        i3d_checkpoint: Path to a pretrained I3D checkpoint for
            video feature extraction. If None and a video path is
            passed to :meth:`score`, the I3D model uses random weights.

    Example::

        scorer = SiLVERScore(variant="phoenix")
        score = scorer.score("sign_video.mp4", text="a person signing hello")
    """

    def __init__(
        self,
        variant: str = "phoenix",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        alpha: Optional[float] = None,
        i3d_checkpoint: Optional[str] = None,
    ):
        self.variant = variant
        self.task_config = get_task_config(variant)
        if alpha is not None:
            self.task_config.alpha = alpha

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self._i3d_checkpoint = i3d_checkpoint
        self._i3d_extractor = None

        self.tokenizer = SimpleTokenizer()
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: Optional[str] = None) -> CLIP4Clip:
        state_dict = None

        if model_path == "none":
            logger.info("Skipping finetuned weights; using CLIP base only.")
        elif model_path is not None:
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            try:
                hub_path = self._download_from_hub()
                state_dict = torch.load(hub_path, map_location="cpu")
            except Exception as e:
                logger.warning(
                    "Could not download finetuned weights (%s); "
                    "using CLIP base weights only.", e
                )

        cross_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cross-base"
        )
        model = CLIP4Clip.from_pretrained(
            cross_model_dir,
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE / "distributed"),
            state_dict=state_dict,
            task_config=self.task_config,
        )
        model.to(self.device)
        return model

    def _download_from_hub(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install it with: pip install huggingface-hub"
            )
        cfg = VARIANT_CONFIGS[self.variant]
        return hf_hub_download(
            repo_id=cfg["hub_model_id"],
            filename=cfg["checkpoint_filename"],
        )

    def _tokenize_text(
        self, text: Union[str, List[str]], max_words: int = 32
    ) -> tuple:
        """Tokenize text into input_ids, segment_ids, attention_mask."""
        if isinstance(text, str):
            text = [text]

        batch_input_ids = []
        batch_segment_ids = []
        batch_attention_mask = []

        for t in text:
            words = self.tokenizer.tokenize(t)
            words = ["<|startoftext|>"] + words[:max_words - 2] + ["<|endoftext|>"]
            word_ids = self.tokenizer.convert_tokens_to_ids(words)

            input_ids = word_ids + [0] * (max_words - len(word_ids))
            segment_ids = [0] * max_words
            attention_mask = [1] * len(word_ids) + [0] * (max_words - len(word_ids))

            batch_input_ids.append(input_ids)
            batch_segment_ids.append(segment_ids)
            batch_attention_mask.append(attention_mask)

        return (
            torch.tensor(batch_input_ids, dtype=torch.long),
            torch.tensor(batch_segment_ids, dtype=torch.long),
            torch.tensor(batch_attention_mask, dtype=torch.long),
        )

    def _prepare_video_features(
        self, features: Union[np.ndarray, torch.Tensor],
        feature_len: int = 64
    ) -> tuple:
        """
        Prepare pre-extracted I3D features for the model.

        Follows the original CiCo dataloader convention:
        - mask length = ``feature_len + 1`` (position 0 = CLS token)
        - ``1`` = padding / empty, ``0`` = valid feature

        Args:
            features: (num_clips, 1024) array of I3D features.
            feature_len: Target number of frames to sample.

        Returns:
            (video_tensor, video_mask) ready for the model.
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if features.dim() == 2:
            features = features.unsqueeze(0)

        batch_size, num_clips, feat_dim = features.shape

        video_tensors = []
        video_masks = []

        for b in range(batch_size):
            feat = features[b]
            n = feat.shape[0]

            if n >= feature_len:
                indices = np.linspace(0, n - 1, feature_len, dtype=int)
                sampled = feat[indices]
                n_valid = feature_len
            else:
                pad = torch.zeros(feature_len - n, feat_dim)
                sampled = torch.cat([feat, pad], dim=0)
                n_valid = n

            video_tensor = sampled.T.unsqueeze(-1)

            video_mask = torch.ones(1, feature_len + 1, dtype=torch.long)
            video_mask[0, 0] = 1  # CLS position always masked
            for i in range(n_valid):
                video_mask[0, i + 1] = 0  # valid feature positions

            video_tensors.append(video_tensor.unsqueeze(0))
            video_masks.append(video_mask)

        return (
            torch.cat(video_tensors, dim=0),
            torch.cat(video_masks, dim=0),
        )

    def _ensure_i3d_extractor(self):
        """Lazily initialise the I3D feature extractor."""
        if self._i3d_extractor is not None:
            return
        from .feature_extractor import I3DFeatureExtractor

        self._i3d_extractor = I3DFeatureExtractor(
            checkpoint_path=self._i3d_checkpoint,
            device=str(self.device),
        )

    def _extract_features(self, video_path: str) -> np.ndarray:
        """Run I3D extraction on a video file."""
        self._ensure_i3d_extractor()
        return self._i3d_extractor.extract(video_path)

    @torch.no_grad()
    def score(
        self,
        video: Union[str, np.ndarray, torch.Tensor],
        text: Union[str, List[str]],
        feature_len: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Compute SiLVERScore between a sign language video and text.

        Args:
            video: One of the following:

                * **str** -- path to a video file (.mp4, .avi, …).
                  I3D features are extracted automatically.
                * **np.ndarray / torch.Tensor** -- pre-extracted I3D
                  features of shape ``(num_clips, 1024)`` for a single
                  video or ``(batch, num_clips, 1024)`` for a batch.
            text: A string or list of strings to compare against.
            feature_len: Number of frames to sample from features.
                Defaults to the variant config value.

        Returns:
            A scalar score (single video + single text) or
            a numpy array of shape ``(num_videos, num_texts)``.
        """
        if isinstance(video, str):
            video_features = self._extract_features(video)
        else:
            video_features = video

        if feature_len is None:
            feature_len = self.task_config.feature_len

        single_text = isinstance(text, str)

        video_tensor, video_mask = self._prepare_video_features(
            video_features, feature_len=feature_len
        )
        video_tensor = video_tensor.to(self.device)
        video_mask = video_mask.to(self.device)

        input_ids, segment_ids, attention_mask = self._tokenize_text(
            text, max_words=self.task_config.max_words
        )
        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        text_mask, seq_hidden, seq_cls = self.model.get_sequence_output(
            input_ids, segment_ids, attention_mask, shaped=True
        )
        video_mask_out, vis_hidden, vis_cls = self.model.get_visual_output(
            video_tensor, video_mask, shaped=True, video_frame=1
        )

        I2T, T2I, _ = self.model.get_similarity_logits(
            seq_hidden, vis_hidden, text_mask, video_mask_out,
            shaped=True, loose_type=self.model.loose_type,
            is_train=True,
            sequence_hidden_aug=seq_hidden,
            text_mask_aug=text_mask,
        )

        combined = (
            self.task_config.dual_mix * I2T
            + (1 - self.task_config.dual_mix) * T2I
        )

        scores = combined.cpu().numpy()

        if scores.shape[0] == 1 and single_text:
            return float(scores[0, 0])
        return scores

    @torch.no_grad()
    def score_from_pkl(
        self,
        pkl_path: str,
        text: Union[str, List[str]],
    ) -> Union[float, np.ndarray]:
        """
        Convenience method: load features from a .pkl file
        (as saved by the I3D feature extractor) and compute score.
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        features = data["feature"]
        return self.score(video=features, text=text)

    def __repr__(self):
        return (
            f"SiLVERScore(variant='{self.variant}', "
            f"device={self.device}, "
            f"alpha={self.task_config.alpha})"
        )
