"""
I3D-based video feature extractor for sign language videos.

Extracts 1024-d embeddings per clip using a sliding window over video frames,
then returns the full feature matrix for use with SiLVERScore.
"""

import logging
import math
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .i3d import InceptionI3d

logger = logging.getLogger(__name__)

# Normalization values used in BSL-1K / CiCo I3D training
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _load_video_cv2(video_path: str) -> np.ndarray:
    """
    Load a video file into a numpy array using OpenCV.

    Returns:
        frames: (num_frames, H, W, 3) uint8 array in BGR format.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video loading. "
            "Install it with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    return np.stack(frames, axis=0)


def _preprocess_frames(
    frames: np.ndarray,
    resize_res: int = 256,
    crop_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess video frames: resize, center crop, normalize.

    Args:
        frames: (T, H, W, 3) uint8 BGR array.
        resize_res: Resize short side to this before cropping.
        crop_size: Center crop to this spatial size.

    Returns:
        tensor: (3, T, crop_size, crop_size) float32 tensor, normalized.
    """
    import cv2

    T, H, W, C = frames.shape

    if H < W:
        new_h = resize_res
        new_w = int(W * resize_res / H)
    else:
        new_w = resize_res
        new_h = int(H * resize_res / W)

    resized = np.empty((T, new_h, new_w, C), dtype=np.uint8)
    for t in range(T):
        resized[t] = cv2.resize(frames[t], (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)

    y_off = (new_h - crop_size) // 2
    x_off = (new_w - crop_size) // 2
    cropped = resized[:, y_off:y_off + crop_size, x_off:x_off + crop_size, :]

    # BGR -> RGB, to float, scale to [0, 1]
    rgb = cropped[:, :, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).float() / 255.0

    # (T, H, W, 3) -> (3, T, H, W)
    tensor = tensor.permute(3, 0, 1, 2)

    # Normalize with mean/std
    mean = torch.tensor(_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


def _slide_windows(num_frames: int, clip_len: int = 16,
                   stride: int = 1) -> list:
    """
    Generate sliding window start indices for feature extraction.

    Returns:
        List of (start_frame, end_frame) tuples.
    """
    if num_frames <= clip_len:
        return [(0, min(clip_len, num_frames))]

    windows = []
    num_clips = math.ceil((num_frames - clip_len) / stride) + 1
    for j in range(num_clips):
        start = j * stride
        end = start + clip_len
        if end > num_frames:
            start = max(0, num_frames - clip_len)
            end = num_frames
        windows.append((start, end))
    return windows


class I3DFeatureExtractor:
    """
    Extract I3D features from sign language videos.

    Args:
        checkpoint_path: Path to the I3D checkpoint (.pth.tar or .pth).
            If None, tries to download from HuggingFace Hub.
        device: Device for inference.
        num_in_frames: Number of frames per clip (default 16).
        stride: Sliding window stride (default 1).
        resize_res: Resize short side to this before cropping (default 256).
        crop_size: Center crop size (default 224).

    Example::

        extractor = I3DFeatureExtractor(checkpoint_path="bsl5k.pth.tar")
        features = extractor.extract("sign_video.mp4")
        # features shape: (num_clips, 1024)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        num_in_frames: int = 16,
        stride: int = 1,
        resize_res: int = 256,
        crop_size: int = 224,
    ):
        self.num_in_frames = num_in_frames
        self.stride = stride
        self.resize_res = resize_res
        self.crop_size = crop_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: Optional[str]) -> InceptionI3d:
        model = InceptionI3d(
            num_classes=981,
            num_in_frames=self.num_in_frames,
            include_embds=True,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Handle DataParallel "module." prefix
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[new_k] = v

            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                logger.info("I3D missing keys: %s", missing[:5])
            if unexpected:
                logger.info("I3D unexpected keys: %s", unexpected[:5])

        model.to(self.device)
        return model

    @torch.no_grad()
    def extract(
        self,
        video_input: Union[str, np.ndarray],
    ) -> np.ndarray:
        """
        Extract I3D features from a video.

        Args:
            video_input: Either a path to a video file (str),
                or a pre-loaded numpy array of frames (T, H, W, 3) in BGR.

        Returns:
            features: (num_clips, 1024) numpy array of I3D embeddings.
        """
        if isinstance(video_input, str):
            frames = _load_video_cv2(video_input)
        elif isinstance(video_input, np.ndarray):
            frames = video_input
        else:
            raise TypeError(
                f"Expected str or np.ndarray, got {type(video_input)}")

        video_tensor = _preprocess_frames(
            frames, resize_res=self.resize_res, crop_size=self.crop_size
        )
        num_frames = video_tensor.shape[1]

        windows = _slide_windows(
            num_frames, clip_len=self.num_in_frames, stride=self.stride
        )

        all_features = []
        for start, end in windows:
            clip = video_tensor[:, start:end, :, :]

            # Pad if clip is shorter than num_in_frames
            if clip.shape[1] < self.num_in_frames:
                pad_size = self.num_in_frames - clip.shape[1]
                clip = F.pad(clip, (0, 0, 0, 0, 0, pad_size))

            clip = clip.unsqueeze(0).to(self.device)
            outputs = self.model(clip)
            embd = outputs["embds"].squeeze().cpu()
            all_features.append(embd)

        features = torch.stack(all_features, dim=0).numpy()
        return features

    def extract_batch(
        self,
        video_paths: list,
        batch_size: int = 4,
    ) -> list:
        """
        Extract features for multiple videos.

        Returns:
            List of (num_clips, 1024) numpy arrays, one per video.
        """
        results = []
        for path in video_paths:
            results.append(self.extract(path))
        return results

    def __repr__(self):
        return (
            f"I3DFeatureExtractor("
            f"num_in_frames={self.num_in_frames}, "
            f"stride={self.stride}, "
            f"device={self.device})"
        )
