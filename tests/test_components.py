"""Tests for individual components: I3D model, tokenizer, feature extractor."""

import numpy as np
import torch
import pytest


@pytest.mark.slow
def test_i3d_forward():
    """Test that InceptionI3d produces correct output shape."""
    from silverscore.i3d import InceptionI3d

    model = InceptionI3d(num_classes=400, num_in_frames=16, include_embds=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert "logits" in output
    assert "embds" in output
    assert output["logits"].shape == (1, 400)
    assert output["embds"].shape[1] == 1024


def test_tokenizer():
    """Test CLIP tokenizer encodes and decodes text."""
    from silverscore.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()

    tokens = tokenizer.encode("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)

    text_tokens = tokenizer.tokenize("hello world")
    assert isinstance(text_tokens, list)
    assert len(text_tokens) > 0

    ids = tokenizer.convert_tokens_to_ids(text_tokens)
    assert ids == tokens


def test_feature_extractor_from_numpy():
    """Test I3D feature extractor with synthetic frames."""
    from silverscore.feature_extractor import (
        _preprocess_frames,
        _slide_windows,
    )

    frames = np.random.randint(0, 255, (32, 240, 320, 3), dtype=np.uint8)
    tensor = _preprocess_frames(frames, resize_res=256, crop_size=224)
    assert tensor.shape == (3, 32, 224, 224)
    assert tensor.dtype == torch.float32

    windows = _slide_windows(num_frames=32, clip_len=16, stride=1)
    assert len(windows) >= 1
    for start, end in windows:
        assert end - start <= 16
        assert start >= 0


def test_slide_windows_short_video():
    """Sliding windows should handle videos shorter than clip_len."""
    from silverscore.feature_extractor import _slide_windows

    windows = _slide_windows(num_frames=8, clip_len=16, stride=1)
    assert len(windows) == 1
    assert windows[0] == (0, 8)


def test_slide_windows_exact():
    """Sliding windows for a video that's exactly clip_len."""
    from silverscore.feature_extractor import _slide_windows

    windows = _slide_windows(num_frames=16, clip_len=16, stride=1)
    assert len(windows) == 1
    assert windows[0] == (0, 16)
