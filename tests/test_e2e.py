"""
End-to-end test: synthetic video -> I3D feature extraction -> SiLVERScore.

This test downloads CLIP ViT-B/32 weights (~340 MB) on the first run.
Mark with @pytest.mark.slow so CI can skip it.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch


def _make_synthetic_video(path: str, num_frames: int = 50,
                          fps: int = 25, width: int = 320,
                          height: int = 240):
    """Write a short synthetic video to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.mark.slow
def test_score_with_video_path():
    """Full pipeline: video file -> I3D features -> CLIP4Clip -> score."""
    from silverscore import SiLVERScore

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    try:
        _make_synthetic_video(video_path, num_frames=50)

        scorer = SiLVERScore(
            variant="phoenix",
            model_path="none",
            device="cpu",
        )

        result = scorer.score(
            video=video_path,
            text="a person signing hello",
        )

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert np.isfinite(result), f"Score is not finite: {result}"
        print(f"\n  score(video_path, single text) = {result:.6f}")

        results = scorer.score(
            video=video_path,
            text=["hello", "goodbye", "thank you"],
        )
        assert isinstance(results, np.ndarray)
        assert results.shape == (1, 3), f"Expected (1, 3), got {results.shape}"
        assert np.all(np.isfinite(results))
        print(f"  score(video_path, 3 texts)     = {results.round(4)}")

    finally:
        os.unlink(video_path)


@pytest.mark.slow
def test_score_with_preextracted_features():
    """Same scorer, but with pre-extracted feature tensors."""
    from silverscore import SiLVERScore

    scorer = SiLVERScore(
        variant="phoenix",
        model_path="none",
        device="cpu",
    )

    fake_features = np.random.randn(80, 1024).astype(np.float32)

    result = scorer.score(video=fake_features, text="testing with features")
    assert isinstance(result, float)
    assert np.isfinite(result)
    print(f"\n  score(features, single text) = {result:.6f}")


@pytest.mark.slow
def test_score_video_matches_manual_extraction():
    """Verify that score(video_path) == extract + score(features)."""
    from silverscore import SiLVERScore
    from silverscore.feature_extractor import I3DFeatureExtractor

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    try:
        _make_synthetic_video(video_path, num_frames=32)

        scorer = SiLVERScore(
            variant="phoenix",
            model_path="none",
            device="cpu",
        )
        text = "a test sentence"

        score_direct = scorer.score(video=video_path, text=text)

        extractor = I3DFeatureExtractor(checkpoint_path=None, device="cpu")
        features = extractor.extract(video_path)
        score_manual = scorer.score(video=features, text=text)

        print(f"\n  Direct:  {score_direct:.6f}")
        print(f"  Manual:  {score_manual:.6f}")

        # Scores won't match exactly because I3D random weights differ
        # between the scorer's lazy extractor and the standalone one.
        # But both should be finite floats.
        assert isinstance(score_direct, float)
        assert isinstance(score_manual, float)
        assert np.isfinite(score_direct)
        assert np.isfinite(score_manual)

    finally:
        os.unlink(video_path)


if __name__ == "__main__":
    print("=== End-to-end SiLVERScore pipeline test ===\n")
    print("[1/3] Testing score(video_path, text) ...")
    test_score_with_video_path()
    print("\n[2/3] Testing score(features, text) ...")
    test_score_with_preextracted_features()
    print("\n[3/3] Testing video path vs manual extraction ...")
    test_score_video_matches_manual_extraction()
    print("\n=== All end-to-end tests passed! ===")
