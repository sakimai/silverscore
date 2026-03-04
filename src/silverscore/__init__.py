"""
SiLVERScore: Sign Language Video Embedding Retrieval Score.

An evaluation metric for sign language generation based on
CLIP-based embedding similarity between video and text.
"""

from .scorer import SiLVERScore
from .config import list_variants, VARIANT_CONFIGS
from .feature_extractor import I3DFeatureExtractor

__version__ = "0.1.0"
__all__ = [
    "SiLVERScore",
    "I3DFeatureExtractor",
    "list_variants",
    "VARIANT_CONFIGS",
]
