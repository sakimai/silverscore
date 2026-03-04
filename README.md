# SiLVERScore

**Si**gn **L**anguage **V**ideo **E**mbedding **Re**presentation **Score** -- an evaluation metric for sign language generation based on CLIP-based embedding similarity between video and text.

## Installation

```bash
pip install silverscore
```

Or install from source:

```bash
git clone https://github.com/sakimai/silverscore.git
cd silverscore
pip install -e .
```

## Quick Start

```python
from silverscore import SiLVERScore

# Initialize with a sign language variant
scorer = SiLVERScore(variant="phoenix")  # German Sign Language (DGS)

# Score a video file directly (I3D features are extracted automatically)
score = scorer.score("sign_video.mp4", text="a person signing hello")
print(f"SiLVERScore: {score:.4f}")

# Or pass pre-extracted I3D features
import numpy as np
features = np.load("video_features.npy")  # shape: (num_clips, 1024)
score = scorer.score(features, text="a person signing hello")
```

### From .pkl feature files

```python
score = scorer.score_from_pkl("path/to/features.pkl", text="a person signing")
```

### Batch scoring

```python
scores = scorer.score(
    video=batch_features,  # (batch, num_clips, 1024)
    text=["sentence one", "sentence two", "sentence three"]
)
# Returns a (num_videos, num_texts) similarity matrix
```

## Available Variants

| Variant | Language | Model ID |
|---------|----------|----------|
| `phoenix` | German Sign Language (DGS) | `sakimai/silverscore-phoenix` |
| `how2sign` | American Sign Language (ASL) | `sakimai/silverscore-how2sign` |
| `csl` | Chinese Sign Language (CSL) | `sakimai/silverscore-csl` |

```python
from silverscore import list_variants
print(list_variants())
```

## Feature Extraction

SiLVERScore expects pre-extracted I3D features (1024-d per clip). Features are extracted using an I3D model with sliding windows of 16 frames:

1. **Domain-agnostic features**: Extracted with a BSL-1K pretrained I3D.
2. **Domain-aware features**: Extracted with a domain-finetuned I3D.

The two feature sets are blended using a weighted combination controlled by the `alpha` parameter.

## How It Works

SiLVERScore uses a CLIP4Clip model with FILIP-style fine-grained token-level similarity:

1. **Text encoding**: Text is tokenized with a CLIP BPE tokenizer and encoded through the CLIP text transformer.
2. **Video encoding**: Pre-extracted I3D features (1024-d) are projected to the CLIP embedding space via a FeatureTransformer (ViT-style).
3. **Similarity**: Token-level cross-modal similarity is computed using softmax-weighted aggregation over text and video tokens.

## Citation

If you use SiLVERScore in your research, please cite:

```bibtex
@inproceedings{imai-etal-2025-silverscore,
    title = "{S}i{LVERS}core: Semantically-Aware Embeddings for Sign Language Generation Evaluation",
    author = "Imai, Saki  and
      Inan, Mert  and
      Sicilia, Anthony B.  and
      Alikhani, Malihe",
    booktitle = "Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing - Natural Language Processing in the Generative AI Era",
    month = sep,
    year = "2025",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2025.ranlp-1.54/",
    pages = "452--461",
}
```
