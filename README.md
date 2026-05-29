# SiLVERScore

**Si**gn **L**anguage **V**ideo **E**mbedding **Re**presentation **Score** — an evaluation metric for sign language generation based on CLIP-based embedding similarity between video and text.

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

# Initialize with a sign language variant and a local checkpoint
scorer = SiLVERScore(variant="phoenix", model_path="/path/to/ph_sota.pth")

# Score a video file directly (I3D features are extracted automatically)
score = scorer.score("sign_video.mp4", text="a person signing hello")
print(f"SiLVERScore: {score:.4f}")

# Or pass pre-extracted I3D features
import numpy as np
features = np.load("video_features.npy")  # shape: (num_clips, 1024)
score = scorer.score(features, text="a person signing hello")
```

By default, returned scores are scaled by **×3.5** (to align with the paper-style
rough 0–100 range convention). To get raw logits instead:

```python
scorer = SiLVERScore(
    variant="phoenix",
    model_path="/path/to/ph_sota.pth",
    score_scale=1.0,
)
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

| Variant | Language | Dataset |
|---------|----------|---------|
| `phoenix` | German Sign Language (DGS) | PHOENIX-2014T |
| `how2sign` | American Sign Language (ASL) | How2Sign |
| `csl` | Chinese Sign Language (CSL-Daily) | CSL-Daily |

```python
from silverscore import list_variants
print(list_variants())
```

### Providing model weights

SiLVERScore requires CLCL checkpoints trained by [Cheng et al. (CiCo, CVPR 2023)](https://github.com/FangyunWei/SLRT/tree/main/CiCo).
Download the pre-trained checkpoints from the links in their repository and pass the path explicitly:

```python
scorer = SiLVERScore(variant="phoenix", model_path="/path/to/ph_sota.pth")
scorer = SiLVERScore(variant="how2sign", model_path="/path/to/H2S_sota.pth")
scorer = SiLVERScore(variant="csl",     model_path="/path/to/csl_sota.pth")
```

## Feature Extraction

SiLVERScore expects pre-extracted I3D features (1024-d per clip). Features are extracted using an I3D model with sliding windows of 16 frames:

1. **Domain-agnostic features**: Extracted with a BSL-1K pretrained I3D ([Albanie et al., ECCV 2020](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/)).
2. **Domain-aware features**: Extracted with CiCo's domain-finetuned I3D encoder.

The two feature sets are blended using a weighted combination controlled by the `alpha` parameter (variant-specific default; e.g. `alpha=0.9` for Phoenix, `alpha=0.8` for How2Sign and CSL).

## How It Works

SiLVERScore adapts the [CiCo / CLCL](https://github.com/FangyunWei/SLRT/tree/main/CiCo) architecture for use as a reference-free evaluation metric:

1. **Text encoding**: Text is tokenized with a CLIP BPE tokenizer and encoded through the CLIP text transformer.
2. **Video encoding**: Pre-extracted I3D features (1024-d) are projected to the CLIP embedding space via a FeatureTransformer (ViT-style).
3. **Similarity**: Token-level cross-modal similarity is computed using FILIP-style softmax-weighted aggregation over text and video tokens (I2T and T2I), blended via `dual_mix`.

## Acknowledgements

SiLVERScore is built on top of two prior works whose code and models we gratefully acknowledge:

- **CiCo** — [Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://github.com/FangyunWei/SLRT/tree/main/CiCo) (Cheng et al., CVPR 2023). The CLCL model architecture and pre-trained checkpoints used by SiLVERScore originate from this work. Please also cite CiCo if you use their checkpoints.
- **BSL-1K / bslattend** — The domain-agnostic I3D encoder pre-trained on large-scale British Sign Language videos ([Albanie et al., ECCV 2020](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/)).

```bibtex
@inproceedings{cheng2023cico,
    title     = {CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning},
    author    = {Yiting Cheng and Fangyun Wei and Jianmin Bao and Dong Chen and Wenqiang Zhang},
    booktitle = {CVPR},
    year      = {2023},
}
```

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

## Paper Reproducibility Notebook

To reproduce the paper-style analysis plots/tables, use:

- `notebooks/reproduce_paper_figures.ipynb`

It recreates:

- Figure 2-style KDE (correct vs random, PHOENIX-14T)
- Figure 3-style KDE (reordered hypotheses, PHOENIX-14T)
- Table 1 overlap/AUC values (PHOENIX-14T and CSL-Daily)

Data location:

- Paper CSVs are included in this repo under `notebooks/paper_data/` for out-of-the-box reproduction.
- The notebook also supports `../signscore` if you run it in a monorepo setup.

Run:

```bash
cd silverscore/notebooks
jupyter notebook reproduce_paper_figures.ipynb
```
