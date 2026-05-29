---
title: SiLVERScore Demo
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Evaluate sign language video ↔ text semantic similarity
tags:
  - sign-language
  - evaluation
  - clip
  - video-text-retrieval
  - nlp
---

# 🤟 SiLVERScore Demo

**SiLVERScore** (Sign Language Video Embedding Representation Score) is a
reference-free evaluation metric for sign language generation. It measures
the semantic alignment between a sign language video and a text description
using CLIP-based cross-modal embeddings.

## How to Use

1. **Upload** a sign language video (`.mp4`, `.avi`, etc.)
2. **Enter** the text description you want to compare against
3. **Select** the sign language variant that matches your video
4. *(Optional)* Upload pre-trained model checkpoints for best accuracy
5. Click **Compute SiLVERScore**

## Model Checkpoints

SiLVERScore uses pre-trained [CiCo / CLCL](https://github.com/FangyunWei/SLRT/tree/main/CiCo)
checkpoints (Cheng et al., CVPR 2023). Download the appropriate checkpoint
and upload it in the **Model Checkpoints** accordion:

| Variant | Checkpoint file |
|---------|----------------|
| Phoenix-2014T (DGS) | `ph_sota.pth` |
| How2Sign (ASL) | `h2s_sota.pth` |
| CSL-Daily (CSL) | `csl_sota.pth` |

Without a checkpoint, the demo runs with base CLIP weights (results may differ
from published scores).

Scores shown in the app are scaled by **×3.5** to align with the paper's
analysis convention (roughly 0–100 style reporting).

## Citation

If you use SiLVERScore in your research, please cite:

```bibtex
@inproceedings{imai-etal-2025-silverscore,
    title = "{S}i{LVERS}core: Semantically-Aware Embeddings for Sign Language Generation Evaluation",
    author = "Imai, Saki and Inan, Mert and Sicilia, Anthony B. and Alikhani, Malihe",
    booktitle = "Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2025",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2025.ranlp-1.54/",
    pages = "452--461",
}
```

Also cite CiCo if you use their checkpoints:

```bibtex
@inproceedings{cheng2023cico,
    title     = {CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning},
    author    = {Yiting Cheng and Fangyun Wei and Jianmin Bao and Dong Chen and Wenqiang Zhang},
    booktitle = {CVPR},
    year      = {2023},
}
```
