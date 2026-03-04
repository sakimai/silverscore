"""
Pre-defined configurations for each sign language variant.
Each variant specifies the HuggingFace Hub model ID, feature paths,
and model hyperparameters used during training.
"""

from argparse import Namespace


VARIANT_CONFIGS = {
    "phoenix": {
        "hub_model_id": "sakimai/silverscore-phoenix",
        "checkpoint_filename": "ph_sota.pth",
        "description": "Phoenix-2014T (German Sign Language)",
        "alpha": 0.9,
        "feature_len": 64,
        "visual_num_hidden_layers": 12,
        "sim_header": "Filip",
        "pretrained_clip_name": "ViT-B/32",
        "cross_num_hidden_layers": 4,
        "linear_patch": "2d",
        "loose_type": True,
        "dual_mix": 0.5,
        "mix_design": "balance",
        "max_words": 32,
    },
    "how2sign": {
        "hub_model_id": "sakimai/silverscore-how2sign",
        "checkpoint_filename": "h2s_sota.pth",
        "description": "How2Sign (American Sign Language)",
        "alpha": 0.8,
        "feature_len": 64,
        "visual_num_hidden_layers": 12,
        "sim_header": "Filip",
        "pretrained_clip_name": "ViT-B/32",
        "cross_num_hidden_layers": 4,
        "linear_patch": "2d",
        "loose_type": True,
        "dual_mix": 0.5,
        "mix_design": "balance",
        "max_words": 32,
    },
    "csl": {
        "hub_model_id": "sakimai/silverscore-csl",
        "checkpoint_filename": "csl_sota.pth",
        "description": "CSL-Daily (Chinese Sign Language)",
        "alpha": 0.8,
        "feature_len": 64,
        "visual_num_hidden_layers": 12,
        "sim_header": "Filip",
        "pretrained_clip_name": "ViT-B/32",
        "cross_num_hidden_layers": 4,
        "linear_patch": "2d",
        "loose_type": True,
        "dual_mix": 0.5,
        "mix_design": "balance",
        "max_words": 32,
    },
}


def get_task_config(variant: str, **overrides) -> Namespace:
    """Build a task_config Namespace for the given variant."""
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available: {list(VARIANT_CONFIGS.keys())}"
        )
    cfg = VARIANT_CONFIGS[variant].copy()
    cfg.update(overrides)
    cfg.setdefault("local_rank", 0)
    cfg.setdefault("not_load_visual", False)
    cfg.setdefault("aug_choose", "t2v")
    return Namespace(**cfg)


def list_variants():
    """Return available variant names and descriptions."""
    return {k: v["description"] for k, v in VARIANT_CONFIGS.items()}
