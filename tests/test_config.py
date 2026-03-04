"""Basic tests for the config system."""

from silverscore.config import (
    get_task_config,
    list_variants,
    VARIANT_CONFIGS,
)


def test_list_variants():
    variants = list_variants()
    assert "phoenix" in variants
    assert "how2sign" in variants
    assert "csl" in variants


def test_get_task_config_defaults():
    cfg = get_task_config("phoenix")
    assert cfg.alpha == 0.9
    assert cfg.sim_header == "Filip"
    assert cfg.feature_len == 64
    assert cfg.local_rank == 0


def test_get_task_config_override():
    cfg = get_task_config("phoenix", alpha=0.5, feature_len=128)
    assert cfg.alpha == 0.5
    assert cfg.feature_len == 128


def test_all_variants_have_required_keys():
    required = {
        "hub_model_id", "checkpoint_filename", "alpha",
        "feature_len", "sim_header", "pretrained_clip_name",
    }
    for name, cfg in VARIANT_CONFIGS.items():
        missing = required - set(cfg.keys())
        assert not missing, f"Variant '{name}' missing keys: {missing}"
