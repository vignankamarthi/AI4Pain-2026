"""Tests for framework.seeds. Spec: FRAMEWORK.md Section 9 decision 4."""
import pytest
from framework import seeds


def test_module_imports():
    assert callable(seeds.default_seed_specs)
    assert callable(seeds.diversify_population)


def test_default_seed_specs_returns_six_seeds():
    specs = seeds.default_seed_specs()
    assert len(specs) == 6


def test_default_seed_families_match_decision():
    """Decision 4: 1d_cnn, bigru, transformer, xgb, lightgbm, multi_stream_bigru."""
    specs = seeds.default_seed_specs()
    families = {s["model"]["family"] for s in specs}
    expected = {"1d_cnn", "bigru", "transformer", "xgb", "lightgbm",
                "multi_stream_bigru"}
    assert families == expected


def test_each_seed_has_required_top_level_keys():
    for spec in seeds.default_seed_specs():
        for key in ("name", "preprocessing", "feature_extraction", "model",
                    "training", "decode"):
            assert key in spec


def test_catch22_seeds_use_feature_extraction():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    assert specs["seed_catch22_xgb"]["feature_extraction"]["family"] == "catch22"
    assert specs["seed_catch22_lightgbm"]["feature_extraction"]["family"] == "catch22"


def test_neural_seeds_have_no_feature_extraction():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    for name in ("seed_bigru", "seed_1d_cnn_resnet",
                 "seed_lightweight_transformer", "seed_multi_stream_bigru"):
        assert specs[name]["feature_extraction"] is None


# ---------- diversify_population ----------

def test_diversify_population_returns_one_list_per_island():
    specs = seeds.default_seed_specs()
    distributed = seeds.diversify_population(specs, island_count=8)
    assert len(distributed) == 8


def test_diversify_population_each_island_nonempty():
    specs = seeds.default_seed_specs()
    distributed = seeds.diversify_population(specs, island_count=8)
    for island in distributed:
        assert len(island) >= 1


def test_diversify_population_rejects_invalid_island_count():
    with pytest.raises(ValueError):
        seeds.diversify_population(seeds.default_seed_specs(), island_count=0)


def test_diversify_population_handles_more_islands_than_seeds():
    specs = seeds.default_seed_specs()  # 6 seeds
    distributed = seeds.diversify_population(specs, island_count=10)
    assert len(distributed) == 10


def test_diversify_population_handles_fewer_islands_than_seeds():
    specs = seeds.default_seed_specs()  # 6 seeds
    distributed = seeds.diversify_population(specs, island_count=3)
    # 6 seeds round-robin across 3 islands -> 2 seeds each.
    assert len(distributed) == 3
    total = sum(len(i) for i in distributed)
    assert total == 6


def test_diversify_population_empty_seeds():
    distributed = seeds.diversify_population([], island_count=3)
    assert len(distributed) == 3
    assert all(len(i) == 0 for i in distributed)
