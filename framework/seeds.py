"""Multi-seed initialization.

Returns the 6 seed program specs locked in FRAMEWORK.md Section 9 decision 4:

  1. 1D-CNN ResNet-style
  2. BiGRU baseline (existing ai4pain.baselines)
  3. Lightweight Transformer encoder
  4. Catch22 + XGBoost
  5. Catch22 + LightGBM
  6. Multi-stream BiGRU (per-channel encoder + late fusion)

NOTE: Only the bigru family is currently runnable end-to-end (its render
entry point lives in framework.render.FAMILY_ENTRY_POINTS). The other 5 specs
will be rejected by render.render_spec_to_code until their entry points are
added. That's intentional: the search starts narrow and widens as we
implement more model families.

`diversify_population` distributes seeds across N islands, filling islands
with fewer seeds by replicating the assigned seed (the loop's first round of
mutation will diversify them in-place).

Gated by HIP-C: Vignan reviews and approves the list before the first iteration.
"""


def default_seed_specs() -> list[dict]:
    """The 6 locked seed specs per FRAMEWORK.md Section 9 decision 4."""
    return [
        {
            "name": "seed_1d_cnn_resnet",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "1d_cnn", "depth": 4,
                      "base_channels": 32, "kernel_size": 7,
                      "use_residual": True},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_bigru",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "bigru", "hidden_size": 64,
                      "num_layers": 1, "dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_lightweight_transformer",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "transformer", "d_model": 64,
                      "num_heads": 4, "num_layers": 2, "ff_dim": 128,
                      "dropout": 0.1},
            "training": {"loss": "ce_class_balanced", "optimizer": "adamw",
                         "lr": 5e-4, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_catch22_xgb",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": {"family": "catch22", "per_channel": True},
            "model": {"family": "xgb", "n_estimators": 500,
                      "max_depth": 6, "learning_rate": 0.05,
                      "subsample": 0.8},
            "training": {"loss": "logloss_class_balanced",
                         "early_stopping_rounds": 30, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_catch22_lightgbm",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": {"family": "catch22", "per_channel": True},
            "model": {"family": "lightgbm", "n_estimators": 500,
                      "num_leaves": 64, "learning_rate": 0.05,
                      "feature_fraction": 0.8, "class_weight": "balanced"},
            "training": {"loss": "multi_logloss",
                         "early_stopping_rounds": 30, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_multi_stream_bigru",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "multi_stream_bigru",
                      "per_channel_hidden": 32, "per_channel_layers": 1,
                      "fusion": "late_concat", "fusion_dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
    ]


def diversify_population(seed_specs: list[dict],
                          island_count: int) -> list[list[dict]]:
    """Distribute the seeds across `island_count` islands.

    If island_count >= len(seed_specs): one seed per island, remaining islands
    get a copy of a randomly cycled seed.
    If island_count < len(seed_specs): pack multiple seeds per island in a
    round-robin.

    Returns a list of length island_count; each element is a list of seed
    spec dicts assigned to that island.
    """
    if island_count < 1:
        raise ValueError(f"island_count must be >= 1, got {island_count}")
    if not seed_specs:
        return [[] for _ in range(island_count)]

    islands: list[list[dict]] = [[] for _ in range(island_count)]
    for i, spec in enumerate(seed_specs):
        islands[i % island_count].append(dict(spec))

    # Fill empty islands by copying from the most-populated ones (deterministic).
    n_seeds = len(seed_specs)
    if island_count > n_seeds:
        for j in range(n_seeds, island_count):
            source_idx = j % n_seeds
            islands[j].append(dict(seed_specs[source_idx]))

    return islands
