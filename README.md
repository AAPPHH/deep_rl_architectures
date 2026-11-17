# deep-rl-architectures

Shared deep RL code: IMPALA (V-trace) and DreamerV3 with self-imitation (SIL) and golden memory. Projects keep only env, reward, curriculum and feature adapters.

- `deep_rl_architectures/dreamer_core.py`: DreamerV3 primitives (twohot, symlog, RSSM, ReturnNorm, Replay, LaProp, clip_agc)
- `deep_rl_architectures/dreamer.py`: WorldModel, Dreamer, generic training loop
- `deep_rl_architectures/impala.py`: V-trace, CAPS losses, entropy controllers, PopArt, RunningMeanStd, impala_targets and impala_loss (optional masks, PopArtAdapter), Ray actor-learner stream plumbing (ray is imported lazily; the `ray` extra is needed only to run the stream)
- `deep_rl_architectures/sil.py`: self-imitation: GoldenMemory (opaque score/group/tie priority, hooks validate, tiebreak, store, weights; global/level/strict capped sampling, expire, purge, refund), compute_sil_loss and compute_sil_loss_batched with a cutoff hook, sil_separate_step with KL guard, sil_clipped_step, sil_mc_returns
- `deep_rl_architectures/lake.py`: data lake (extra `lake`: pyarrow, duckdb, pandas; import explicitly, not pulled in by the package): `Lake(root)` with runs (`lake.run(...)` context manager, streams as arrow IPC live files rolled to zstd parquet, run_meta.json, checkpoints), datasets (versioned parquet shards + meta.json, append-only registry.parquet, last row per name/version wins) and `lake.query(sql)` (duckdb, each stream across all runs as a view with a run_id column)

Consumers: any project that keeps its own env, reward and feature adapters.

Install a released version by tag, without letting pip touch the torch or numpy of the environment:

    pip install --no-deps "deep-rl-architectures @ git+https://github.com/AAPPHH/deep_rl_architectures@v0.2.0"

Editable installs are for a dev environment only:

    pip install --no-deps -e .

Tests: `python -m unittest deep_rl_architectures.tests.test_deep_rl_architectures deep_rl_architectures.tests.test_sil deep_rl_architectures.tests.test_lake`

Changelog

- 0.3.0: add lake module (runs, streams, datasets, registry, duckdb query)
- 0.2.0: package renamed to deep-rl-architectures (formerly actor-critic), no behaviour change
- 0.1.0: IMPALA and DreamerV3 core, SIL and golden memory
