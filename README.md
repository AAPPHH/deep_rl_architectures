# actor-critic

Shared actor-critic code: IMPALA (V-trace) and DreamerV3 with self-imitation (SIL) and golden memory. Projects keep only env, reward, curriculum and feature adapters.

- `actor_critic/dreamer_core.py`: DreamerV3 primitives (twohot, symlog, RSSM, ReturnNorm, Replay, LaProp, clip_agc)
- `actor_critic/dreamer.py`: WorldModel, Dreamer, generic training loop
- `actor_critic/impala.py`: V-trace, CAPS losses, entropy controllers, PopArt, RunningMeanStd, impala_targets and impala_loss (optional masks, PopArtAdapter), Ray actor-learner stream plumbing (ray is imported lazily; the `ray` extra is needed only to run the stream)
- `actor_critic/sil.py`: self-imitation: GoldenMemory (opaque score/group/tie priority, hooks validate, tiebreak, store, weights; global/level/strict capped sampling, expire, purge, refund), compute_sil_loss and compute_sil_loss_batched with a cutoff hook, sil_separate_step with KL guard, sil_clipped_step, sil_mc_returns

Consumers: any project that keeps its own env, reward and feature adapters.

Install a released version by tag, without letting pip touch the torch or numpy of the environment:

    pip install --no-deps "actor-critic @ git+https://github.com/AAPPHH/actor-critic@v0.1.0"

Editable installs are for a dev environment only:

    pip install --no-deps -e .

Tests: `python -m unittest actor_critic.tests.test_actor_critic actor_critic.tests.test_sil`

Changelog

- 0.1.0: IMPALA and DreamerV3 core, SIL and golden memory
