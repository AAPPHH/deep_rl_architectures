import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deep_rl_architectures import dreamer as DR
from deep_rl_architectures.tests.test_deep_rl_architectures import A, B, CFG, O, P, T, batch

GOLDEN = Path(__file__).with_name("golden_continuous.pt")


class ToyVec:
    def __init__(self, task, n, seed, discrete, ctx=2, length=8, arms=None):
        self.n, self.discrete, self.ctx, self.length = n, discrete, ctx, length
        self.obs_dim, self.act_dim, self.priv_dim = ctx, ((arms or ctx) if discrete else 1), 1
        self.rng = np.random.default_rng(seed)
        self.c, self.t = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)

    def obs(self, i):
        o = np.zeros(self.ctx, np.float32)
        o[self.c[i]] = 1.0
        return o, np.zeros(1, np.float32)

    def reset(self, i):
        self.c[i], self.t[i] = self.rng.integers(0, self.ctx), 0
        return self.obs(i)

    def step(self, act):
        r, tr = np.zeros(self.n, np.float32), np.zeros(self.n, bool)
        for i in range(self.n):
            r[i] = float(np.argmax(act[i]) == self.c[i] % self.act_dim) if self.discrete else float((act[i, 0] > 0) == (self.c[i] == 1))
            self.c[i] = self.rng.integers(0, self.ctx)
            self.t[i] += 1
            tr[i] = self.t[i] >= self.length
        o, p = zip(*[self.obs(i) for i in range(self.n)])
        return np.stack(o), np.stack(p), r, np.zeros(self.n, bool), tr


def flat(sd):
    return torch.cat([v.detach().float().flatten() for v in sd.values()])


def metrics(m):
    return {k: torch.as_tensor(v).detach().float().flatten() for k, v in m.items()}


def continuous_trace():
    torch.set_num_threads(1)
    out = {}
    variants = {"base": lambda ag: None, "adv": lambda ag: ag.configure_switches(adv_norm="adv", imag_prior="mode"),
                "mix": lambda ag: ag.configure_switches(actor_grad="mix", actor_grad_mix=0.5),
                "anchor": lambda ag: (ag.set_anchor(), ag.set_data_anchor({"coef": 0.5, "all_rows": True}), ag.configure_switches(target_nats=0.1))}
    for name, setup in variants.items():
        torch.manual_seed(0)
        ag = DR.Dreamer(dict(CFG), O, A, P)
        setup(ag)
        for i in range(3):
            for k, v in metrics(ag.update(batch(i))).items():
                out[f"{name}/m{i}/{k}"] = v
        out[f"{name}/params"] = flat(ag.state_dict())
        obs, first, prev = torch.randn(4, O), torch.ones(4), torch.zeros(4, A)
        for g in (False, True):
            a, (d, s) = ag.act(obs, first, prev, None, greedy=g)
            out[f"{name}/act{g}"], out[f"{name}/deter{g}"] = a, d
        ag.noise = DR.Noise(legacy=False)
        sd, ss = ag.rssm.initial(5)
        for legacy in (True, False):
            ag.noise = DR.Noise(legacy=legacy)
            f, a = ag.rollout(sd, ss)
            out[f"{name}/rollf{legacy}"], out[f"{name}/rolla{legacy}"] = f, a
    with tempfile.TemporaryDirectory() as d:
        cfg = {**CFG, "logdir": d, "envs": 2, "eval_episodes": 2, "steps": 120, "prefill": 40, "replay_cap": 400, "train_ratio": 64, "eval_every": 60, "log_every": 1, "batch": 2, "T": 8}
        DR.train(cfg, lambda task, n, seed: ToyVec(task, n, seed, False))
        out["train/latest"] = flat(torch.load(Path(d) / "latest.pt", weights_only=False)["model"])
        out["train/metrics"] = torch.tensor([v for r in map(json.loads, (Path(d) / "metrics.jsonl").read_text().splitlines()) for k, v in r.items() if isinstance(v, float) and k != "time"])
        out["train/eval"] = torch.tensor([json.loads(l)["ret"] for l in (Path(d) / "eval.jsonl").read_text().splitlines()])
    return out


class ContinuousBitIdentity(unittest.TestCase):
    def test_matches_golden_recorded_before_discrete_support(self):
        gold, now = torch.load(GOLDEN, weights_only=False), continuous_trace()
        self.assertEqual(set(gold), set(now))
        for k in gold:
            self.assertTrue(torch.equal(gold[k], now[k]), k)


DCFG = {**CFG, "action_space": "discrete"}
NA = 3


def dbatch(seed=0):
    b = batch(seed)
    b["act"] = torch.nn.functional.one_hot(torch.randint(0, NA, (B, T), generator=torch.Generator().manual_seed(seed)), NA).float()
    return b


class DiscreteActor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ag = DR.Dreamer(dict(DCFG), O, NA, P)

    def test_head_width_and_dist(self):
        self.assertEqual(self.ag.actor.out.out_features, NA)
        feat = torch.randn(5, 4, self.ag.actor.body[0].in_features)
        d = self.ag.actor_dist(feat)
        self.assertEqual(d.probs.shape, (5, 4, NA))
        self.assertTrue(torch.allclose(d.probs.sum(-1), torch.ones(5, 4)))
        self.assertGreaterEqual(float(d.probs.min().detach()), 0.01 / NA - 1e-9)
        a = d.sample()
        self.assertEqual(a.shape, (5, 4, NA))
        self.assertTrue(torch.equal(a.sum(-1), torch.ones(5, 4)))
        self.assertEqual(d.log_prob(a).shape, (5, 4))
        self.assertTrue(torch.allclose(d.log_prob(a), d.logp.gather(-1, a.argmax(-1, keepdim=True))[..., 0]))
        ref = torch.distributions.Categorical(probs=d.probs)
        self.assertTrue(torch.allclose(d.entropy(), ref.entropy(), atol=1e-6))
        self.assertTrue(torch.equal(d.mode.argmax(-1), d.probs.argmax(-1)))

    def test_act_sample_and_greedy_are_one_hot(self):
        obs, first, prev = torch.randn(4, O), torch.ones(4), torch.zeros(4, NA)
        for g in (False, True):
            a, _ = self.ag.act(obs, first, prev, None, greedy=g)
            self.assertTrue(torch.equal(a.sum(-1), torch.ones(4)) and set(a.unique().tolist()) <= {0.0, 1.0})

    def test_reinforce_gradient_sign(self):
        logit = torch.zeros(1, NA, requires_grad=True)
        d = DR.OneHotCat(logit, 0.0)
        a = torch.nn.functional.one_hot(torch.tensor([1]), NA).float()
        for adv, sign in ((1.0, 1.0), (-1.0, -1.0)):
            g, = torch.autograd.grad(-(d.log_prob(a) * adv).sum(), logit, retain_graph=True)
            self.assertLess(float(g[0, 1]) * sign, 0.0)
            self.assertTrue(float(g[0, 0]) * sign > 0 and float(g[0, 2]) * sign > 0)

    def test_update_finite_and_logs_entropy(self):
        for i in range(2):
            m = self.ag.update(dbatch(i))
        self.assertIsNotNone(m)
        for k in ("policy", "value", "recon", "dyn", "actor_entropy"):
            self.assertTrue(math.isfinite(float(m[k])), k)
        self.assertNotIn("actor_sigma_mean", m)
        self.assertLessEqual(float(m["actor_entropy"]), math.log(NA) + 1e-5)

    def test_rollout_noise_modes_one_hot(self):
        sd, ss = self.ag.rssm.initial(5)
        for legacy in (True, False):
            self.ag.noise = DR.Noise(legacy=legacy)
            f, a = self.ag.rollout(sd, ss)
            self.assertEqual(a.shape, (5, DCFG["horizon"] + 1, NA))
            self.assertTrue(torch.equal(a.sum(-1), torch.ones(5, DCFG["horizon"] + 1)))

    def test_continuous_only_features_rejected(self):
        for f in (lambda: self.ag.configure_switches(actor_grad="dynamics"), lambda: self.ag.configure_switches(actor_grad="mix"), self.ag.set_anchor,
                  lambda: self.ag.set_data_anchor({"coef": 1.0}), lambda: self.ag.set_sigma(0.3), lambda: self.ag.remap_std(0.1, 1.0)):
            with self.assertRaises(ValueError):
                f()
        self.ag.set_data_anchor(None)
        self.ag.configure_switches(adv_norm="adv")

    def test_bad_action_space(self):
        with self.assertRaises(ValueError):
            DR.Dreamer({**CFG, "action_space": "box"}, O, NA, P)
        with self.assertRaises(ValueError):
            DR.Dreamer(dict(DCFG), O, 1, P)

    def test_continuous_default_unchanged_head(self):
        self.assertEqual(DR.Dreamer(dict(CFG), O, A, P).actor.out.out_features, 2 * A)


class DiscreteTrain(unittest.TestCase):
    def test_learns_bandit_on_cpu_and_beats_random(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            cfg = {**DCFG, "logdir": d, "envs": 4, "eval_episodes": 8, "steps": 1200, "prefill": 400, "replay_cap": 4000, "train_ratio": 128, "eval_every": 300, "log_every": 50,
                   "batch": 8, "T": 16, "horizon": 6, "lr": 2e-3, "ent": 3e-3, "seed": 1, "warmup": 100}
            DR.train(cfg, lambda task, n, seed: ToyVec(task, n, seed, True, ctx=1, arms=NA))
            ev = [json.loads(l)["ret"] for l in (Path(d) / "eval.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(ev[-1], 7.0, ev)
        self.assertLess(ev[0], 5.0, ev)


if __name__ == "__main__":
    unittest.main()
