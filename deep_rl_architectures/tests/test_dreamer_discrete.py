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
    def __init__(self, task, n, seed, discrete, ctx=2, length=8):
        self.n, self.discrete, self.ctx, self.length = n, discrete, ctx, length
        self.obs_dim, self.act_dim, self.priv_dim = ctx, (ctx if discrete else 1), 1
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
            r[i] = float(np.argmax(act[i]) == self.c[i]) if self.discrete else float((act[i, 0] > 0) == (self.c[i] == 1))
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


if __name__ == "__main__":
    unittest.main()
