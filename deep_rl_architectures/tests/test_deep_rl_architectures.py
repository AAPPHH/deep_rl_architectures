import copy
import math
import unittest
from queue import Full

import numpy as np
import torch

from deep_rl_architectures import dreamer_core as C
from deep_rl_architectures import dreamer as DR
from deep_rl_architectures import impala as IM

CFG = {
    "seed": 0, "device": "cpu", "cpus": 1, "repval_detach": True, "task": "synthetic", "batch": 3, "T": 6,
    "deter": 32, "hidden": 32, "units": 32, "stoch": 4, "discrete": 4, "layers": 2, "head_layers": 1, "unimix": 0.01,
    "bins": 255, "bin_lim": 20.0, "min_std": 0.1, "max_std": 1.0, "log_sigma_lim": (-5.0, 2.0),
    "free": 1.0, "horizon": 3, "lam": 0.95, "disc_horizon": 333, "ent": 3e-4,
    "lr": 1e-3, "agc": 0.3, "pmin": 1e-3, "eps": 1e-20, "b1": 0.9, "b2": 0.999, "warmup": 1, "slow_frac": 0.02,
    "ret_decay": 0.99, "ret_lo": 0.05, "ret_hi": 0.95, "max_nonfinite": 1000,
    "scales": {"dyn": 0.5, "rep": 0.1, "recon": 1.0, "rew": 1.0, "con": 1.0, "priv": 1.0, "policy": 1.0, "value": 1.0, "repval": 0.3},
    "critic_priv": "none", "priv_stop_grad": False, "head_action": False, "priv_norm": "symlog", "fast": False, "graph": True, "graph_ac": False, "graph_strict": False, "amp": None,
    "actor_in": "latent", "gru_hidden": 128, "bc_into_wm": False, "bc_wm_scale": 1.0, "priv_aux": False, "priv_aux_scale": 5.0,
    "action_repeat": 1, "reward_ensemble": 0, "dyn_ensemble": 0, "pess_coef": 1.0, "dyn_pess_coef": 1.0, "trunc_pct": 0.0, "trunc_decay": 0.99, "ens_boot": 0.8, "ens_graph": False,
}
B, T, O, A, P = 3, 6, 2, 1, 3


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    first = torch.zeros(B, T, 1)
    first[:, 0] = 1
    term = torch.zeros(B, T, 1)
    term[1, 4] = 1
    return {"obs": torch.randn(B, T, O, generator=g), "priv": torch.randn(B, T, P, generator=g), "act": torch.rand(B, T, A, generator=g) * 2 - 1,
            "rew": torch.randn(B, T, 1, generator=g), "first": first, "term": term}


class VTrace(unittest.TestCase):
    def test_on_policy_no_done_matches_nstep_return(self):
        g = torch.Generator().manual_seed(0)
        Bn, Tn, gamma = 2, 5, 0.9
        lp = torch.randn(Bn, Tn, generator=g)
        rew, val, boot = torch.randn(Bn, Tn, generator=g), torch.randn(Bn, Tn, generator=g), torch.randn(Bn, generator=g)
        vs, adv, rho, diag = IM.compute_vtrace(lp, lp, rew, val, torch.zeros(Bn, Tn), boot, gamma)
        ref = torch.zeros(Bn, Tn)
        nxt = boot
        for t in reversed(range(Tn)):
            nxt = rew[:, t] + gamma * nxt
            ref[:, t] = nxt
        self.assertTrue(torch.allclose(vs.view(Bn, Tn), ref, atol=1e-5))
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertEqual(adv.shape, (Bn * Tn,))

    def test_done_cuts_bootstrap(self):
        lp = torch.zeros(1, 3)
        rew, val = torch.tensor([[1.0, 2.0, 3.0]]), torch.zeros(1, 3)
        dones = torch.tensor([[0.0, 1.0, 0.0]])
        vs, _, _, _ = IM.compute_vtrace(lp, lp, rew, val, dones, torch.tensor([10.0]), 0.5)
        self.assertAlmostEqual(float(vs[1]), 2.0, places=6)
        self.assertAlmostEqual(float(vs[0]), 1.0 + 0.5 * 2.0, places=6)

    def test_rho_clip(self):
        beh, cur = torch.zeros(1, 2), torch.full((1, 2), 3.0)
        _, _, rho, diag = IM.compute_vtrace(beh, cur, torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1), 0.9, rho_bar=1.5)
        self.assertAlmostEqual(rho, 1.5, places=6)
        self.assertGreater(float(diag["is_ratio"].min()), 1.5)

    def test_caps_identical_zero(self):
        lg = torch.randn(1, 4, 9).expand(6, 4, 9).reshape(6, 4, 9)
        loss, per, masked = IM.compute_caps_loss(lg, torch.zeros(6), 1, 6)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(masked, 0)


class ImpalaLoss(unittest.TestCase):
    CFG = {"gamma": 0.9, "rho_bar": 1.0, "c_bar": 1.0, "value_coef": 0.5, "alpha": 0.01}

    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.B, self.T = 2, 4
        n = self.B * self.T
        self.blp = torch.randn(self.B, self.T, generator=g) * 0.3
        self.rew = torch.randn(self.B, self.T, generator=g)
        self.dones = torch.zeros(self.B, self.T)
        self.dones[0, 2] = 1
        self.boot = torch.randn(self.B, generator=g)
        self.lp = (self.blp + torch.randn(self.B, self.T, generator=g) * 0.2).reshape(n).requires_grad_(True)
        self.ent = torch.rand(n, generator=g).requires_grad_(True)
        self.vn = torch.randn(n, generator=g).requires_grad_(True)

    def targets(self, popart):
        vals = (self.vn.detach() * (popart.sigma if popart is not None else 1.0) + (popart.mu if popart is not None else 0.0)).view(self.B, self.T)
        return IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, vals, self.dones, self.boot, popart, self.CFG)

    def test_no_popart_matches_naive_reference(self):
        tg = self.targets(None)
        out = IM.impala_loss(tg, self.lp, self.ent, self.vn, self.CFG)
        gamma, n = self.CFG["gamma"], self.B * self.T
        vals = self.vn.detach().view(self.B, self.T)
        vs = torch.zeros(self.B, self.T)
        adv = torch.zeros(self.B, self.T)
        for b in range(self.B):
            nxt_v, nxt_vs = float(self.boot[b]), float(self.boot[b])
            for t in reversed(range(self.T)):
                nd = 1.0 - float(self.dones[b, t])
                ratio = float(torch.exp(self.lp.detach().view(self.B, self.T)[b, t] - self.blp[b, t]))
                rho = c = min(ratio, 1.0)
                v = float(vals[b, t])
                r = float(self.rew[b, t])
                vs[b, t] = v + rho * (r + gamma * nd * nxt_v - v) + gamma * nd * c * (nxt_vs - nxt_v)
                adv[b, t] = rho * (r + gamma * nd * nxt_vs - v)
                nxt_v, nxt_vs = v, float(vs[b, t])
        adv = adv.reshape(n)
        self.assertTrue(torch.allclose(tg["vtargets"], vs.reshape(n), atol=1e-5))
        pol = -(adv * self.lp.detach()).mean()
        val = 0.5 * ((self.vn.detach() - vs.reshape(n)) ** 2).mean()
        ent = -0.01 * self.ent.detach().mean()
        self.assertTrue(torch.allclose(out["policy"], pol, atol=1e-5))
        self.assertTrue(torch.allclose(out["value"], val, atol=1e-5))
        self.assertTrue(torch.allclose(out["loss"], pol + 0.5 * val + ent, atol=1e-5))

    def test_mask_updates_popart_and_rho_on_masked_only(self):
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[0, 2:] = False
        full = self.targets(None)
        pa, pb = IM.PopArtLayer(3, 1), IM.PopArtLayer(3, 1)
        tg = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, pa, self.CFG, mask=mask)
        pb.update_stats(tg["vtargets"][mask.reshape(-1)])
        self.assertTrue(torch.equal(pa.mu, pb.mu) and torch.equal(pa.sigma, pb.sigma))
        none = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, self.CFG, mask=mask)
        rho = full["diag"]["is_ratio"].clamp(max=1.0)[mask].mean().item()
        self.assertEqual(none["avg_rho"], rho)
        keep = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, self.CFG, mask=torch.ones_like(mask))
        self.assertEqual(keep["avg_rho"], full["avg_rho"])
        with self.assertRaises(ValueError):
            IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, self.CFG, mask=mask.reshape(-1))

    def test_masked_loss_equals_loss_on_selected(self):
        mask = torch.ones(self.B * self.T, dtype=torch.bool)
        mask[[1, 5]] = False
        tg = self.targets(None)
        a = IM.impala_loss(tg, self.lp[mask], self.ent[mask], self.vn[mask], self.CFG, mask=mask)["loss"]
        b = IM.impala_loss({**tg, "advantages": tg["advantages"][mask], "normalized_targets": tg["normalized_targets"][mask]}, self.lp[mask], self.ent[mask], self.vn[mask], self.CFG)["loss"]
        self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp, self.ent, self.vn, self.CFG, mask=mask)
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp[mask], self.ent[mask], self.vn[mask], self.CFG, sl=slice(0, 2), mask=mask)

    def test_popart_adapter_uses_normalize_target(self):
        class Legacy:
            def __init__(self):
                self.sigma = torch.tensor([2.0])
                self.mu = torch.tensor([0.0])
                self.seen = None

            def update_stats(self, x):
                self.seen = x

            def normalize_target(self, x):
                return x / self.sigma

        lg = Legacy()
        tg = self.targets(lg)
        self.assertTrue(torch.equal(tg["normalized_targets"], tg["vtargets"] / 2.0))
        self.assertTrue(torch.equal(lg.seen, tg["vtargets"]))
        self.assertTrue(torch.equal(IM.PopArtAdapter(lg).sigma, lg.sigma))

    def test_popart_shapes_grads_and_state(self):
        pa = IM.PopArtLayer(3, 1)
        pa.mu.fill_(0.5)
        pa.sigma.fill_(2.0)
        tg = self.targets(pa)
        self.assertEqual(tg["vtargets"].shape, (self.B * self.T,))
        self.assertEqual(tg["normalized_targets"].shape, (self.B * self.T,))
        self.assertNotEqual(float(pa.mu), 0.5)
        out = IM.impala_loss(tg, self.lp, self.ent, self.vn, {**self.CFG, "policy_coef": 0.1})
        out["loss"].backward()
        for t in (self.lp, self.ent, self.vn):
            self.assertTrue(torch.isfinite(t.grad).all())
        self.assertTrue(all(torch.isfinite(v) for v in out.values()))

    def test_chunks_sum_to_whole(self):
        tg = self.targets(None)
        whole = IM.impala_loss(tg, self.lp, self.ent, self.vn, self.CFG)["loss"]
        h = self.B * self.T // 2
        parts = sum(IM.impala_loss(tg, self.lp[s], self.ent[s], self.vn[s], self.CFG, sl=s, weight=0.5)["loss"] for s in (slice(0, h), slice(h, None)))
        self.assertTrue(torch.allclose(whole, parts, atol=1e-6))


class ImpalaFlags(unittest.TestCase):
    CFG = ImpalaLoss.CFG
    setUp = ImpalaLoss.setUp
    targets = ImpalaLoss.targets

    def test_log_rho_clip_none_is_bitwise_default(self):
        a = IM.compute_vtrace(self.blp, self.lp.detach().view(self.B, self.T), self.rew, torch.zeros(self.B, self.T), self.dones, self.boot, 0.9)
        b = IM.compute_vtrace(self.blp, self.lp.detach().view(self.B, self.T), self.rew, torch.zeros(self.B, self.T), self.dones, self.boot, 0.9, log_rho_clip=None)
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_log_rho_clip_clamps_before_exp(self):
        beh, cur = torch.zeros(1, 2), torch.tensor([[50.0, -50.0]])
        args = (beh, cur, torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1), 0.9)
        _, _, _, diag = IM.compute_vtrace(*args, rho_bar=1e9, c_bar=1e9, log_rho_clip=2.0)
        self.assertTrue(torch.allclose(diag["is_ratio"], torch.exp(torch.tensor([[2.0, -2.0]]))))
        _, _, _, diag = IM.compute_vtrace(*args, rho_bar=1e9, c_bar=1e9)
        self.assertGreater(float(diag["is_ratio"].max()), 1e20)

    def test_targets_defaults_for_bars(self):
        cfg = {"gamma": 0.9, "value_coef": 0.5, "alpha": 0.01}
        a = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, cfg)
        b = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, self.CFG)
        self.assertTrue(torch.equal(a["advantages"], b["advantages"]) and torch.equal(a["vtargets"], b["vtargets"]))

    def test_log_rho_clip_through_cfg(self):
        cfg = {**self.CFG, "rho_bar": 1e9, "c_bar": 1e9, "log_rho_clip": 0.01}
        tg = IM.impala_targets(self.blp, self.lp.detach().view(self.B, self.T), self.rew, self.vn.detach().view(self.B, self.T), self.dones, self.boot, None, cfg)
        self.assertLessEqual(float(tg["diag"]["is_ratio"].max()), math.exp(0.01) + 1e-6)

    def test_adv_sigma_norm(self):
        pa = IM.PopArtLayer(3, 1)
        pa.sigma.fill_(4.0)
        tg = self.targets(pa)
        on = IM.impala_loss(tg, self.lp, self.ent, self.vn, self.CFG)["policy"]
        off = IM.impala_loss(tg, self.lp, self.ent, self.vn, {**self.CFG, "adv_sigma_norm": False})["policy"]
        raw = -(tg["advantages"] * self.lp).mean()
        self.assertTrue(torch.allclose(off, raw, atol=1e-6))
        self.assertTrue(torch.allclose(on, raw / (tg["sigma"] + 1e-8), atol=1e-6))
        self.assertFalse(torch.allclose(on, off))

    def test_value_loss_form(self):
        tg = self.targets(None)
        half = IM.impala_loss(tg, self.lp, self.ent, self.vn, self.CFG)["value"]
        mse = IM.impala_loss(tg, self.lp, self.ent, self.vn, {**self.CFG, "value_loss_form": "mse"})["value"]
        self.assertTrue(torch.allclose(mse, torch.nn.functional.mse_loss(self.vn, tg["normalized_targets"]), atol=1e-7))
        self.assertTrue(torch.allclose(2 * half, mse, atol=1e-6))
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp, self.ent, self.vn, {**self.CFG, "value_loss_form": "huber"})

    def test_shape_asserts(self):
        tg = self.targets(None)
        n = self.B * self.T
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp, self.ent, self.vn.view(n, 1), self.CFG)
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp.view(1, n), self.ent, self.vn, self.CFG)
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp, self.ent, self.vn[:-1], self.CFG)
        with self.assertRaises(ValueError):
            IM.impala_loss(tg, self.lp[:-1], self.ent[:-1], self.vn[:-1], self.CFG)


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.task = None

    def set_weights(self, w):
        self.calls.append(("weights", w))

    def set_task(self, task):
        self.calls.append(("task", dict(task)))
        self.task = task

    def collect_trajectory(self):
        self.calls.append(("collect",))
        return {"task": self.task, "n": len(self.calls)}


class FakeQueue:
    def __init__(self, full_at=None):
        self.items = []
        self.full_at = full_at

    def put(self, item, timeout):
        if self.full_at is not None and len(self.items) >= self.full_at:
            raise Full
        self.items.append(item)


def draw_sched(sched, rng):
    p = sched["probs"]
    return {"pick": 0 if len(p) == 1 else int(rng.choice(len(p), p=p)), "scale": sched["scale"]}


class Stream(unittest.TestCase):
    def loop(self, store, q, seed=0, **kw):
        return IM.StreamLoop(store.poll, q.put, lambda w: copy.deepcopy(w[0]), np.random.RandomState(seed), draw_sched, sleep=lambda s: None, **kw)

    def test_store_publish_and_poll(self):
        s = IM.ParamStore()
        self.assertEqual(s.poll(-1, -1), {"stop": False})
        self.assertEqual(s.publish(0, ["w0"], {"probs": [0.5, 0.5], "scale": 0.1}), 0)
        self.assertEqual(s.poll(-1, -1), {"stop": False, "version": 0, "weights": ["w0"], "sched_version": 0, "sched": {"probs": [0.5, 0.5], "scale": 0.1}})
        s.publish(0, ["w0"], {"probs": [0.5, 0.5], "scale": 0.1})
        self.assertEqual(s.poll(0, 0), {"stop": False})
        s.publish(1, ["w1"], {"probs": [0.25, 0.75], "scale": 0.1})
        self.assertEqual(s.poll(0, 0), {"stop": False, "version": 1, "weights": ["w1"], "sched_version": 1, "sched": {"probs": [0.25, 0.75], "scale": 0.1}})
        s.stop()
        self.assertTrue(s.poll(1, 1)["stop"])

    def test_publish_coerces_array_sched_to_lists(self):
        s = IM.ParamStore()
        s.publish(0, ["w"], {"probs": np.array([0.5, 0.5]), "scale": torch.tensor([0.1]), "n": np.float64(2.0)})
        self.assertEqual(s.sched, {"probs": [0.5, 0.5], "scale": [float(torch.tensor([0.1]).item())], "n": 2.0})
        self.assertIsInstance(s.sched["probs"], list)
        s.publish(0, ["w"], {"probs": np.array([0.5, 0.5]), "scale": torch.tensor([0.1]), "n": np.float64(2.0)})
        self.assertEqual(s.sched_version, 0)
        s.publish(1, ["w"], {"probs": np.array([0.4, 0.6]), "scale": torch.tensor([0.1]), "n": np.float64(2.0)})
        self.assertEqual(s.sched_version, 1)

    def test_idle_until_published_then_task_weights_collect_order(self):
        s, q, w = IM.ParamStore(), FakeQueue(), FakeWorker()
        loop = self.loop(s, q)
        self.assertTrue(loop.step(w))
        self.assertEqual(w.calls, [])
        s.publish(3, [{"x": 1}], {"probs": [1.0], "scale": 0.5})
        self.assertTrue(loop.step(w))
        self.assertEqual([c[0] for c in w.calls], ["weights", "task", "collect"])
        self.assertEqual(w.calls[1][1], {"pick": 0, "scale": 0.5})
        meta, traj = q.items[0]
        self.assertEqual((meta["version"], meta["seq"], meta["task"]), (3, 0, {"pick": 0, "scale": 0.5}))
        self.assertEqual(traj["task"], meta["task"])
        self.assertTrue(loop.step(w))
        self.assertEqual([c[0] for c in w.calls[3:]], ["task", "collect"])
        self.assertEqual(q.items[1][0]["seq"], 1)

    def test_rng_draws_only_for_multi_choice_and_follow_sched_updates(self):
        s, q, w = IM.ParamStore(), FakeQueue(), FakeWorker()
        loop = self.loop(s, q, seed=5)
        ref = np.random.RandomState(5)
        s.publish(0, [0], {"probs": [1.0], "scale": 0.0})
        loop.step(w)
        s.publish(0, [0], {"probs": [0.2, 0.3, 0.5], "scale": 0.0})
        want = [int(ref.choice(3, p=[0.2, 0.3, 0.5])) for _ in range(20)]
        for _ in range(20):
            loop.step(w)
        self.assertEqual([m["task"]["pick"] for m, _ in q.items[1:]], want)
        self.assertEqual(loop.rng.random_sample(), ref.random_sample())

    def test_full_queue_retries_then_stop_ends_loop(self):
        s, q, w = IM.ParamStore(), FakeQueue(full_at=0), FakeWorker()
        s.publish(0, [0], {"probs": [1.0], "scale": 0.0})
        loop = self.loop(s, q, put_timeout=0.0)
        polls = []
        base = s.poll
        def poll(v, sv):
            polls.append(1)
            if len(polls) >= 3:
                s.stop()
            return base(v, sv)
        loop.poll = poll
        self.assertFalse(loop.step(w))
        self.assertEqual(loop.produced, 0)

    def test_feed_returns_task_and_version_and_idle_fraction(self):
        from queue import Queue
        q = Queue()
        for i in range(3):
            q.put(({"version": i, "task": {"k": i}, "seq": i, "collect_s": 1.0, "wait_s": 1.0}, {"i": i}))
        q.qsize = q.qsize
        q.get_nowait_batch = lambda n: [q.get_nowait() for _ in range(n)]
        feed = IM.StreamFeed(q, lambda: None, timeout=0.01)
        got = [feed.next() for _ in range(3)]
        self.assertEqual([(g[1], g[2]) for g in got], [({"k": i}, i) for i in range(3)])
        self.assertEqual(feed.backlog(), 0)
        self.assertAlmostEqual(feed.idle_frac(), 0.5)

    def test_module_imports_without_ray(self):
        import subprocess, sys
        code = "import sys; sys.modules['ray'] = None; import deep_rl_architectures.impala as m; m.check_stream_loops([], wait=lambda l, **k: ([], l), get=lambda r: 0)"
        self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True).returncode, 0)


class Numerics(unittest.TestCase):
    def test_symlog_roundtrip(self):
        x = torch.tensor([-1e6, -3.0, 0.0, 0.5, 40.0, 1e5])
        self.assertTrue(torch.allclose(C.symexp(C.symlog(x)), x, rtol=1e-5, atol=1e-6))

    def test_twohot_roundtrip(self):
        edges = C.bin_edges(255, 20.0)
        self.assertEqual(edges.numel(), 255)
        self.assertEqual(float(edges[127]), 0.0)
        x = torch.tensor([-500.0, -3.2, 0.0, 0.3, 17.0, 1234.5])
        th = C.twohot(x, edges)
        self.assertTrue(torch.allclose(th.sum(-1), torch.ones(6), atol=1e-6))
        self.assertTrue(((th > 0).sum(-1) <= 2).all())
        self.assertTrue(torch.allclose((th * edges).sum(-1), x, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(C.twohot_decode(torch.log(th + 1e-30), edges), x, rtol=1e-3, atol=1e-3))

    def test_lambda_return_lam1_is_discounted_sum(self):
        rew, boot, disc = torch.ones(2, 3), torch.full((2, 4), 5.0), torch.full((2, 3), 0.5)
        r = C.lambda_return(rew, boot, disc, 1.0)
        self.assertAlmostEqual(float(r[0, 0]), 1 + 0.5 * (1 + 0.5 * (1 + 0.5 * 5.0)), places=5)


class Components(unittest.TestCase):
    def test_rssm_observe_shapes_and_reset(self):
        torch.manual_seed(0)
        rssm = C.RSSM(CFG, 8, 1)
        emb, act, first = torch.randn(2, 5, 8), torch.rand(2, 5, 1), torch.zeros(2, 5)
        first[:, 0] = 1
        deter, stoch, pl, ql = rssm.observe(emb, act, first, mode=True)
        self.assertEqual(deter.shape, (2, 5, 32))
        self.assertEqual(stoch.shape, (2, 5, 4, 4))
        self.assertEqual(ql.shape, (2, 5, 16))
        self.assertTrue(torch.allclose(stoch.sum(-1), torch.ones(2, 5, 4)))
        first2 = first.clone()
        first2[:, 3] = 1
        d2 = rssm.observe(emb, act, first2, mode=True)[0]
        self.assertTrue(torch.equal(deter[:, :3], d2[:, :3]))
        self.assertFalse(torch.equal(deter[:, 3:], d2[:, 3:]))

    def test_return_norm(self):
        rn = C.ReturnNorm(0.99, 0.05, 0.95)
        x = torch.linspace(0, 10, 101)
        s = rn(x)
        self.assertAlmostEqual(float(s), 9.0, places=4)
        self.assertEqual(float(rn(torch.zeros(10))), max(1.0, float(rn.v[1] - rn.v[0])))
        self.assertGreaterEqual(float(rn(torch.zeros(10))), 1.0)

    def test_laprop_descends_quadratic(self):
        p = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
        opt = C.LaProp([p], lr=0.1, betas=(0.9, 0.99), eps=1e-8)
        for _ in range(200):
            p.grad = 2 * p.detach()
            opt.step()
        self.assertLess(float(p.detach().abs().max()), 0.3)

    def test_clip_agc(self):
        p = torch.nn.Parameter(torch.ones(4))
        p.grad = torch.full((4,), 100.0)
        C.clip_agc([p], 0.3, 1e-3)
        self.assertAlmostEqual(float(p.grad.norm()), 0.6, places=3)
        q = torch.nn.Parameter(torch.ones(4))
        q.grad = torch.full((4,), 0.01)
        C.clip_agc([q], 0.3, 1e-3)
        self.assertTrue(torch.equal(q.grad, torch.full((4,), 0.01)))

    def test_entropy_controller_moves_alpha(self):
        ctl = IM.EntropyController(target=1.0)
        a0 = ctl.alpha
        for _ in range(5):
            ctl.update(2.0)
        self.assertLess(ctl.alpha, a0)
        self.assertEqual(IM.FixedEntropy(0.01).update(5.0), 0.01)

    def test_popart_preserves_outputs(self):
        pa = IM.PopArtLayer(6, 1)
        x = torch.randn(4, 6)
        before = pa(x)
        pa.update_stats(torch.randn(16) * 10 + 3)
        self.assertTrue(torch.allclose(before, pa(x), atol=1e-4))


class TinyDreamer(unittest.TestCase):
    def test_update_finite_and_changes_params(self):
        torch.manual_seed(0)
        ag = DR.Dreamer(dict(CFG), O, A, P)
        p0 = {k: v.clone() for k, v in ag.named_parameters()}
        for i in range(3):
            m = ag.update(batch(i))
        self.assertIsNotNone(m)
        for k in ("policy", "value", "recon", "dyn"):
            self.assertTrue(math.isfinite(float(m[k])), k)
        self.assertTrue(any(not torch.equal(v, p0[k]) for k, v in ag.named_parameters()))

    def test_act_shape(self):
        torch.manual_seed(0)
        ag = DR.Dreamer(dict(CFG), O, A, P)
        a, state = ag.act(torch.randn(4, O), torch.ones(4), torch.zeros(4, A), None)
        self.assertEqual(a.shape, (4, A))
        self.assertTrue(((a >= -1) & (a <= 1)).all())

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        a, b = DR.Dreamer(dict(CFG), O, A, P), DR.Dreamer(dict(CFG), O, A, P)
        b.load_state_dict(a.state_dict())
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))


if __name__ == "__main__":
    unittest.main()
