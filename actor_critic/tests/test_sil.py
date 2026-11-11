import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from actor_critic.sil import (GoldenMemory, compute_sil_loss, compute_sil_loss_batched, gm_cap_probs, sil_clipped_step, sil_full_cutoff,
                                sil_mc_returns, sil_policy_kl, sil_separate_step)

KW = dict(conf_prob=0.9, conf_heads=2, adv_max_sigma=3.0, conf_sigma=0.05)


def seg(n=6, score=1, seed=0):
    g = np.random.RandomState(seed)
    r = np.zeros(n, dtype=np.float32)
    r[-1] = float(score)
    d = np.zeros(n, dtype=np.float32)
    d[-1] = 1.0
    return {"temporal": g.randn(n, 3).astype(np.float32), "state": g.randn(n, 2).astype(np.float32), "bin_indices": g.randint(0, 4, (n, 2)),
            "rewards": r, "dones": d}


class Head(nn.Module):
    head_type = "categorical"
    recurrent = False

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(5, 2 * 4 + 1)

    def forward(self, obs):
        return self.net(torch.cat([obs["temporal"], obs["state"]], dim=-1))


class Pop(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.tensor(0.5))
        self.register_buffer("sigma", torch.tensor(2.0))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.popart = Pop()


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.actor, self.critic = Head(), Critic()

    def evaluate(self, obs, indices):
        out = self.actor(obs)
        logits, v = out[:, :8].reshape(-1, 2, 4), out[:, 8:]
        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(-1, indices.unsqueeze(-1)).squeeze(-1).sum(-1)
        return lp, None, -(logp.exp() * logp).sum(-1), v * 2.0 + 0.5, v, None, logits


class Memory(unittest.TestCase):
    def gm(self, **kw):
        return GoldenMemory(3, 2, base=4.0, max_p=0.8, **kw)

    def test_eviction_prefers_low_score_and_rejects_worse(self):
        gm = self.gm()
        for s in (3, 1, 2):
            self.assertTrue(gm.add(seg(), 1.0, s))
        self.assertFalse(gm.add(seg(), 1.0, 1))
        self.assertTrue(gm.add(seg(), 1.0, 5))
        self.assertEqual(sorted(gm.score[:3].tolist()), [2, 3, 5])
        self.assertEqual(gm.pop_events()["evicted"], 1)

    def test_group_outranks_score_and_tie_breaks(self):
        gm = self.gm()
        gm.add(seg(), 1.0, 1, 1)
        gm.add(seg(), 1.0, 9, 0)
        gm.add(seg(), 1.0, 5, 0)
        self.assertTrue(gm.add(seg(), 1.0, 1, 2))
        self.assertEqual(sorted(gm.group[:3].tolist()), [0, 1, 2])
        self.assertEqual(sorted(gm.score[:3].tolist()), [1, 1, 9])

    def test_used_segments_are_recycled_first_and_refund_restores_uses(self):
        gm = self.gm()
        for s in (3, 2, 1):
            gm.add(seg(), 1.0, s)
        np.random.seed(0)
        gm.sample(3)
        gm.sample(3)
        self.assertEqual(gm.uses[:3].tolist(), [2, 2, 2])
        self.assertTrue(gm.add(seg(), 1.0, 0))
        self.assertEqual((gm.recycled_consumed, gm.score[2]), (1, 0))
        gm.sample(3)
        gm.refund_last_sample()
        self.assertEqual((gm.uses_saved, gm.uses[:3].tolist()), (1, [2, 2, 0]))

    def test_floor_protects_best_per_group(self):
        class Floor(GoldenMemory):
            FLOOR = 1
        gm = Floor(2, 9, base=2.0, max_p=1.0)
        gm.add(seg(), 1.0, 1, 0)
        gm.add(seg(), 1.0, 1, 1)
        self.assertFalse(gm.add(seg(), 1.0, 0, 0))
        self.assertEqual(gm.size, 2)

    def test_validate_hook_counts_rejections(self):
        class Strict(GoldenMemory):
            def validate(self, seg, ret, score):
                return "no_score" if score < 1 else None
        gm = Strict(2, 2, base=2.0, max_p=1.0)
        self.assertFalse(gm.add(seg(), 1.0, 0))
        self.assertEqual((gm.rejected["no_score"], gm.pop_events()["rejected"]), (1, {"no_score": 1}))

    def test_expire_purge_and_sampling_stats(self):
        gm = GoldenMemory(4, 3, base=2.0, max_p=0.9, max_age=2)
        for i in range(3):
            gm.current_update = i
            gm.add(seg(), 1.0, i + 1)
        gm.current_update = 4
        self.assertEqual(gm.expire(), 2)
        np.random.seed(1)
        out = gm.sample(5)
        self.assertEqual((len(out), gm.last_valid_available), (1, 1))
        self.assertEqual(gm.purge(), 1)
        self.assertEqual(gm.sample(2), [])

    def test_unweighted_without_max_p_and_override_weights(self):
        class Flat(GoldenMemory):
            def weights(self, score):
                return np.ones(len(score))
        gm = Flat(5, 9)
        for s in range(5):
            gm.add(seg(), 1.0, s)
        np.random.seed(2)
        gm.draw(3)
        self.assertAlmostEqual(gm.last_weight_max, 0.2)

    def test_level_sampling_needs_level_probs_and_follows_them(self):
        gm = GoldenMemory(6, 9, base=2.0, max_p=0.9, sampling="level", num_levels=2)
        for i in range(6):
            gm.add(seg(), 1.0, i + 1, i % 2)
        with self.assertRaises(ValueError):
            gm.sample(2)
        np.random.seed(0)
        gm.sample(3, [1.0, 0.0])
        self.assertTrue((gm.last_sampled_group == 0).all())

    def test_cap_probs_respect_the_cap(self):
        p, capped, infeasible = gm_cap_probs(np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.4)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertLessEqual(float(p.max()), 0.4 + 1e-9)
        self.assertTrue(capped[0] and not infeasible)
        self.assertTrue(gm_cap_probs(np.ones(2), 0.2)[2])


class Loss(unittest.TestCase):
    def test_mc_returns_cut_at_dones(self):
        r, d = torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(sil_mc_returns(r, d, 0.5), torch.tensor([[2.0, 2.0, 5.0, 4.0]])))
        self.assertTrue(torch.allclose(sil_mc_returns(r, d, 0.5, torch.tensor([8.0]))[:, :2], torch.tensor([[2.0, 2.0]])))
        self.assertTrue(torch.allclose(sil_mc_returns(r, torch.zeros(1, 4), 0.5, torch.tensor([8.0]))[:, -1], torch.tensor([8.0])))

    def test_batched_matches_per_episode_and_cutoff_hook_truncates(self):
        segs = [seg(8, 5, 1), seg(6, 3, 2), seg(9, 4, 3)]
        a, b = Agent(), Agent()
        la, sa = compute_sil_loss(a, segs, "cpu", 0.9, sil_full_cutoff, **KW)
        lb, sb = compute_sil_loss_batched(b, segs, "cpu", 0.9, sil_full_cutoff, **KW)
        self.assertAlmostEqual(float(la.detach()), float(lb.detach()), places=5)
        self.assertEqual((sa["sil_used_steps"], sa["sil_cutoff_fallback"]), (23.0, 0.0))
        la.backward()
        lb.backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-5))
        half = lambda s, T: (T // 2, True)
        for fn in (compute_sil_loss, compute_sil_loss_batched):
            _, st = fn(Agent(), segs, "cpu", 0.9, half, **KW)
            self.assertEqual((st["sil_used_steps"], st["sil_cutoff_fallback"]), (4.0 + 3.0 + 4.0, 3.0))

    def test_empty_batch(self):
        loss, st = compute_sil_loss_batched(Agent(), [], "cpu", 0.9, sil_full_cutoff, **KW)
        self.assertEqual((float(loss), st), (0.0, {}))

    def test_kl_zero_for_identical_and_positive_otherwise(self):
        x = torch.randn(2, 3, 2, 4)
        valid = torch.ones(2, 3, dtype=torch.bool)
        self.assertAlmostEqual(sil_policy_kl(x, x, valid, False), 0.0, places=6)
        self.assertGreater(sil_policy_kl(x, torch.randn(2, 3, 2, 4), valid, False), 0.0)

    def test_separate_step_updates_and_kl_guard_rolls_back(self):
        segs = [seg(8, 5, 1), seg(6, 3, 2)]
        for max_kl, skipped in ((1e9, False), (1e-12, True)):
            agent = Agent()
            opt = torch.optim.Adam(agent.actor.parameters(), lr=1e-2)
            before = [p.detach().clone() for p in agent.actor.parameters()]
            st = sil_separate_step(agent, opt, GradScaler(enabled=False), lambda: compute_sil_loss_batched(agent, segs, "cpu", 0.9, sil_full_cutoff, keep_policy=True, **KW),
                                   1.0, max_kl, group_norms=lambda a: {"n": 1.0})
            self.assertEqual((st["sil_kl_skipped"], st["group_norms"]), (skipped, {"n": 1.0}))
            self.assertEqual(all(torch.equal(b, p) for b, p in zip(before, agent.actor.parameters())), skipped)
        with self.assertRaises(ValueError):
            sil_separate_step(agent, opt, GradScaler(enabled=False), lambda: compute_sil_loss_batched(agent, segs, "cpu", 0.9, sil_full_cutoff, **KW), 1.0, 1.0)

    def test_clipped_step_returns_norm_and_steps_only_when_finite(self):
        m = nn.Linear(2, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        w0 = m.weight.detach().clone()
        raw, norms = sil_clipped_step(opt, GradScaler(enabled=False), m(torch.ones(1, 2)).sum(), list(m.parameters()), 0.5, lambda: {"g": 2})
        self.assertGreater(raw, 0.0)
        self.assertEqual(norms, {"g": 2})
        self.assertFalse(torch.equal(w0, m.weight))
        w1 = m.weight.detach().clone()
        raw, _ = sil_clipped_step(opt, GradScaler(enabled=False), m(torch.full((1, 2), float("inf"))).sum(), list(m.parameters()), 0.5)
        self.assertFalse(np.isfinite(raw))
        self.assertTrue(torch.equal(w1, m.weight))


if __name__ == "__main__":
    unittest.main()
