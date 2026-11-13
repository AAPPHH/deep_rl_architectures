import contextlib
import copy
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

GM_SAMPLINGS = ("global", "level")
GM_CAP_MODES = ("renorm", "strict")


def gm_cap_probs(w, max_p):
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    p = w / w.sum()
    capped = p > max_p
    if n * max_p < 1.0:
        return np.full(n, 1.0 / n), capped, True
    fixed = np.zeros(n, dtype=bool)
    while True:
        over = ~fixed & (p > max_p * (1.0 + 1e-12))
        if not over.any():
            return p, capped, False
        fixed |= over
        free = ~fixed
        rest = max(1.0 - max_p * np.count_nonzero(fixed), 0.0)
        wf = w[free].sum()
        p = np.full(n, max_p)
        if free.any():
            p[free] = w[free] * (rest / wf) if wf > 0 else rest / np.count_nonzero(free)


class GoldenMemory:
    FLOOR = 0

    def __init__(self, capacity, max_uses, base=None, max_p=None, device="cpu", sampling="global", max_age=0, cap_mode="renorm",
                 num_levels=1, labels=None):
        if sampling not in GM_SAMPLINGS:
            raise ValueError(f"unknown golden memory sampling {sampling!r}, expected one of {list(GM_SAMPLINGS)}")
        if cap_mode not in GM_CAP_MODES:
            raise ValueError(f"unknown golden memory cap_mode {cap_mode!r}, expected one of {list(GM_CAP_MODES)}")
        self.num_levels = int(num_levels)
        self.labels = [f"L{i}" for i in range(self.num_levels)] if labels is None else list(labels)
        self.sampling = sampling
        self.max_age = int(max_age)
        self.cap_mode = cap_mode
        self.capacity = capacity
        self.max_uses = max_uses
        self.base = None if base is None else float(base)
        self.max_p = None if max_p is None else float(max_p)
        self.device = torch.device(device)
        self.buffer = [None] * capacity
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.score = np.zeros(capacity, dtype=np.float32)
        self.group = np.zeros(capacity, dtype=np.int32)
        self.uses = np.zeros(capacity, dtype=np.int32)
        self.added_at = np.zeros(capacity, dtype=np.int64)
        self.ep_lens = np.zeros(capacity, dtype=np.int32)
        self.tie = np.zeros(capacity, dtype=np.int32)
        self.size = 0
        self.current_update = 0
        self.purged = 0
        self.expired = 0
        self.recycled_consumed = 0
        self.uses_saved = 0
        self.rejected = Counter()
        self._ev_rejected = Counter()
        self._ev_add_attempts = 0
        self._ev_added = 0
        self._ev_evicted = 0
        self._ev_ev_score_sum = 0.0
        self._ev_ev_ret_sum = 0.0
        self._ev_ev_age_sum = 0.0
        self.reset_sample_stats()

    def reset_sample_stats(self):
        self.last_cap_infeasible = False
        self.last_valid_available = 0
        self.last_sampled_score = np.zeros(0, dtype=np.float32)
        self.last_sampled_lens = np.zeros(0, dtype=np.int32)
        self.last_sampled_idx = np.zeros(0, dtype=np.int64)
        self.last_sampled_group = np.zeros(0, dtype=np.int32)
        self.last_sampled_age = np.zeros(0, dtype=np.int64)
        self.last_weight_max = 0.0
        self.last_weight_entropy = 0.0
        self.last_capped_count = 0

    def validate(self, seg, ret, score):
        return None

    def tiebreak(self, seg):
        return 0

    def length(self, seg):
        return len(seg["rewards"])

    def store(self, seg):
        return seg

    def write(self, idx, seg):
        self.buffer[idx] = self.store(seg)

    def weights(self, score):
        return self.base ** (score.astype(np.float64) - float(score.max()))

    def _protected_mask(self):
        prot = np.zeros(self.size, dtype=bool)
        for lvl in np.unique(self.group[:self.size]):
            idxs = np.where(self.group[:self.size] == lvl)[0]
            if len(idxs) <= self.FLOOR:
                prot[idxs] = True
            else:
                order = np.lexsort((self.tie[idxs], -self.score[idxs]))
                prot[idxs[order[:self.FLOOR]]] = True
        return prot

    def _quality(self, idx):
        return (int(self.group[idx]), float(self.score[idx]), -float(self.tie[idx]))

    def _recycle(self, used):
        order = np.lexsort((-self.tie[used], self.score[used], self.group[used]))
        return int(used[order[0]])

    def add(self, seg, ret, score, group=0):
        self._ev_add_attempts += 1
        reason = self.validate(seg, ret, score)
        if reason:
            self.rejected[reason] += 1
            self._ev_rejected[reason] += 1
            return False
        tie = int(self.tiebreak(seg))
        new_q = (int(group), float(score), -float(tie))
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            used = np.where(self.uses[:self.size] >= self.max_uses)[0]
            if len(used) > 0:
                idx = self._recycle(used)
                self.recycled_consumed += 1
            else:
                prot = self._protected_mask()
                cand = np.where(~prot)[0]
                if len(cand) == 0:
                    return False
                order = np.lexsort((-self.tie[cand], self.score[cand], self.group[cand]))
                idx = int(cand[order[0]])
                if new_q <= self._quality(idx):
                    return False
            self._ev_evicted += 1
            self._ev_ev_score_sum += float(self.score[idx])
            self._ev_ev_ret_sum += float(self.returns[idx])
            self._ev_ev_age_sum += float(self.current_update - self.added_at[idx])
        self.write(idx, seg)
        self.returns[idx] = ret
        self.score[idx] = score
        self.group[idx] = group
        self.uses[idx] = 0
        self.added_at[idx] = self.current_update
        self.ep_lens[idx] = self.length(seg)
        self.tie[idx] = tie
        self._ev_added += 1
        return True

    def rebuild_tie(self):
        for i in range(self.size):
            self.tie[i] = self.tiebreak(self.buffer[i])

    def _level_p(self, valid, level_probs):
        if level_probs is None:
            raise ValueError("golden memory sampling 'level' needs level_probs, the academy level distribution")
        probs = np.asarray(level_probs, dtype=np.float64)
        uniq, inv = np.unique(self.group[valid], return_inverse=True)
        lp = np.array([probs[l] if l < len(probs) else 0.0 for l in uniq])
        if not lp.sum() > 0:
            return valid[:0], np.zeros(0), np.zeros(0, dtype=bool)
        keep = lp[inv] > 0
        valid, inv = valid[keep], inv[keep]
        lp = lp / lp.sum()
        g = self.score[valid].astype(np.float64)
        g_max = np.full(len(uniq), -np.inf)
        np.maximum.at(g_max, inv, g)
        w = self.base ** (g - g_max[inv])
        if self.cap_mode == "strict":
            p = np.zeros(len(valid))
            capped = np.zeros(len(valid), dtype=bool)
            for j in np.unique(inv):
                m = inv == j
                p[m], capped[m], _ = gm_cap_probs(w[m], self.max_p)
            return valid, lp[inv] * p, capped
        p = w / np.bincount(inv, weights=w, minlength=len(uniq))[inv]
        capped = p > self.max_p
        p = np.minimum(p, self.max_p)
        return valid, lp[inv] * p / np.bincount(inv, weights=p, minlength=len(uniq))[inv], capped

    def _draw_levels(self, pool, p, level_probs, n):
        chosen = np.zeros(n, dtype=np.int64)
        for i in range(n):
            k = np.random.choice(len(pool), p=p)
            chosen[i] = pool[k]
            pool = np.delete(pool, k)
            if i + 1 < n:
                pool, p, _ = self._level_p(pool, level_probs)
        return chosen

    def _draw_strict(self, pool, w, n):
        chosen = np.zeros(n, dtype=np.int64)
        p = None
        for i in range(n):
            pi, capped_i, _ = gm_cap_probs(w, self.max_p)
            if p is None:
                p, capped = pi, capped_i
            k = np.random.choice(len(pool), p=pi)
            chosen[i] = pool[k]
            pool, w = np.delete(pool, k), np.delete(w, k)
        return chosen, p, capped

    def _keep(self, keep):
        idx = np.where(keep)[0]
        n = len(idx)
        for a in (self.returns, self.score, self.group, self.uses, self.added_at, self.ep_lens, self.tie):
            a[:n] = a[:self.size][idx]
        self.buffer[:self.size] = [self.buffer[i] for i in idx] + [None] * (self.size - n)
        dropped = self.size - n
        self.size = n
        self.last_sampled_idx = np.zeros(0, dtype=np.int64)
        return dropped

    def purge(self):
        n = self._keep(np.zeros(self.size, dtype=bool))
        self.purged += n
        return n

    def expire(self):
        if self.max_age <= 0 or self.size == 0:
            return 0
        n = self._keep(self.current_update - self.added_at[:self.size] <= self.max_age)
        self.expired += n
        return n

    def draw(self, n, level_probs=None):
        self.reset_sample_stats()
        empty = np.zeros(0, dtype=np.int64)
        if self.size == 0:
            return empty
        valid = np.where(self.uses[:self.size] < self.max_uses)[0]
        if len(valid) == 0:
            return empty
        g = self.score[:self.size]
        if self.sampling == "level":
            valid, p, capped = self._level_p(valid, level_probs)
            if len(valid) == 0:
                return empty
            n = min(n, len(valid))
            chosen = self._draw_levels(valid, p, level_probs, n)
        else:
            w = self.weights(g[valid])
            n = min(n, len(valid))
            if self.cap_mode == "strict":
                chosen, p, capped = self._draw_strict(valid, w, n)
            else:
                p = w / w.sum()
                capped = np.zeros(len(p), dtype=bool)
                if self.max_p is not None:
                    capped = p > self.max_p
                    p = np.minimum(p, self.max_p)
                    p = p / p.sum()
                chosen = np.random.choice(valid, size=n, replace=False, p=p)
        self.last_valid_available = int(len(valid))
        self.last_sampled_score = g[chosen].copy()
        self.last_sampled_lens = self.ep_lens[chosen].copy()
        self.last_sampled_group = self.group[chosen].copy()
        self.last_sampled_age = self.current_update - self.added_at[chosen]
        if self.cap_mode == "strict":
            counts = np.unique(self.group[valid], return_counts=True)[1] if self.sampling == "level" else [len(valid)]
            self.last_cap_infeasible = bool(min(counts) * self.max_p < 1.0)
        self.last_weight_max = float(p.max())
        self.last_weight_entropy = float(-(p * np.log(p)).sum())
        self.last_capped_count = int(np.count_nonzero(capped))
        self.uses[chosen] += 1
        self.last_sampled_idx = chosen.astype(np.int64).copy()
        return chosen

    def sample(self, n, level_probs=None):
        return [self.buffer[i] for i in self.draw(n, level_probs)]

    def refund_last_sample(self):
        if len(self.last_sampled_idx):
            self.uses[self.last_sampled_idx] -= 1
            self.uses_saved += int(len(self.last_sampled_idx))
            self.last_sampled_idx = np.zeros(0, dtype=np.int64)

    def pop_events(self):
        ev = {"add_attempts": self._ev_add_attempts, "added": self._ev_added, "rejected": dict(self._ev_rejected), "evicted": self._ev_evicted,
              "evicted_score_sum": self._ev_ev_score_sum, "evicted_return_sum": self._ev_ev_ret_sum, "evicted_age_sum": self._ev_ev_age_sum}
        self._ev_rejected = Counter()
        self._ev_add_attempts = self._ev_added = self._ev_evicted = 0
        self._ev_ev_score_sum = self._ev_ev_ret_sum = self._ev_ev_age_sum = 0.0
        return ev


def sil_confident_gauss(params, conf_sigma):
    return (params[..., 1] < conf_sigma).all(dim=-1)


def sil_positive_advantage(mc_returns, values_norm, mu, sigma, adv_max):
    a_norm = (mc_returns - mu) / (sigma + 1e-8) - values_norm
    return torch.clamp(a_norm, min=0.0, max=adv_max)


def sil_mc_returns(rew, done, gamma, bootstrap=None):
    nd = 1.0 - done
    R = 0.0 if bootstrap is None else bootstrap
    cols = []
    for t in reversed(range(rew.shape[-1])):
        R = rew[..., t] + gamma * R * nd[..., t]
        cols.append(R)
    return torch.stack(cols[::-1], dim=-1)


def sil_full_cutoff(seg, T):
    return T, False


def sil_policy_kl(pre, post, valid, gauss):
    pre, post = pre.float(), post.float()
    if gauss:
        m1, s1, m2, s2 = pre[..., 0], pre[..., 1], post[..., 0], post[..., 1]
        kl = (torch.log(s2 / s1) + (s1.pow(2) + (m1 - m2).pow(2)) / (2.0 * s2.pow(2)) - 0.5).sum(dim=-1)
    else:
        l1, l2 = torch.log_softmax(pre, dim=-1), torch.log_softmax(post, dim=-1)
        kl = (l1.exp() * (l1 - l2)).sum(dim=-1).sum(dim=-1)
    return float(kl[valid].mean())


def sil_row_normalize(mean_rows, scale_rows):
    def normalize(x):
        if x.shape[0] != mean_rows.shape[0]:
            raise ValueError(f"per-row normalization expects {mean_rows.shape[0]} rows, got {x.shape[0]}")
        shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        return (x - mean_rows.view(shape)) / scale_rows.view(shape)
    return normalize


def sil_ensure(v, device, dtype=torch.float32):
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(v, dtype=dtype, device=device)


def sil_stats(loss, pi, v, adv, frac, used, total, cutoff_mean, fallback, conf_frac, bypass, kept, mean_conf, an_np, adv_max, extra, adv_np, w_np, lp_pm, ent_pm, ep_stats):
    out = {"sil_loss": loss.item(), "sil_pi": pi.item(), "sil_v": v.item(), "sil_adv": adv, "sil_frac": frac, "sil_used_steps": float(used),
           "sil_truncated_fraction": float(total - used) / float(max(total, 1)), "sil_cutoff_mean": cutoff_mean, "sil_cutoff_fallback": float(fallback),
           "sil_conf_clipped_fraction": float(conf_frac), "sil_conf_bypass": 1.0 if bypass else 0.0,
           "sil_spear_weight_mean": float(kept) / float(max(used, 1)), "sil_mean_conf_prob": mean_conf, **extra,
           "sil_adv_norm_mean": float(an_np.mean()), "sil_adv_norm_max": float(an_np.max()), "sil_adv_clamp_frac": float((an_np >= adv_max - 1e-6).mean()),
           "adv_raw_np": adv_np, "weights_np": w_np, "logprob_pm": lp_pm, "entropy_pm": ent_pm, "ep_stats": ep_stats}
    return out


def compute_sil_loss(agent, segs, device, gamma, cutoff_fn, conf_prob, conf_heads, adv_max_sigma, conf_sigma):
    if not segs:
        return torch.tensor(0.0, device=device), {}
    pa_mu = float(agent.critic.popart.mu.detach())
    pa_sigma = float(agent.critic.popart.sigma.detach())
    ep_data = []
    used_steps = 0
    total_steps = 0
    cutoff_fracs = []
    fallback = 0
    gauss = getattr(getattr(agent, "actor", None), "head_type", "categorical") == "gaussian"
    act_dtype = torch.float32 if gauss else torch.int64
    for seg in segs:
        rewards = sil_ensure(seg["rewards"], device)
        dones = sil_ensure(seg["dones"], device)
        T = len(rewards)
        cutoff, fell_back = cutoff_fn(seg, T)
        fallback += int(fell_back)
        obs = {
            "temporal": sil_ensure(seg["temporal"], device)[:cutoff],
            "state": sil_ensure(seg["state"], device)[:cutoff],
        }
        if "mask_bits" in seg:
            obs["mask_bits"] = sil_ensure(seg["mask_bits"], device, torch.uint8)[:cutoff]
        if getattr(getattr(agent, "actor", None), "recurrent", False):
            obs["h0"] = agent.init_hidden(1, obs["temporal"].device)
            obs["dones"] = dones[:cutoff]
        indices = sil_ensure(seg["bin_indices"], device, act_dtype)[:cutoff]
        log_probs, _, entropy_pm, values, values_norm, _, logits = agent.evaluate(obs, indices)
        values = values.squeeze(-1)
        values_norm = values_norm.squeeze(-1)
        mc_returns = sil_mc_returns(rewards.float(), dones.float(), gamma)[:cutoff]
        with torch.no_grad():
            a_norm = sil_positive_advantage(mc_returns, values_norm.detach(), pa_mu, pa_sigma, adv_max_sigma)
            if gauss:
                p_head = torch.full(indices.shape, float("nan"), device=indices.device)
                clip_mask = sil_confident_gauss(logits.detach().float(), conf_sigma) & (a_norm > 0)
            else:
                lg = logits.detach().float()
                lg = lg - lg.logsumexp(dim=-1, keepdim=True)
                p_head = lg.gather(-1, indices.unsqueeze(-1)).squeeze(-1).exp()
                clip_mask = ((p_head > conf_prob).sum(dim=-1) >= conf_heads) & (a_norm > 0)
        used_steps += int(cutoff)
        total_steps += int(T)
        cutoff_fracs.append(cutoff / T)
        ep_data.append((log_probs, values.detach(), mc_returns, clip_mask, a_norm, p_head, entropy_pm, logits, indices, int(cutoff)))
    clipped_steps = int(sum(int(d[3].sum()) for d in ep_data))
    conf_frac = clipped_steps / float(max(used_steps, 1))
    bypass = conf_frac > 0.9
    kept_steps = used_steps if bypass else used_steps - clipped_steps
    total_pi_loss = 0.0
    total_v_loss = 0.0
    adv_raw_parts = []
    weight_parts = []
    a_norm_parts = []
    lp_pm_parts = []
    ent_pm_parts = []
    ep_records = []
    conf_prob_parts = []
    for log_probs, values, mc_returns, clip_mask, a_norm, p_head, entropy_pm, logits, indices, cutoff in ep_data:
        keep = torch.ones_like(clip_mask) if bypass else ~clip_mask
        adv_raw = mc_returns - values
        if keep.any():
            pos_adv = a_norm[keep]
            total_pi_loss += -(log_probs[keep] * pos_adv).mean()
            total_v_loss += 0.5 * pos_adv.pow(2).mean()
        else:
            pos_adv = torch.zeros(0, device=device)
        with torch.no_grad():
            adv_raw_parts.append(adv_raw.detach().float())
            weight_parts.append(pos_adv.detach().float())
            a_norm_parts.append(a_norm.float())
            conf_prob_parts.append(p_head.reshape(-1))
            if gauss:
                lp_pm_parts.append(agent.actor.distribution(logits.detach().float()).log_prob(indices).mean(dim=0))
            else:
                lp_pm_parts.append(torch.log_softmax(logits.detach().float(), dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1).mean(dim=0))
            ent_pm_parts.append(entropy_pm.detach().float().mean(dim=0))
            if len(pos_adv):
                pf = (pos_adv.detach() > 0).float().mean()
                pm = pos_adv.detach().float().mean()
            else:
                pf = torch.zeros((), device=device)
                pm = torch.zeros((), device=device)
            ep_records.append((cutoff, pf, pm, adv_raw.detach().float().mean()))
    n = len(segs)
    loss = (total_pi_loss + total_v_loss) / n
    with torch.no_grad():
        adv_np = torch.cat(adv_raw_parts).cpu().numpy()
        w_np = torch.cat(weight_parts).cpu().numpy()
        an_np = torch.cat(a_norm_parts).cpu().numpy()
        lp_pm = torch.stack(lp_pm_parts).mean(dim=0).cpu().numpy()
        ent_pm = torch.stack(ent_pm_parts).mean(dim=0).cpu().numpy()
        ep_scalars = torch.stack([torch.stack([pf, pm, am]) for _, pf, pm, am in ep_records]).cpu().numpy()
        ep_stats = [(ep_records[i][0], float(ep_scalars[i, 0]), float(ep_scalars[i, 2])) for i in range(n)]
    return loss, sil_stats(loss, total_pi_loss / n, total_v_loss / n, float(ep_scalars[:, 1].mean()), float(ep_scalars[:, 0].mean()), used_steps,
                           total_steps, float(np.mean(cutoff_fracs)), fallback, conf_frac, bypass, kept_steps, float(torch.cat(conf_prob_parts).mean()),
                           an_np, adv_max_sigma, {}, adv_np, w_np, lp_pm, ent_pm, ep_stats)


def compute_sil_loss_batched(agent, segs, device, gamma, cutoff_fn, conf_prob, conf_heads, adv_max_sigma,
                             conf_sigma, keep_policy=False, sigma_grad=True, conf_mode="all"):
    if not segs:
        return torch.tensor(0.0, device=device), {}
    pa_mu = float(agent.critic.popart.mu.detach())
    pa_sigma = float(agent.critic.popart.sigma.detach())
    actor = getattr(agent, "actor", None)
    recurrent = getattr(actor, "recurrent", False)
    gauss = getattr(actor, "head_type", "categorical") == "gaussian"
    rews, dns, temps, states, idxs, cutoffs, lens, masks = [], [], [], [], [], [], [], []
    fallback = 0
    for seg in segs:
        rewards = sil_ensure(seg["rewards"], device)
        T = len(rewards)
        cutoff, fell_back = cutoff_fn(seg, T)
        fallback += int(fell_back)
        rews.append(rewards)
        dns.append(sil_ensure(seg["dones"], device))
        temps.append(sil_ensure(seg["temporal"], device)[:cutoff])
        states.append(sil_ensure(seg["state"], device)[:cutoff])
        if "mask_bits" in seg:
            masks.append(sil_ensure(seg["mask_bits"], device, torch.uint8)[:cutoff])
        idxs.append(sil_ensure(seg["bin_indices"], device, torch.float32 if gauss else torch.int64)[:cutoff])
        cutoffs.append(int(cutoff))
        lens.append(int(T))
    n = len(segs)
    Tc = max(cutoffs)
    pad = nn.utils.rnn.pad_sequence
    obs = {"temporal": pad(temps, batch_first=True).reshape(n * Tc, *temps[0].shape[1:]),
           "state": pad(states, batch_first=True).reshape(n * Tc, *states[0].shape[1:])}
    if masks:
        obs["mask_bits"] = pad(masks, batch_first=True).reshape(n * Tc, *masks[0].shape[1:])
    indices = pad(idxs, batch_first=True).reshape(n * Tc, *idxs[0].shape[1:])
    cut_t = torch.as_tensor(cutoffs, device=device)
    valid = torch.arange(Tc, device=device).unsqueeze(0) < cut_t.unsqueeze(1)
    if recurrent:
        d_model = pad([d[:c] for d, c in zip(dns, cutoffs)], batch_first=True, padding_value=1.0)
        d_model[torch.arange(n, device=d_model.device), cut_t.to(d_model.device) - 1] = 1.0
        obs["h0"] = agent.init_hidden(n, obs["temporal"].device)
        obs["dones"] = d_model.reshape(-1)
    owners = [m for m in (actor, getattr(agent, "critic", None)) if isinstance(m, nn.Module) and m.training]
    rms = [m for o in owners for m in o.modules() if {"mean", "var", "count"} <= set(dict(m.named_buffers(recurse=False)))]
    snaps = [[] for _ in rms]
    if rms:
        for tp, st in zip(temps, states):
            if actor in owners:
                actor.update_input_stats({"temporal": tp, "state": st})
            if agent.critic in owners:
                agent.critic.state_rms.update(st)
            for j, m in enumerate(rms):
                snaps[j].append((m.mean.clone(), m.var.sqrt() + 1e-8))

    def _evaluate(obs=obs, indices=indices):
        flags = {}
        if rms:
            flags = {m: m.training for m in agent.modules()}
            for m, sn in zip(rms, snaps):
                m.normalize = sil_row_normalize(torch.stack([s[0] for s in sn]).repeat_interleave(Tc, dim=0),
                                                torch.stack([s[1] for s in sn]).repeat_interleave(Tc, dim=0))
            agent.eval()
        try:
            return agent.evaluate(obs, indices)
        finally:
            for m in rms:
                del m.normalize
            for m, f in flags.items():
                m.training = f

    log_probs, _, entropy_pm, values, values_norm, _, logits = _evaluate()
    log_probs = log_probs.reshape(n, Tc)
    values = values.squeeze(-1).reshape(n, Tc)
    values_norm = values_norm.squeeze(-1).reshape(n, Tc)
    entropy_pm = entropy_pm.reshape(n, Tc, -1)
    logits = logits.reshape(n, Tc, *logits.shape[1:])
    indices = indices.reshape(n, Tc, -1)
    dim_drop = None
    if gauss and (not sigma_grad or conf_mode == "per_dim"):
        lp_src = logits.float()
        sig = lp_src[..., 1] if sigma_grad else lp_src[..., 1].detach()
        lp_dims = agent.actor.distribution(torch.stack([lp_src[..., 0], sig], dim=-1)).log_prob(indices.float())
        if conf_mode == "per_dim":
            dim_drop = lp_src[..., 1].detach() < conf_sigma
            lp_dims = torch.where(dim_drop, 0.0, lp_dims)
        log_probs = lp_dims.sum(dim=-1)
    mc_returns = sil_mc_returns(pad(rews, batch_first=True), pad(dns, batch_first=True, padding_value=1.0), gamma)[:, :Tc]
    with torch.no_grad():
        a_norm = sil_positive_advantage(mc_returns, values_norm.detach(), pa_mu, pa_sigma, adv_max_sigma)
        if gauss:
            p_head = torch.full(indices.shape, float("nan"), device=indices.device)
            clip_mask = sil_confident_gauss(logits.detach().float(), conf_sigma) & (a_norm > 0) & valid
        else:
            lg = logits.detach().float()
            lg = lg - lg.logsumexp(dim=-1, keepdim=True)
            p_head = lg.gather(-1, indices.unsqueeze(-1)).squeeze(-1).exp()
            clip_mask = ((p_head > conf_prob).sum(dim=-1) >= conf_heads) & (a_norm > 0) & valid
    used_steps = sum(cutoffs)
    total_steps = sum(lens)
    clipped_steps = int(clip_mask.sum())
    conf_frac = clipped_steps / float(max(used_steps, 1))
    bypass = conf_frac > 0.9
    kept_steps = used_steps if bypass else used_steps - clipped_steps
    keep = valid if bypass else valid & ~clip_mask
    n_keep = keep.sum(dim=1).clamp(min=1)
    total_pi_loss = (-(torch.where(keep, log_probs * a_norm, 0.0).sum(dim=1) / n_keep)).sum()
    total_v_loss = (0.5 * (torch.where(keep, a_norm.pow(2), 0.0).sum(dim=1) / n_keep)).sum()
    loss = (total_pi_loss + total_v_loss) / n
    with torch.no_grad():
        adv_raw = (mc_returns - values.detach()).float()
        a_f = a_norm.float()
        pf = torch.where(keep, (a_f > 0).float(), 0.0).sum(dim=1) / n_keep
        pm = torch.where(keep, a_f, 0.0).sum(dim=1) / n_keep
        am = torch.where(valid, adv_raw, 0.0).sum(dim=1) / cut_t
        ep_scalars = torch.stack([pf, pm, am], dim=1).cpu().numpy()
        adv_np = adv_raw[valid].cpu().numpy()
        w_np = a_f[keep].cpu().numpy()
        an_np = a_f[valid].cpu().numpy()
        vmask = valid.unsqueeze(-1)
        if gauss:
            lpr = agent.actor.distribution(logits.detach().float()).log_prob(indices)
        else:
            lpr = torch.log_softmax(logits.detach().float(), dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        lp_pm = (torch.where(vmask, lpr, 0.0).sum(dim=1) / cut_t.unsqueeze(1)).mean(dim=0).cpu().numpy()
        ent_pm = (torch.where(vmask, entropy_pm.detach().float(), 0.0).sum(dim=1) / cut_t.unsqueeze(1)).mean(dim=0).cpu().numpy()
        ep_stats = [(cutoffs[i], float(ep_scalars[i, 0]), float(ep_scalars[i, 2])) for i in range(n)]
        mean_conf = float(p_head[valid].mean())
    extra = {"sil_conf_dim_fraction": float(dim_drop[valid].float().mean()) if dim_drop is not None else 0.0}
    stats = sil_stats(loss, total_pi_loss / n, total_v_loss / n, float(ep_scalars[:, 1].mean()), float(ep_scalars[:, 0].mean()), used_steps, total_steps,
                      float(np.mean([c / t for c, t in zip(cutoffs, lens)])), fallback, conf_frac, bypass, kept_steps, mean_conf, an_np, adv_max_sigma,
                      extra, adv_np, w_np, lp_pm, ent_pm, ep_stats)
    if keep_policy:
        stats["policy"] = {"pre": logits.detach(), "valid": valid, "gauss": gauss,
                           "forward": lambda shape=logits.shape: _evaluate()[6].reshape(shape)}
    return loss, stats


def sil_clipped_step(optimizer, scaler, loss, params, max_norm, group_norms=None):
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    raw = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
    norms = group_norms() if group_norms else None
    if math.isfinite(raw):
        scaler.step(optimizer)
    scaler.update()
    return raw, norms


def sil_separate_step(agent, sil_opt, scaler, loss_fn, coef, max_kl=None, amp=None, group_norms=None):
    amp = amp or contextlib.nullcontext
    params = [p for g in sil_opt.param_groups for p in g["params"]]
    for p in agent.parameters():
        p.grad = None
    with amp():
        loss, stats = loss_fn()
    scaler.scale(coef * loss).backward()
    scaler.unscale_(sil_opt)
    grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    raw = float(torch.stack(grads).norm()) if grads else 0.0
    if group_norms:
        stats["group_norms"] = group_norms(agent)
    stats["sil_grad_raw"] = stats["sil_grad_clipped"] = raw if math.isfinite(raw) else None
    stats["sil_kl"] = None
    stats["sil_kl_skipped"] = False
    stats["sil_param_delta"] = 0.0
    pol = stats.pop("policy", None)
    if max_kl is not None and pol is None:
        raise ValueError("the SIL KL guard needs the policy of the SIL forward, call the loss with keep_policy=True")
    if not math.isfinite(raw):
        scaler.update()
        for p in agent.parameters():
            p.grad = None
        return stats
    before = [p.detach().clone() for p in params]
    opt_state = {p: {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in st.items()}
                 for p, st in sil_opt.state.items()} if max_kl is not None else None
    scale_state = scaler.state_dict() if max_kl is not None else None
    scaler.step(sil_opt)
    scaler.update()
    for p in agent.parameters():
        p.grad = None
    with torch.no_grad():
        stats["sil_param_delta"] = float(torch.stack([(p.detach() - b).float().norm() for p, b in zip(params, before)]).norm())
        if pol is not None:
            with amp():
                post = pol["forward"]()
            stats["sil_kl"] = sil_policy_kl(pol["pre"], post, pol["valid"], pol["gauss"])
    if max_kl is not None and not stats["sil_kl"] <= max_kl:
        with torch.no_grad():
            for p, b in zip(params, before):
                p.copy_(b)
        sil_opt.state.clear()
        sil_opt.state.update(opt_state)
        if scale_state:
            scaler.load_state_dict(scale_state)
        stats["sil_kl_skipped"] = True
    return stats
