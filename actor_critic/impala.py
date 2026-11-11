import math
import time
from collections import deque
from queue import Empty as QueueEmpty, Full as QueueFull

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(epsilon))
        self.frozen = False
        self.hold_tail = 0

    @torch.no_grad()
    def update(self, x):
        if self.frozen or x.shape[0] < 2:
            return
        h = self.hold_tail
        keep = (self.mean[-h:].clone(), self.var[-h:].clone()) if h else None
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total
        if h:
            self.mean[-h:], self.var[-h:] = keep

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class PopArtLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=1e-2):
        super().__init__()
        self.beta = beta
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        normalized = F.linear(x, self.weight, self.bias)
        return normalized * self.sigma + self.mu

    def forward_normalized(self, x):
        return F.linear(x, self.weight, self.bias)

    def normalize_targets(self, targets):
        return (targets - self.mu) / (self.sigma + 1e-8)

    @torch.no_grad()
    def update_stats(self, targets):
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()
        new_mu = targets.mean()
        new_sigma = targets.std().clamp(min=1e-4)
        self.mu.copy_((1 - self.beta) * self.mu + self.beta * new_mu)
        self.sigma.copy_(((1 - self.beta) * self.sigma + self.beta * new_sigma).clamp(1e-4, 1e6))
        self.weight.data.mul_(old_sigma / self.sigma)
        self.bias.data.copy_((old_sigma * self.bias.data + old_mu - self.mu) / self.sigma)


def policy_lag_stats(versions, current):
    if not len(versions):
        return {"lag_mean": float("nan"), "lag_max": float("nan")}
    lags = current - np.asarray(versions, dtype=np.float64)
    return {"lag_mean": float(lags.mean()), "lag_max": float(lags.max())}


class EntropyController:
    def __init__(self, target, alpha_init=0.1, alpha_min=5e-3, alpha_max=1.0,
                 lr=0.03, tau=10.0, d_window=6, e_clamp=2.0):
        self.target = float(target)
        self.log_min = math.log(alpha_min)
        self.log_max = math.log(alpha_max)
        self.log_alpha = min(max(math.log(alpha_init), self.log_min), self.log_max)
        self.lr = float(lr)
        self.tau = float(tau)
        self.d_window = int(d_window)
        self.e_clamp = float(e_clamp)
        self._h_hist = deque(maxlen=self.d_window)
        self.last_e = 0.0
        self.last_dh = 0.0
        self.last_e_eff = 0.0
        self.last_delta_applied = 0.0

    @property
    def alpha(self):
        return math.exp(self.log_alpha)

    def set_target(self, target):
        self.target = float(target)

    def _slope(self):
        n = len(self._h_hist)
        if n < self.d_window:
            return 0.0
        y = np.asarray(self._h_hist, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        xm = x.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom <= 0.0:
            return 0.0
        return float(((x - xm) * (y - y.mean())).sum() / denom)

    def update(self, entropy_mean):
        h = float(entropy_mean)
        self._h_hist.append(h)
        e = h - self.target
        dh = self._slope()
        e_eff = max(-self.e_clamp, min(self.e_clamp, e + self.tau * dh))
        before = self.log_alpha
        self.log_alpha = min(max(self.log_alpha - self.lr * e_eff, self.log_min), self.log_max)
        self.last_e = e
        self.last_dh = dh
        self.last_e_eff = e_eff
        self.last_delta_applied = self.log_alpha - before
        return self.alpha

    def diagnostics(self):
        la = self.log_alpha
        return {
            "log_alpha": la,
            "alpha_error": self.last_e,
            "alpha_loss": float("nan"),
            "alpha_grad": self.last_e,
            "alpha_at_min": float(la <= self.log_min + 1e-6),
            "alpha_at_max": float(la >= self.log_max - 1e-6),
            "ent_e": self.last_e,
            "ent_dh": self.last_dh,
            "ent_e_eff": self.last_e_eff,
            "ent_delta_applied": self.last_delta_applied,
        }


class FixedEntropy:
    def __init__(self, coef):
        self.coef = float(coef)
        self.target = float("nan")
        self.log_alpha = math.log(max(self.coef, 1e-12))
        self.log_min = self.log_alpha
        self.log_max = self.log_alpha
        self.last_e = float("nan")
        self.last_dh = 0.0
        self.last_e_eff = 0.0
        self.last_delta_applied = 0.0

    @property
    def alpha(self):
        return self.coef

    def set_target(self, target):
        pass

    def update(self, entropy_mean):
        return self.coef

    def diagnostics(self):
        return {
            "log_alpha": self.log_alpha,
            "alpha_error": float("nan"),
            "alpha_loss": float("nan"),
            "alpha_grad": float("nan"),
            "alpha_at_min": 0.0,
            "alpha_at_max": 0.0,
            "ent_e": float("nan"),
            "ent_dh": 0.0,
            "ent_e_eff": 0.0,
            "ent_delta_applied": 0.0,
        }


def make_entropy_controller(cfg):
    ent = cfg["entropy"]
    if ent["mode"] == "fixed":
        return FixedEntropy(ent["coef"])
    c = ent["controller"]
    return EntropyController(
        target=c["target"], alpha_init=c["alpha_init"],
        alpha_min=c["alpha_clamp"][0], alpha_max=c["alpha_clamp"][1],
        lr=c["lr"], tau=c["tau"], d_window=c["d_window"], e_clamp=c["e_clamp"],
    )


def compute_vtrace(behavior_log_probs, current_log_probs, rewards, values,
                   dones, bootstrap_values, gamma, rho_bar=1.0, c_bar=1.0, log_rho_clip=None):
    B, T = rewards.shape
    log_rho = current_log_probs - behavior_log_probs
    if log_rho_clip is not None:
        log_rho = log_rho.clamp(-log_rho_clip, log_rho_clip)
    is_ratio = torch.exp(log_rho)
    rho = is_ratio.clamp(max=rho_bar)
    c = is_ratio.clamp(max=c_bar)
    values_ext = torch.cat([values, bootstrap_values.unsqueeze(1)], dim=1)
    vs = torch.zeros(B, T + 1, device=rewards.device)
    vs[:, T] = bootstrap_values
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        delta = rho[:, t] * (rewards[:, t] + gamma * not_done * values_ext[:, t + 1] - values_ext[:, t])
        vs[:, t] = values_ext[:, t] + delta + gamma * not_done * c[:, t] * (vs[:, t + 1] - values_ext[:, t + 1])
    vtrace_targets = vs[:, :T]
    next_vs = torch.cat([vtrace_targets[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    td = rewards + gamma * (1.0 - dones) * next_vs - values_ext[:, :T]
    advantages = rho * td
    diag = {"is_ratio": is_ratio, "td": td, "rho_bar": rho_bar, "c_bar": c_bar}
    return vtrace_targets.reshape(-1), advantages.reshape(-1), rho.mean().item(), diag

def compute_caps_loss(logits, dones, B, T, denom=None):
    lg = logits.float().view(B, T, logits.shape[-2], logits.shape[-1])
    lg = lg - lg.logsumexp(dim=-1, keepdim=True)
    cur = lg[:, :-1]
    nxt = lg[:, 1:].detach()
    kl = (cur.exp() * (cur - nxt)).sum(dim=-1)
    valid = 1.0 - dones.view(B, T)[:, :-1]
    if denom is None:
        denom = valid.sum().clamp_min(1.0)
    kl_head = (kl * valid.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    masked_pairs = int((valid < 0.5).sum().item())
    return kl_head.mean(), kl_head, masked_pairs


def compute_caps_loss_mu(params, dones, B, T, denom=None):
    mu = params.float().view(B, T, params.shape[-2], 2)[..., 0]
    d2 = (mu[:, :-1] - mu[:, 1:].detach()).pow(2)
    valid = 1.0 - dones.view(B, T)[:, :-1]
    if denom is None:
        denom = valid.sum().clamp_min(1.0)
    per_head = (d2 * valid.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    masked_pairs = int((valid < 0.5).sum().item())
    return per_head.mean(), per_head, masked_pairs


def compute_caps_loss_gauss(params, dones, B, T, denom=None):
    pr = params.float().view(B, T, params.shape[-2], 2)
    cur = pr[:, :-1]
    nxt = pr[:, 1:].detach()
    mu1, s1 = cur[..., 0], cur[..., 1]
    mu2, s2 = nxt[..., 0], nxt[..., 1]
    kl = torch.log(s2 / s1) + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2.0 * s2.pow(2)) - 0.5
    valid = 1.0 - dones.view(B, T)[:, :-1]
    if denom is None:
        denom = valid.sum().clamp_min(1.0)
    kl_head = (kl * valid.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    masked_pairs = int((valid < 0.5).sum().item())
    return kl_head.mean(), kl_head, masked_pairs


class PopArtAdapter:
    def __init__(self, popart):
        self.popart = popart

    @property
    def sigma(self):
        return self.popart.sigma

    def update_stats(self, targets):
        self.popart.update_stats(targets)

    def normalize_targets(self, targets):
        return self.popart.normalize_target(targets)


def impala_targets(behavior_log_probs, current_log_probs, rewards, values, dones, bootstrap_values, popart, cfg, mask=None):
    if popart is not None and not hasattr(popart, "normalize_targets") and hasattr(popart, "normalize_target"):
        popart = PopArtAdapter(popart)
    if mask is not None:
        mask = mask.to(torch.bool)
        if mask.shape != rewards.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must equal rewards shape {tuple(rewards.shape)}")
    with torch.no_grad():
        vtargets, advantages, avg_rho, diag = compute_vtrace(
            behavior_log_probs, current_log_probs.float(), rewards, values.float(), dones, bootstrap_values,
            cfg["gamma"], cfg.get("rho_bar", 1.0), cfg.get("c_bar", 1.0), cfg.get("log_rho_clip"),
        )
    if mask is not None:
        avg_rho = diag["is_ratio"].clamp(max=diag["rho_bar"])[mask].mean().item()
    if popart is None:
        return {"vtargets": vtargets, "advantages": advantages, "normalized_targets": vtargets, "sigma": None,
                "avg_rho": avg_rho, "diag": diag}
    popart.update_stats(vtargets if mask is None else vtargets[mask.reshape(-1)])
    return {"vtargets": vtargets, "advantages": advantages, "normalized_targets": popart.normalize_targets(vtargets),
            "sigma": popart.sigma.detach(), "avg_rho": avg_rho, "diag": diag}


def impala_loss(targets, log_probs, entropy, normalized_values, cfg, sl=slice(None), weight=1.0, mask=None):
    idx = sl
    if mask is not None:
        if sl != slice(None):
            raise ValueError("impala_loss takes either sl or mask, not both")
        idx = mask.to(torch.bool).reshape(-1)
    adv = targets["advantages"][idx]
    tgt = targets["normalized_targets"][idx].detach()
    for name, x in (("log_probs", log_probs), ("normalized_values", normalized_values), ("targets", tgt), ("advantages", adv)):
        if x.dim() != 1 or x.shape != log_probs.shape:
            raise ValueError(f"impala_loss needs 1-D tensors of equal length, {name} has shape {tuple(x.shape)} against log_probs {tuple(log_probs.shape)}")
    if targets["sigma"] is not None and cfg.get("adv_sigma_norm", True):
        adv = adv / (targets["sigma"] + 1e-8)
    policy = -(adv * log_probs).mean()
    form = cfg.get("value_loss_form", "half_mse")
    if form == "half_mse":
        value = 0.5 * (normalized_values - tgt).pow(2).mean()
    elif form == "mse":
        value = F.mse_loss(normalized_values, tgt)
    else:
        raise ValueError(f"value_loss_form must be 'half_mse' or 'mse', got {form!r}")
    entropy_loss = -cfg["alpha"] * entropy.mean()
    loss = (cfg.get("policy_coef", 1.0) * policy + cfg["value_coef"] * value + entropy_loss) * weight
    return {"loss": loss, "policy": policy, "value": value, "entropy": entropy_loss}


def _plain(v):
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


class ParamStore:
    def __init__(self):
        self.version = -1
        self.weights = None
        self.sched_version = -1
        self.sched = None
        self.stopped = False

    def publish(self, version, weights, sched):
        if version != self.version:
            self.version = int(version)
            self.weights = weights
        sched = {k: _plain(v) for k, v in dict(sched).items()}
        if sched != self.sched:
            self.sched = sched
            self.sched_version += 1
        return self.version

    def stop(self):
        self.stopped = True

    def poll(self, version, sched_version):
        out = {"stop": self.stopped}
        if version != self.version:
            out["version"] = self.version
            out["weights"] = self.weights
        if sched_version != self.sched_version:
            out["sched_version"] = self.sched_version
            out["sched"] = dict(self.sched)
        return out


class StreamLoop:
    def __init__(self, poll, put, resolve, rng, sample_task, sleep=time.sleep, clock=time.perf_counter,
                 put_timeout=1.0, idle_sleep=0.05):
        self.poll = poll
        self.put = put
        self.resolve = resolve
        self.rng = rng
        self.sample_task = sample_task
        self.sleep = sleep
        self.clock = clock
        self.put_timeout = put_timeout
        self.idle_sleep = idle_sleep
        self.version = -1
        self.sched_version = -1
        self.sched = None
        self.produced = 0
        self.last_end = None

    def _sync(self, worker):
        info = self.poll(self.version, self.sched_version)
        if info["stop"]:
            return False
        if "version" in info:
            worker.set_weights(self.resolve(info["weights"]))
            self.version = info["version"]
        if "sched_version" in info:
            self.sched_version = info["sched_version"]
            self.sched = info["sched"]
        return True

    def _deliver(self, item):
        while True:
            try:
                self.put(item, self.put_timeout)
                return True
            except QueueFull:
                if self.poll(self.version, self.sched_version)["stop"]:
                    return False

    def step(self, worker):
        if not self._sync(worker):
            return False
        if self.version < 0 or self.sched is None:
            self.sleep(self.idle_sleep)
            return True
        task = self.sample_task(self.sched, self.rng)
        worker.set_task(task)
        t0 = self.clock()
        traj = worker.collect_trajectory()
        t1 = self.clock()
        meta = {"version": self.version, "task": task, "seq": self.produced, "collect_s": t1 - t0,
                "wait_s": 0.0 if self.last_end is None else t0 - self.last_end}
        self.last_end = t1
        if not self._deliver((meta, traj)):
            return False
        self.produced += 1
        return True

    def run(self, worker):
        while self.step(worker):
            pass
        return self.produced


class StreamFeed:
    def __init__(self, queue, check, timeout=1.0):
        self.queue = queue
        self.check = check
        self.timeout = timeout
        self.buf = deque()
        self.received = 0
        self.wait_s = 0.0
        self.collect_s = 0.0

    def next(self):
        while not self.buf:
            self.check()
            n = self.queue.qsize()
            if n:
                self.buf.extend(self.queue.get_nowait_batch(n))
                continue
            try:
                self.buf.append(self.queue.get(block=True, timeout=self.timeout))
            except QueueEmpty:
                pass
        meta, traj = self.buf.popleft()
        self.received += 1
        self.wait_s += meta["wait_s"]
        self.collect_s += meta["collect_s"]
        return traj, meta["task"], meta["version"]

    def backlog(self):
        return self.queue.qsize() + len(self.buf)

    def idle_frac(self):
        total = self.wait_s + self.collect_s
        frac = self.wait_s / total if total > 0 else float("nan")
        self.wait_s = 0.0
        self.collect_s = 0.0
        return frac


def check_stream_loops(loops, wait=None, get=None):
    if wait is None or get is None:
        import ray
        wait, get = wait or ray.wait, get or ray.get
    done, _ = wait(loops, num_returns=1, timeout=0)
    if done:
        produced = get(done[0])
        raise RuntimeError(f"stream worker loop ended while the driver was consuming ({produced} trajectories delivered)")


def shutdown_stream(store, queue, workers):
    import ray
    store.stop.remote()
    for w in workers:
        ray.kill(w)
    queue.shutdown(force=True)
    ray.kill(store)
