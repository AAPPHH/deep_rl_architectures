import contextlib
import copy
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dreamer_core import (ACTOR_INS, DIST, Encoder, Fp32, Frozen, HEAD_PREFIX, LaProp, MLP, Noise, OpTrace, RSSM, Replay, ReturnNorm, bin_edges, cat_probs, clip_agc, f32, gauss_nll, kl_terms, lambda_return, symexp, symlog, trunc_init, twohot_decode, twohot_nll)


class WorldModel(nn.Module):
    def __init__(self, cfg, obs_dim, act_dim, priv_dim):
        super().__init__()
        self.cfg, self.A, self.P = cfg, act_dim, priv_dim
        self.noise = Noise()
        c, u, hl = cfg, cfg["units"], cfg["head_layers"]
        feat = c["stoch"] * c["discrete"] + c["deter"]
        hin = feat + (act_dim if c["head_action"] else 0)
        self.register_buffer("edges", bin_edges(c["bins"], c["bin_lim"]))
        if c["priv_norm"] == "standard":
            self.register_buffer("priv_mean", torch.zeros(priv_dim))
            self.register_buffer("priv_std", torch.ones(priv_dim))
        self.enc = Encoder(obs_dim, u, c["layers"])
        self.rssm = RSSM(c, u, act_dim)
        self.dec = MLP(feat, obs_dim, u, c["layers"])
        self.rew = MLP(hin, c["bins"], u, hl, outscale=0.0)
        self.con = MLP(hin, 1, u, hl)
        self.priv = MLP(feat, 2 * priv_dim, u, hl)
        self.__dict__["teacher"] = None

    def init_opt(self):
        c = self.cfg
        self.opt = LaProp(self.trainable(), c["lr"], (c["b1"], c["b2"]), c["eps"])
        self.steps, self.skips, self.wm_lr_scale, self.actor_lr_scale, self.critic_lr_scale = 0, 0, 1.0, 1.0, 1.0
        self.lr_ramp, self.nonfinite = None, 0

    def split_group(self, ids, tag):
        g0 = self.opt.param_groups[0]
        sel = [p for p in g0["params"] if id(p) in ids]
        if sel:
            g0["params"] = [p for p in g0["params"] if id(p) not in ids]
            self.opt.add_param_group({**{k: v for k, v in g0.items() if k != "params"}, "params": sel, tag: True})

    def set_wm_lr_scale(self, scale):
        self.wm_lr_scale = float(scale)
        if self.wm_lr_scale == 1.0:
            return
        self.split_group({id(p) for n, p in self.named_parameters() if not n.startswith(HEAD_PREFIX)}, "wm")

    def set_head_lr_scales(self, actor=1.0, critic=1.0):
        self.actor_lr_scale, self.critic_lr_scale = float(actor), float(critic)
        for tag, sc, pre in (("actor", self.actor_lr_scale, ("actor.", "actor_")), ("critic", self.critic_lr_scale, ("critic.",))):
            if sc != 1.0:
                self.split_group({id(p) for n, p in self.named_parameters() if n.startswith(pre)}, tag)

    def trainable(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("slow.")]

    def feat(self, deter, stoch):
        return torch.cat([stoch.flatten(-2), deter], -1)

    def head_in(self, feat, act):
        return torch.cat([feat, act], -1) if self.cfg["head_action"] else feat

    def priv_head(self, feat):
        mu, ls = self.priv(feat.detach() if self.cfg["priv_stop_grad"] or not self.cfg.get("priv_wm_grad", True) else feat).chunk(2, -1)
        return mu, ls.clamp(*self.cfg["log_sigma_lim"])

    def priv_enc(self, priv):
        return (priv - self.priv_mean) / self.priv_std if self.cfg["priv_norm"] == "standard" else symlog(priv)

    def priv_dec(self, x):
        return x * self.priv_std + self.priv_mean if self.cfg["priv_norm"] == "standard" else symexp(x)

    def world_loss(self, b, mode=False):
        c = self.cfg
        obs, priv, act, rew, first, term = (b[k] for k in ("obs", "priv", "act", "rew", "first", "term"))
        deter, stoch, plogit, qlogit = self.rssm.observe(self.enc(obs), act, first[..., 0], b.get("prev"), mode, self.noise)
        feat = self.feat(deter, stoch)
        hin = self.head_in(feat, act)
        tg = self.teacher_targets(obs, act, first[..., 0], b.get("prev"), mode) if self.teacher is not None else None
        mask = b.get("mask")
        mean = (lambda x: x.mean()) if mask is None else (lambda x: (x * mask).sum() / mask.sum().clamp(min=1.0))
        L = {}
        dyn, rep = kl_terms(qlogit, plogit, c["stoch"], c["discrete"], c["unimix"], c["free"], c.get("free_bits_mode", "sum"))
        L["dyn"], L["rep"] = mean(dyn), mean(rep)
        L["recon"] = mean(((self.dec(feat) - symlog(obs)) ** 2).sum(-1))
        if tg is None:
            L["rew"] = mean(twohot_nll(self.rew(hin), rew[..., 0], self.edges))
            con = self.con(hin)[..., 0]
            L["con"] = F.binary_cross_entropy_with_logits(con, 1 - term[..., 0]) if mask is None else mean(F.binary_cross_entropy_with_logits(con, 1 - term[..., 0], reduction="none"))
        else:
            L["rew"] = mean(-(tg[0] * F.log_softmax(self.rew(hin), -1)).sum(-1))
            L["con"] = mean(F.binary_cross_entropy_with_logits(self.con(hin)[..., 0], tg[1], reduction="none"))
            L["anc"] = mean(((torch.cat([cat_probs(qlogit, c["stoch"], c["discrete"], c["unimix"]).flatten(-2), deter], -1) - tg[2]) ** 2).mean(-1))
        mu, ls = self.priv_head(feat)
        psym = self.priv_enc(priv)
        L["priv"] = mean(gauss_nll(mu, ls, psym))
        if c.get("wm_multistep"):
            L.update(self.multistep_losses(b, deter, stoch, qlogit))
        return L, deter, stoch, feat, mu, ls, psym

    def multistep_losses(self, b, deter, stoch, qlogit):
        c = self.cfg
        ms = c["wm_multistep"]
        k, every = int(ms["k"]), int(ms.get("every", 1))
        obs, priv, act, rew, first, term = (b[x] for x in ("obs", "priv", "act", "rew", "first", "term"))
        prev = b.get("prev")
        if prev is None:
            prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
        B, T = obs.shape[:2]
        starts = list(range(0, T - k, every))
        if not starts:
            raise ValueError(f"wm_multistep k={k} needs a window longer than {k} steps, got T={T}")
        n = len(starts)
        st = torch.arange(0, T - k, every, device=obs.device)
        idx = st[:, None] + 1 + torch.arange(k, device=obs.device)
        d, s = deter[:, st].flatten(0, 1), stoch[:, st].flatten(0, 1)
        ds, ss, ls_ = [], [], []
        for j in range(k):
            d, s, lg = self.rssm.img_step(d, s, prev[:, idx[:, j]].flatten(0, 1), self.noise)
            ds.append(d)
            ss.append(s)
            ls_.append(lg)
        unf = lambda xs: torch.stack(xs, 1).unflatten(0, (B, n))
        pd, ps, pl = unf(ds), unf(ss), unf(ls_)
        feat = self.feat(pd, ps)
        hin = self.head_in(feat, act[:, idx])
        w = (first[..., 0][:, idx].cumsum(-1) == 0).float()
        m = b.get("mask")
        if m is not None:
            w = w * m[:, idx] * m[:, st][..., None]
        mean = lambda x: (x * w).sum() / w.sum().clamp(min=1.0)
        L = {"ms_recon": mean(((self.dec(feat) - symlog(obs[:, idx])) ** 2).sum(-1)),
             "ms_rew": mean(twohot_nll(self.rew(hin), rew[..., 0][:, idx], self.edges)),
             "ms_con": mean(F.binary_cross_entropy_with_logits(self.con(hin)[..., 0], 1 - term[..., 0][:, idx], reduction="none"))}
        mu, lsg = self.priv_head(feat)
        L["ms_priv"] = mean(gauss_nll(mu, lsg, self.priv_enc(priv)[:, idx]))
        if float(ms.get("kl", 0.0)) > 0:
            L["ms_kl"] = mean(kl_terms(qlogit[:, idx], pl, c["stoch"], c["discrete"], c["unimix"], c["free"], c.get("free_bits_mode", "sum"))[0])
        return L

    @torch.no_grad()
    def teacher_targets(self, obs, act, first, prev, mode):
        t, c = self.teacher, self.cfg
        deter, stoch, _, qlogit = t.rssm.observe(t.enc(obs), act, first, prev, mode, self.noise)
        hin = t.head_in(t.feat(deter, stoch), act)
        return F.softmax(t.rew(hin), -1), torch.sigmoid(t.con(hin)[..., 0]), torch.cat([cat_probs(qlogit, c["stoch"], c["discrete"], c["unimix"]).flatten(-2), deter], -1)

    def learn(self, b, mode=False):
        if self.cfg.get("fast"):
            return self.fast_step(("wm", mode), lambda bb: self.backward_of(self.learn_loss(bb, mode)), b)
        total, out = self.learn_loss(b, mode)
        if not self.step_opt(total):
            return None
        self.post_step()
        return out

    def backward_of(self, loss):
        loss[0].backward()
        return loss[1]

    def post_step(self):
        pass

    def zero_grads(self):
        self.opt.zero_grad(set_to_none=True)

    def fast_state(self):
        return []

    def fast_build(self, fn, b, sig):
        can_graph = self.cfg.get("graph") and (sig[0][0] == "wm" or self.cfg.get("graph_ac"))
        buf = {k: v.clone() for k, v in b.items()}
        noise = Noise(legacy=False)
        st = {"sig": sig, "buf": buf, "graph": None, "out": None, "noise": noise}
        dev = next(self.parameters()).device
        snap = [t.clone() for t in self.fast_state()]
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if dev.type == "cuda" else None
        self.noise = noise
        noise.start_record()
        self.zero_grads()
        fn(buf)
        noise.finish_record(dev)
        self.zero_grads()
        if dev.type == "cuda" and can_graph and not getattr(self, "graph_failed", False):
            err, tracer = None, OpTrace()
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self.zero_grads()
                        noise.fill()
                        fn(buf)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.zero_grads()
                noise.fill()
                g = torch.cuda.CUDAGraph()
                with tracer, torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), capture_error_mode="thread_local"):
                    try:
                        st["out"] = fn(buf)
                    except Exception as e:
                        err = e
                if err is None:
                    st["graph"] = g
            except Exception as e:
                err = err or e
            if err is not None:
                self.graph_failed = True
                self.graph_error = "".join(traceback.format_exception(err)) + f" | last op before failure: {tracer.last}"
                print(f"cuda graph capture failed: {err!r} after {tracer.last}")
                st["graph"], st["out"] = None, None
                try:
                    torch.cuda.synchronize()
                except Exception:
                    self.cfg["graph"] = False
                self.zero_grads()
                st["buf"] = {k: v.clone() for k, v in b.items()}
        for t, s0 in zip(self.fast_state(), snap):
            t.copy_(s0)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            try:
                torch.cuda.set_rng_state(cuda_rng)
            except Exception as e:
                print(f"cuda rng restore failed: {e!r}")
        self.noise = Noise()
        if DIST.on and DIST.any(getattr(self, "graph_error", None)) and not getattr(self, "graph_error", None):
            self.graph_failed, self.graph_error = True, "cuda graph capture failed on another rank"
            st["graph"], st["out"] = None, None
        if getattr(self, "graph_error", None) and self.cfg.get("graph_strict"):
            raise RuntimeError(f"cuda graph capture failed and graph_strict forbids the eager fallback: {self.graph_error}")
        return st

    def fast_step(self, key, fn, b):
        gs = self.__dict__.setdefault("graphs", {})
        sig = (key, tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(b.items())))
        if sig not in gs:
            gs[sig] = self.fast_build(fn, b, sig)
        st = gs[sig]
        for k, v in b.items():
            st["buf"][k].copy_(v)
        self.noise = st["noise"]
        st["noise"].fill()
        if st["graph"] is not None:
            st["graph"].replay()
        else:
            self.zero_grads()
            try:
                st["out"] = fn(st["buf"])
            finally:
                self.noise = Noise()
        self.noise = Noise()
        if not self.apply_grads():
            return None
        self.post_step()
        return {k: v.clone() for k, v in st["out"].items()}

    def amp(self):
        mode = self.cfg.get("amp")
        if mode is None:
            return contextlib.nullcontext()
        dev = self.edges.device.type
        if "amp_ok" not in self.__dict__:
            self.__dict__["amp_ok"] = mode == "bf16" and (dev != "cuda" or torch.cuda.is_bf16_supported())
        if not self.amp_ok:
            raise RuntimeError(f"amp {mode!r} is not supported on {dev}")
        return torch.autocast(dev, dtype=torch.bfloat16, cache_enabled=False)

    def learn_loss(self, b, mode=False):
        with self.amp():
            return self.learn_loss_body(b, mode)

    def learn_loss_body(self, b, mode=False):
        L, deter, stoch, feat, mu, ls, psym = self.world_loss(b, mode)
        total = sum(self.wm_weight(k) * L[k] for k in L)
        with torch.no_grad():
            err = (mu - psym) ** 2
            m = b.get("mask")
            rmse = (err.mean((0, 1)) if m is None else (err * m[..., None]).sum((0, 1)) / m.sum().clamp(min=1.0)).sqrt()
        out = {k: x.detach() for k, x in L.items()}
        out.update(priv_rmse_n=rmse.mean(), priv_rmse_dims=rmse, sigma=ls.detach().exp().mean())
        return total, out

    def wm_weight(self, k):
        if k.startswith("ms_"):
            ms = self.cfg["wm_multistep"]
            return float(ms.get("weight", 1.0)) * (float(ms.get("kl", 0.0)) if k == "ms_kl" else 1.0)
        w = self.cfg["scales"][k]
        return w * float(self.cfg.get("priv_aux_scale", 5.0)) if k == "priv" and self.cfg.get("priv_aux", False) else w

    def step_opt(self, total):
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        return self.apply_grads()

    def sync_state(self):
        return []

    def apply_grads(self):
        c = self.cfg
        params = self.trainable()
        if DIST.on:
            DIST.mean_([p.grad for p in params if p.grad is not None])
            DIST.bcast_(self.sync_state())
        norms = torch._foreach_norm([p.grad for p in params if p.grad is not None])
        if not bool(torch.isfinite(torch.stack(norms)).all()):
            self.skips += 1
            self.nonfinite += 1
            if self.skips > c["max_nonfinite"]:
                raise RuntimeError("non-finite gradients")
            return False
        self.skips = 0
        clip_agc(params, c["agc"], c["pmin"])
        lr0 = c["lr"] if self.lr_ramp is None else self.lr_ramp
        for g in self.opt.param_groups:
            g["lr"] = lr0 * min(1.0, (self.steps + 1) / c["warmup"]) if c["warmup"] else lr0
            for tag, sc in (("wm", self.wm_lr_scale), ("actor", self.actor_lr_scale), ("critic", self.critic_lr_scale)):
                if g.get(tag):
                    g["lr"] *= sc
        self.opt.step()
        self.steps += 1
        return True


class Dreamer(WorldModel):
    def __init__(self, cfg, obs_dim, act_dim, priv_dim):
        super().__init__(cfg, obs_dim, act_dim, priv_dim)
        c, u = cfg, cfg["units"]
        feat = c["stoch"] * c["discrete"] + c["deter"]
        cin = feat + (priv_dim if c["critic_priv"] != "none" else 0)
        self.actor_in = c.get("actor_in", "latent")
        if self.actor_in not in ACTOR_INS:
            raise ValueError(f"model.actor_in must be one of {ACTOR_INS}, got {self.actor_in!r}")
        self.gru_hidden = int(c.get("gru_hidden", 128))
        ain = feat + (obs_dim if self.actor_in == "latent_u" else 0) + (self.gru_hidden if self.actor_in == "gru_u" else 0)
        if self.actor_in != "latent":
            self.cfg = {**c, "graph": False, "graph_ac": False}
            self.register_buffer("actor_u_mean", torch.zeros(obs_dim))
            self.register_buffer("actor_u_std", torch.ones(obs_dim))
            self.register_buffer("actor_u_fit", torch.zeros(1))
        if self.actor_in == "gru_u":
            self.actor_gru = nn.GRUCell(obs_dim, self.gru_hidden)
        self.actor = MLP(ain, 2 * act_dim, u, c["layers"], outscale=0.01)
        self.critic = MLP(cin, c["bins"], u, c["layers"], outscale=0.0)
        self.slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.retnorm = ReturnNorm(c["ret_decay"], c["ret_lo"], c["ret_hi"])
        self.gamma = (1.0 - 1.0 / c["disc_horizon"]) ** (1 if c.get("gamma_per_step") else int(c.get("action_repeat", 1)))
        self.register_buffer("ac_ctl", torch.tensor([float(c["scales"]["policy"]), 0.0, float(c["ent"])]), persistent=False)
        self.ent_t, self.ent_val = False, float(c["ent"])
        self.bc_beta = 0.0
        self.kl_units = "raw"
        self.advnorm = ReturnNorm(c["ret_decay"], c["ret_lo"], c["ret_hi"])
        self.adv_mode, self.adv_floor, self.dual, self.imag_clamp = "return", 0.05, None, False
        self.imag_prior, self.actor_grad, self.actor_rho = None, "reinforce", 1.0
        self.register_buffer("dual_lb", torch.zeros(1), persistent=False)
        self.register_buffer("dual_tgt", torch.zeros(1), persistent=False)
        self.register_buffer("env_ctl", torch.zeros(2, 2), persistent=False)
        self.register_buffer("env_seen", torch.zeros(2), persistent=False)
        self.__dict__["bc_actor"] = None
        self.ens_k, self.dyn_k = int(c.get("reward_ensemble", 0)), int(c.get("dyn_ensemble", 0))
        if self.ens_active():
            hl, hin = c["head_layers"], feat + (act_dim if c["head_action"] else 0)
            self.ens_rew = nn.ModuleList([MLP(hin, c["bins"], u, hl, outscale=0.0) for _ in range(self.ens_k)])
            self.ens_dyn = nn.ModuleList([self.dyn_head(c) for _ in range(self.dyn_k)])
            self.register_buffer("pess_ctl", torch.zeros(2), persistent=False)
            self.ens_log = {}
            if not c.get("ens_graph"):
                self.cfg = {**self.cfg, "graph_ac": False}
        self.init_opt()

    def ens_active(self):
        return self.ens_k > 0 or self.dyn_k > 0

    @staticmethod
    def dyn_head(c):
        out = nn.Linear(c["hidden"], c["stoch"] * c["discrete"])
        trunc_init(out, 1.0)
        return nn.Sequential(RSSM.block(c["deter"], c["hidden"]), out, Fp32())

    def rew_members(self, hin):
        return torch.stack([twohot_decode(m(hin), self.edges) for m in [self.rew, *self.ens_rew]])

    def dyn_std(self, deter):
        c = self.cfg
        p = torch.stack([cat_probs(l, c["stoch"], c["discrete"], c["unimix"]) for l in [self.rssm.prior_logit(deter), *[h(deter) for h in self.ens_dyn]]])
        return p.var(0, unbiased=False).sum(-1).mean(-1).sqrt()

    def ens_stats(self, feat, act):
        r = self.rew_members(self.head_in(feat, act))
        mean, std = r.mean(0), r.std(0, unbiased=False)
        if self.dyn_k:
            std = std + float(self.cfg["dyn_pess_coef"]) * self.dyn_std(feat[..., -self.cfg["deter"]:])
        return mean, std

    def pess_update(self, unc):
        pct = float(self.cfg["trunc_pct"])
        if pct <= 0:
            return
        q = torch.quantile(unc.detach().flatten().float(), pct / 100.0)
        d = float(self.cfg["trunc_decay"])
        self.pess_ctl[0].copy_(torch.where(self.pess_ctl[1] > 0, d * self.pess_ctl[0] + (1 - d) * q, q))
        self.pess_ctl[1].fill_(1.0)

    def pess_heads(self, ifeat, iact):
        c, H = self.cfg, self.cfg["horizon"]
        f, a = (ifeat[:, :H], iact[:, :H].clamp(-1, 1)) if c["head_action"] else (ifeat[:, 1:], None)
        mean, unc = self.ens_stats(f, a)
        idisc = self.gamma * torch.sigmoid(self.con(self.head_in(f, a))[..., 0])
        cut = (unc > self.pess_ctl[0]) & (self.pess_ctl[1] > 0) if float(c["trunc_pct"]) > 0 else torch.zeros_like(unc, dtype=torch.bool)
        self.ens_log = {"ens_std_imag": unc.mean(), "trunc_frac": cut.float().mean(), "pess_thr": self.pess_ctl[0].clone()}
        return mean - float(c["pess_coef"]) * unc, idisc * (~cut).float()

    def ens_losses(self, batch, deter, feat):
        c = self.cfg
        rew, act, obs = batch["rew"][..., 0], batch["act"], batch["obs"]
        hin = self.head_in(feat.detach(), act)
        out = {}
        if self.ens_k:
            u = self.noise.uniform((self.ens_k, *rew.shape), rew)
            m = (u < float(c["ens_boot"])).float()
            nll = torch.stack([twohot_nll(h(hin), rew, self.edges) for h in self.ens_rew])
            out["ens_rew"] = ((m * nll).sum((1, 2)) / m.sum((1, 2)).clamp(min=1.0)).mean()
        if self.dyn_k:
            with torch.no_grad():
                q = self.rssm.post_logit(deter.detach(), self.enc(obs))
            out["ens_dyn"] = torch.stack([kl_terms(q, h(deter.detach()), c["stoch"], c["discrete"], c["unimix"], 0.0)[0].mean() for h in self.ens_dyn]).mean()
        return out

    def freeze_wm(self):
        for n, p in self.named_parameters():
            if not n.startswith(HEAD_PREFIX):
                p.requires_grad_(False)
        self.init_opt()

    def set_anchor(self, state=None):
        a = copy.deepcopy(self.actor).requires_grad_(False)
        if state is not None:
            a.load_state_dict(state)
        self.__dict__["bc_actor"] = a

    data_anchor = None

    def set_data_anchor(self, cfg):
        self.data_anchor = None if not cfg else {"coef": float(cfg["coef"]), "follow_beta": bool(cfg.get("follow_beta", True)), "replace_kl": bool(cfg.get("replace_kl", False)), "target": cfg.get("target", "sample"), "sigma": float(cfg.get("sigma", 0.3)), "feat_ref": cfg.get("feat_ref"), "all_rows": bool(cfg.get("all_rows", False))}
        self.__dict__["frozen_wm"] = None
        if self.data_anchor is not None and self.data_anchor["target"] not in ("sample", "teacher_mu", "teacher_live"):
            raise ValueError(f"data anchor target must be 'sample', 'teacher_mu' or 'teacher_live', got {self.data_anchor['target']!r}")
        if self.data_anchor is not None and self.data_anchor["feat_ref"] not in (None, "frozen_wm"):
            raise ValueError(f"data anchor feat_ref must be None or 'frozen_wm', got {self.data_anchor['feat_ref']!r}")
        if self.data_anchor is not None and self.data_anchor["feat_ref"] and self.data_anchor["target"] != "teacher_live":
            raise ValueError("data anchor feat_ref needs target teacher_live")
        if self.data_anchor is not None and self.data_anchor["feat_ref"]:
            self.__dict__["frozen_wm"] = (copy.deepcopy(self.enc).requires_grad_(False), copy.deepcopy(self.rssm).requires_grad_(False))

    def mode_feats(self, batch, enc, rssm):
        with torch.no_grad():
            deter, stoch, _, _ = rssm.observe(enc(batch["obs"]), batch["act"], batch["first"][..., 0], batch.get("prev"), True, self.noise)
            return self.feat(deter, stoch)

    @staticmethod
    def censored_logp(dist, a):
        n = dist.base_dist
        hi, lo = a >= 1.0 - 1e-6, a <= -1.0 + 1e-6
        inner = n.log_prob(a.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        up, dn = torch.special.log_ndtr((n.loc - 1.0) / n.scale), torch.special.log_ndtr((-1.0 - n.loc) / n.scale)
        return torch.where(hi, up, torch.where(lo, dn, inner)).sum(-1)

    def set_sigma(self, sigma, net=None):
        lo, hi = float(self.cfg["min_std"]), float(self.cfg["max_std"])
        if not lo < float(sigma) < hi:
            raise ValueError(f"sigma_init {sigma} must lie strictly between min_std {lo} and max_std {hi}")
        p = (float(sigma) - lo) / (hi - lo)
        out = (self.actor if net is None else net).out
        half = out.out_features // 2
        with torch.no_grad():
            out.weight[half:].zero_()
            out.bias[half:].fill_(math.log(p / (1.0 - p)) - 2.0)

    def reset_actor_opt(self):
        for n, p in self.named_parameters():
            if n.startswith(("actor.", "actor_")):
                self.opt.state.pop(p, None)

    @torch.no_grad()
    def remap_std(self, lo0, hi0, probe=4096):
        lo, hi, out = float(self.cfg["min_std"]), float(self.cfg["max_std"]), self.actor.out
        half = out.out_features // 2
        h = self.actor.body(0.5 * torch.randn(probe, self.actor.body[0].in_features, generator=torch.Generator().manual_seed(0)).to(out.weight.device))

        def sigma(raw, lo_, hi_):
            return (hi_ - lo_) * torch.sigmoid(raw + 2.0) + lo_

        raw0 = out(h)[:, half:]
        r0 = raw0.mean(0)
        p0, s0 = torch.sigmoid(r0 + 2.0), sigma(r0, lo0, hi0)
        q = ((s0 - lo) / (hi - lo)).clamp(1e-4, 1.0 - 1e-4)
        slope = (hi0 - lo0) * p0 * (1.0 - p0) / ((hi - lo) * q * (1.0 - q))
        out.weight[half:] *= slope[:, None]
        out.bias[half:] = torch.log(q / (1.0 - q)) - 2.0 - slope * (r0 - out.bias[half:])
        return {"sigma_before": float(sigma(raw0, lo0, hi0).mean()), "sigma_after": float(sigma(out(h)[:, half:], lo, hi).mean())}

    def anchor_dist(self, feat):
        mu, raw = self.bc_actor(feat).chunk(2, -1)
        std = (self.cfg["max_std"] - self.cfg["min_std"]) * torch.sigmoid(raw + 2.0) + self.cfg["min_std"]
        return torch.distributions.Independent(torch.distributions.Normal(torch.tanh(mu), std, validate_args=False), 1, validate_args=False)

    def configure_switches(self, adv_norm="return", adv_floor=0.05, target_nats=None, dual_lr=0.01, beta0=1.0, imag_clamp=False, dual_floor=0.0, dyn_target=False, imag_prior=None, actor_grad="reinforce", actor_grad_mix=1.0):
        if adv_norm not in ("return", "adv"):
            raise ValueError(f"train.adv_norm must be 'return' or 'adv', got {adv_norm!r}")
        if imag_prior not in (None, "mode"):
            raise ValueError(f"train.imag_prior must be None or 'mode', got {imag_prior!r}")
        if actor_grad not in ("reinforce", "dynamics", "mix"):
            raise ValueError(f"train.actor_grad must be 'reinforce', 'dynamics' or 'mix', got {actor_grad!r}")
        if not 0.0 <= float(actor_grad_mix) <= 1.0:
            raise ValueError(f"train.actor_grad_mix must be in [0, 1], got {actor_grad_mix!r}")
        if actor_grad != "reinforce" and (self.ens_active() or self.actor_in != "latent"):
            raise ValueError("train.actor_grad dynamics/mix supports neither reward/dyn ensembles nor model.actor_in other than 'latent'")
        self.imag_prior, self.actor_grad = imag_prior, actor_grad
        self.actor_rho = {"reinforce": 1.0, "dynamics": 0.0, "mix": float(actor_grad_mix)}[actor_grad]
        self.adv_mode, self.adv_floor, self.imag_clamp = adv_norm, float(adv_floor), bool(imag_clamp)
        self.dual = None if target_nats is None else {"target": float(target_nats), "lr": float(dual_lr), "lo": max(1e-3, float(dual_floor)), "hi": float(beta0) * 10.0, "dyn": bool(dyn_target)}
        if self.dual is not None:
            self.dual_tgt.fill_(float(target_nats))
        self.dual_reset(beta0)

    def reset_opt_state(self):
        self.opt.state.clear()

    def set_kl_target(self, target):
        self.dual["target"] = float(target)
        self.dual_tgt.fill_(float(target))

    def dual_reset(self, beta0=None):
        if self.dual is not None:
            if beta0 is not None:
                self.dual["beta0"] = float(beta0)
            self.dual_lb.fill_(math.log(self.dual["beta0"]))

    def env_update(self, k, x):
        x, d, e = x.detach(), self.cfg["ret_decay"], self.env_ctl[k]
        b = torch.stack([x.amin(), x.amax()])
        slow = torch.stack([d * e[0] + (1 - d) * b[0], d * e[1] + (1 - d) * b[1]])
        e.copy_(torch.where(self.env_seen[k] > 0, torch.stack([torch.minimum(b[0], slow[0]), torch.maximum(b[1], slow[1])]), b))
        self.env_seen[k].fill_(1.0)

    def env_clip(self, k, x):
        lo, hi = self.env_ctl[k][0], self.env_ctl[k][1]
        m, on = 0.1 * (hi - lo), self.env_seen[k] > 0
        inf = torch.full_like(lo, float("inf"))
        return torch.maximum(torch.minimum(x, torch.where(on, hi + m, inf)), torch.where(on, lo - m, -inf))

    def sync_state(self):
        st = self.fast_state()
        return [t for t in st[:len(st) - len(list(self.slow.parameters()))] if t.is_floating_point()]

    def fast_state(self):
        dev = next(self.parameters()).device
        self.retnorm.ensure(dev)
        st = [self.retnorm.v, self.retnorm.seen]
        if self.adv_mode == "adv":
            self.advnorm.ensure(dev)
            st += [self.advnorm.v, self.advnorm.seen]
        if self.dual is not None:
            st.append(self.dual_lb)
            if self.dual["dyn"]:
                st.append(self.dual_tgt)
        if self.imag_clamp:
            st += [self.env_ctl, self.env_seen]
        if self.ens_active():
            st.append(self.pess_ctl)
        return st + list(self.slow.parameters())

    def post_step(self):
        with torch.no_grad():
            torch._foreach_lerp_(list(self.slow.parameters()), list(self.critic.parameters()), self.cfg["slow_frac"])

    def fit_u_norm(self, u):
        x = symlog(u.reshape(-1, u.shape[-1]).float())
        self.actor_u_mean.copy_(x.mean(0))
        self.actor_u_std.copy_(x.std(0).clamp_min(1e-2))
        self.actor_u_fit.fill_(1.0)

    def u_norm(self, u):
        return (symlog(u) - self.actor_u_mean) / self.actor_u_std

    def actor_inp(self, feat, u=None, g=None):
        if self.actor_in == "latent":
            return feat
        if u is None or (self.actor_in == "gru_u" and g is None):
            raise ValueError(f"model.actor_in {self.actor_in!r} reads the raw u (and the actor GRU state), the imagination and latent-only paths support only 'latent'")
        return torch.cat([feat, self.u_norm(u)], -1) if self.actor_in == "latent_u" else torch.cat([feat, g], -1)

    def actor_gru_seq(self, u, first):
        x, g, out = self.u_norm(u), torch.zeros(u.shape[0], self.gru_hidden, device=u.device), []
        for t in range(u.shape[1]):
            g = f32(self.actor_gru(x[:, t], torch.where(first[:, t, None] > 0, torch.zeros_like(g), g)))
            out.append(g)
        return torch.stack(out, 1)

    def actor_seq_dist(self, feat, u, first):
        if self.actor_in == "latent":
            return self.actor_dist(feat)
        return self.actor_dist(feat, u, self.actor_gru_seq(u, first) if self.actor_in == "gru_u" else None)

    def actor_dist(self, feat, u=None, g=None):
        mu, raw = self.actor(self.actor_inp(feat, u, g)).chunk(2, -1)
        std = (self.cfg["max_std"] - self.cfg["min_std"]) * torch.sigmoid(raw + 2.0) + self.cfg["min_std"]
        return torch.distributions.Independent(torch.distributions.Normal(torch.tanh(mu), std, validate_args=False), 1, validate_args=False)

    def critic_in(self, feat, priv_sym):
        return feat if self.cfg["critic_priv"] == "none" else torch.cat([feat, priv_sym], -1)

    def value(self, net, x):
        return twohot_decode(net(x), self.edges)

    @torch.no_grad()
    def act(self, obs, first, prev_act, state, greedy=False):
        if state is None:
            state = self.rssm.initial(obs.shape[0])
        deter, stoch, _, _ = self.rssm.obs_step(state[0], state[1], prev_act, first, self.enc(obs))
        dist = self.actor_dist(self.feat(deter, stoch))
        return (dist.mean if greedy else dist.sample()).clamp(-1, 1), (deter, stoch)

    @torch.no_grad()
    def imagine(self, deter, stoch):
        return self.rollout(deter, stoch)

    def frozen(self):
        return Frozen([p for n, p in self.named_parameters() if p.requires_grad and not n.startswith(("actor.", "actor_gru."))])

    @staticmethod
    def st_clamp(a):
        return a.clamp(-1, 1).detach() + (a - a.detach())

    def rollout(self, deter, stoch, grad=False):
        H, feats, acts = self.cfg["horizon"], [], []
        for h in range(H + 1):
            feat = self.feat(deter, stoch)
            dist = self.actor_dist(feat)
            if self.noise.legacy:
                a = dist.rsample() if grad else dist.sample()
            else:
                a = dist.mean + dist.base_dist.scale * self.noise.normal(dist.mean.shape, dist.mean)
            feats.append(feat)
            acts.append(a)
            if h < H:
                deter, stoch, _ = self.rssm.img_step(deter, stoch, self.st_clamp(a) if grad else a.clamp(-1, 1), self.noise, self.imag_prior == "mode")
        return torch.stack(feats, 1), torch.stack(acts, 1)

    def dyn_return(self, ifeat, iact):
        c = self.cfg
        with self.frozen():
            irew, idisc = self.imag_heads(ifeat, self.st_clamp(iact))
            ipriv = self.priv_head(ifeat)[0] if c["critic_priv"] != "none" else None
            v = self.value(self.critic, self.critic_in(ifeat, ipriv))
            if self.imag_clamp:
                irew, v = self.env_clip(0, irew), self.env_clip(1, v)
            return lambda_return(irew, v, idisc, c["lam"])

    def imag_heads(self, ifeat, iact):
        if self.ens_active():
            return self.pess_heads(ifeat, iact)
        H = self.cfg["horizon"]
        if self.cfg["head_action"]:
            hin = self.head_in(ifeat[:, :H], iact[:, :H].clamp(-1, 1))
            return twohot_decode(self.rew(hin), self.edges), self.gamma * torch.sigmoid(self.con(hin)[..., 0])
        return twohot_decode(self.rew(ifeat), self.edges)[:, 1:], self.gamma * torch.sigmoid(self.con(ifeat)[..., 0])[:, 1:]

    def replay_returns(self, rew, term, first, boot):
        c = self.cfg
        if c["head_action"]:
            rdisc = self.gamma * (1 - term[:, :-1, 0]) * (1 - first[:, 1:, 0])
            trunc = (1 - term[:, :-1, 0]) * first[:, 1:, 0]
            return lambda_return(rew[:, :-1, 0], boot, rdisc, c["lam"], trunc), 1 - trunc
        rdisc = self.gamma * (1 - term[:, 1:, 0]) * (1 - first[:, 1:, 0])
        return lambda_return(rew[:, 1:, 0], boot, rdisc, c["lam"]), (1 - term[:, :-1, 0]) * (1 - first[:, 1:, 0])

    def enable_ent_tensor(self):
        if self.__dict__.get("graphs"):
            raise RuntimeError("the entropy coefficient must become a device tensor before the first captured step")
        self.ent_t = True

    def ent_c(self):
        return self.ac_ctl[2] if self.ent_t else self.cfg["ent"]

    def update(self, batch):
        self.ac_ctl[0].fill_(float(self.cfg["scales"]["policy"]))
        if self.ent_t:
            self.ac_ctl[2].fill_(float(self.ent_val))
        if self.dual is None:
            self.ac_ctl[1].fill_(float(self.bc_beta))
        if self.cfg.get("fast"):
            return self.fast_step("ac", lambda bb: self.backward_of(self.update_loss(bb)), batch)
        total, out = self.update_loss(batch)
        if not self.step_opt(total):
            return None
        self.post_step()
        return out

    def update_loss(self, batch):
        with self.amp():
            return self.update_loss_body(batch)

    def update_loss_body(self, batch):
        c, s = self.cfg, self.cfg["scales"]
        priv, rew, first, term = (batch[k] for k in ("priv", "rew", "first", "term"))
        B, T = batch["obs"].shape[:2]
        L, deter, stoch, feat, mu, ls, psym = self.world_loss(batch)
        with torch.no_grad():
            err = self.priv_dec(mu) - priv
            priv_rmse = err.pow(2).mean((0, 1)).sqrt()
        start_deter, start_stoch = deter.detach().flatten(0, 1), stoch.detach().flatten(0, 1)
        if self.ens_active():
            with torch.no_grad():
                u_real = self.ens_stats(feat.detach(), batch["act"])[1]
                self.pess_update(u_real)
        dyn = self.actor_grad != "reinforce"
        if dyn:
            with self.frozen():
                ifeat_g, iact_g = self.rollout(start_deter, start_stoch, True)
        with torch.no_grad():
            ifeat, iact = (ifeat_g.detach(), iact_g.detach()) if dyn else self.imagine(start_deter, start_stoch)
            H = c["horizon"]
            irew, idisc = self.imag_heads(ifeat, iact)
            w = torch.cat([torch.ones_like(idisc[:, :1]), idisc.cumprod(1)], 1)[:, :H]
            ipriv = self.priv_head(ifeat)[0] if c["critic_priv"] != "none" else None
            slow_i = self.value(self.slow, self.critic_in(ifeat, ipriv))
        vlogit = self.critic(self.critic_in(ifeat, ipriv))
        v = twohot_decode(vlogit, self.edges).detach()
        if self.imag_clamp:
            self.env_update(0, rew)
            irew, v = self.env_clip(0, irew), self.env_clip(1, v)
            ret_raw = lambda_return(irew, v, idisc, c["lam"])
            ret = self.env_clip(1, ret_raw)
            clamp_frac = (ret != ret_raw).float().mean()
        else:
            ret = lambda_return(irew, v, idisc, c["lam"])
        scale = self.retnorm(ret)
        if self.adv_mode == "adv":
            adv_scale = self.advnorm(ret - v[:, :H], floor=3.29 * self.adv_floor * scale.detach()) / 3.29
            adv = ((ret - v[:, :H]) / adv_scale).detach()
        else:
            adv = ((ret - v[:, :H]) / scale).detach()
        dist = self.actor_dist(ifeat[:, :H])
        if dyn:
            dyn_t = self.dyn_return(ifeat_g, iact_g)[:, :H] / scale.detach()
            L["policy"] = -(w * (self.actor_rho * dist.log_prob(iact[:, :H]) * adv + (1 - self.actor_rho) * dyn_t + self.ent_c() * dist.entropy())).mean()
        else:
            L["policy"] = -(w * (dist.log_prob(iact[:, :H]) * adv + self.ent_c() * dist.entropy())).mean()
        a_sigma = dist.base_dist.scale.detach()
        L["value"] = (w * (twohot_nll(vlogit[:, :H], ret, self.edges) + twohot_nll(vlogit[:, :H], slow_i[:, :H], self.edges))).mean()
        rpriv = {"none": None, "decoded": mu.detach(), "mixed": psym}[c["critic_priv"]]
        rin = self.critic_in(feat.detach() if c["repval_detach"] else feat, rpriv)
        with torch.no_grad():
            slow_r = self.value(self.slow, rin)
            rret, rmask = self.replay_returns(rew, term, first, ret[:, 0].reshape(B, T))
        if self.imag_clamp:
            self.env_update(1, rret)
        rlogit = self.critic(rin)
        L["repval"] = (rmask * (twohot_nll(rlogit[:, :-1], rret, self.edges) + twohot_nll(rlogit[:, :-1], slow_r[:, :-1], self.edges))).mean()
        if self.bc_actor is not None:
            with torch.no_grad():
                ref_i, ref_r = self.anchor_dist(ifeat[:, :H]), self.anchor_dist(feat.detach())
            kl_i = torch.distributions.kl_divergence(dist, ref_i)
            kl_r = torch.distributions.kl_divergence(self.actor_dist(feat.detach()), ref_r)
            L["kl_bc"] = 0.5 * ((w * kl_i).mean() + kl_r.mean())
            kl_ref = L["kl_bc"].detach()
            if self.kl_units == "return" and self.dual is None:
                L["kl_bc"] = L["kl_bc"] / scale.detach()
            mu_bc_diff = (dist.base_dist.loc.detach() - ref_i.base_dist.loc).abs().mean()
            if self.data_anchor is not None and self.data_anchor.get("replace_kl"):
                del L["kl_bc"]
        if self.data_anchor is not None and ("clean" in batch or self.data_anchor["all_rows"]):
            cm = torch.ones_like(rew[..., 0]) if self.data_anchor["all_rows"] else batch["clean"][..., 0]
            if self.data_anchor["target"] == "teacher_live":
                fr = self.frozen_wm
                f_loss, f_ref = (self.mode_feats(batch, self.enc, self.rssm), self.mode_feats(batch, *fr)) if fr is not None else (feat.detach(),) * 2
                with torch.no_grad():
                    mu_ref = self.anchor_dist(f_ref).base_dist.loc
                nll = (0.5 * ((self.actor_dist(f_loss).base_dist.loc - mu_ref) / self.data_anchor["sigma"]) ** 2).sum(-1)
            elif self.data_anchor["target"] == "teacher_mu":
                nll = (0.5 * ((self.actor_dist(feat.detach()).base_dist.loc - batch["mu_ref"]) / self.data_anchor["sigma"]) ** 2).sum(-1)
            else:
                nll = -self.censored_logp(self.actor_dist(feat.detach()), batch["act"].clamp(-1, 1))
            L["bc_data"] = float(self.data_anchor["coef"]) * (nll * cm).sum() / cm.sum().clamp(min=1.0)
        beta = self.ac_ctl[1]
        if self.dual is not None:
            beta = self.dual_lb.exp().clamp(self.dual["lo"], self.dual["hi"])[0]
        total = sum((self.ac_ctl[0] if k == "policy" else beta if k == "kl_bc" or (k == "bc_data" and self.data_anchor.get("follow_beta", True)) else 1.0 if k == "bc_data" else self.wm_weight(k)) * L[k] for k in L)
        if self.ens_active():
            lens = self.ens_losses(batch, deter, feat)
            total = total + sum(lens.values())
            L.update(lens)
        if self.dual is not None and self.bc_actor is not None:
            d = self.dual
            tgt = self.dual_tgt if d["dyn"] else d["target"]
            step = (self.ac_ctl[0] > 0).float() * d["lr"] * (kl_ref - tgt) / tgt
            self.dual_lb.copy_((self.dual_lb + step).clamp(math.log(d["lo"]), math.log(d["hi"])))
        out = {k: x.detach() for k, x in L.items()}
        out.update(ret_scale=scale, priv_rmse=priv_rmse.mean(), priv_rmse_dims=priv_rmse, wm_sigma=ls.detach().exp().mean(), actor_sigma_mean=a_sigma.mean(),
                   actor_sigma_min=a_sigma.amin(), adv_std=adv.std(),
                   ret_mean=ret[:, :H].mean().detach(), v_mean=v[:, :H].mean(), adv_mean=adv.mean(), v_ev=1.0 - (ret[:, :H] - v[:, :H]).var() / ret[:, :H].var().clamp(min=1e-8))
        if self.bc_actor is not None:
            out["mu_bc_diff"] = mu_bc_diff
            if "kl_bc" not in L:
                out["kl_bc"] = kl_ref
        if self.ens_active():
            out.update(self.ens_log)
            out["ens_std_real"] = u_real.mean()
        if self.adv_mode == "adv":
            out["adv_scale"] = adv_scale
        if self.imag_clamp:
            out["clamp_frac"] = clamp_frac
        if self.dual is not None and self.bc_actor is not None:
            out["beta"] = beta.detach()
        return total, out

    def grad_probe(self, batch, n=64):
        c, H, dev = self.cfg, self.cfg["horizon"], next(self.parameters()).device
        params = list(self.actor.parameters())
        norm = lambda g: float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))
        with torch.enable_grad(), torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            with torch.no_grad():
                _, deter, stoch, feat, _, _, _ = self.world_loss(batch)
                sd, ss = deter.detach().flatten(0, 1), stoch.detach().flatten(0, 1)
                pick = torch.randperm(sd.shape[0], device=sd.device)[:n]
                ifeat, iact = self.imagine(sd[pick], ss[pick])
                irew, idisc = self.imag_heads(ifeat, iact)
                w = torch.cat([torch.ones_like(idisc[:, :1]), idisc.cumprod(1)], 1)[:, :H]
                ipriv = self.priv_head(ifeat)[0] if c["critic_priv"] != "none" else None
                v = self.value(self.critic, self.critic_in(ifeat, ipriv))
                ret = lambda_return(irew, v, idisc, c["lam"])
                adv = (ret - v[:, :H]) / (self.retnorm.v[1] - self.retnorm.v[0]).clamp(min=1.0)
            dist = self.actor_dist(ifeat[:, :H])
            pol = -(w * (dist.log_prob(iact[:, :H]) * adv + self.ent_c() * dist.entropy())).mean()
            out = {"gn_policy": norm(torch.autograd.grad(pol, params, retain_graph=True, allow_unused=True)), "gn_kl": float("nan")}
            if self.bc_actor is not None:
                with torch.no_grad():
                    ref = self.anchor_dist(ifeat[:, :H])
                kl = (w * torch.distributions.kl_divergence(dist, ref)).mean()
                out["gn_kl"] = norm(torch.autograd.grad(kl, params, allow_unused=True))
            out["gn_policy_eff"] = float(self.ac_ctl[0]) * out["gn_policy"]
            out["gn_kl_eff"] = float(self.dual_lb.exp().clamp(self.dual["lo"], self.dual["hi"]) if self.dual is not None else self.ac_ctl[1]) * out["gn_kl"]
        return out


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def evaluate(agent, vec, dev, episodes):
    n = vec.n
    obs, _ = zip(*[vec.reset(i) for i in range(n)])
    obs = np.stack(obs)
    first, prev, state, ret, done, out = np.ones((n, 1), np.float32), np.zeros((n, agent.A), np.float32), None, np.zeros(n), np.zeros(n, bool), []
    while not done.all():
        a, state = agent.act(torch.as_tensor(obs, device=dev), torch.as_tensor(first[:, 0], device=dev), torch.as_tensor(prev, device=dev), state, greedy=True)
        a = a.cpu().numpy()
        obs, _, r, term, trunc = vec.step(a)
        ret += r * ~done
        done |= term | trunc
        first[:] = 0
        prev = a
        for i in np.flatnonzero(term | trunc):
            obs[i], _ = vec.reset(i)
    return float(ret[:episodes].mean())


def train(cfg, make_vec):
    torch.set_num_threads(int(cfg["cpus"]))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu" if cfg["device"] == "auto" else cfg["device"])
    logdir = Path(cfg["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(json.dumps(cfg, default=str), encoding="utf-8")
    E = cfg["envs"]
    vec, evec = make_vec(cfg["task"], E, cfg["seed"]), make_vec(cfg["task"], cfg["eval_episodes"], cfg["seed"] + 1000)
    O, A, P = vec.obs_dim, vec.act_dim, vec.priv_dim
    agent = Dreamer(cfg, O, A, P).to(dev)
    replay = Replay(E, cfg["replay_cap"] // E, {"obs": O, "priv": P, "act": A, "rew": 1, "first": 1, "term": 1})
    obs, priv = (np.stack(x) for x in zip(*[vec.reset(i) for i in range(E)]))
    rew, first, term = np.zeros((E, 1), np.float32), np.ones((E, 1), np.float32), np.zeros((E, 1), np.float32)
    prev, state = np.zeros((E, A), np.float32), None
    ep_ret, ep_len = np.zeros(E), np.zeros(E, dtype=np.int64)
    env_steps, credit, updates, next_eval, t0 = 0, 0.0, 0, 0, time.time()
    bt = cfg["batch"] * cfg["T"]
    while env_steps < cfg["steps"]:
        if env_steps < cfg["prefill"]:
            act = rng.uniform(-1, 1, size=(E, A)).astype(np.float32)
        else:
            a, state = agent.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), torch.as_tensor(first[:, 0], device=dev), torch.as_tensor(prev, device=dev), state)
            act = a.cpu().numpy()
        for e in range(E):
            replay.add(e, {"obs": obs[e], "priv": priv[e], "act": act[e], "rew": rew[e], "first": first[e], "term": term[e]})
        obs, priv, r, tm, tr = vec.step(act)
        env_steps += E
        ep_ret += r
        ep_len += 1
        prev, first[:], rew[:, 0], term[:, 0] = act, 0.0, r, tm
        for e in np.flatnonzero(tm | tr):
            replay.add(e, {"obs": obs[e], "priv": priv[e], "act": np.zeros(A, np.float32), "rew": rew[e], "first": first[e], "term": term[e]})
            append_jsonl(logdir / "episodes.jsonl", {"step": env_steps, "ret": float(ep_ret[e]), "len": int(ep_len[e])})
            ep_ret[e], ep_len[e] = 0.0, 0
            obs[e], priv[e] = vec.reset(e)
            rew[e], first[e], term[e], prev[e] = 0.0, 1.0, 0.0, 0.0
        if env_steps >= cfg["prefill"] and replay.ready(cfg["T"]):
            credit += E * cfg["train_ratio"]
            while credit >= bt:
                credit -= bt
                batch = {k: torch.as_tensor(v, device=dev) for k, v in replay.sample(cfg["batch"], cfg["T"], rng).items()}
                m = agent.update(batch)
                updates += 1
                if m is not None and updates % cfg["log_every"] == 0:
                    append_jsonl(logdir / "metrics.jsonl", {"update": updates, "step": env_steps, "time": time.time() - t0, **{k: (v.tolist() if v.ndim else float(v)) for k, v in m.items()}})
        if env_steps >= next_eval and env_steps >= cfg["prefill"]:
            next_eval += cfg["eval_every"]
            append_jsonl(logdir / "eval.jsonl", {"step": env_steps, "ret": evaluate(agent, evec, dev, cfg["eval_episodes"])})
    torch.save({"model": agent.state_dict(), "cfg": cfg, "updates": updates, "env_steps": env_steps}, logdir / "latest.pt")
    return agent
