import ast
import copy
import math
from collections import deque
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.utils._python_dispatch
import torch.nn.functional as F

ACTOR_INS = ("latent", "latent_u", "gru_u")
HEAD_PREFIX = ("actor.", "actor_", "critic.", "slow.", "ens_")


def f32(x):
    return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x


class Fp32(nn.Module):
    def forward(self, x):
        return f32(x)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def bin_edges(n, lim):
    half = symexp(torch.linspace(0, lim, n // 2 + 1)[1:])
    return torch.cat([-half.flip(0), torch.zeros(1), half])


def twohot(x, edges):
    n = edges.numel()
    x = x.clamp(edges[0], edges[-1])
    below = ((edges <= x[..., None]).sum(-1) - 1).clamp(0, n - 2)
    lo, hi = edges[below], edges[below + 1]
    w_hi = (x - lo) / (hi - lo)
    return F.one_hot(below, n) * (1 - w_hi)[..., None] + F.one_hot(below + 1, n) * w_hi[..., None]


def twohot_decode(logits, edges):
    p, m = F.softmax(f32(logits), -1), edges.numel() // 2
    return (edges[m + 1:] * (p[..., m + 1:] - p[..., :m].flip(-1))).sum(-1) + p[..., m] * edges[m]


def twohot_nll(logits, target, edges):
    return -(twohot(f32(target), edges) * F.log_softmax(f32(logits), -1)).sum(-1)


def cat_probs(logit, stoch, discrete, unimix):
    p = F.softmax(f32(logit).reshape(*logit.shape[:-1], stoch, discrete), -1)
    return (1 - unimix) * p + unimix / discrete


def kl_terms(post_logit, prior_logit, stoch, discrete, unimix, free, mode="sum"):
    p = cat_probs(post_logit, stoch, discrete, unimix)
    q = cat_probs(prior_logit, stoch, discrete, unimix)
    lp, lq = p.log(), q.log()
    dyn = p.detach() * (lp.detach() - lq)
    rep = p * (lp - lq.detach())
    if mode == "group":
        return dyn.sum(-1).clamp(min=free / stoch).sum(-1), rep.sum(-1).clamp(min=free / stoch).sum(-1)
    return dyn.sum((-1, -2)).clamp(min=free), rep.sum((-1, -2)).clamp(min=free)


def gauss_nll(mu, log_sigma, y):
    return (0.5 * ((y - mu) / log_sigma.exp()) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)).sum(-1)


def lambda_return(reward, boot, disc, lam, reseed=None):
    out = boot[:, -1]
    rets = []
    for t in reversed(range(reward.shape[1])):
        out = reward[:, t] + disc[:, t] * ((1 - lam) * boot[:, t + 1] + lam * out)
        if reseed is not None:
            out = torch.where(reseed[:, t] > 0, boot[:, t], out)
        rets.append(out)
    return torch.stack(rets[::-1], 1)


class Noise:
    def __init__(self, legacy=True):
        self.legacy, self.mode, self.rec, self.bufs, self.i = legacy, "inline", [], [], 0

    def draw(self, kind, shape, ref):
        if self.mode == "replay":
            t = self.bufs[self.i]
            self.i += 1
            if tuple(t.shape) != tuple(shape):
                raise RuntimeError(f"noise slot {self.i - 1} shape {tuple(t.shape)} != {tuple(shape)}")
            return t
        if self.mode == "record":
            self.rec.append((kind, tuple(shape)))
        return self.fresh(kind, shape, ref.device, ref.dtype)

    @staticmethod
    def fresh(kind, shape, device, dtype=torch.float32):
        return torch.rand(shape, device=device, dtype=dtype) if kind == "u" else torch.randn(shape, device=device, dtype=dtype)

    def uniform(self, shape, ref):
        return self.draw("u", shape, ref)

    def normal(self, shape, ref):
        return self.draw("n", shape, ref)

    def start_record(self):
        self.mode, self.rec, self.bufs, self.i = "record", [], [], 0

    def finish_record(self, device):
        self.bufs = [torch.empty(sh, device=device) for _, sh in self.rec]
        self.mode, self.i = "replay", 0

    def begin(self):
        self.i = 0

    def fill(self):
        for (kind, _), b in zip(self.rec, self.bufs):
            b.uniform_() if kind == "u" else b.normal_()
        self.i = 0


class OpTrace(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = deque(maxlen=5)

    @property
    def last(self):
        return list(self.ops)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def gumbel_argmax(p, u):
    g = -torch.log(-torch.log(u.clamp(min=1e-20, max=1.0 - 2.0 ** -24)))
    return (p.log() + g).argmax(-1)


def float_key(x):
    b = x.view(torch.int32).to(torch.int64)
    return torch.where(b < 0, b ^ 0x7FFFFFFF, b)


def key_float(k):
    return torch.where(k < 0, k ^ 0x7FFFFFFF, k).to(torch.int32).view(torch.float32)


def kth_smallest(x, ranks):
    key = float_key(x)[None]
    lo = torch.full((ranks.numel(), 1), -(2 ** 31), dtype=torch.int64, device=x.device)
    hi = torch.full_like(lo, 2 ** 31 - 1)
    need = ranks[:, None] + 1
    for _ in range(32):
        mid = (lo + hi) >> 1
        ok = (key <= mid).sum(1, keepdim=True) >= need
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid + 1)
    return key_float(hi[:, 0])


class Frozen:
    def __init__(self, params):
        self.params = params

    def __enter__(self):
        for p in self.params:
            p.requires_grad_(False)

    def __exit__(self, *exc):
        for p in self.params:
            p.requires_grad_(True)


class Dist:
    def __init__(self):
        self.rank, self.world, self.dev = 0, 1, torch.device("cpu")

    @property
    def on(self):
        return self.world > 1

    @property
    def lead(self):
        return self.rank == 0

    def init(self, rank, world, port, device, timeout_s):
        import torch.distributed as td
        if device.type == "cuda":
            if torch.cuda.device_count() != 1:
                raise ValueError(f"each rank must see exactly one GPU through CUDA_VISIBLE_DEVICES, sees {torch.cuda.device_count()}")
            torch.cuda.set_device(0)
        td.init_process_group("nccl" if device.type == "cuda" else "gloo", init_method=f"tcp://127.0.0.1:{int(port)}", rank=int(rank), world_size=int(world), timeout=timedelta(seconds=float(timeout_s)))
        self.rank, self.world, self.dev = int(rank), int(world), torch.device("cuda", 0) if device.type == "cuda" else device

    def shutdown(self):
        if self.on:
            import torch.distributed as td
            td.destroy_process_group()
        self.rank, self.world = 0, 1

    def flat(self, ts):
        return torch.cat([t.detach().reshape(-1).float() for t in ts]) if ts else torch.zeros(0, device=self.dev)

    @torch.no_grad()
    def unflat(self, flat, ts):
        for t, x in zip(ts, flat.split([t.numel() for t in ts])):
            t.copy_(x.view_as(t))

    def mean_(self, ts):
        if not self.on or not ts:
            return
        import torch.distributed as td
        flat = self.flat(ts)
        td.all_reduce(flat)
        self.unflat(flat.div_(self.world), ts)

    def mean_dict(self, d):
        if not self.on:
            return d
        out = {k: torch.as_tensor(v).detach().clone() for k, v in sorted(d.items())}
        self.mean_(list(out.values()))
        return out

    def bcast_(self, ts):
        if not self.on or not ts:
            return
        import torch.distributed as td
        flat = self.flat(ts)
        td.broadcast(flat, 0)
        self.unflat(flat, ts)

    def bcast_module(self, module):
        self.bcast_([t for t in [*module.parameters(), *module.buffers()] if t.is_floating_point()])

    def bcast(self, obj=None):
        if not self.on:
            return obj
        import torch.distributed as td
        box = [obj]
        td.broadcast_object_list(box, src=0, device=self.dev if self.dev.type == "cuda" else None)
        return box[0]

    def gather(self, obj):
        if not self.on:
            return [obj]
        import torch.distributed as td
        out = [None] * self.world
        td.all_gather_object(out, obj)
        return out

    def any(self, flag):
        if not self.on:
            return bool(flag)
        import torch.distributed as td
        x = torch.full((1,), float(bool(flag)), device=self.dev)
        td.all_reduce(x, op=td.ReduceOp.MAX)
        return bool(x.item())

    def barrier(self):
        if self.on:
            import torch.distributed as td
            td.barrier()

    def verify(self, module):
        if not self.on:
            return
        bits = self.flat(list(module.parameters())).view(torch.int32).long()
        mine = (int(bits.sum()), int((bits * torch.arange(1, bits.numel() + 1, device=bits.device)).sum()), int(bits.numel()))
        hashes = self.gather(mine)
        if any(h != hashes[0] for h in hashes):
            raise RuntimeError(f"ranks diverged: parameter hashes {hashes}")


DIST = Dist()


class ReturnNorm:
    def __init__(self, decay, lo, hi):
        self.decay, self.q, self.v, self.seen, self.ranks = decay, (lo, hi), None, None, {}

    def ensure(self, device):
        if self.v is None or self.v.device != device:
            self.v, self.seen = torch.zeros(2, device=device), torch.zeros(1, device=device)

    def quantiles(self, x):
        x = x.detach().flatten().float()
        n = x.numel()
        poss = [q * (n - 1) for q in self.q]
        los = [int(math.floor(p)) for p in poss]
        his = [min(lo + 1, n - 1) for lo in los]
        key = (n, x.device)
        if key not in self.ranks:
            self.ranks[key] = torch.tensor(los + his, device=x.device)
        v = kth_smallest(x, self.ranks[key])
        k = len(poss)
        return torch.stack([v[i] + (v[k + i] - v[i]) * (poss[i] - los[i]) for i in range(k)])

    def __call__(self, ret, floor=1.0):
        self.ensure(ret.device)
        q = self.quantiles(ret)
        self.v.copy_(torch.where(self.seen > 0, self.decay * self.v + (1 - self.decay) * q, q))
        self.seen.fill_(1.0)
        return (self.v[1] - self.v[0]).clamp(min=floor)


def trunc_init(layer, outscale):
    fan = (layer.in_features + layer.out_features) / 2
    std = math.sqrt(1.0 / fan) / 0.87962566103423978
    with torch.no_grad():
        if outscale == 0.0:
            layer.weight.zero_()
        else:
            nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
            layer.weight.mul_(outscale)
        if layer.bias is not None:
            layer.bias.zero_()


class MLP(nn.Module):
    def __init__(self, inp, out, units, layers, outscale=1.0, symlog_in=False):
        super().__init__()
        mods, d = [], inp
        for _ in range(layers):
            lin = nn.Linear(d, units, bias=False)
            trunc_init(lin, 1.0)
            mods += [lin, nn.LayerNorm(units), nn.SiLU()]
            d = units
        self.body, self.out, self.symlog_in = nn.Sequential(*mods), nn.Linear(d, out), symlog_in
        trunc_init(self.out, outscale)

    def forward(self, x):
        return f32(self.out(self.body(symlog(x) if self.symlog_in else x)))


class Encoder(nn.Module):
    def __init__(self, inp, units, layers):
        super().__init__()
        self.body = MLP(inp, units, units, layers, symlog_in=True).body

    def forward(self, x):
        return self.body(symlog(x))


class RSSM(nn.Module):
    def __init__(self, c, emb, act):
        super().__init__()
        self.S, self.D, self.deter, self.unimix, self.fast = c["stoch"], c["discrete"], c["deter"], c["unimix"], bool(c.get("fast"))
        sd, h = self.S * self.D, c["hidden"]
        self.img_in = self.block(sd + act, h)
        self.gru = nn.Linear(h + self.deter, 3 * self.deter, bias=False)
        trunc_init(self.gru, 1.0)
        self.gru_norm = nn.LayerNorm(3 * self.deter)
        self.prior_h, self.prior_out = self.block(self.deter, h), nn.Linear(h, sd)
        self.post_h, self.post_out = self.block(self.deter + emb, h), nn.Linear(h, sd)
        trunc_init(self.prior_out, 1.0)
        trunc_init(self.post_out, 1.0)
        self.init_w = nn.Parameter(torch.zeros(self.deter))

    @staticmethod
    def block(i, o):
        lin = nn.Linear(i, o, bias=False)
        trunc_init(lin, 1.0)
        return nn.Sequential(lin, nn.LayerNorm(o), nn.SiLU())

    def prior_logit(self, deter):
        return f32(self.prior_out(self.prior_h(deter)))

    def post_logit(self, deter, embed):
        return f32(self.post_out(self.post_h(torch.cat([deter, embed], -1))))

    def sample(self, logit, gen=None):
        p = cat_probs(logit, self.S, self.D, self.unimix)
        if isinstance(gen, Noise) and gen.legacy:
            gen = None
        if isinstance(gen, Noise):
            idx = gumbel_argmax(p, gen.uniform(p.shape, p))
        elif gen is None and not self.fast:
            idx = torch.distributions.Categorical(probs=p, validate_args=False).sample()
        else:
            idx = torch.multinomial(p.reshape(-1, self.D), 1, generator=gen, replacement=gen is None).reshape(p.shape[:-1])
        oh = F.one_hot(idx, self.D).to(p.dtype)
        return oh + p - p.detach()

    def initial(self, b):
        deter = torch.tanh(self.init_w).expand(b, -1)
        p = cat_probs(self.prior_logit(deter), self.S, self.D, self.unimix)
        return deter, F.one_hot(p.argmax(-1), self.D).to(p.dtype)

    def step_deter(self, deter, stoch, action):
        x = self.img_in(torch.cat([stoch.flatten(-2), action], -1))
        r, cand, u = self.gru_norm(self.gru(torch.cat([x, deter], -1))).chunk(3, -1)
        cand = torch.tanh(torch.sigmoid(r) * cand)
        u = torch.sigmoid(u - 1)
        return f32(u * cand + (1 - u) * deter)

    def img_step(self, deter, stoch, action, gen=None, mode=False):
        deter = self.step_deter(deter, stoch, action)
        logit = self.prior_logit(deter)
        if mode:
            p = cat_probs(logit, self.S, self.D, self.unimix)
            return deter, F.one_hot(p.argmax(-1), self.D).to(p.dtype) + p - p.detach(), logit
        return deter, self.sample(logit, gen), logit

    def mode(self, logit):
        return F.one_hot(cat_probs(logit, self.S, self.D, self.unimix).argmax(-1), self.D).to(logit.dtype)

    def obs_step(self, deter, stoch, action, first, embed, mode=False, init=None, gen=None):
        i_deter, i_stoch = init or self.initial(deter.shape[0])
        m = first[:, None]
        deter, stoch = torch.where(m > 0, i_deter, deter), torch.where(m[..., None] > 0, i_stoch, stoch)
        action = action * (1 - m)
        deter = self.step_deter(deter, stoch, action)
        plogit, qlogit = self.prior_logit(deter), self.post_logit(deter, embed)
        return deter, (cat_probs(qlogit, self.S, self.D, self.unimix) if mode == "soft" else self.mode(qlogit)) if mode else self.sample(qlogit, gen), plogit, qlogit

    def observe(self, embed, action, first, prev=None, mode=False, gen=None):
        b, t = embed.shape[:2]
        init = self.initial(b)
        deter, stoch = init
        if prev is None:
            prev = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], 1)
        outs = []
        for i in range(t):
            deter, stoch, pl, ql = self.obs_step(deter, stoch, prev[:, i], first[:, i], embed[:, i], mode, init, gen)
            outs.append((deter, stoch, pl, ql))
        return [torch.stack(x, 1) for x in zip(*outs)]


class Replay:
    def __init__(self, envs, cap, dims):
        self.cap, self.n, self.ptr = cap, np.zeros(envs, dtype=np.int64), np.zeros(envs, dtype=np.int64)
        self.data = {k: np.zeros((envs, cap, d), dtype=np.float32) for k, d in dims.items()}

    def add(self, e, rec):
        for k, v in rec.items():
            self.data[k][e, self.ptr[e]] = v
        self.ptr[e] = (self.ptr[e] + 1) % self.cap
        self.n[e] = min(self.n[e] + 1, self.cap)

    def ready(self, T):
        return bool((self.n >= T).any())

    def sample(self, b, T, rng):
        ok = np.flatnonzero(self.n >= T)
        es = rng.choice(ok, size=b)
        start = np.array([rng.integers(0, self.n[e] - T + 1) for e in es])
        idx = (((self.ptr[es] - self.n[es]) + start) % self.cap)[:, None] + np.arange(T)[None]
        idx %= self.cap
        return {k: v[es[:, None], idx] for k, v in self.data.items()}


class LaProp(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            by_t = {}
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1
                by_t.setdefault(st["t"], []).append(p)
            for t, ps in by_t.items():
                gr, v, m = [p.grad for p in ps], [self.state[p]["v"] for p in ps], [self.state[p]["m"] for p in ps]
                torch._foreach_mul_(v, b2)
                torch._foreach_addcmul_(v, gr, gr, value=1 - b2)
                denom = torch._foreach_div(v, 1 - b2 ** t)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, g["eps"])
                torch._foreach_mul_(m, b1)
                torch._foreach_addcdiv_(m, gr, denom, value=1 - b1)
                torch._foreach_add_(ps, m, alpha=-g["lr"] / (1 - b1 ** t))


def clip_agc(params, clip, pmin):
    ps = [p for p in params if p.grad is not None]
    if not ps:
        return
    grads = [p.grad for p in ps]
    limit = torch.stack(torch._foreach_norm([p.detach() for p in ps])).clamp(min=pmin) * clip
    gn = torch.stack(torch._foreach_norm(grads))
    scale = torch.where(gn > limit, limit / (gn + 1e-6), torch.ones_like(gn))
    torch._foreach_mul_(grads, list(scale.unbind()))


def apply_overrides(cfg, argv):
    cfg = copy.deepcopy(cfg)
    for a in argv:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        d, *rest = k.split(".")
        node, key = cfg, d
        for r in rest:
            node, key = node[key], r
        if key not in node:
            raise KeyError(f"unknown config key {k!r}")
        try:
            node[key] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            node[key] = v
    return cfg


class StepReplay:
    def __init__(self, envs, cap, dims, device, seed=0):
        self.envs, self.cap, self.dev = int(envs), int(cap), torch.device(device)
        self.data = {k: torch.zeros(self.envs, self.cap, d, device=self.dev) for k, d in dims.items()}
        self.n = self.ptr = 0
        self.anchor, self.anchor_share, self.anchor_grid = None, 0.0, 0
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(seed))

    def add(self, rec):
        for k, v in rec.items():
            self.data[k][:, self.ptr] = v.reshape(self.envs, -1).to(self.data[k].dtype)
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def ready(self, length):
        return self.n >= length

    def set_anchor(self, steps, share, extra=None, grid=0):
        self.anchor, self.anchor_share, self.anchor_grid = {k: v[:, :int(steps)].clone() for k, v in self.data.items()}, float(share), int(grid)
        if extra:
            self.anchor.update({k: v[:, :int(steps)].clone().to(self.dev) for k, v in extra.items()})

    def sample(self, b, length):
        na = int(round(self.anchor_share * b)) if self.anchor is not None else 0
        out = self.sample_ring(b - na, length)
        if self.anchor is not None:
            for k, v in self.anchor.items():
                if k not in out:
                    out[k] = torch.zeros(b - na, length, v.shape[-1], device=self.dev)
        if na:
            a = self.sample_anchor(na, length)
            out = {k: torch.cat([out[k], a[k]]) for k in out}
        return out

    def sample_anchor(self, b, length):
        n = next(iter(self.anchor.values())).shape[1]
        es = torch.randint(0, self.envs, (b,), generator=self.gen, device=self.dev)
        start = (torch.rand(b, generator=self.gen, device=self.dev) * (n - length + 1)).long().clamp(max=n - length)
        if self.anchor_grid:
            start = start // self.anchor_grid * self.anchor_grid
        idx = start[:, None] + torch.arange(length, device=self.dev)[None]
        return {k: v[es[:, None], idx] for k, v in self.anchor.items()}

    def sample_ring(self, b, length):
        es = torch.randint(0, self.envs, (b,), generator=self.gen, device=self.dev)
        start = (torch.rand(b, generator=self.gen, device=self.dev) * (self.n - length + 1)).long().clamp(max=self.n - length)
        idx = (self.ptr - self.n + start[:, None] + torch.arange(length, device=self.dev)[None]) % self.cap
        return {k: v[es[:, None], idx] for k, v in self.data.items()}
