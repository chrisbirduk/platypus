"""STANDARD CONTINUAL-LEARNING BASELINES on the same trunk, data and schedule (14 Nov 2026; README open question 1). All off by default.
  bl_ewc > 0       EWC (online, Kirkpatrick et al. 2017; Schwarz et al. 2018): at the end of every visit a diagonal empirical Fisher from
                   bl_ewc_batches batches of the visited text is added to the running Fisher and the weights are anchored there; the training
                   loss gets bl_ewc * sum F (w - w*)^2. Units added by widening after an anchor are unprotected (sliced to the anchored shape).
  bl_er > 0        STORED-EXAMPLE REPLAY (experience replay, Chaudhry et al. 2019): replay_buffer_batches batches per visit are kept as raw
                   bytes; every training step adds bl_er x the loss on a random stored batch of another slot (through that slot's head).
  bl_adapter > 0   PER-EPISODE ADAPTERS (Houlsby et al. 2019; frozen backbone): the first visit trains the trunk as usual; every later visit
                   trains only a residual low-rank adapter per block for its slot (rank bl_adapter, B zero-initialised) and the slot's head,
                   with Adam (lr bl_ad_lr), the trunk frozen. Evaluation runs each slot through its own adapters.
The protections of the system itself (dreams, Fisher-whitened Muon, current-batch KL, n-gram protection, conduit replay, the sleep layer)
are switched off by the run's options for a plain baseline; this file only adds the three methods."""
import torch
import torch.nn.functional as F


def _named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def ewc_update(model, cfg, get, slot):
    """end of a visit: running Fisher += this visit's diagonal empirical Fisher; anchor = the weights now"""
    nb = int(getattr(cfg, 'bl_ewc_batches', 8)); was = model.training; model.train(False)
    Fs = {n: torch.zeros_like(p) for n, p in _named(model)}
    for _ in range(nb):
        x, y = get(); model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            l = F.cross_entropy(model(x, slot).reshape(-1, 256).float(), y.reshape(-1)); l.backward()
        for n, p in _named(model):
            if p.grad is not None: Fs[n] += p.grad.detach() ** 2 / nb
    model.zero_grad(set_to_none=True); model.train(was)
    old = model.__dict__.get('_ewc_F') or {}
    for n, f in Fs.items():
        o = old.get(n)
        if o is not None:
            sl = tuple(slice(0, min(a, b)) for a, b in zip(o.shape, f.shape)); f[sl] += o[sl]
    model.__dict__['_ewc_F'] = Fs; model.__dict__['_ewc_P'] = {n: p.detach().clone() for n, p in _named(model)}
    tot = sum(float(f.sum()) for f in Fs.values())
    print(f"  [BASELINE EWC] slot {slot}: Fisher from {nb} batches added (running total {tot:.3g}); anchor = the weights after this visit", flush=True)


def ewc_penalty(model, cfg):
    Fs, Ps = model.__dict__.get('_ewc_F'), model.__dict__.get('_ewc_P')
    if not Fs: return None
    pen = 0.0
    for n, p in _named(model):
        f, p0 = Fs.get(n), Ps.get(n)
        if f is None or p0 is None or p.dim() != f.dim(): continue
        sl = tuple(slice(0, min(a, b)) for a, b in zip(f.shape, p.shape))
        pen = pen + (f[sl] * (p[sl] - p0[sl]) ** 2).sum()
    return float(cfg.bl_ewc) * pen


def er_loss(model, cfg, slot):
    """a random stored batch of another slot, through its own head"""
    buf = [b for b in model.__dict__.get('_replay_buffer', []) if int(b[2]) != int(slot)]
    if not buf: return None
    bx, by, bd = buf[torch.randint(len(buf), (1,)).item()]
    dv = model.embed.weight.device
    sv = model.__dict__.get('_bl_slot_of'); bd_ = (sv or {}).get(int(bd), int(bd))
    return float(cfg.bl_er) * F.cross_entropy(model(bx.long().to(dv), bd_).reshape(-1, 256).float(), by.long().to(dv).reshape(-1))


def ad_apply(model, i, h):
    """the slot's adapter after block i (residual, low rank)"""
    A = model.__dict__.get('_bl_ad', {}).get((model.__dict__.get('_s0_dom'), i))
    if A is None: return h
    a, b = A
    return h + torch.relu(F.layer_norm(h.float(), h.shape[-1:]) @ a) @ b


def ad_train(model, cfg, get, slot):
    """a later visit: train only this slot's adapters and head; the trunk stays as it is"""
    r = int(cfg.bl_adapter); dv = model.embed.weight.device; D = model.embed.weight.shape[1]; AD = model.__dict__.setdefault('_bl_ad', {})
    for i in range(len(model.blocks)):
        if (slot, i) not in AD:
            AD[(slot, i)] = (torch.nn.Parameter(torch.randn(D, r, device=dv) / D ** 0.5), torch.nn.Parameter(torch.zeros(r, D, device=dv)))
    hp = [] if getattr(cfg, 'shared_head', False) else list(model.heads[slot].parameters())   # a shared head stays frozen (it serves every slot)
    ps = [p for i in range(len(model.blocks)) for p in AD[(slot, i)]] + hp
    keep = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters(): p.requires_grad_(False)
    for p in hp: p.requires_grad_(True)
    op = model.__dict__.setdefault('_bl_ad_opt', {}).get(slot)
    if op is None: op = torch.optim.Adam(ps, lr=float(cfg.bl_ad_lr)); model.__dict__['_bl_ad_opt'][slot] = op
    model.train(); ls = []
    try:
        for s in range(int(cfg.steps_per_task)):
            x, y = get(); l = F.cross_entropy(model(x, slot).reshape(-1, 256).float(), y.reshape(-1))
            op.zero_grad(set_to_none=True); l.backward(); op.step(); ls.append(float(l))
    finally:
        for n, p in model.named_parameters(): p.requires_grad_(keep.get(n, True))
    k = max(1, len(ls) // 10)
    print(f"  [BASELINE ADAPTERS] slot {slot}: {len(ls)} steps on the slot's adapters (rank {r}, {sum(p.numel() for p in ps)} parameters with "
          f"the head), trunk frozen | loss {sum(ls[:k]) / k / 0.6931:.3f} -> {sum(ls[-k:]) / k / 0.6931:.3f} bits per byte", flush=True)
