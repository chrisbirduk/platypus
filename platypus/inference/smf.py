"""SPARSE MEMORY CONSOLIDATION IN THE TRUNK (inf_smf; 16 Nov 2026). After Lin et al. 2025 ('Continual Learning via Sparse Memory
Finetuning') and Berges et al. 2024 ('Memory Layers at Scale'): a product-key memory layer (Lample et al. 2019) beside the middle block's
MLP - n x n value slots, the top inf_smf_k read per byte - trained with the trunk on ordinary text for the first inf_smf_pre visits (its
'pretraining'), then frozen for text. From then on the question rows (taught facts, the written real-text inferences that the judge has
kept, the retraction rows) write ONLY to this memory, and only to the inf_smf_t slots with the highest TF-IDF where the rows' ANSWER bytes
are predicted (accesses at answer positions x log(text batches / text batches using the slot)): every other trunk weight is untouched by
them (inf_distill_wake_params 'smem'; previously 'mlp_out', every block's MLP output matrix).
Offline (tools/sparse_mem.py, 6k stand-in trunk, facts mixed with text as on the GPU): recall 0.70 at +0.03 bits of forgetting, against
0.37 for mlp_out and 0.50 for full fine-tuning at the same forgetting; 0.88 at +0.08.
The layer's tensors are BUFFERS of the block (saved with the model, copied with it, outside the main optimiser and the protections); the
memory has its own row-wise Adam (moments only for rows it updates: unselected rows never move)."""
import math, time
import numpy as np
import torch
import torch.nn.functional as F


def smf_init(block, cfg):
    """the layer's tensors on the block (called once at model construction when inf_smf)"""
    d = block.d_model; n = int(getattr(cfg, 'inf_smf_n', 256)); dk = int(getattr(cfg, 'inf_smf_dk', 64)); g = torch.Generator().manual_seed(int(getattr(cfg, 'seed', 0)) + 4747)
    block.register_buffer('smf_lnw', torch.ones(d)); block.register_buffer('smf_lnb', torch.zeros(d))
    block.register_buffer('smf_qw', torch.randn(2 * dk, d, generator=g) / math.sqrt(d)); block.register_buffer('smf_qb', torch.zeros(2 * dk))
    block.register_buffer('smf_k1', torch.randn(n, dk, generator=g) / math.sqrt(dk)); block.register_buffer('smf_k2', torch.randn(n, dk, generator=g) / math.sqrt(dk))
    block.register_buffer('smf_V', torch.randn(n * n, d, generator=g) * 0.02)
    block._smf_n = n; block._smf_dk = dk; block._smf_k = int(getattr(cfg, 'inf_smf_k', 16)); block._smf_last = None; block._smf_off = False


_KEYS = ('smf_lnw', 'smf_lnb', 'smf_qw', 'smf_qb', 'smf_k1', 'smf_k2')


def smf_read(s, hn):
    """the memory's output for the block's MLP input hn (B, T, d); the slots read are kept on the block (detached) for TF-IDF"""
    n, dk, K = s._smf_n, s._smf_dk, s._smf_k
    x = F.layer_norm(hn.float(), (hn.shape[-1],), s.smf_lnw, s.smf_lnb); q = F.linear(x, s.smf_qw, s.smf_qb)
    q1, q2 = F.layer_norm(q[..., :dk], (dk,)), F.layer_norm(q[..., dk:], (dk,))
    a1, i1 = (q1 @ s.smf_k1.t()).topk(K, -1); a2, i2 = (q2 @ s.smf_k2.t()).topk(K, -1)
    sc = (a1[..., :, None] + a2[..., None, :]).flatten(-2); ix = (i1[..., :, None] * n + i2[..., None, :]).flatten(-2)
    top, j = sc.topk(K, -1); idx = ix.gather(-1, j); w = F.softmax(top, -1)
    s._smf_last = idx.detach()
    return (w[..., None] * F.embedding(idx, s.smf_V)).sum(-2)


def _blk(model):
    for b in model.blocks:
        if getattr(b, 'smf_V', None) is not None: return b
    return None


def _state(model, cfg):
    st = model.__dict__.get('_smf_st')
    if st is None:
        b = _blk(model); N = b.smf_V.shape[0]; dev = b.smf_V.device
        st = dict(bg=torch.zeros(N, device=dev), nbg=0, ans=torch.zeros(N, device=dev), m=torch.zeros_like(b.smf_V), v=torch.zeros_like(b.smf_V),
                  t=torch.zeros(N, device=dev), wrote=torch.zeros(N, dtype=torch.bool, device=dev), steps_t=0, steps_q=0, rows_q=[], kopt=None, vis=0, ql=[])
        model.__dict__['_smf_st'] = st
    return st


def _row_adam(b, st, rows, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam on the given rows of V only (moments and step counts per row; every other row untouched)"""
    if rows.numel() == 0: return
    g = grad[rows].float(); st['t'][rows] += 1; t = st['t'][rows][:, None]
    st['m'][rows] = b1 * st['m'][rows] + (1 - b1) * g; st['v'][rows] = b2 * st['v'][rows] + (1 - b2) * g * g
    mh = st['m'][rows] / (1 - b1 ** t); vh = st['v'][rows] / (1 - b2 ** t)
    with torch.no_grad(): b.smf_V[rows] -= lr * mh / (vh.sqrt() + eps)


def smf_text_step(model, cfg, nstat=None):
    """after the real step's backward: in the first inf_smf_pre visits the layer learns from text (keys and query by Adam, the value rows
    read by row-wise Adam); after them it is frozen for text. Background slot use (text batches using each slot) is counted throughout."""
    b = _blk(model)
    if b is None: return
    st = _state(model, cfg); pre = st['vis'] < int(getattr(cfg, 'inf_smf_pre', 1))
    if pre and not b.smf_V.requires_grad: smf_phase(model, cfg)                # the first step (buffers cannot require grad before .to(device))
    ix = b._smf_last
    if ix is not None:
        ix_ = ix[:nstat] if (nstat is not None and nstat < ix.shape[0]) else ix
        st['bg'][torch.unique(ix_.reshape(-1))] += 1; st['nbg'] += 1
    if pre:
        if st['kopt'] is None: st['kopt'] = torch.optim.Adam([getattr(b, k) for k in _KEYS], lr=float(cfg.lr) * float(getattr(cfg, 'inf_smf_klr', 1.0)))
        if all(getattr(b, k).grad is not None for k in _KEYS):
            st['kopt'].step()
        g = b.smf_V.grad
        if g is not None:
            rows = torch.nonzero(g.abs().sum(-1) > 0).flatten(); _row_adam(b, st, rows, g, float(cfg.lr) * float(getattr(cfg, 'inf_smf_vlr', 10.0))); st['steps_t'] += 1
    for k in _KEYS + ('smf_V',): getattr(b, k).grad = None


def smf_phase(model, cfg):
    """requires_grad of the layer's tensors: on while it learns from text, off after (the question rows switch the values on for themselves)"""
    b = _blk(model)
    if b is None: return
    st = _state(model, cfg); pre = st['vis'] < int(getattr(cfg, 'inf_smf_pre', 1))
    for k in _KEYS + ('smf_V',):
        t = getattr(b, k)
        if t.requires_grad != pre: t.requires_grad_(pre)


def smf_visit(model, cfg, vis):
    """at each sleep: the visit count (ends the text phase)"""
    if _blk(model) is None: return
    st = _state(model, cfg); st['vis'] = int(vis) + 1; smf_phase(model, cfg)


def smf_grad_on(model):
    """make the value table the only thing the question rows can change (before their forward)"""
    b = _blk(model)
    if b is None: return None
    for k in _KEYS: getattr(b, k).requires_grad_(False)
    b.smf_V.requires_grad_(True); b.smf_V.grad = None; st = model.__dict__.get('_smf_st')
    if st is not None: st['ans'].zero_()
    return b


def smf_note(model, rows):
    """after a question-row forward: count the slots read at the ANSWER positions; rows = [(row index, first answer-predicting position, n)]"""
    b = _blk(model); st = model.__dict__.get('_smf_st')
    if b is None or st is None or b._smf_last is None: return
    ix = b._smf_last
    for r, p0, n in rows:
        if n > 0 and r < ix.shape[0]: st['ans'][ix[r, p0:p0 + n].reshape(-1)] += 1


def smf_q_step(model, cfg):
    """the question rows' update: TF-IDF over the answer positions' slot reads, the top inf_smf_t rows only, row-wise Adam"""
    b = _blk(model); st = model.__dict__.get('_smf_st')
    if b is None or st is None: return
    g = b.smf_V.grad
    try:
        if g is None or st['ans'].sum() <= 0: return
        idf = torch.log((st['nbg'] + 1.0) / (st['bg'] + 1.0)); tf = st['ans'] * idf
        T = int(getattr(cfg, 'inf_smf_t', 1024)); nz = int((st['ans'] > 0).sum())
        rows = tf.topk(min(T, nz)).indices
        _row_adam(b, st, rows, g, float(getattr(cfg, 'inf_smf_lr', 1e-2))); st['wrote'][rows] = True; st['steps_q'] += 1; st['rows_q'].append(int(rows.numel()))
    finally:
        b.smf_V.grad = None


@torch.no_grad()
def smf_readout(model, cfg, vals, slot, vis):
    """[TRUNK MEMORY LAYER] slots used by text, slots written by the question rows, bits on held-out text with the layer on and off (does the
    trunk use it), the share of text reads landing on slots the question rows wrote"""
    b = _blk(model); st = model.__dict__.get('_smf_st')
    if b is None or st is None: return
    try:
        from platypus.inference.closed_form import _ws_run
        g = vals; acc = {True: [0.0, 0], False: [0.0, 0]}; hit = [0, 0]
        if hasattr(g, 'reset'): g.reset()
        was = model.training; model.eval()
        try:
            for _ in range(4):
                x, y = g()
                for on in (True, False):
                    b._smf_off = not on
                    try: lg = model(x, slot)
                    finally: b._smf_off = False
                    T = y.shape[1]; lg = lg[:, -T:]
                    nl = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1), reduction='sum') / math.log(2)
                    acc[on][0] += float(nl); acc[on][1] += int(y.numel())
                    if on and b._smf_last is not None: hit[0] += int(st['wrote'][b._smf_last].any(-1).sum()); hit[1] += int(b._smf_last.shape[0] * b._smf_last.shape[1])
        finally: model.train(was)
        N = b.smf_V.shape[0]; on_ = acc[True][0] / max(1, acc[True][1]); off_ = acc[False][0] / max(1, acc[False][1])
        print(f"  [TRUNK MEMORY LAYER | visit {vis + 1}] {b._smf_n}x{b._smf_n} = {N} slots, top {b._smf_k} a byte; {'LEARNING FROM TEXT' if st['vis'] < int(getattr(cfg, 'inf_smf_pre', 1)) else 'frozen for text'} "
              f"({st['steps_t']} text steps so far) | slots used by text {int((st['bg'] > 0).sum())} ({st['nbg']} batches counted) | QUESTION ROWS: {st['steps_q']} steps, "
              f"{int(st['wrote'].sum())} slots ever written (mean {np.mean(st['rows_q'][-200:]) if st['rows_q'] else 0:.0f} a step) | held-out bits per byte of the visited domain WITH the layer "
              f"{on_:.4f}, layer OFF {off_:.4f} ({off_ - on_:+.4f}: how much the trunk leans on it) | text positions reading a question-written slot {hit[0] / max(1, hit[1]):.4f}", flush=True)
    except Exception as e:
        from platypus.common import _fail
        _fail('inference.smf:readout'); print(f"  [TRUNK MEMORY LAYER] readout unavailable ({type(e).__name__}: {e})"[:240], flush=True)
