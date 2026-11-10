"""CARRY THE SUBJECT (inf_carry, 10 Nov; off by default). An auxiliary training loss that teaches the trunk to keep the latest name's identity
readable at the bytes after it - the path the trunk lacks (RESULTS.md: from the state where an answer starts, a linear map recovers the
subject for subjects it was fitted on but not for new ones, 0.01-0.05, so facts in the weights cannot be reached from a new wording).
On real-text training windows: SUBJECT REGISTER as in the name memory (the latest capitalised word of 3+ letters ending before the byte);
at up to inf_carry_n sampled bytes 1..inf_carry_dist after a name, a small trainable projection of the edited layer's input (the same
place the name memory reads keys) must pick out THAT name's state (the same layer's input at the name's last byte, gradient stopped) among
the names sampled in the batch - InfoNCE, temperature inf_carry_tau, every occurrence of the same name a positive. The contrastive form
cannot be met by collapsing states. Weight inf_carry on top of the language-model loss; the projections have their own Adam."""
import re
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail

_CL = {'arm': False, 'K': None, 'hooked': None, 'P': None, 'opt': None, 'acc': [], 'n': 0}
_CL_WORD = re.compile(rb"[A-Za-z]+")


def _cl_isname(b): return len(b) >= 3 and 65 <= b[0] <= 90 and all(65 <= x <= 90 or 97 <= x <= 122 for x in b)


def _cl_hook(m, inp, out):
    if _CL['arm']: _CL['K'] = inp[0]


def carry_arm(model, cfg):
    """called right before the training forward of a real-text window; installs the capture hook once"""
    if float(getattr(cfg, 'inf_carry', 0.0)) <= 0: return
    mo = model.blocks[int(getattr(cfg, 'inf_cf_layer', -1))].mlp_out
    if _CL['hooked'] is not mo:
        if _CL.get('h') is not None: _CL['h'].remove()
        _CL['h'] = mo.register_forward_hook(_cl_hook); _CL['hooked'] = mo
    _CL['arm'] = True; _CL['K'] = None


def carry_disarm(): _CL['arm'] = False


def carry_loss(model, cfg, x):
    """the auxiliary loss for the window batch x (B, T) whose forward has just run (keys captured); 0.0 when nothing to do"""
    K = _CL.get('K'); _CL['K'] = None; _CL['arm'] = False
    if K is None or K.dim() != 3 or K.shape[:2] != x.shape[:2]: return 0.0
    try:
        B, T, H = K.shape; dev = K.device; dmax = int(getattr(cfg, 'inf_carry_dist', 32)); nmax = int(getattr(cfg, 'inf_carry_n', 128))
        Xc = x.detach().to('cpu', torch.uint8).numpy(); qs, ns, ids = [], [], []
        for b in range(B):
            txt = bytes(Xc[b].tolist()); last = None
            for m in _CL_WORD.finditer(txt):
                w = m.group(0)
                if not _cl_isname(w): continue
                ne = m.end() - 1; nxt = len(txt)
                for m2 in _CL_WORD.finditer(txt, m.end()):
                    if _cl_isname(m2.group(0)): nxt = m2.end() - 1; break
                for t in range(ne + 1, min(T, ne + 1 + dmax, nxt + 1)): qs.append((b, t)); ns.append((b, ne)); ids.append(w)
        if len(set(ids)) < 4: return 0.0
        rs = np.random.RandomState(_CL['n']); _CL['n'] += 1
        pick = rs.permutation(len(qs))[:nmax]
        qb = torch.tensor([qs[i][0] for i in pick], device=dev); qt = torch.tensor([qs[i][1] for i in pick], device=dev)
        nb = torch.tensor([ns[i][0] for i in pick], device=dev); nt = torch.tensor([ns[i][1] for i in pick], device=dev)
        wid = [ids[i] for i in pick]; uq = {w: k for k, w in enumerate(sorted(set(wid)))}; lab = torch.tensor([uq[w] for w in wid], device=dev)
        if _CL['P'] is None or _CL['P'][0].in_features != H:
            d = int(getattr(cfg, 'inf_carry_dim', 128))
            _CL['P'] = (torch.nn.Linear(H, d).to(dev), torch.nn.Linear(H, d).to(dev))
            _CL['opt'] = torch.optim.Adam(list(_CL['P'][0].parameters()) + list(_CL['P'][1].parameters()), lr=1e-3)
        else:                                                                  # step the projections on the gradients of the previous call
            if _CL['P'][0].weight.grad is not None: _CL['opt'].step()
            _CL['opt'].zero_grad()
        Pq, Pk = _CL['P']
        q = F.normalize(Pq(K[qb, qt].float()), dim=-1)                          # the state at the byte: gradient reaches the trunk
        k = F.normalize(Pk(K[nb, nt].detach().float()), dim=-1)                # the name's own state: no gradient to the trunk
        s = q @ k.T / float(getattr(cfg, 'inf_carry_tau', 0.1))
        pos = (lab[:, None] == lab[None, :]).float()
        loss = -(torch.logsumexp(s.masked_fill(pos == 0, -1e9), 1) - torch.logsumexp(s, 1)).mean()
        with torch.no_grad(): _CL['acc'].append(float((pos.gather(1, s.argmax(1)[:, None])[:, 0]).mean()))
        return loss
    except Exception:
        _fail('inference.carry'); return 0.0


def carry_report(tag):
    """training accuracy of the carry task since the last report (the true name's state first among the batch's names)"""
    a = _CL['acc']; _CL['acc'] = []
    if a: print(f"  [CARRY THE SUBJECT | {tag}] training batches {len(a)}: the byte's state picks its own name among the batch's names {np.mean(a):.2f} (last 50: {np.mean(a[-50:]):.2f})", flush=True)
