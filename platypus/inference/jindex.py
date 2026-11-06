"""JEPA INDEX (cg26s, inf_jx; readouts only - the trunk is never trained by it).
Each sleep a small byte-level text encoder is trained V-JEPA style on the sentences in the store: the ONLINE encoder reads a sentence cut
at a word boundary (the rest withheld, as in a question 'A reports to '), a predictor maps its pooled code to what a MOVING-AVERAGE
TARGET encoder (momentum inf_jx_mom, stop-gradient) makes of the WHOLE sentence; L1 loss, no negatives (inf_jx_nce > 0 adds an InfoNCE
term over the batch). Every stored sentence is then keyed by the target encoder, and a question finds its sentence by the nearest key (L1)
to the predictor's output - the index the toy tools/toy_jsel.py found to fix question-time retrieval (0.88-0.98 against 0.05-0.35 by
answer confidence).
Readouts each sleep (probe chains, store off for every answer):
  RETRIEVAL   is the true fact sentence (it names A and B) the top-1 / among the top-5 of all live stored sentences for 'A r0 '; the same
              for the question in a HELD-OUT wording (inf_eval paraphrases); NAME LOOKUP baseline: the most recent stored sentence
              naming A; HYBRID: the stored sentences naming A, ranked by the JEPA key (names pick the candidates, the index the fact).
  ONE-HOP     answer B among 20 candidates from the weights alone, with the RETRIEVED sentence in context, with the TRUE sentence in
              context (ceiling); held-out wording with the retrieved sentence.
  TWO-HOP     by retrieval: hop 1 retrieves for 'A r0 ', reads B' from that sentence (20 candidates), hop 2 retrieves for 'B' r1 ' and
              reads C; control: hop 2 asked about a RANDOM name instead of B' (must fall to chance); ceiling: the true sentences."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.inference.probes import (
    inf_probes,
)
from platypus.inference.evaluate import (
    _inf_lps,
)
_JX = {}                                                                     # the encoder, its EMA target and predictor live here across sleeps
class _JXEnc(nn.Module):
    """byte transformer encoder (bidirectional) -> mean over the text positions -> LayerNorm"""
    def __init__(self, d_, nl_, L_):
        super().__init__(); self.L = L_; self.emb = nn.Embedding(257, d_); self.pos = nn.Parameter(torch.randn(L_, d_) * 0.02)
        lay_ = nn.TransformerEncoderLayer(d_, max(1, d_ // 64), 4 * d_, 0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(lay_, nl_, enable_nested_tensor=False); self.n = nn.LayerNorm(d_)
    def forward(self, X_):
        pad_ = (X_ == 256); h_ = self.tr(self.emb(X_) + self.pos[:X_.shape[1]], src_key_padding_mask=pad_)
        m_ = (~pad_).float().unsqueeze(-1); return self.n((h_ * m_).sum(1) / m_.sum(1).clamp(min=1))
def _jx_batch(txts_, L_, dev_):
    bs_ = [bytes(t_)[:L_] for t_ in txts_]; Lm_ = max(4, max(len(b_) for b_ in bs_)); X_ = torch.full((len(bs_), Lm_), 256, dtype=torch.long)
    for j_, b_ in enumerate(bs_): X_[j_, :len(b_)] = torch.tensor(list(b_), dtype=torch.long)
    return X_.to(dev_)
def _jx_cut(s_, rs_):
    """the question form of a stored sentence: cut at a word boundary after the first two words (half the time before the last word)"""
    w_ = [i_ + 1 for i_, c_ in enumerate(s_) if c_ == 32]
    if len(w_) < 2: return s_[:max(1, len(s_) // 2)]
    return s_[:w_[-1]] if rs_.rand() < 0.5 else s_[:w_[int(rs_.randint(1, len(w_)))]]
def _jx_live(S_, cfg):
    it_ = [i_ for i_, x_ in enumerate(S_['it']) if x_.get('s') and not x_.get('gone')]
    return it_[-int(getattr(cfg, 'inf_jx_pool', 200000)):]
def _jx_train(model, cfg, S_):
    dev_ = model.embed.weight.device; L_ = int(getattr(cfg, 'inf_jx_len', 128)); d_ = int(getattr(cfg, 'inf_jx_dim', 128)); nl_ = int(getattr(cfg, 'inf_jx_layers', 3))
    if not _JX:
        with torch.random.fork_rng(devices=[dev_] if dev_.type == 'cuda' else []):   # its own random stream: the run itself is unchanged
            torch.manual_seed(int(getattr(cfg, 'seed', 0)) + 4242)
            on_ = _JXEnc(d_, nl_, L_).to(dev_); tg_ = _JXEnc(d_, nl_, L_).to(dev_); tg_.load_state_dict(on_.state_dict())
            pr_ = nn.Sequential(nn.Linear(d_, 2 * d_), nn.GELU(), nn.Linear(2 * d_, d_)).to(dev_)
        for p_ in tg_.parameters(): p_.requires_grad_(False)
        _JX.update(on=on_, tg=tg_, pr=pr_, op=torch.optim.AdamW(list(on_.parameters()) + list(pr_.parameters()), lr=float(getattr(cfg, 'inf_jx_lr', 5e-4)), weight_decay=0.01), steps=0)
    on_, tg_, pr_, op_ = _JX['on'], _JX['tg'], _JX['pr'], _JX['op']; mom_ = float(getattr(cfg, 'inf_jx_mom', 0.996)); nce_ = float(getattr(cfg, 'inf_jx_nce', 0.0))
    lv_ = _jx_live(S_, cfg); rs_ = np.random.RandomState(9100 + int(model.__dict__.get('_visit', 0))); B_ = int(getattr(cfg, 'inf_jx_rows', 64)); ls_ = []
    if len(lv_) < B_: return None
    on_.train(); pr_.train()
    with torch.enable_grad():
        for _ in range(int(getattr(cfg, 'inf_jx_steps', 200))):
            ss_ = [S_['it'][int(i_)]['s'] for i_ in rs_.choice(lv_, B_, replace=False)]
            with torch.no_grad(): z_ = tg_(_jx_batch(ss_, L_, dev_))
            p_ = pr_(on_(_jx_batch([_jx_cut(s_, rs_) for s_ in ss_], L_, dev_))); l_ = (p_ - z_).abs().mean()
            if nce_ > 0: l_ = l_ + nce_ * F.cross_entropy(-torch.cdist(p_, z_, p=1) / max(1e-3, float(z_.abs().mean()) * d_ * 0.1), torch.arange(B_, device=dev_))
            op_.zero_grad(); l_.backward(); op_.step(); ls_.append(float(l_))
            with torch.no_grad():
                for a_, b_ in zip(tg_.parameters(), on_.parameters()): a_.mul_(mom_).add_(b_, alpha=1 - mom_)
    on_.eval(); pr_.eval(); _JX['steps'] += len(ls_)
    return ls_[0], float(np.mean(ls_[-20:])), len(lv_)
@torch.no_grad()
def _jx_keys(cfg, S_, dev_):
    L_ = int(getattr(cfg, 'inf_jx_len', 128)); lv_ = _jx_live(S_, cfg); K_ = []
    for a_ in range(0, len(lv_), 2048): K_.append(_JX['tg'](_jx_batch([S_['it'][i_]['s'] for i_ in lv_[a_:a_ + 2048]], L_, dev_)))
    return lv_, torch.cat(K_)
@torch.no_grad()
def _jx_find(cfg, q_, K_, dev_, k_=5):
    p_ = _JX['pr'](_JX['on'](_jx_batch([q_.encode('latin-1', 'replace')], int(getattr(cfg, 'inf_jx_len', 128)), dev_)))
    return torch.topk(-(K_ - p_).abs().sum(-1), min(k_, K_.shape[0])).indices.tolist()
def _jx_readout(model, cfg, S_, tag_):
    """retrieval, one-hop and two-hop readouts of the index (see the module note)"""
    dev_ = model.embed.weight.device; lv_, K_ = _jx_keys(cfg, S_, dev_); txt_ = lambda i_: S_['it'][lv_[i_]]['s']
    ch_ = [c_ for c_ in inf_probes(cfg) if len(c_['names']) >= 3 and not c_.get('upd')][:int(getattr(cfg, 'inf_jx_n', 60))]
    alln_ = sorted({n_ for c_ in inf_probes(cfg) for n_ in c_['names']}); rs_ = np.random.RandomState(9200)
    t2s_ = model.__dict__.get('_t2s', {}); has_ = lambda s_, a_, b_: a_.encode('latin-1') in s_ and b_.encode('latin-1') in s_
    import re as _re                                                          # probe names -> live stored sentences naming them (one regex pass)
    rx_ = _re.compile(rb'\b(' + b'|'.join(_re.escape(n_.encode('latin-1')) for n_ in sorted(alln_, key=len, reverse=True)) + rb')\b'); NX_ = {}
    for i_ in lv_:
        for m_ in set(rx_.findall(S_['it'][i_]['s'])): NX_.setdefault(m_.decode('latin-1'), []).append(i_)
    has_ = lambda s_, a_, b_: {a_.encode('latin-1'), b_.encode('latin-1')} <= set(rx_.findall(s_))   # both names, as whole words
    def ans_(ctx_, q_, tgt_, oth_, sl_):
        cs_ = [tgt_] + oth_; ps_ = (ctx_.decode('latin-1', 'replace') + ' ' + q_) if ctx_ else q_
        return cs_[int(np.argmax(_inf_lps(model, cfg, ps_, cs_, sl_, False)))]
    R_ = {k_: [] for k_ in ('r1', 'r5', 'ho1', 'ho5', 'name', 'hyb', 'ho_hyb', 'w', 'ret', 'true', 'ho_ret', 'h1', 'two', 'two_rand', 'two_true')}
    pos_ = {i_: j_ for j_, i_ in enumerate(lv_)}
    def hyb_(q_, A_):                                                         # HYBRID: the sentences that name A, ranked by the JEPA key
        c9_ = [pos_[i_] for i_ in NX_.get(A_, []) if i_ in pos_]
        if not c9_: return None
        p9_ = _JX['pr'](_JX['on'](_jx_batch([q_.encode('latin-1', 'replace')], int(getattr(cfg, 'inf_jx_len', 128)), dev_)))
        return c9_[int((K_[c9_] - p9_).abs().sum(-1).argmin())]
    for c_ in ch_:
        A_, B_, C_ = c_['names'][:3]; r0_, r1_ = c_['rels'][0], c_['rels'][1]; sl_ = int(t2s_.get(int(c_['facts'][0][0]), 0)); sl2_ = int(t2s_.get(int(c_['facts'][1][0]), 0))
        tr1_ = next((S_['it'][i_]['s'] for i_ in reversed(NX_.get(A_, [])) if has_(S_['it'][i_]['s'], A_, B_)), None)
        tr2_ = next((S_['it'][i_]['s'] for i_ in reversed(NX_.get(B_, [])) if has_(S_['it'][i_]['s'], B_, C_)), None)
        if tr1_ is None: continue                                            # the fact is not (yet) in the store: nothing to retrieve
        q_ = f"{A_} {r0_} "; top_ = _jx_find(cfg, q_, K_, dev_); R_['r1'].append(has_(txt_(top_[0]), A_, B_)); R_['r5'].append(any(has_(txt_(i_), A_, B_) for i_ in top_))
        nm_ = S_['it'][NX_[A_][-1]]['s'] if NX_.get(A_) else None; R_['name'].append(bool(nm_) and has_(nm_, A_, B_))
        with torch.no_grad(): hb_ = hyb_(q_, A_)
        R_['hyb'].append(hb_ is not None and has_(txt_(hb_), A_, B_))
        oth_ = list(rs_.choice([x_ for x_ in alln_ if x_ not in c_['names']], 19, replace=False))
        R_['w'].append(ans_(None, q_, B_, oth_, sl_) == B_); R_['ret'].append(ans_(txt_(top_[0]), q_, B_, oth_, sl_) == B_); R_['true'].append(ans_(tr1_, q_, B_, oth_, sl_) == B_)
        if r0_ in _INF_PARA:
            qh_ = f"{A_} {_INF_PARA[r0_]} "; th_ = _jx_find(cfg, qh_, K_, dev_); R_['ho1'].append(has_(txt_(th_[0]), A_, B_)); R_['ho5'].append(any(has_(txt_(i_), A_, B_) for i_ in th_))
            R_['ho_ret'].append(ans_(txt_(th_[0]), qh_, B_, oth_, sl_) == B_)
            with torch.no_grad(): hh_ = hyb_(qh_, A_)
            R_['ho_hyb'].append(hh_ is not None and has_(txt_(hh_), A_, B_))
        if tr2_ is None: continue
        Bh_ = ans_(txt_(top_[0]), q_, B_, oth_, sl_); R_['h1'].append(Bh_ == B_)                            # hop 1: retrieve, read B'
        oth2_ = list(rs_.choice([x_ for x_ in alln_ if x_ not in c_['names']], 19, replace=False))
        q2_ = f"{Bh_} {r1_} "; t2_ = _jx_find(cfg, q2_, K_, dev_, 1); R_['two'].append(ans_(txt_(t2_[0]), q2_, C_, oth2_, sl2_) == C_)
        Br_ = alln_[int(rs_.randint(len(alln_)))]; q3_ = f"{Br_} {r1_} "; t3_ = _jx_find(cfg, q3_, K_, dev_, 1); R_['two_rand'].append(ans_(txt_(t3_[0]), q3_, C_, oth2_, sl2_) == C_)
        Bt_ = ans_(tr1_, q_, B_, oth_, sl_); R_['two_true'].append(ans_(tr2_, f"{Bt_} {r1_} ", C_, oth2_, sl2_) == C_)
    f_ = lambda k_: f"{np.mean(R_[k_]):.2f} (n {len(R_[k_])})" if R_[k_] else "- (n 0)"
    print(f"  [JEPA INDEX | {tag_}] {len(lv_)} stored sentences keyed | RETRIEVAL of the fact sentence for 'A r0 ': top-1 {f_('r1')}, top-5 {f_('r5')} | HELD-OUT wording top-1 {f_('ho1')}, top-5 {f_('ho5')} | "
          f"NAME LOOKUP (most recent sentence naming A) {f_('name')} | HYBRID (sentences naming A, ranked by the JEPA key) {f_('hyb')}, held-out wording {f_('ho_hyb')} || ONE-HOP answer (20 candidates, store off): weights only {f_('w')} | RETRIEVED sentence in context {f_('ret')} | "
          f"TRUE sentence in context {f_('true')} | held-out wording, retrieved {f_('ho_ret')} || TWO-HOP BY RETRIEVAL: hop 1 {f_('h1')}, two-hop {f_('two')} | control, hop 2 about a RANDOM name {f_('two_rand')} | "
          f"true sentences {f_('two_true')} (chance 0.05)", flush=True)
def _jx_sleep(model, cfg, S_):
    """inf_jx: train the index on the store, then the readouts (every inf_jx_every visits)"""
    try:
        r_ = _jx_train(model, cfg, S_); v_ = int(model.__dict__.get('_visit', 0))
        if r_ is None: print(f"  [JEPA INDEX] skipped: fewer stored sentences than one batch", flush=True); return
        with torch.no_grad(): sp_ = float(_jx_keys(cfg, S_, model.embed.weight.device)[1][:4096].std(0).mean())
        print(f"  [JEPA INDEX] visit {v_ + 1}: {int(getattr(cfg, 'inf_jx_steps', 200))} steps on {r_[2]} stored sentences (total {_JX['steps']}) | L1 {r_[0]:.3f} -> {r_[1]:.3f} | key spread {sp_:.3f} (0 = collapsed)"
              + (f" | InfoNCE weight {float(getattr(cfg, 'inf_jx_nce', 0.0)):g}" if float(getattr(cfg, 'inf_jx_nce', 0.0)) > 0 else ""), flush=True)
        if (v_ + 1) % max(1, int(getattr(cfg, 'inf_jx_every', 1))) == 0: _jx_readout(model, cfg, S_, f"visit {v_ + 1}")
    except Exception as e_:
        print(f"  [JEPA INDEX] unavailable ({type(e_).__name__}: {e_})"[:260], flush=True)

# names used at run time from modules that load after this one
_LATE = {
    '_INF_PARA': 'platypus.inference.qtime',
}
