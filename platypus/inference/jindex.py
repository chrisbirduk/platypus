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
              reads C; control: hop 2 asked about a RANDOM name instead of B' (must fall to chance); ceiling: the true sentences.
cg26t additions (all readouts; each off by default):
  inf_jx_pair  share of each batch trained on PAIRED WORDINGS: two different stored sentences naming the same two names (mostly one fact
               told twice) - the cut first sentence predicts the key of the SECOND; the store mines the pairs itself (no labels).
  inf_jx_arm_nce > 0   a second index arm 'C' with InfoNCE at that weight (the toy: helps when facts share a subject).
  inf_jx_gen   a third arm 'G', the GENERIC TRACE: every name the store's own name finder sees is masked before encoding (questions too);
               the names are kept as slots. Retrieval = the sentences whose slots hold the question's name, ranked by the name-free
               structure key (toy tools/toy_jgeneric.py: with the names masked the trace is name-invariant 1.00, keeps relation and
               wording 1.00 and two encoders agree 0.87 against 0.43 raw).
  The extra arms report retrieval (top-1/top-5 and hybrid, written and held-out wordings); the answer readouts use arm V.
cg26u: inf_jx_route - JEPA ROUTING readout (routing unchanged): each visit's stored sentences, joined into ~1 KB windows, routed by a byte
  profile, by arm V's JEPA codes and by both, to the nearest slot prototype; accuracy on revisits, agreement with the router and novelty
  AUC for first visits (tools/toy_jroute.py: locally bytes win at 1 KB, JEPA reaches 1.00 from 4-8 windows, both detect a new domain at
  AUC 0.95-1.00 from 4 windows; masking names hurt routing)."""
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
from platypus.inference.units import (
    _INF_NAME,
)
_JX = {}                                                                     # the encoder, its EMA target and predictor live here across sleeps
class _JXEnc(nn.Module):
    """byte transformer encoder (bidirectional) -> mean over the text positions -> LayerNorm"""
    def __init__(self, d_, nl_, L_):
        super().__init__(); self.L = L_; self.emb = nn.Embedding(258, d_); self.pos = nn.Parameter(torch.randn(L_, d_) * 0.02)
        lay_ = nn.TransformerEncoderLayer(d_, max(1, d_ // 64), 4 * d_, 0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(lay_, nl_, enable_nested_tensor=False); self.n = nn.LayerNorm(d_)
    def forward(self, X_):
        pad_ = (X_ == 256); h_ = self.tr(self.emb(X_) + self.pos[:X_.shape[1]], src_key_padding_mask=pad_)
        m_ = (~pad_).float().unsqueeze(-1); return self.n((h_ * m_).sum(1) / m_.sum(1).clamp(min=1))
def _jx_batch(txts_, L_, dev_):
    """rows of bytes (or of token ids: 0-255 bytes, 257 a masked name) -> (n, len) ids, 256 = padding"""
    bs_ = [list(t_)[:L_] for t_ in txts_]; Lm_ = max(4, max(len(b_) for b_ in bs_)); X_ = torch.full((len(bs_), Lm_), 256, dtype=torch.long)
    for j_, b_ in enumerate(bs_): X_[j_, :len(b_)] = torch.tensor(b_, dtype=torch.long)
    return X_.to(dev_)
def _jx_ids(s_, gen_):
    """a sentence as ids; gen_: every name the store's name finder sees replaced by the mask id 257 (byte for byte)"""
    s_ = bytes(s_); ids_ = list(s_)
    if gen_:
        for m_ in _INF_NAME.finditer(s_):
            for k_ in range(m_.start(), m_.end()): ids_[k_] = 257
    return ids_
def _jx_arms(cfg):
    """the index arms: V (V-JEPA, as cg26s), C (+ InfoNCE), G (generic trace: names masked)"""
    A_ = [('V', float(getattr(cfg, 'inf_jx_nce', 0.0)), False)]
    if float(getattr(cfg, 'inf_jx_arm_nce', 0.0)) > 0: A_.append(('C', float(getattr(cfg, 'inf_jx_arm_nce', 0.0)), False))
    if int(getattr(cfg, 'inf_jx_gen', 0)):                                     # G's InfoNCE weight: inf_jx_gen_nce if set (cg26v: on the GPU the
        A_.append(('G', float(getattr(cfg, 'inf_jx_gen_nce', 0.0)) or float(getattr(cfg, 'inf_jx_nce', 0.0)), True))   #   L1-only arms collapsed in the first sleeps)
    return A_
def _jx_pairs(S_, lv_):
    """PAIRED WORDINGS mined from the store: groups of different live sentences naming the same two names"""
    G_ = {}
    for i_ in lv_[-50000:]:
        s_ = S_['it'][i_]['s']; ns_ = sorted(set(_INF_NAME.findall(s_)))
        if len(ns_) == 2: G_.setdefault(tuple(ns_), {}).setdefault(s_, i_)
    return [list(v_.values()) for v_ in G_.values() if len(v_) >= 2]
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
    lv_ = _jx_live(S_, cfg); B_ = int(getattr(cfg, 'inf_jx_rows', 64)); mom_ = float(getattr(cfg, 'inf_jx_mom', 0.996))
    if len(lv_) < B_: return None
    pf_ = float(getattr(cfg, 'inf_jx_pair', 0.0)); PG_ = _jx_pairs(S_, lv_) if pf_ > 0 else []; npr_ = int(round(pf_ * B_)) if PG_ else 0
    out_ = {}
    for ai_, (nm_, nce_, gen_) in enumerate(_jx_arms(cfg)):
        if nm_ not in _JX:
            with torch.random.fork_rng(devices=[dev_] if dev_.type == 'cuda' else []):   # its own random stream: the run itself is unchanged
                torch.manual_seed(int(getattr(cfg, 'seed', 0)) + 4242 + ai_)
                on_ = _JXEnc(d_, nl_, L_).to(dev_); tg_ = _JXEnc(d_, nl_, L_).to(dev_); tg_.load_state_dict(on_.state_dict())
                pr_ = nn.Sequential(nn.Linear(d_, 2 * d_), nn.GELU(), nn.Linear(2 * d_, d_)).to(dev_)
            for p_ in tg_.parameters(): p_.requires_grad_(False)
            _JX[nm_] = dict(on=on_, tg=tg_, pr=pr_, op=torch.optim.AdamW(list(on_.parameters()) + list(pr_.parameters()), lr=float(getattr(cfg, 'inf_jx_lr', 5e-4)), weight_decay=0.01), steps=0, gen=gen_)
        X9_ = _JX[nm_]; on_, tg_, pr_, op_ = X9_['on'], X9_['tg'], X9_['pr'], X9_['op']
        rs_ = np.random.RandomState(9100 + int(model.__dict__.get('_visit', 0))); ls_ = []
        on_.train(); pr_.train()
        with torch.enable_grad():
            for _ in range(int(getattr(cfg, 'inf_jx_steps', 200))):
                src_ = [S_['it'][int(i_)]['s'] for i_ in rs_.choice(lv_, B_ - npr_, replace=False)]; tgt_ = list(src_)
                for _p in range(npr_):                                         # paired wordings: the cut first sentence predicts the second
                    g_ = PG_[int(rs_.randint(len(PG_)))]; a9_, b9_ = rs_.choice(len(g_), 2, replace=False)
                    src_.append(S_['it'][g_[a9_]]['s']); tgt_.append(S_['it'][g_[b9_]]['s'])
                with torch.no_grad(): z_ = tg_(_jx_batch([_jx_ids(t_, gen_) for t_ in tgt_], L_, dev_))
                p_ = pr_(on_(_jx_batch([_jx_ids(_jx_cut(t_, rs_), gen_) for t_ in src_], L_, dev_))); l_ = (p_ - z_).abs().mean()
                if nce_ > 0: l_ = l_ + nce_ * F.cross_entropy(-torch.cdist(p_, z_, p=1) / max(1e-3, float(z_.abs().mean()) * d_ * 0.1), torch.arange(len(src_), device=dev_))
                op_.zero_grad(); l_.backward(); op_.step(); ls_.append(float(l_))
                with torch.no_grad():
                    for a_, b_ in zip(tg_.parameters(), on_.parameters()): a_.mul_(mom_).add_(b_, alpha=1 - mom_)
        on_.eval(); pr_.eval(); X9_['steps'] += len(ls_); out_[nm_] = (ls_[0], float(np.mean(ls_[-20:])))
    return out_, len(lv_), len(PG_), npr_
@torch.no_grad()
def _jx_keys(cfg, S_, dev_, nm_='V'):
    L_ = int(getattr(cfg, 'inf_jx_len', 128)); lv_ = _jx_live(S_, cfg); K_ = []; X9_ = _JX[nm_]
    for a_ in range(0, len(lv_), 2048): K_.append(X9_['tg'](_jx_batch([_jx_ids(S_['it'][i_]['s'], X9_['gen']) for i_ in lv_[a_:a_ + 2048]], L_, dev_)))
    return lv_, torch.cat(K_)
@torch.no_grad()
def _jx_pred(cfg, q_, dev_, nm_='V'):
    X9_ = _JX[nm_]; return X9_['pr'](X9_['on'](_jx_batch([_jx_ids(q_.encode('latin-1', 'replace'), X9_['gen'])], int(getattr(cfg, 'inf_jx_len', 128)), dev_)))
@torch.no_grad()
def _jx_find(cfg, q_, K_, dev_, k_=5, nm_='V'):
    p_ = _jx_pred(cfg, q_, dev_, nm_)
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
    def lps_(ctx_, q_, cs_, sl_, raw_=False):
        if isinstance(ctx_, bytes): ctx_ = ctx_.decode('latin-1', 'replace')
        return _inf_lps(model, cfg, (ctx_ + ' ' + q_) if ctx_ else q_, cs_, sl_, False, raw_=raw_)
    def ans_(ctx_, q_, tgt_, oth_, sl_):                                      # (scored as every readout: inf_score, per-byte mean since cg26w)
        cs_ = [tgt_] + oth_; return cs_[int(np.argmax(lps_(ctx_, q_, cs_, sl_)))]
    def ans2_(ctx_, q_, tgt_, oth_, sl_):                                     # CONTEXT DIAGNOSTIC: the pick by summed and by MEAN byte log-probability
        cs_ = [tgt_] + oth_; lp_, nb_ = lps_(ctx_, q_, cs_, sl_, True)
        return cs_[int(np.argmax(lp_))], cs_[int(np.argmax(lp_ / nb_))]
    R_ = {k_: [] for k_ in ('r1', 'r5', 'ho1', 'ho5', 'name', 'hyb', 'ho_hyb', 'w', 'ret', 'true', 'ho_ret', 'h1', 'two', 'two_rand', 'two_true')}
    DG_ = {k_: [] for k_ in ('w_s', 'true_s', 'w_m', 'true_m', 'fact', 'fact_m', 'true_is_fact', 'short', 'ex')} if int(getattr(cfg, 'inf_jx_ctxdiag', 0)) else None
    pos_ = {i_: j_ for j_, i_ in enumerate(lv_)}
    def hyb_(q_, A_):                                                         # HYBRID: the sentences that name A, ranked by the JEPA key
        c9_ = [pos_[i_] for i_ in NX_.get(A_, []) if i_ in pos_]
        if not c9_: return None
        p9_ = _jx_pred(cfg, q_, dev_, 'V')
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
        if DG_ is not None:                                                   # which sentence is 'true', and does the scoring favour short names?
            fs_ = c_['facts'][0][1]; ws_, wm_ = ans2_(None, q_, B_, oth_, sl_); ts_, tm_ = ans2_(tr1_, q_, B_, oth_, sl_); fs2_, fm_ = ans2_(fs_, q_, B_, oth_, sl_)
            DG_['w_s'].append(ws_ == B_); DG_['true_s'].append(ts_ == B_); DG_['w_m'].append(wm_ == B_); DG_['true_m'].append(tm_ == B_); DG_['fact'].append(fs2_ == B_); DG_['fact_m'].append(fm_ == B_)
            DG_['true_is_fact'].append(f"{A_} {r0_} {B_}".encode('latin-1') in tr1_)
            for p9_ in (ts_, fs2_):
                if p9_ != B_: DG_['short'].append(len(p9_) < len(B_))
            if len(DG_['ex']) < 2: DG_['ex'].append(f"Q '{q_}' -> {B_} | true sentence '{tr1_[:90].decode('latin-1', 'replace')}' picked {ts_} (mean {tm_}) | fact '{fs_}' picked {fs2_} (mean {fm_})")
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
    ex_ = []                                                                  # the other arms: retrieval only
    for nm_, nc9_, gen_ in _jx_arms(cfg)[1:]:
        _, K2_ = _jx_keys(cfg, S_, dev_, nm_); E_ = {k_: [] for k_ in ('r1', 'r5', 'ho1', 'ho5', 'hyb', 'ho_hyb')}
        for c_ in ch_:
            A_, B_ = c_['names'][:2]; r0_ = c_['rels'][0]
            if not any(has_(S_['it'][i_]['s'], A_, B_) for i_ in NX_.get(A_, [])): continue
            c9_ = [pos_[i_] for i_ in NX_.get(A_, []) if i_ in pos_]
            for w_, qq_ in (('', f"{A_} {r0_} "),) + ((('ho', f"{A_} {_INF_PARA[r0_]} "),) if r0_ in _INF_PARA else ()):
                p9_ = _jx_pred(cfg, qq_, dev_, nm_)
                if not gen_:
                    tp_ = torch.topk(-(K2_ - p9_).abs().sum(-1), min(5, K2_.shape[0])).indices.tolist()
                    E_[w_ + '1' if w_ else 'r1'].append(has_(txt_(tp_[0]), A_, B_)); E_[w_ + '5' if w_ else 'r5'].append(any(has_(txt_(i_), A_, B_) for i_ in tp_))
                if c9_: E_['ho_hyb' if w_ else 'hyb'].append(has_(txt_(c9_[int((K2_[c9_] - p9_).abs().sum(-1).argmin())]), A_, B_))
        g9_ = lambda k_: f"{np.mean(E_[k_]):.2f} (n {len(E_[k_])})" if E_[k_] else "- (n 0)"
        ex_.append(f"ARM {nm_} ({('GENERIC: names masked' + (', + InfoNCE' if nc9_ > 0 else '') + ', ranked among the sentences whose slots hold A') if gen_ else 'V-JEPA + InfoNCE'}): "
                   + ("" if gen_ else f"top-1 {g9_('r1')}, top-5 {g9_('r5')}, held-out top-1 {g9_('ho1')}, top-5 {g9_('ho5')}, ") + f"HYBRID {g9_('hyb')}, held-out {g9_('ho_hyb')}")
    if DG_ is not None and DG_['w_m']:
        d9_ = lambda k_: f"{np.mean(DG_[k_]):.2f} (n {len(DG_[k_])})" if DG_[k_] else "- (n 0)"
        ex_.append(f"CONTEXT DIAGNOSTIC (one-hop, 20 candidates; sum = summed byte log-probability, mean = per byte): weights only sum {d9_('w_s')} / mean {d9_('w_m')} | "
                   f"TRUE sentence sum {d9_('true_s')} / mean {d9_('true_m')} (it holds the exact fact 'A r0 B' {d9_('true_is_fact')}) | the PLANTED FACT sentence itself sum {d9_('fact')} / mean {d9_('fact_m')} | "
                   f"wrong picks with context that are SHORTER names than B {d9_('short')} | e.g. " + " || e.g. ".join(DG_['ex']))
    print(f"  [JEPA INDEX | {tag_}] {len(lv_)} stored sentences keyed | RETRIEVAL of the fact sentence for 'A r0 ': top-1 {f_('r1')}, top-5 {f_('r5')} | HELD-OUT wording top-1 {f_('ho1')}, top-5 {f_('ho5')} | "
          f"NAME LOOKUP (most recent sentence naming A) {f_('name')} | HYBRID (sentences naming A, ranked by the JEPA key) {f_('hyb')}, held-out wording {f_('ho_hyb')} || ONE-HOP answer (20 candidates, store off): weights only {f_('w')} | RETRIEVED sentence in context {f_('ret')} | "
          f"TRUE sentence in context {f_('true')} | held-out wording, retrieved {f_('ho_ret')} || TWO-HOP BY RETRIEVAL: hop 1 {f_('h1')}, two-hop {f_('two')} | control, hop 2 about a RANDOM name {f_('two_rand')} | "
          f"true sentences {f_('two_true')} (chance 0.05)" + ("".join(" || " + e_ for e_ in ex_)), flush=True)
def _jx_bytes(bs_):
    """byte unigram + hashed byte-pair profile, square-rooted (the routing toy's BYTES router)"""
    out_ = np.zeros((len(bs_), 256 + 4096), dtype=np.float32)
    for i_, b_ in enumerate(bs_):
        a_ = np.frombuffer(bytes(b_), dtype=np.uint8).astype(np.int64)
        if a_.size < 2: continue
        out_[i_, :256] = np.bincount(a_, minlength=256); out_[i_, 256:] = np.bincount((a_[:-1] * 257 + a_[1:]) % 4096, minlength=4096)
    out_[:, :256] /= out_[:, :256].sum(1, keepdims=True) + 1e-9; out_[:, 256:] /= out_[:, 256:].sum(1, keepdims=True) + 1e-9
    return torch.tensor(np.sqrt(out_))
@torch.no_grad()
def _jx_route(model, cfg, S_):
    """JEPA ROUTING READOUT (inf_jx_route; routing itself unchanged): this visit's stored sentences are joined into ~1 KB windows and coded by
    BYTES (byte profile), JEPA (arm V's target encoder, mean over 128-byte chunks) and BYTES+JEPA (median-normalised distances added);
    each method routes the visit, with 1 and 4 windows of context, to the nearest prototype of the slots seen so far (prototype = mean code
    of the windows of the visits routed there). Scored against the true domain (evaluation only) and the router's own slot; NOVELTY = the
    distance to the nearest prototype, first visits of a domain against revisits (AUC over the run so far)."""
    v_ = int(model.__dict__.get('_visit', 0)); sl_ = int(model.__dict__.get('_cur_dom', -1)); td_ = int(model.__dict__.get('_s2t', {}).get(sl_, sl_))
    it_ = [x_['s'] for x_ in S_['it'] if x_.get('s') and not x_.get('gone') and int(x_.get('v', -1)) == v_]
    if len(it_) < 40: print(f"  [JEPA ROUTING | visit {v_ + 1}] skipped: {len(it_)} stored sentences from this visit", flush=True); return
    rs_ = np.random.RandomState(9300 + v_); W_ = []
    for _ in range(32):                                                      # 32 windows of ~1 KB from the visit's sentences
        o_ = rs_.permutation(len(it_)); w_ = b''
        for i_ in o_:
            w_ += bytes(it_[i_]) + b' '
            if len(w_) >= 1024: break
        W_.append(w_[:1024])
    dev_ = model.embed.weight.device; L_ = int(getattr(cfg, 'inf_jx_len', 128))
    cs_ = [w_[i_:i_ + L_] for w_ in W_ for i_ in range(0, len(w_), L_)]; nc_ = [len(range(0, len(w_), L_)) for w_ in W_]
    zc_ = _JX['V']['tg'](_jx_batch(cs_, L_, dev_)).float().cpu(); zj_ = torch.stack([c_.mean(0) for c_ in torch.split(zc_, nc_)])
    F_ = {'BYTES': _jx_bytes(W_), 'JEPA': zj_}
    R9_ = _JX.setdefault('route', {'proto': {}, 'hist': []}); P_ = R9_['proto']; seen_ = {d_ for _, d_, _ in R9_['hist']}
    first_ = td_ not in seen_; row_ = {'v': v_, 'td': td_, 'first': first_, 'slot': sl_}
    if P_:
        sls_ = sorted(P_)
        for k_ in (1, 4):
            idx_ = torch.tensor(rs_.randint(0, len(W_), size=(16, k_)))
            D_ = {}
            for nm_, X_ in F_.items():
                C_ = X_[idx_].mean(1); Pm_ = torch.stack([P_[s9_][nm_][0] / P_[s9_][nm_][1] for s9_ in sls_]); D_[nm_] = torch.cdist(C_, Pm_)
            D_['BYTES+JEPA'] = D_['BYTES'] / D_['BYTES'].median().clamp_min(1e-9) + D_['JEPA'] / D_['JEPA'].median().clamp_min(1e-9)
            for nm_, Dm_ in D_.items():
                ch_ = [sls_[int(j_)] for j_ in Dm_.argmin(1)]
                row_[(nm_, k_, 'dom')] = float(np.mean([P_[c_]['td'] == td_ for c_ in ch_])); row_[(nm_, k_, 'agree')] = float(np.mean([c_ == sl_ for c_ in ch_]))
                row_[(nm_, k_, 'nov')] = float(Dm_.min(1).values.mean() / (Dm_.median().clamp_min(1e-9)))
    for nm_, X_ in F_.items():                                               # this visit's windows join its slot's prototype
        e_ = P_.setdefault(sl_, {'td': td_}); s_, n_ = e_.get(nm_, (0.0, 0)); e_[nm_] = (s_ + X_.sum(0), n_ + X_.shape[0]); e_['td'] = td_
    R9_['hist'].append((v_, td_, row_))
    H_ = [r_ for _, _, r_ in R9_['hist'] if ('JEPA', 1, 'dom') in r_]
    def auc_(p9_, n9_):
        if not p9_ or not n9_: return float('nan')
        a9_ = np.concatenate([p9_, n9_]).argsort().argsort().astype(float) + 1
        return (a9_[:len(p9_)].sum() - len(p9_) * (len(p9_) + 1) / 2) / (len(p9_) * len(n9_))
    parts_ = []
    for nm_ in ('BYTES', 'JEPA', 'BYTES+JEPA'):
        for k_ in (1, 4):
            rv_ = [r_[(nm_, k_, 'dom')] for r_ in H_ if not r_['first']]; ag_ = [r_[(nm_, k_, 'agree')] for r_ in H_ if not r_['first']]
            nv_ = auc_([r_[(nm_, k_, 'nov')] for r_ in H_ if r_['first']], [r_[(nm_, k_, 'nov')] for r_ in H_ if not r_['first']])
            parts_.append(f"{nm_} {k_} KB: revisits to their domain {np.mean(rv_) if rv_ else float('nan'):.2f} (n {len(rv_)}), agrees with the router {np.mean(ag_) if ag_ else float('nan'):.2f}, novelty AUC {nv_:.2f}")
    now_ = " | ".join(f"{nm_} {k_} KB -> {row_[(nm_, k_, 'dom')]:.2f}" for nm_ in ('BYTES', 'JEPA', 'BYTES+JEPA') for k_ in (1, 4) if (nm_, k_, 'dom') in row_)
    print(f"  [JEPA ROUTING | visit {v_ + 1}] true domain {td_} ({'FIRST visit' if first_ else 'revisit'}), router slot {sl_}, {len(it_)} stored sentences in {len(W_)} windows | this visit to its domain: {now_ or 'no prototypes yet'} || "
          f"SO FAR: " + " || ".join(parts_), flush=True)
def _jx_sleep(model, cfg, S_):
    """inf_jx: train the index arms on the store, then the readouts (every inf_jx_every visits); its time is kept out of the sleep budget"""
    t0_ = __import__('time').time()
    try:
        _jx_sleep_(model, cfg, S_)
    finally: model.__dict__['_jx_secs'] = model.__dict__.get('_jx_secs', 0.0) + __import__('time').time() - t0_
def _jx_sleep_(model, cfg, S_):
    try:
        r_ = _jx_train(model, cfg, S_); v_ = int(model.__dict__.get('_visit', 0))
        if r_ is None: print(f"  [JEPA INDEX] skipped: fewer stored sentences than one batch", flush=True); return
        L9_, n9_, pg9_, npr9_ = r_; dev_ = model.embed.weight.device
        sp_ = {nm_: float(_jx_keys(cfg, S_, dev_, nm_)[1][:4096].std(0).mean()) for nm_ in L9_}
        print(f"  [JEPA INDEX] visit {v_ + 1}: {int(getattr(cfg, 'inf_jx_steps', 200))} steps per arm on {n9_} stored sentences (total {_JX['V']['steps']}) | "
              + " | ".join(f"arm {nm_}: L1 {a_:.3f} -> {b_:.3f}, key spread {sp_[nm_]:.3f}" for nm_, (a_, b_) in L9_.items())
              + (f" | InfoNCE weight {float(getattr(cfg, 'inf_jx_nce', 0.0)):g}" if float(getattr(cfg, 'inf_jx_nce', 0.0)) > 0 else "")
              + (f" | PAIRED WORDINGS: {pg9_} groups of different sentences naming the same two names, {npr9_} pair rows per batch" if float(getattr(cfg, 'inf_jx_pair', 0.0)) > 0 else ""), flush=True)
        if (v_ + 1) % max(1, int(getattr(cfg, 'inf_jx_every', 1))) == 0: _jx_readout(model, cfg, S_, f"visit {v_ + 1}")
        if int(getattr(cfg, 'inf_jx_route', 0)):
            try: _jx_route(model, cfg, S_)
            except Exception as e9_: print(f"  [JEPA ROUTING] unavailable ({type(e9_).__name__}: {e9_})"[:240], flush=True)
    except Exception as e_:
        print(f"  [JEPA INDEX] unavailable ({type(e_).__name__}: {e_})"[:260], flush=True)

# names used at run time from modules that load after this one
_LATE = {
    '_INF_PARA': 'platypus.inference.qtime',
}
