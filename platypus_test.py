# =====================================================================================================================
# PLATYPUS: a language model that learns without forgetting and thinks in its sleep
# =====================================================================================================================
# Copyright 2026 CHRISTIAN BIRD
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# =====================================================================================================================
#
# WHAT THIS FILE IS
#   The full-size (GPU) build of Platypus, a byte-level continual-learning language model. It reads ten Pile domains
#   one after another over several passes and is evaluated on how much it keeps, how much it learns and what it can
#   infer. Build: cg20s. The run configuration and a full change history are in NAME and LAB_OVR near the end.
#
# WHAT IT DOES
#   * Continual learning without stored data: domains are learned in sequence; old domains are protected by
#     rehearsing the model's own dreams (stored self-generated windows with their top-32 predictions), curvature-aware
#     (Fisher-whitened) updates, and small per-domain capacity (heads, private n-gram rows, private units, FiLM codes).
#   * A strict closed loop: after the first pass, no real text. The model learns only from replays of its memory traces
#     that it can verify exactly against what it originally read (verified prefix credit, repair at the first wrong
#     byte, confidence gate, trace retirement).
#   * Label-free routing: the domain is never given; visits are assigned to slots by prediction quality.
#   * Inference in sleep: after every visit it links facts met in different places (bridge-centred chaining with hub
#     abstention), replays salient facts (NREM-like), proposes links from memory and keeps only grounded ones
#     (REM-like), revises beliefs (recency rule) and retracts dependent conclusions (cascading retraction).
#   * A fast store outside the weights: an entity-banked, drift-free-keyed delta-rule logit corrector, written to
#     convergence with do-no-harm anchors and applied only where it is confident, so it can say "I don't know".
#   * Readouts: probe recall from the weights and with the store, answer-or-abstain, sources, belief revision,
#     store cost on ordinary text, splice-and-appraise, and multi-hop question-time chaining.
#
# ARCHITECTURE
#   1024-byte windows; byte + hashed 2/3/4-gram embeddings (sparse); block 0 a causal local filter; blocks 1-3 gated
#   delta-rule mixers (8 heads, beta up to 2, preconditioned write key, WY-form parallel solve); residual width 1088;
#   one soft-capped head per domain. About 171M trunk parameters plus 223M hashed n-gram parameters.
#
# USAGE
#   Run as a script (e.g. in Colab with a GPU). Options are set through LAB_OVR (JSON) and the Config defaults.
#   Logs report per-visit BPC, forgetting since each visit, routing, and the inference readouts.
#
# STATUS
#   This is research code. It is bulky: it carries options, readouts and experiment paths from many iterations that are
#   switched off in this configuration, and it needs a cleanup (dead options removed, modules split, tests added) and
#   further optimisation (evaluation frequency, store writes, data loading and generation speed) before wider use.
# =====================================================================================================================
import contextlib
import os as _os0
_os0.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')   # less fragmentation (only takes effect before CUDA starts)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, random, time, contextlib, os, weakref as _weakref
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LOG2=math.log(2)

class Config:
    vocab=256; d_model=384; d_state=384; n_layers=4   # bigger base (diagnostic: capacity-starved)
    d_state_max=2048        # widening cap (raised: all domains incl. later ones get fresh room)
    widen_chunk=64          # units added per widening
    widen_per_difficulty=512  # units per unit of (difficulty - easiest)
    lock_after_domains=2    # accumulate distinctness reference over this many domains (prose cluster) before locking
    seed=0                          # set per run by the driver at the bottom
    threshold_mode='relative'       # 'layer' = drive vs layer mean (v2); 'relative' = drive vs the unit's own typical drive
    channel_thresholds=False        # B/C: thresholds on every d_model channel too (embeddings, pos, LayerNorms, residual writes)
    route_mode='forward'            # 'forward' (B), 'write' (write-only routing), 'none' (no distinctness routing)
    route_writes=False              # derived from route_mode in run()
    gate_mode='threshold'           # 'threshold' = independent per-unit sigmoid (B); 'competitive' = impedance competition
    write_budget=0.20               # competitive: fraction of units (per layer) written each step - the lowest-impedance path
    comp_temp=0.05                  # softness of the budget cutoff
    comp_floor=0.0                  # write for units outside the budget (0 = eliminated)
    unit_budget=None                # competitive budget for state/MLP units (None -> write_budget)
    channel_budget=None             # competitive budget for channels/embeddings (None -> write_budget)
    channel_gate_mode=None          # 'threshold' or 'competitive' for channels/embeddings (None -> gate_mode)
    route_width=0.1                 # write-route selectivity: exp(-(distinct-affinity)^2/width)
    hebb_rate=0.01                  # affinity drift toward the distinctness of content that drives each unit
    log_locality=True               # measure orientation clustering / damage locality / neighbour effect after each domain
    locality_k=8                    # nearest neighbours (by input-weight orientation) for the neighbour test
    drive_ema=0.99                  # smoothing for each unit's typical drive (relative mode)
    drive_ref='ema'                 # SIGNAL 1: 'ema' (D) | 'slow' | 'max' | 'peak'
    drive_slow=0.9995               # slow EMA of raw drive ('slow'/'max')
    peak_decay=0.9998               # decay of the running peak of the fast EMA ('peak')
    use_sens=False                  # SIGNAL 2: loss sensitivity on earlier domains raises thresholds
    sens_alpha=1.0                  # weight of sensitivity relative to usage importance
    theta_dist='uniform'            # base impedance distribution: 'uniform' | 'lognormal' | 'bimodal' | 'constant'
    theta_sigma=0.5                 # lognormal spread
    theta_reserve=0.3               # bimodal: fraction of hard-to-write reserve neurons
    emb_thresholds=True             # gate embedding/position columns (with channel_thresholds)
    imp_transform='relu'            # 'relu' (D) | 'raw' | 'sqrt' | 'log' | 'rank'
    imp_top=0.2                     # 'rank': fraction of neurons made important per domain
    imp_decay=1.0                   # importance kept per domain end (1 = permanent)
    imp_revisit=None                # REVISIT-AWARE WRITE GATES (repeated passes): None = importance and sensitivity add up per VISIT (a revisited
    imp_combine='sum'               # how the domains' importances combine: 'sum' = add up per visit (original); 'max' = each domain keeps ONE
    imp_recency_tau=0.0             # with imp_combine='max': RECENCY - a domain's protection is weighted c + (1-c) exp(-age/tau), age = visits
    imp_recency_floor=0.5           #   since it was last trained (0 = the very next visit): fresh writes fully protected, consolidated ones at c
    imp_floor=0.0                   # GUARANTEED PLASTICITY: in every group the least-important imp_floor share of neurons gate as if fresh
    imp_online_every=0              # >0: grow importance DURING a domain every N steps
    log_every=200
    orth_gate=False                 # ORTHOGONAL-GATED UPDATE: threshold gates only the in-old-subspace part of writes
    og_energy=0.5                   # old-subspace energy protected (0.5 = the dominant shared directions)
    og_in=0.0                       # in-subspace part written at gate*og_in (0 = removed, 1 = gated as usual)
    og_orth='gate'                  # orthogonal part: 'gate' (usual threshold gate) or a fixed strength (1.0 = free)
    og_orth_boost=1.0               # orthogonal part written at min(gate*boost, 1)
    og_common=1.0                   # extra write scale for common-mode params (biases, LayerNorm, beta) once old domains exist
    og_common_always=False          # apply the common-mode scale even without orthogonal gating
    replay_buffer_batches=0         # >0: dedicated replay store of this many batches per domain (else the small prior_buffer)
    shared_head=False               # one output head shared by all domains (instead of one per domain)
    head_select_eval=()             # also evaluate with INFERRED heads: ('bytes','features') - training unchanged
    scratch_domain=None             # FORWARD-TRANSFER reference: train a fresh model on this one domain only
    fwt_zeroshot=False              # score each domain before training on it (earlier heads, inferred from bytes)
    gate_start_domain=0             # no write gating before this domain index (1 = first domain ungated)
    hs_alpha=0.1                    # byte-pair count smoothing
    hs_temp=1.0                     # posterior sharpness (scales the cumulative log-likelihood)
    dual_rate=False                 # FAST/SLOW WEIGHTS in every gated param (from domain 2)
    shunt=False                     # SHUNT & CONSOLIDATE: interfering (old-subspace) part of writes kept in a shunt store,
    shunt_every=200                 # steps between consolidations (make it divide steps_per_task: last one at domain end)
    shunt_batches=4                 # recent current-domain batches used for distillation
    shunt_T=1.0; shunt_old_weight=1.0
    shunt_mode='subspace'           # 'subspace' = only the interfering part goes to the shunt; 'all' = all new learning (fast store)
    shunt_consol='periodic'         # 'periodic' = separate distillation bursts; 'continuous' = old memories (main weights)
    shunt_kl_mode='fixed'           # 'fixed' | 'learned' (per-old-domain weight learned, uncertainty weighting) | 'confidence'
    dream=False                     # DREAM PROBES (no stored data): each old head samples text from the MAIN weights (which
    dream_n=64; dream_len=128; dream_batch=8; dream_weight=1.0; dream_temp=1.0
    dream_adapt=False               # ADAPTIVE dream weight per old domain: multiplied by exp(lr*(dreamKL/target-1)) each
    dream_rollback=False            # DAMAGE ALARM + ROLLBACK: the shunt is snapshotted every rb_every steps; if an old domain's
    topic_mode=False                # DYNAMIC TOPICS (no domain labels in training): the byte-context topic tracker spawns topics
    og_dream_n=16                   # statistics go stale at the upper blocks); covariance = mix*dream + (1-mix)*stored
    recon_blocks=(2,3)              # of recon_blocks, each layer's output on an old domain's dreams is pulled back towards what
    recon_lr=0.1     # the main weights produce: W += lr*(v_old - W k)k^T / (N*mean|k|^2), k = live layer input
    recon_layers=('Wk','Wv','gate','Wout','mlp_in','mlp_out')   # v2: correct only these layers (output projections: Wout, mlp_out)
    recon_new_energy=0.5
    recon_cap=1.0                   #  (4) robust weight 1/(1+KL_t/cap) per dream token (as the dream KL): degenerate dreams count less
    recon_sleep=0                   #  (3) >0: no corrections during training; this many SLEEP steps at the end of each domain,
    stale_diag=False                # at each domain start: overlap of the stored / dream-refreshed old-input subspaces with the
    lr_sched=None                   # LEARN EACH DOMAIN BETTER: 'cosine' = per-domain warm-up (lr_warmup steps) then cosine decay
    lr_warmup=50; lr_min_frac=0.1   # to lr_min_frac x lr by the domain's end (re-warmed at every new domain)
    dream_decay=0.0                 # >0: dream weight falls linearly over each domain to (1 - dream_decay) x dream_weight
    dream_recency=None              # SYSTEMS CONSOLIDATION - dreams guard RECENT memories: old domain d's share of the dream budget
    dream_graduate=None             # domains of age >= this GRADUATE: dream share dream_grad_floor (0 = no dreams, structural
    dream_grad_floor=0.0            # protection only; >0 = occasional maintenance dreams)
    block_attr=False                # DATA-FREE BLOCK ATTRIBUTION (works for any future domain/topic): every block_attr_every steps,
    block_auto=False                # use the attribution to set each block's write strength automatically: blocks whose damage
    block_write=None                # e.g. (1,1,0.5,0.5): write strength per block once there is something to protect
    forget_diag=False               # at each domain end: which part (backbone / each block / heads / routing) lost each earlier domain
    fd_batches=5
    logit_cap=0.0                   # ARCHITECTURE: >0 soft-caps every head's logits at +-logit_cap (cap*tanh(z/cap))
    gate_ln=False                   # ARCHITECTURE: the SSM output gate reads LayerNorm(x) instead of the raw residual x
    dream_cap=0.0                   # >0: ROBUST dream KL, per-token KL -> cap*log(1+KL/cap) (bounded influence of outliers)
    dream_filter=False              # generate 1.5x dreams and keep the most varied (distinct bytes, no long single-byte runs)
    dream_refresh=50                # every N steps, dream_batch fresh sequences replace the oldest in one old domain's pool
    cross_lr=0.01; cross_min=-1.0; cross_max=2.0
    pg_mode=None                    # POLARISATION gating of weight writes: None | 'weight' | 'proj' | 'proj_uniform' | 'proj_seqnov'
    polar_diag=False                # print, once per domain, whether token angles separate old from new content (no training effect)
    data_free=False                 # NO stored samples: sensitivity from each domain's last live batches, orthogonal
    cov_every=10; cov_tokens=512    # data-free covariance sampling: every N steps, token subsample size
    topic_diag=False                # TOPICS from byte context (diagnostic only)
    topic_k=16; topic_h=1024; topic_window=128
    topic_center='running'          # profiles centred on a running mean: 'running' (centroids stored centred: drift as the mean
    topic_cover=0.05                # head-coverage: a topic 'knows' a domain if it got >= this share of its training contexts
    topic_temp=0.03                 # softness of topic assignment (centred cosine / temp)
    topic_new=0.2                   # a context is 'unmatched' when its best centred cosine is below this
    topic_jump=0.05; topic_gap=50   # open a topic when the unmatched share jumps this far above its long-term level
    topic_lr=0.2; topic_mature=300.0
    reserve_mult=0                  # 0 = off (growth as before); 16 = x16 state and MLP units allocated at init
    init_state_mult=2.0             # initially active state units = d_state * this (2.0 ~ first-domain width of the growth runs)
    init_mlp_mult=1.0               # initially active MLP units = 2*d_model * this
    mix_heads=1                     # MIXER: number of delta-rule heads (keys normalised per head; d_state split evenly)
    mix_head_dim=0                  # MIXER: >0 = head dimension cap instead of a fixed head count: each block uses the largest head dim <= this that
    lean_stats=False                # SPEED: skip the per-forward unit / MLP / channel drive statistics (only the write gates, growth and importance
    muon_gram=False                 # SPEED: Gram Newton-Schulz for Muon (iterate on the n x n Gram matrix, one restart, fp16 GEMMs) for weights with aspect >= 2
    sleep_kmax_pass=()              # per-pass dream floor, e.g. (3,4,5,6,6): while every old domain is under budget, dream every k-th step in pass p
    noise_diag=0                    # >0: every noise_diag steps, gradient noise scales of the real and the dream loss (4 half-batch backward passes)
    dream_mom=0.0                   # DREAM MOMENTUM beta (>0): running average of the trunk's dream-loss gradient (split from the real rows' at no
    kf_dream=0.0                    # LIVE OLD-DOMAIN KEYS (0..1): Kalman input factors mixed with the dream rows' CURRENT input factor (the old domains'
    kf_dream_every=200              #   kf_dream_stride-th dream step (merged forward, input side only)
    kf_dream_stride=3
    sleep_fixed=0.0                 # >0: OPEN LOOP - constant sleep pressure (the GPU loop sat at sleep_pmin 0.5 all run) and the per-pass dream floor
    head_dream_k=False              # True: old heads (gradient only on dream steps; Adam skips them otherwise and cancels the x k weight) step k x
    mix_precond=0.0                 # >0: PRECONDITIONED DELTA RULE (diagonal RLS write, cf. OSDN 2026): each write uses the key rescaled per feature by
    mix_precond_chunk=0             # >0: the preconditioner's statistics are updated once per block of this many positions (block c uses the keys of
    mix_precond_detach=False        # True: no gradient through the preconditioner (D and the normaliser are statistics): a linear rescale of the keys
    kf_share_a=False                # True: Linears reading the SAME input (Wk, Wv, Wb, Wa, gate) share one input-covariance computation in the hooks and
    muon_mom_dream=0.0              # >0: SPLIT MOMENTUM - the dream part of the trunk gradient (row-split weight gradient) gets its own Muon momentum
    kf_rank=0                       # >0: LOW-RANK + ISOTROPIC whitening factors: top kf_rank eigen-directions kept, the rest of the spectrum replaced by its
    bank_hold=0                     # >0: the first bank_hold rows of every bank are never trained on; at each visit end [bank holdout] prints the
    dream_lazy_rows=False           # SPEED: the dream decision (closed loop / drift budget) is taken BEFORE the merged forward, so on skipped steps
    ema_every=1                     # SPEED: tail average + averaging pull applied every k steps with k-fold rates (same horizon; 1 = every step)
    mix_beta_tok=False              # MIXER: per-token, per-head write strength beta_t = sigmoid(W_b h) (starts at the old constant)
    mix_conv=0                      # MIXER: >0 = short causal depthwise convolution of this width on keys/values(/queries)
    mix_query=False                 # MIXER: separate query projection (starts as a copy of the key projection); output read
    mix_beta_max=1.0                # REFLECTIONS: write strength beta_t = mix_beta_max * sigmoid(.); 2.0 lets I - beta k k^T reach a Householder
    cons3=False                     # THREE TIMESCALES (Benna-Fusi-style chain): shunt (fast) -> MID store -> main (slow). At each domain end
    cons3_age=2                     # Old domains visited >= cons3_age domains ago (graduated) are held to - and dream from - the SLOW
    n_passes=1                      # REPEATED PASSES over the whole domain sequence (revisits train the domain's own head; protection
    extra_domains=()                # further Pile subsets appended after the base five (pile_set_name values)
    nov_pool=0                      # NOVELTY RESERVE: number of closed MLP slices per block (0 = off). A slice opens only on a visit whose text is
    nov_frac=0.25                   # slice width as a share of the active MLP width (2 d_model init_mlp_mult)
    nov_token=False                 # PER-TOKEN gate: each slice fires only on positions whose n-gram (table nov_table, either hash) was rare
    lean=False                      # LEAN RUN: minimal evaluation - every domain is scored only at the end of each pass (current domain after each
    no_revisit=False                # TRAINING TEXT NEVER REVISITED: each visit reads its shard as non-overlapping windows in random order, each window
    eval_batch=0                    # >0: evaluation batches are stacked into forwards of up to this many sequences (same text, same result, faster)
    pass_shard_bytes=4_000_000      # with pass_fresh_data: training text per domain per pass (one shard per pass)
    pass_fresh_data=False           # with n_passes > 1: every pass trains on UNSEEN text - each domain's training text is collected n_passes times
    mix_decay=False                 # MIXER: per-token, per-head DECAY (Gated DeltaNet): S_t = a_t S_{t-1} + beta_t (v_t - a_t S_{t-1} k_t) k_t^T,
    mix_decay_bias=-8.0             #        log a_t = -softplus(W_a h + b); starts at a ~ 0.9997 (almost no forgetting inside a sequence)
    grow_blocks_at=None             # DEPTH: domain indices at whose start a new TOP block is added (identity at first: zero output
    stat_every=25                   # SPEED: host-side diagnostics (worst dream etc.) only every this many steps
    amp=None                        # SPEED: 'bf16' = blocks run under bfloat16 autocast (delta-rule solve, norms, heads stay fp32)
    timing=True; timing_every=10    # SPEED: per-phase wall-clock per step, measured on regular steps (step % timing_every == 3)
    mix_chunk=0                     # >0: CHUNKED delta-rule solve (chunks of mix_chunk positions, state passed between chunks) instead of the full T x T
    mix_lowp=False                  # True (GPU, with mix_chunk): the chunked solve's two d x d-sized state products per chunk in bf16 on tensor cores; the
    mix_kernel='torch'              # 'fla' = Triton chunked Gated DeltaNet kernel (flash-linear-attention) for the delta-rule solve, via
    fast_teacher=False             # FROZEN TEACHERS: the main weights (live minus shunt) and, with three timescales, the slow weights (minus
    fast_kl=False                   # FUSED OLD-HEAD KL: all old heads stacked into one matrix -> one matmul / softmax for every old domain
    fast_dreams=False               # RECURRENT DREAM SAMPLER: dreams generated token by token with the recurrent form of the mixer (O(T)
    borrow_w=0.0                    # FORWARD BORROWING: the CURRENT head also learns from old domains' dreams (soft targets = the old teacher), each
    borrow_mode='posterior'         # 'posterior' = relevance from the label-free posterior; 'grad' = BORROW ONLY IF IT HELPS: every borrow_check steps,
    borrow_check=25
    borrow_first=False              # borrow only on a domain's FIRST visit (when it has the least data of its own)
    em_w=0.0                        # RESPONSIBILITY-WEIGHTED LEARNING (EM): old heads (readouts only, features detached) learn from the CURRENT
    dream_stack=False               # FASTER STREAM GENERATION: every row of a refill (warm and cold streams, every teacher) in ONE recurrent loop -
    fast_dream_chunk=256            # max sequences per sampler call (memory: recurrent state per block = rows x d_state^2 / heads)
    dream_cache=False               # CACHED DREAM TARGETS: the teacher log-probabilities computed while sampling are stored with each dream
    diag_light=False                # single pass: label-free evaluation of the current domain per visit + all domains at the end (the only
    emb_shunt=False                 # 1. EMBEDDINGS IN THE SHUNT: byte/position embedding changes are tracked in the shunt like the blocks', so the
    surprise_mod=False              # 2. SURPRISE-MODULATED WRITES (predictive coding / neuromodulated plasticity): each block's update is scaled by
    cons_chain_keep=0.75            #    domain end L1 <- (1-r1) L1 + shunt, L2 <- (1-r2) L2 + r1 L1, ..., the slow weights get r_K L_K. Each old domain
    gate_dream_relax=0.0            # RESIDUAL GATES: each unit's write gate is opened by relax x c_u, c_u = min(1, the unit's mean |activity| on
    joint_train=False               # REFERENCE: JOINT TRAINING - every batch mixes rows from ALL domains (each row through its own head, so every
    lwf_posterior=False             # POSTERIOR-WEIGHTED CURRENT-BATCH KL (the exact minimiser of the KL-proximal objective is the density-weighted
    lwf_post_floor=0.1              #   mixture f* ~ rho_new q_new + lambda rho_old f_old): each old head's per-token KL is weighted by
    dream_onpolicy=0.0              # ON-POLICY DREAMS (GKD / self-distillation): this share of every dream batch is sampled from the LIVE network (old
    dream_burnin=0                  # DREAM BURN-IN: the KL on dreams ignores the first dream_burnin positions (they start from a newline with no
    feat_anchor=0.0                 # FEATURE ANCHORING: weight of a relative-MSE term matching block feat_anchor_block's output on DREAMS to the
    feat_anchor_block=2             # frozen teacher's (function-space protection inside the layers that forget; dreams only, so the new domain is free)
    dream_pool_mult=1.0             # LARGER DREAM POOLS: every old domain's pool (and initial generation) x this, same per-step dream batch -> each
    dream_stream=False              # CONTINUOUS DREAM STREAMS: instead of pools, parallel per-domain streams from the frozen teacher; every chunk = a
    dream_stream_prompt=64          #   dream_stream_prompt-byte prompt (the tail of the stream's previous chunk: mid-text, like a real window start; no KL)
    dream_stream_block=100          #   + fresh bytes up to dream_len; chunks for the next dream_stream_block steps generated in one batched call; each
    dream_every=1                   # >1: the dream KL (and dream borrowing) only every dream_every steps, weight x dream_every (Adam's momentum
    share_mode=None                 # ADAPTIVE DREAM SHARES (dream bank): 'drift' = old domain d's share of the dream rows ~ EMA of its dream KL (how far it is
    dream_temp_low=1.0              # v9.8 cg10: dream temperature for LOW-ENTROPY domains (BPC on the first visit's last real batches below dream_temp_bpc),
    dream_temp_bpc=0.0              #   decided once at the first visit (no real text read later); 0 = off
    share_floor=0.3                 #   share_mode 'sim': SIMILARITY-WEIGHTED REPLAY (McClelland et al. 2020) - old domains' dream rows ~ their relevance to the
    share_reweight=False            # True: rows reweighted by (1/n_old)/p_d - same objective as equal shares, only the sampling changes
    dream_reuse=False               # with dream_every=k: the dream term at NORMAL weight on its steps (own forward/backward), its gradient cached and
    bank_rec=False                  # dream bank written with the RECURRENT sampler (O(T): needed at 1024-byte windows) from the visit-end weights
    bank_chunk=512                  # rows per recurrent-sampler call when writing the bank (memory: rows x the recurrent state)
    anchor_mode=None                # LONG-MEMORY ANCHOR on the dream rows (added to the usual KL to the visit-start teacher):
    anchor_replace=False            #   True: the anchor's log-probabilities REPLACE the visit-start teacher's targets (one KL, not two)
    state_prior=False               # PER-DOMAIN INITIAL DELTA-RULE STATE: every block's memory starts each sequence from a learned
    lsmem=False                     # EXACT LEAST-SQUARES MEMORY: a linear map M[d] from FIXED random n-gram keys (orders 1..lsm_orders, hashed,
    dream_head_grad=None            # None: old heads frozen in the dream KL (only the trunk is pulled back); 'both': old domains' heads ALSO
    proj_lambda=0.0                 # >0: CLOSED-FORM PROTECTION (OWM / gradient projection): each trunk Linear keeps an EMA covariance C of its
    proj_eps=0.05                   #   covariance EMA rate per collected dream batch
    proj_nodream=False              #   True: no dream KL at all (dreams only feed the covariance)
    ngp=()                          # PRIVATE PER-DOMAIN N-GRAM ROWS, e.g. (2,3,4): each domain has its own small hashed n-gram tables whose vector is
    ngp_rows=4096
    infer_ctx=False                 # LABEL-FREE CONTEXT: at each visit the domain is NOT given - the first infer_batches batches are scored by every existing
    infer_batches=4                 #   slot (head + private rows); a revisit of the best slot if its BPC is within infer_delta of that slot's reference
    infer_gap_eps=0.0               # cg13: >0 = GAP RULE when every slot is used (see infer_slot_rep); 0.1 replays 10/10 on cg9 and cg10
    infer_proto_guard=0.15          # cg13: with the gap rule, a clear prototype overrides only within this many bits of the lowest BPC (cg13 CPU
                                    #   check: 0.28 bits off -> wrong; cg9 / cg10 replays stay 10/10 at 0.15)
    infer_tiebreak=True             # cg13: False = no prototype tie-break between two slots within 0.1 bits (cg10: a near-tied prototype sent PubMed
                                    #   into the USPTO slot, USPTO forgot +0.171; replayed on cg10 routing without it: 19/20)
    infer_delta=0.3                 #   (its BPC at the end of its last visit), else a NEW slot. Heads, private rows, banks, dreams all follow the slot
    infer_rep=False                 # v9.8 cg9 REPRESENTATION ROUTING (with infer_ctx): every slot keeps a PROTOTYPE, the mean top-block output on its visit's last
                                    #   batches (read through that slot's own codes / private units); the candidate slot is the nearest prototype (1 - cosine,
                                    #   the new batches read through each slot); REVISIT if it is clearly nearest (distance <= infer_margin x the second nearest)
                                    #   AND its BPC is within infer_delta_rep of its reference, else a NEW slot (cg8: the representation jump flagged every
                                    #   revisit, the loss jump only half; the loss jump flagged every new domain)
    infer_margin=0.6
    infer_delta_rep=0.6
    infer_proto_gap=0.0             # cg14d: >0 = with the gap rule, a clear prototype overrides only if ITS slot's BPC is within this many bits of that slot's
                                    #   own reference. All 4 overrides so far: correct ones (Gutenberg) +0.013 / +0.088 / +0.102, the wrong one (cg14 visit 14:
                                    #   PubMed -> USPTO's slot, the gap rule had picked PubMed's own slot at -0.022) +0.417 -> 0.25
    infer_delta_new=0.0             # cg13c: revisit threshold used ONLY while slots are still free (0 = use infer_delta). cg13b visit 3: PubMed +0.290
                                    #   with agreeing prototypes merged into Wikipedia's slot under 0.3. All logged free-slot decisions: correct revisits
                                    #   gap <= 0.105, new domains >= 0.22 (FreeLaw 0.22-0.235, PubMed 0.243-0.29, StackE 0.40-0.45) -> 0.16 sits between
    pass_shuffle=False              # passes after the first visit the domains in a fresh random order (seeded): routing cannot rely on the order
    ctx_carry=False                 # CONTINUOUS CONTEXT: every batch row is a lane that reads its part of the shard in order, and each block's delta-rule
    ctx_lanes=False                 #   control: the same lanes (consecutive segments) without carrying the memory
    ctx_bptt=1                      # WRITE-PATH TRAINING ON REAL TEXT (with ctx_carry): every training row is K = ctx_bptt CONSECUTIVE segments of its lane
    ctx_bptt_from=1                 #   CURRICULUM (short first - essential in the passkey test): visits 1..ctx_bptt_from train single segments, later
    oracle_mem=False                # ORACLE LINEAR MEMORY (diagnostic, at the end of the run): the best a linear (delta-rule) memory could do with the
    oracle_batches=64
    oracle_lams=(1e-2,1e-1,1.0,10.0)
    oracle_scales=(0.25,0.5,1.0)        #   S* also scaled by each of these (the trunk was trained reading memories of the delta rule's own scale)
    oracle_gd=0
    oracle_gd_lr=0.01
    oracle_persist=True             #   also the PERSISTENT-READ oracle: S* read at EVERY position (out += scale S* k~, no decay) instead of as the window's
    oracle_pscales=(0.1,0.25,0.5,1.0)   #   initial state (which the short-horizon decay gates fade within the window); oracle_gd optimises it too
    oracle_adapter=()               #   ranks of a LOW-RANK ADAPTER ORACLE: per domain, x += B (A hn) beside every block's MLP (A random, B zero, as LoRA),
    oracle_ad_steps=300
    oracle_ad_lr=0.003
    real_w_pass=()                  # REAL-TEXT LOSS WEIGHT PER PASS, e.g. (1.0, 0.0): in pass 2 the weights get NO gradient from the real text's labels (the
    mk_dream=0                      # MEMORY-KEYED DREAMS (context distillation from episodic memory): every mk_every steps the delta-rule memories at the end
    mk_every=4
    mk_rows=8
    mk_pool=128
    mk_w=1.0
    mk_teacher='mem'                #   'mem' = teacher reads with the snapshot memory; 'none' = control (teacher without memory: self-distillation)
    mk_always=False
    recon_w=0.0                     # REPLAY TRAINING (memory as a CONDUIT): with real-text gradient on, recon_rows windows per step are encoded into their
    recon_rows=8                    #   end-of-window delta-rule memory (differentiable) and the model is trained to REGENERATE each window starting from its
    recon_mark=False                # REPLAY TRAINING v2 (harness encoder/decoder): each of recon_rows random cd_seg-byte SEGMENTS is encoded into its
    cd_w=0.0                        # CONDUIT REPLAY (Stage C, the harness recipe): in passes whose real_w_pass is 0, cd_rows random cd_seg-byte segments of the
    cd_rows=16                      #   current real batch are encoded into delta-rule traces by a FROZEN teacher (the visit-start weights), replayed free-running
    cd_seg=64                       #   from the trace (cd_mark, temperature cd_temp), checked for EXACTNESS by re-encoding the replay (trace mismatch <= eps; up to
    cd_tries=4                      #   cd_tries draws for the inexact rows), and the student learns the teacher's trace-conditioned targets reading the replay as
    cd_temp=0.7                     #   ORDINARY text (KL, weight cd_w; inexact rows count 1/4). The weights never get a gradient from the real bytes.
    cd_eps=0.0                      #   exactness tolerance (relative trace mismatch); 0 = calibrated every step as 10 x the mismatch of re-encoding the same
    cd_mark=1                       #   segments (the numerical floor of the kernels)
    cd_chain=1                      # conduit: windows of cd_chain x cd_seg bytes, each sub-segment with its own trace, joined for the student
    cd_ngp_frozen=False             # conduit student pass reads the private n-gram rows without training them (10x lr: they blew up)
    cd_fac=False                    # FACTORED TRACES: a segment's memory kept as its per-token write terms (U, K) with S = U^T K, exact (the
                                    #   carry path's own end-state sum, unsummed): 2 x T x dh numbers per head instead of dh^2 (41x smaller at
                                    #   8-byte segments); the replay reads S k = U^T (K k); the exactness test compares states through Gram matrices
    cd_lr=1.0                       # conduit update scale: conduit-only steps move the weights by cd_lr x a normal step (Muon + clipping make the
                                    #   step size independent of the loss scale); on dream steps the conduit loss is scaled by cd_lr instead.
                                    #   < 0: AUTO = min(1, conduit bytes / real batch bytes)
    cd_gen_chunk=1024               # conduit replay generated in chunks of this many segments (x tries) - bounds memory at full-window volume
    vp_nreal=True                   # False: no closed-form head refit on the real labels in passes with real_w_pass 0 (closes the label leak)
    cd_inc=False                    # INCREMENTAL REPLAY GENERATION: each new byte runs the local-filter block over the prefix (or the last
                                    #   cd_inc_win bytes) and the memory blocks over the NEW byte only, their memory carried from the previous
                                    #   byte (exact: the same arithmetic as the full-prefix pass); the generation logits are the teacher targets
    cd_inc_win=0                    #   > 0: window of the local-filter block (must cover its width + the n-gram orders); 0: the whole prefix
    cd_greedy=False                 # first try GREEDY for every segment; the remaining cd_tries - 1 tries SAMPLED, only for segments still inexact
    cd_eps_every=50                 # the self-mismatch tolerance re-measured every cd_eps_every conduit steps (cached in between)
    cd_skip_real=False              # conduit steps (real_w 0): the zero-weighted real-text loss gets no trunk backward (features without grad)
    cd_recon_w=0.0                  # > 0: REPLAY TRAINING ON EXACT REPLAYS in conduit passes: cd_recon_rows exact replayed segments are encoded by the
    cd_recon_rows=16                #   LIVE model (gradient into the encoder) and decoded [mark] + segment[:-1] -> segment from that memory, as in pass 1;
    cd_fok=False                    # v7 CONFIDENCE GATE (parallel sampled tries): each try's confidence = sum log p(chosen byte) during generation; a
    cd_fok_far=0.005                #   piece whose most confident try clears its domain's threshold is accepted WITHOUT the re-encode check; uncertain
    cd_fok_chk=0.05                 #   pieces + a random cd_fok_chk share of confident ones are checked; threshold = lowest confidence with <= cd_fok_far
    cd_fok_min=20000                #   inexact tries above it among the checked tries (after cd_fok_min checked tries for that domain)
    cd_fok_1try=0.0                 # > 0: ONE try per piece once the domain's running exact share reaches this (two below it)
    cd_runs=False                   # v7 EXACT RUNS: where an exact run follows an inexact piece, the student's memory is RESET cd_run_skip bytes into
    cd_run_skip=7                   #   the run (log decay -30), once the local-filter block's 7-byte reach (filter width 4 + n-grams up to 4) lies inside
                                    #   the run; the run's first cd_run_skip + 1 bytes do not teach. Nothing before the run reaches later positions
    cd_runs_any=False               # v9h (with cd_runs): reset after EVERY inexact piece (partial-credit prefixes get clean context too)
    cd_runs_mask=False              # v9 FIX 1 (with cd_runs): each exact run after an inexact piece is read as its OWN CLEAN SEGMENT - the local filter
    cd_run_excl=0                   #   and the n-gram contexts do not reach across the run start, the memory is reset AT it; the run's first byte (+
                                    #   cd_run_excl more) does not teach. Replaces the cd_run_skip dead zone (one whole 8-byte piece per run start)
    cd_kl_norm=False                # v9 FIX 2a: the conduit KL averaged over the positions that TEACH (not all positions): its share of the clipped
                                    #   update no longer shrinks when few positions teach
    cd_lr_teach=False               # v9 FIX 2b: the conduit step scaled by the share of positions that teach (parameter interpolation, as cd_lr)
    cd_reuse=1                      # v9 ROUTE 3: replays generated every cd_reuse-th conduit step; the steps between learn again from the last batch
    cd_repair=False                 # v9e: GREEDY first try + key-row error localisation + cd_repair_n repairs at the first wrong byte
    cd_repair_n=1                   #   repairs per piece (each excludes the byte found wrong, greedy after it)
    cd_hint_frac=0.0                # cg12b HINTS (local hint test: readable traces 81 -> 281 of 300): when a piece's trace is written, the true
                                    #   bytes at the lowest teacher-forced margins (trace-conditioned, true prefix) are stored with it, cd_hint_frac
                                    #   of all bytes per conduit call; replays are generated with those bytes forced. Verification is unchanged
                                    #   (the whole piece is re-encoded and key-checked), so a hinted piece still has to reproduce its trace
    cd_hint_piece=2                 #   at most this many hints per piece
    cd_hint_adapt=0.0               # cg13: >0 = hint budget per call = this x the teacher-forced wrong bytes (one host read per call), clamped to
    cd_hint_min=0.02                #   [cd_hint_min, cd_hint_max] of the replayed bytes (replaces cd_hint_frac)
    cd_hint_max=0.08
    cd_partial=True                 #   (with cd_repair) the key-verified prefix of an inexact piece teaches (partial credit)
    cd_keytol=0.02                  #   key-row tolerance floor (relative row distance); max(this, 10 x the re-encode noise)
    cd_first_block=False            # v9e: checks re-encode only up to the first delta block; exact iff no key row differs (with cd_repair)
    cd_floor_lo=0.0                 # v9e LOW-EXACT FLOOR (cd_floor_hi > cd_floor_lo): conduit weight 0 at a running verified-position share
    cd_floor_hi=0.0                 #   <= lo, full at >= hi, linear between
    cd_floor_every=4                #   at weight 0: only every this-many-th conduit call generates (to keep measuring), the others skip
    cd_buf=0.0                      # v9f REPLAY BUFFER: share of each pass-2 conduit step's windows taken (at random) from a short-lived buffer
    cd_buf_visits=2                 #   of OTHER domains' verified replay windows, kept for this many later visits
    cd_buf_per_step=1               #   windows stored per conduit step
    cd_buf_cap=400                  #   max windows kept per domain visit
    cd_buf_detach=False             # v9g: buffered windows read with their domain's head + private units DETACHED (they train the shared trunk only)
    cd_graph=False                  # SPEED: conduit replay generation recorded as CUDA GRAPHS (one per byte position and batch size; replayed with one
                                    #   launch instead of hundreds). The SAME model forward as the eager path (same kernels for the same batch size);
                                    #   only the launching changes. Startup + periodic self-checks against the eager forward; any failure -> eager.
    cd_graph_pad=True               #   pad variable batch sizes (the repair pass) up to a power of two (the extra rows copy row 0 and are discarded),
                                    #   so a few graphs cover every size. False: only exact batch sizes, recorded after cd_graph_min uses (bit-identical
                                    #   to eager, but the variable-size repair pass then mostly stays eager)
    cd_graph_min=2                  #   (unpadded sizes) uses of a batch size before it is recorded
    cd_graph_check=25               #   self-check (graph vs eager logits, same inputs) on the first call of every new graph set and then every N calls
    cd_graph_tol=0.1                #   self-check tolerance: max |logit difference| / max |logit|; above it (or < 95% same top byte) graphs are switched off
    cd_graph_rows=512               #   graphs are recorded for at most this many rows; larger batches run as several replays of the same graphs
                                    #   (graph memory ~ proportional to rows: 512 keeps it near a quarter of a 2048-row graph)
    cd_graph_max_rec=800            #   safety: more recordings than this in a run means graphs are being rebuilt too often -> eager
    cd_graph_free_gb=6.0            #   MEMORY GUARD: a new graph set is recorded only while at least this much GPU memory is free (else that batch size
                                    #   stays eager); an out-of-memory error anywhere in the graph path switches graphs off and frees their memory
    kf_warm_batches=8               # v9.8: dream batches read at visit start for the live input factors (kf_drop_a)
    kf_drop_a=True                  # v9.8 (g3): with kf_dream >= 1 the old domains' stored INPUT factors are neither collected nor stored; the visit
                                    #   starts from the dream rows' live input factors (one forward pass); gradient factors stored as before
    dstep=False                     # v9.8 cg8 DIRECT LEARNED STEP (pass-1 visits with old domains, real text on): every dstep_every steps the gradient of a
                                    #   HELD-OUT objective J (new domain's held-out text CE + KL to the stored targets on held-out bank rows of the old domains) is
                                    #   taken before and after the update u; along u, J is fitted by a 1-D quadratic (slopes d0 = gJ(0).u, d1 = gJ(1).u) and the
                                    #   best step length s* = -d0 / (d1 - d0) clipped to [0.5, 2] (d0 >= 0: 0.5; no curvature: 2); the visit's lr multiplier is
                                    #   multiplied by s*^dstep_k, kept in [0.25, 4], fresh at every visit (per domain). Local harness (theory test HD vs P):
                                    #   law 1.902 / old 2.197 vs lr x1 2.004 / 2.182 and lr x2 1.928 / 2.203 (above the lr curve)
    dstep_every=10
    dstep_k=0.5
    dstep_rows=4                    #   held-out text rows per check (drawn once per visit from the stream, never trained on)
    dstep_drows=4                   #   held-out bank rows per check (bank rows < dstep_hold are never drawn for training)
    dstep_hold=32
    dstep_pass1=True                #   only visits with the real-text loss on (strict closed loop in pass 2: no real text, multiplier 1)
    cdet_log=False                  # v9.8 cg8 CHANGE-DETECTION LOG (no effect on training): every step the batch-mean top-block representation is compared with
                                    #   its running mean (1 - cosine) and the real-text loss with its running mean; at each visit end: the switch step's
                                    #   jump vs the visit's own steps (z-score, share of steps at or above it) for both signals
    samp_graph=False                # v9.8 cg8 CUDA GRAPH for the recurrent bank sampler: after samp_graph_warm eager bytes every byte step (embedding, blocks with
    samp_graph_warm=64              #   the windowed state written in place at a device-side slot index, heads) is one graph replay; sampling stays eager (same
                                    #   generator, same draws); self-check against the eager step on cloned state at the first bytes of every call
    cd_slot=False                   # v9.8 cg5 SLOT MEMORY for generation: each segment's factored memory sits in a FIXED-SIZE buffer (trace factors + one slot per
                                    #   generated byte, each slot with its own decay offset), written IN PLACE (the eager path rescaled and concatenated every row's
                                    #   whole memory at every byte, and the cg4 graphs copied it in, cloned and concatenated it again); with cd_graph: graphs per
                                    #   (row chunk, byte position) bound to one persistent full-size buffer: no state copies, clones or concatenations at all
    cd_slot_check=0                 #   TEST: the first N generation calls also run the old path on the same inputs and report the largest log-probability difference
    cd_graph_check2=250             #   self-check interval after the first 40 self-checks (cd_graph_check before that)
    cd_sync_find=-1                 #   >= 0: at this conduit call, report every operation that makes the CPU wait for the GPU (code line and count)
    cd_graph_fake=False             #   TEST ONLY (CPU): the graph plumbing (static buffers, padding, state passing) with the forward re-run instead of replayed
    cd_nosync=False                 # v9d: sync-free conduit bookkeeping (gate calibration, statistics, slicing) - same decisions, fewer CPU-GPU syncs
    cd_mis32=False                  # v9c: exactness-check Grams in fp32 (TF32 off) + fp64 combination, not whole traces cast to fp64
    cd_gen_chunk1=0                 # v9c: > 0: generation chunk (pieces) for domains on ONE try (cd_fok_1try reached)
    cd_prof=False                   # v9: profile of a conduit GENERATION step at the conduit calls in cd_prof_calls (first generation step from there)
    cd_prof_calls=(50,1650)         #   (50: the first pass-2 domain; 1650: the 5th, typically high exactness - single try, confidence gate on)
    cd_ngp_mask=False               # v8b (with cd_ngp_frozen False): private n-gram rows learn in conduit passes only from the positions that TEACH
    rs_sel=False                    # v8 SELECTIVE REPLAY TRAINING: of the recon_rows (pass 1) / cd_recon_rows (conduit) replay-training pieces, a rs_frac
    rs_frac=0.5                     #   share is sampled from a pool rs_pool x larger in proportion to p(1 - p), p = exp(-L x the live model's
    rs_pool=4                       #   teacher-forced replay loss per byte) (scored without gradient: the BORDERLINE pieces), the rest at random
                                    #   the replays are verified copies of the text, so the model keeps practising replay without the real bytes
    self_prot_w=0.0                 # cg15b: SELF-PROTECTION ON REVISITS - the revisited slot's own dream bank joins the protected set, its share x this
                                    #   (cg14e2: Github misrouted to StackExchange's slot left StackExchange with NO dream protection, +0.016 forgetting)
    self_dream_nreal=False          # in a pass with real_w_pass 0 the visited domain is protected by its own stored dreams as well
    cd_par=False                    # conduit: all tries of all rows generated as ONE batch (GPU: one model call per byte instead of one per byte per try)
    cd_eps_floor=1e-9               # exactness tolerance floor (GPU batch-size-dependent numerics: 1e-4; a single wrong byte gives ~1e-2)
    cd_inexact_w=0.25               # conduit weight of an INEXACT replay row (exact rows 1; 0 = learn from exact replays only)
    cd_teacher='frozen'             # 'live': NO TEACHER COPY - the live model encodes, replays, checks and makes the targets in every step | 'domain': PER-DOMAIN TEACHER - a visit's conduit teacher is the model saved at the END OF THAT SLOT'S LAST VISIT
                                    #   (fixed within the visit, so traces stay readable; fresh about its own domain, before later visits overwrote it)
    cd_fixed=False                  # conduit teacher FIXED for the whole no-real-text phase: the weights at its first conduit step (= end of the last
                                    #   real-text pass; the harness 'hippocampus'), instead of each visit's start (whose replay skill erodes)
    recon_diag=False                # at the run end: REPLAY FIDELITY per domain - held-out BPC of a window read starting from ITS OWN snapshot vs from none
    mk_gen='mem'                    #   'mem' = dreams generated starting from the snapshot memory; 'none' = from an empty memory (pure self-dreams control)
    ctx_jump=0                      #   >0: every ctx_jump segments all lanes jump to fresh random places (memory reset): episodes of ctx_jump segments,
    ctx_jump_keep=False             #   True: the memory is NOT reset at a jump - each episode starts from the memory of another document of the same
    slot_mem=False                  # MEDIUM-TIMESCALE MEMORY PER SLOT: every slot d keeps a persistent delta-rule state S_d per block; its real rows start
    slot_mem_gamma=0.98
    dream_pair_full=False           # WRITE-PATH TRAINING (with dream_pair > 0 for the 2 x length bank): every dream row is a WHOLE 2 x seq_len dream
    dream_pair_both=False           # True: each pair spends two of the dream rows - the first half A as a normal row AND the second half B carried
    slot_mem_reenc=0                # >0: at every visit end each other slot's memory is RE-ENCODED with the current weights: S_d = mean segment-end
    dream_pair=0.0                  # PAIRED-DREAM EPISODES (learning to use carried memory): the bank holds 2 x seq_len dreams with stored targets from the
    eval_ctx=0                      # evaluation windows of (1 + eval_ctx) x seq_len bytes, scored on the last seq_len bytes only (long-context reading)
    guard=False                     # EARLY DRIFT GUARD (needs bank_hold >= guard_rows): in the first guard_span of a visit, every guard_every (fractions of
    guard_rows=32                   #   the visit) the KL(stored targets || current) of each old domain is measured on guard_rows held-out bank rows (never
    guard_span=0.4                  #   trained on: an unbiased sensor). An old domain is flagged when its KL exceeds max(guard_ratio x its visit-start KL,
    guard_every=0.025               #   visit-start KL + guard_min); for guard_dur of the visit after a flag it gets guard_boost x its dream-row share
    guard_ratio=2.0                 #   (the other domains give up rows: same dream cost) and its Fisher factors count guard_kf x in the Kalman whitening
    guard_min=0.02                  #   (stronger protection of its directions; the whitening is rebuilt when the flagged set changes)
    guard_dur=0.05
    guard_boost=3.0
    guard_kf=1.0
    guard_mode='ratio'              # 'atr': VOLATILITY-SCALED (average true range, as in trading): per old domain ATR = EMA over guard_atr_n checks of the
    guard_atr_n=5                   #   true range |KL_t - KL_t-1| of its held-out KL; flag on a channel breakout KL > EMA(KL) + guard_atr_k x ATR (EMA over the same
    guard_atr_k=3.0                 #   with guard_atr_share the dream-row shares follow ATR_d / mean ATR (clipped to [0.5, guard_boost]) all the time - the
    guard_atr_share=True            #   domains whose function is moving fastest get more of the (unchanged) dream rows
    guard_atr_rows=False            # ATR-CONTROLLED DREAM ROWS: rows per dream step = dream_batch x (mean ATR now / its running mean over the run so far), clipped
    guard_rows_min=0.5              #   to [guard_rows_min, guard_rows_max] x dream_batch: more rows while the old domains' functions move fast, fewer when calm
    guard_rows_max=2.0              #   (the ratio averages ~1: about the same total dream rows)
    stream=False                    # NO VISIT BOUNDARIES (implies infer_ctx): one byte stream; the model is never told where a domain ends. It watches its
    stream_k=0.5                    #   CUSUM on z = (l - level) / noise: level = median of the last stream_win in-control losses, noise = the batch-to-batch
    stream_h=5.0                    #   spread (1.4826 median |l_t - l_t-1| / sqrt 2: blind to the slow fall of the loss while learning); S = max(0, S + z - k),
    stream_sig_floor=0.02           #   alarm at S > stream_h (a jump of d noise units is caught after ~ h / (d - k) batches). Floor on the noise (bits)
    stream_win=16
    stream_trace=False              #   diagnostic: print the detector's statistics around every true boundary
    stream_warm=20                  #   steps at a visit start that only feed the statistics
    stream_rep=False                #   cg15: alarm on the REPRESENTATION JUMP (1 - cos(batch-mean top-block output, its running mean)) instead of the
                                    #   loss CUSUM (cdet log, cg12b/cg13c: 15/15 switches flagged by the jump, no step inside a visit at or above it; loss 12/15)
    stream_rep_z=12.0               #   alarm when the jump is this many robust noise units (1.4826 MAD of the visit's in-control jumps) above their median
    stream_rep_or=True              #   with stream_rep: alarm on the jump OR the loss CUSUM
    stream_rep_floor=1e-4           #   floor on that noise
    stream_cap=3.0                  #   longest visit, x steps_per_task (only a safety bound: visits end by detection)
    ema_tail=0.0                    # >0: TAIL AVERAGING - an exponential moving average of all weights (private rows included) with horizon
    sleep_keep=0.0                  # >0: WAKE / SLEEP - every fresh batch is scored (no-grad forward, head + private rows); its top sleep_keep share of
    sleep_merge=1                   #   rows per training step = sleep_merge x batch_size (merged batch)
    sleep_lr=0.0                    #   lr multiplier on merged steps (0: sqrt(rows / batch_size))
    sleep_dream=True                #   dream rows scaled with the merged batch (same real : dream ratio as a normal step)
    sleep_wake=True                 #   rows not queued train their domain's private rows (head-only backward)
    sleep_random=False              #   control: queued rows chosen at random (no scoring forward)
    sleep_score='loss'              #   'loss' = rows ranked by loss; 'mem' = by MEMORY GAIN: loss on the window's second half read WITHOUT the first half minus
    sleep_revisit=False             #   True: wake / sleep only on REVISITS (a domain's first visit trains normally)
    ngo_lr_mult=1.0                 #   x ngram_lr_mult
    ngram_in_mult=1.0               # learning-rate multiplier of the INPUT-side n-gram tables only (lazy optimiser), e.g. 0.1 = slow input features
    gn_diag=0                       # >0: GRADIENT-NOISE DIAGNOSTIC every gn_diag steps: per-row (window) gradients of the real-text loss on the current batch
    varpro_k=0                      # >0: VARIABLE PROJECTION (head first): before the loss, the current head is refit on this batch's (detached) features;
    varpro_mode='adam'              #   the trunk gradient is then taken at the refit head (envelope theorem). 'adam': varpro_k head-only Adam steps (lr x varpro_lr);
    varpro_lr=1.0                   #   'gn': varpro_k Gauss-Newton steps dW = varpro_lr (C + lam I)^-1 H'(Y - P)/n, C = running mean of H'H/n (online natural
    varpro_lam=1e-2                 #   gradient for the head, prior from all earlier batches), lam relative to mean diag C
    varpro_beta=0.95                #   running-covariance EMA factor of the 'gn' refit (per domain, kept across visits)
    profile_visit=0                 # >0 (GPU): torch.profiler over steps 100-110 of this visit (1 = the first domain); prints the top operations by GPU
    whiten_bf16=False               # True (GPU): the Fisher whitening products around Muon's Newton-Schulz in bf16 (tensor cores) - ~0.6% change of the step
    muon_batch=False                # True: Muon's Newton-Schulz for all same-shape trunk matrices in ONE batched call per shape (same arithmetic, ~29 small
    varpro_batch=False              #   True: the refits solve (C + lam I) X = H'(P - Y) / n directly (LU), all old heads of a step in ONE batched call
    varpro_inv_every=1              #   the refits' inverse (C + lam I)^-1 recomputed every this many uses per head (cached in between: GPU time)
    varpro_old=False                # VARPRO FOR EVERY HEAD: each OLD head also gets one 'gn' refit per step on its own dream rows towards their stored
    head_center=False               # SOFTMAX SHIFT SYMMETRY: after every step each head's rows (and bias) are centred (adding a vector to every row
    muon=False                      # MUON for the trunk: every 2-D weight in the blocks gets orthogonalised Nesterov momentum (5 Newton-Schulz steps),
    muon_lr=1.0                     #   update RMS matched to Adam (0.2 sqrt(max(m,n)); Moonshot 2025) at lr x muon_lr; Adam for everything else.
    muon_precise=False              # v9.8 cg11: PRECISE SHARED UPDATE (local PX_PS): the shared block matrices step along the Fisher-whitened momentum
                                    #   itself, rescaled to Muon's size (no Newton-Schulz orthogonalisation); the per-domain private units (pu_*) and
                                    #   FiLM matrices keep Muon. Local d128: with private units 2.006 -> 1.960 law at the same near-zero forgetting
    muon_ns_steps=5                 # cg12a: Newton-Schulz iterations per Muon step (5 = standard; 3 = cheaper, weak directions amplified ~41x instead of ~490x)
    muon_precise_cap=0.0            # >0 (cg11b): cap the precise step's largest singular value at this multiple of Muon's (1.0 = Muon's spectral size)
    muon_precise_pit=2              #   power-iteration steps per update (warm-started)
    muon_mom=0.95                   #   Zero rows / columns of a gradient stay zero (private / reserve units keep their isolation).
    dec_prot=0.0                    # >0: DECOUPLED PROTECTION (AdamW-style): the dream-KL gradient of the trunk's 2-D weights is taken OUT of the Muon
    dec_mode='move'                 #   normalisation and applied as a plain step lr x muon_lr x dec_prot x 0.2 g / rms_long(g) (per-matrix scalar RMS over ~100
    pmuon=False                     # PROTECTED MUON (use with muon): real-text KRONECKER FISHER memory. Every kf_every steps one extra forward/backward on the
    kf_every=10
    kf_lam=1.0
    kf_kalman=0.0                   # >0: KALMAN-CONSISTENT strength: the whitening's lam becomes kf_lam x kf_kalman / (visits the OTHER domains have contributed),
    kf_kalman_max=0                 #   >0: the visit count is capped here (discounted Kalman: precision saturates instead of freezing the trunk on long runs)
    probe_visits=()                 # DREAM-SOURCE PROBE (diagnostic, training unchanged): at these visits, before and after training, each old domain's real-text
    sleep_eps=0.0                   # >0: CLOSED-LOOP DREAMING (sleep pressure): each old domain d has a pressure p_d (init 1, kept across visits) multiplying
    sleep_eta=0.02                  #   its dream-KL weight; after every dream step p_d <- clip(p_d exp(sleep_eta (KL_d / sleep_eps - 1)), sleep_pmin, sleep_pmax)
    sleep_pmin=0.25                 #   with KL_d the domain's own dream KL this step (dual ascent on the budget KL_d <= sleep_eps nats/byte): protection binds
    sleep_pmax=8.0                  #   only where a domain actually drifts; fragile domains get more, stable ones less
    sleep_rel=0.0                   #   >0: RELATIVE budgets instead: domain d's budget = sleep_rel x its own reference KL (mean of its first 20 dream measurements
    bank_keep=0                     # with bank_roll: the first bank_keep rows of every bank are never re-dreamed (a small persistent ANCHOR with the domain's
    dctl=''                         # DRIFT-BUDGET DREAM CONTROLLER (bank-free closed loop; replaces sleep pressure): dream RATE r (weight fixed at 1, the
    dctl_rel=1.5
    dctl_rmin=0.125                 #   lowest dream rate (1 in 8 steps)
    dctl_calwin=0.0                 #   >0: PER-VISIT CALIBRATION - dream every step for the first dctl_calwin of each visit; each domain's allowance is then
    sleep_ref_frac=0.0              #   >0: the relative budget's reference = the first sleep_ref_frac x steps_per_task dream measurements (default: 20).
    sleep_kmax=1                    #   >1: ADAPTIVE DREAM RATE - dream only every sleep_kmax steps (weight x k) while every domain's smoothed KL is under its
    pu_k_base=0                     # >0 (with pu_k > pu_k_base): SURPRISE-TRIGGERED GROWTH (neurogenesis): every domain starts with pu_k_base private units;
    pu_grow_rel=1.25                #   a NEW domain whose best old head scores more than pu_grow_rel x the median of earlier new domains' scores on its first
    kf_sketch_tok=4096              #   rows of each collected batch used for the sketch (strided subsample; 200 collections per visit)
    bank_roll=0.0                   # >0: ROLLING DREAM REFRESH: at every visit end, bank_roll of every OTHER old domain's bank rows (oldest first, cyclic) are
    ema_pull_rel=0.0                #   >0: the pull scaled to the averaging window instead: per step 1 / (ema_pull_rel x horizon), horizon = ema_tail x visit
    kf_maxdim=4096                  #   factors larger than this are kept diagonal (memory)
    kf_source='sample'              #   'sample': extra forward/backward with model-sampled labels (true Fisher); 'main': the factors come FREE from the
    pass_lr_noold=0.0               # cg13: >0 = lr multiplier for visits with NO old domain (the run's first visit); replaces the direct step's
    cd_reuse_k=0                    # >0: SPACED REPLAY REUSE (same domain), see cd_reuse_mode
    cd_reuse_lag=15
    cd_reuse_mode='uniform'         #   'uniform' | 'mix' (half by fragility, half uniform) | 'upw' (fragile pieces x cd_reuse_upw) | 'loop' (maintenance loop)
    cd_reuse_upw=2.0
    cd_loop_easy=0.005              # 'loop': degradation (share of positions where the teacher's top byte now differs) <= easy: interval x2; > fail: second chance,
    cd_loop_fail=0.05               #   then dropped; else interval halved; dropped when consolidated (KL < kldrop) or after maxuse re-checks
    cd_loop_maxuse=4
    cd_loop_kldrop=0.02
    cd_win_keep=0.0                 # 0<k<1: WINDOW-LEVEL SURPRISE GATING of the batch rows (fewer generations)
    cd_win_rand=False               #   control: chosen at random
    cd_gate_mode='surprise'         # with cd_gate_keep: 'surprise' (most surprising pieces) | 'kpp' (DIVERSITY: k-means++ / BADGE-style sampling on each
                                    #   piece's teaching-gradient embedding sum(p_student - p_target), origin as first centre so tiny-signal pieces are rarely
                                    #   chosen) | 'rand' (random control)
    cd_prot_rows=8                  # 'prot' gate: dream-bank rows per old domain for the old domains' head-input covariance A_old (per visit)
    cd_prot_lam=0.1                 # 'prot' gate: damping lam = cd_prot_lam x mean diag(A_old)
    dream_sim=''                    # SIMILARITY-WEIGHTED DREAMS (SWIL, McClelland et al. 2020): within each old domain, the dream row is picked among
                                    #   dream_sim_k candidates by a RANK-1 (LoRA-like) gradient signature at the head input, (mean z, mean dz) per row, against
                                    #   the current learning's signature: 'harm' = predicted loss increase under the current step; 'abs' = |similarity|
    dream_sim_k=4
    dream_sim_floor=0.25            #   share of dream rows still drawn at random (frequency)
    cd_store_cap=0                  # >0: BOUNDED LONG-TERM STORE (hippocampus): windows admitted by REDUCIBLE SIGNAL (+ random share), kept ACROSS visits,
                                    #   re-checked on a spaced schedule (any domain, its own head), retired once consolidated, capped (evict the most consolidated)
    cd_store_adm=0.25               #   share of each call's rows admitted by the highest positive reducible signal
    cd_store_rand=0.0625            #   + each row admitted at random with this probability (~20% of admissions: frequency)
    cd_store_k=8                    #   stored windows re-checked (and re-taught) per conduit step
    cd_store_retire=0.25            #   retire when the window's remaining signal (positive part, per piece) has fallen to this share of its value at admission
    cd_store_admit='rho'            #   'rho' (reducible signal) | 'rand' (control: the same number admitted at random)
    inf_on=False                    # cg20 OFFLINE INFERENCE IN SLEEP: probes planted in the text, a sentence store (real text in real passes, VERIFIED
    inf_probes=150                  #   bytes only in closed-loop passes), bridge-centred chaining after every visit, conclusions consolidated (see inf_sleep)
    inf_hops3=0.33                  #   share of three-hop probe chains
    inf_plant_k=6                   #   copies of each probe fact sentence per pass in its domain's text
    inf_probe_file=''               #   optional JSONL from the API probe script (natural-language chains); default: templated chains, invented names
    inf_store_cap=200000            #   sentences kept (only sentences with >= 2 candidate names)
    inf_df=40                       #   a name is 'rare' (bridge / partner) if seen in at most this many windows
    inf_maxfan=6                    #   ABSTAIN: a bridge with more partners than this is a hub and gives no conclusions
    inf_budget=600                  #   conclusions per sleep
    inf_steps=60                    #   consolidation steps per sleep (conclusions planted alone into kept verified windows)
    inf_rows=16                     #   rows per step
    inf_rowlen=256                  #   bytes per consolidation row (conclusions need little context; 4x cheaper than full windows)
    inf_lr=0.1                      #   lr multiplier
    inf_target=30                   #   a conclusion is rehearsed until it has been in this many consolidation rows
    inf_mix=0.5                     #   share of conclusion rows (the rest: anchor windows, interleaving)
    inf_anchor_w=1.0                #   weight of the anchors' KL to the frozen pre-sleep model
    inf_eval_every=1                #   probe recall from the weights every this many visits (0: only at the end)
    inf_cons='w'                    #   cg20h: 'w' consolidate into the weights (Adam, KL anchors) | 'mem' into a FAST STORE: sparse error-memory corrector on the pre-head features
    inf_mem_dx=4096                 #   fast store: expansion units
    inf_mem_act=0.02                #   fast store: share of units active per position
    inf_mem_beta=0.5                #   fast store: delta-rule write strength
    inf_mem_gain=1.0                #   fast store: logit gain of the correction
    inf_mem_nofill=0.5              #   fast store: share of conclusion rows written with no context before them
    pu_cons_steps=0                 # cg19 (1) PRIVATE-INTO-SHARED CONSOLIDATION after every closed-loop visit: this many offline steps in which a
                                    #   FROZEN copy of the model (the visit's private units ON) teaches the SHARED block weights to reproduce it with
                                    #   that domain's private units OFF, on the visit's VERIFIED windows (the NREM reservoir; needs nrem_res > 0)
    pu_cons_lr=0.3                  #   lr multiplier (plain Adam over the blocks' shared weights, fresh per phase)
    pu_cons_rows=16                 #   rows per step
    pu_cons_anchor=0.25             #   share of rows from OLD domains' kept verified windows, taught to keep the frozen copy's output (interleaving)
    pu_cons_shrink=0.0              #   >0: after the phase the domain's private-unit outputs x (1 - shrink) (frees capacity; 0 = keep)
    pu_cons_retire=0.25             # cg19 (3) TRACE-LEVEL RETIREMENT: a window is retired (not kept) once its KL with the private units off falls below
                                    #   this share of its value at the phase start (the shared weights reproduce it: absorbed); <0 = never
    pu_cons_cap=0                   # cg19 (3) CAPACITY CAP: >0 = each domain keeps at most this many verified windows (weakest-first eviction:
                                    #   the most absorbed, lowest remaining KL, go first)
    pu_cons_eval=4                  #   held-out batches per domain for the phase's before/after diagnostics
    trace_lesion=()                 # cg19 (2) TRACE LESION: true domains whose visits are SKIPPED from pass trace_lesion_pass (0-based) on: no traces,
    trace_lesion_pass=2             #   no replay (their old dreams stay) - does the knowledge hold without the hippocampus?
    sleep_phase=0.0                 # >0: OFFLINE SLEEP after every no-real-text visit: sleep_phase x steps_per_task extra steps with NO new experience
    sleep_nrem=0.5                  #   share of them NREM (hippocampal: re-replay the long-term store's tagged windows, highest signal first; retire)
                                    #   the rest REM (cortical: dreams only, every step); needs cd_store_cap > 0 for NREM
    sleep_ctrl=False                #   control: the same extra steps, but awake (the visit simply continues)
    lm_rank=-1                      # >=0: LOW-RANK DREAM MERGE (REM-like weight-space consolidation) after every no-real-text visit: for each old domain
                                    #   fit the shared-trunk update that restores its dreams' stored targets (lm_steps Adam steps), truncate it to rank lm_rank
                                    #   (0 = full rank), merge the domains' updates (lm_merge), and apply the largest scale that passes the NO-HARM CHECK
    lm_steps=20
    lm_lr=1e-3
    lm_rows=16                      #   dream rows per fitting step
    lm_val=32                       #   held-out bank rows per domain for the check (never fitted)
    lm_merge='ties'                 #   'ties' (trim 80% smallest, elect sign, mean of agreeing) | 'mean' | 'sum'
    lm_scales=(1.0,0.5,0.25)
    lm_eps=0.0                      #   no old domain's held-out dream KL may rise above (1 + lm_eps) x its value before
    lm_store_check=False            #   the no-harm check also on the LONG-TERM STORE's real windows of the old domains (needs cd_store_cap)
    lm_cur_eps=0.01                 #   and the visited domain's last verified replays may not move by more than this KL (nats/byte)
    sem_share=0.0                   # >0: SEMANTIC ROWS (shared memory): every episode admitted to the long-term store also leaves a TRACE-FREE copy in its
                                    #   domain's semantic bank (real verified bytes + the trace-reading teacher's evidence-blended targets, top-k); in dream
                                    #   batches each old-domain row is a semantic row with this probability (real-text protection, ~1000x smaller than a trace)
    sem_tgt='teacher'               #   'student': at the visit's end its semantic rows are RE-SCORED by the current model (fresh targets, as dream banks)
    sem_cap=1000                    #   semantic rows kept per domain (oldest replaced first)
    sem_minver=0.9                  #   only windows whose verified share is at least this become semantic rows
    nrem_res=0                      # >0: NREM v1 - a RESERVOIR of the visit's verified windows (bytes + the frozen teacher's evidence-blended targets, top-k,
                                    #   + log q(y)); in the sleep phase (sleep_phase, sleep_nrem) they are re-taught offline, no traces needed (frozen teacher)
    nrem_tgt='teacher'              #   'teacher': KL to the stored teacher targets | 'bytes': cross-entropy on the stored verified bytes only (no teacher information in sleep)
    nrem_mix=0.0                    #   CROSS-DOMAIN NREM: this share of every NREM batch is drawn from OTHER domains' kept samples (each domain keeps the sample of
                                    #   its latest visit; replaced when the domain is visited again); each window is taught under its own domain
    nrem_mix_sel='signal'           #   'signal': other-domain windows with the highest remaining signal | 'rand': uniform
    nrem_own_sel='signal'           #   current-visit windows: 'signal' (highest remaining) | 'rand' (uniform; use with nrem_retire<0)
    nrem_mix_add=False              #   True: other-domain windows are ADDED (nrem_mix x nrem_k extra) instead of replacing current-visit ones; the current
                                    #   visit's teaching weight is unchanged (normalised by its own windows)
    nrem_real=False                 #   DIAGNOSTIC (breaks the closed loop): the sleep sample holds random REAL pass-1 windows of the domain (same length and
                                    #   teaching weights) instead of recalled ones - plain replay vs recall
    nrem_real_cap=512               #   pass-1 rows kept per domain for nrem_real
    sleep_ctrl_bytes=False          #   with sleep_ctrl: the extra AWAKE steps teach the fresh checked replays by next-byte cross-entropy only (no teacher
                                    #   targets, no upkeep loss) - 'sleep rebuilt from traces'
    cd_bytes_all=False              # pass-2 replay steps teach the checked replays by next-byte cross-entropy on the verified bytes INSTEAD of the KL to
                                    #   the frozen teacher's (evidence-blended) targets; the teacher only regenerates and checks (the upkeep loss is kept)
    cd_bytes_w=1.0                  #   with cd_bytes_all: the share of the byte CE; the rest stays the KL to the teacher's targets (0.9 = mostly bytes)
    nrem_k=8                        #   windows per NREM step, highest remaining reducible signal first
    nrem_retire=0.25                #   retire a window when its remaining signal falls to this share of its value at storage
    cd_gate_keep=0.0                # >0: SURPRISE-GATED REPLAY (test): only the most surprising share of each call's replayed pieces teaches (surprise = the plain
                                    #   model's loss on the piece); the rest get weight 0. Diagnostic: surprise predicts the replay signal (r 0.80-0.99 locally)
    cd_evid_adapt=0.0               # >0: ADAPTIVE EVIDENCE WEIGHT (acetylcholine-like): at a disagreeing position the blend is cd_evid_w + (cd_evid_adapt - cd_evid_w)
                                    #   x the share of disagreeing positions in its piece (novel pieces trust the evidence more)
    prot_decay=1.0                  # <1: PROTECTION DECAYS WITH DISUSE: an old domain's dream rows are weighted prot_decay ** (visits since its last visit)
    cd_sig_every=0                  # >0: REPLAY-SIGNAL DIAGNOSTIC every this many conduit calls (no training effect): distribution of the learning signal per
                                    #   replayed piece, how concentrated it is, and whether the surprise at experience predicts it (gating before replaying)
    cd_evid_w=0.0                   # EVIDENCE TARGETS (pass-2 closed loop): at positions where the teacher's top byte differs from the VERIFIED replayed byte,
                                    #   the target becomes (1 - w) x teacher + w x that byte (the model's own verified reconstruction, no real text). Fact test:
                                    #   with teacher-only targets a corrected fact never overwrote an entrenched one (self-distillation confirmation bias)
    cd_evid_all=False               #   True: blend at every teaching position, not only where the teacher disagrees
    fact_spec=()                    # UPDATE-OR-PROTECT TEST: synthetic facts ' Name:1234;' injected into training text, entries [domain, pass, version, every_bytes]
                                    #   (version 0 = the original values, 1 = the changed values); evaluated through every fact domain's slot at each visit end
    fact_n=8                        #   number of facts (names and 4-digit values drawn from fact_seed)
    fact_seed=5
    dream_lesion=()                 # SCHEMA / LESION TEST: true domains whose dream bank is switched off after their visit in pass >= dream_lesion_pass
    dream_lesion_pass=1             #   (the hippocampal-lesion analogue of Tse et al. 2007: does the domain's knowledge survive without its replay?)
    dream_prio=0.0                  # cg16: PRIORITISED DREAM ROWS within each old domain's bank: this share of rows is drawn in proportion to the row's KL
                                    #   the last time it was used (rows never used: the domain's mean; no extra forward), the rest uniformly. Local dream_test:
                                    #   forgetting -8% (4 rows/step), -2% (every 2nd step), -3% (16 rows), no cost to learning
    dream_prio_sync=8               #   priorities copied to the host every this many dream steps (one sync)
    bank_anchor=0.0                 # cg16: ANCHORED BANK: this share of each domain's bank is its FIRST bank's rows, kept for the whole run; at every later
                                    #   bank write only the rest is re-sampled, and ALL rows get fresh targets from the current weights (the GPU writes a
                                    #   domain's bank right after its own visit, so stale pass-1 targets would pull towards the weaker pass-1 model). Local
                                    #   prot2 test (anchored inputs AND targets): drift +0.031 vs +0.119 regenerated, +0.010 fully fixed
    dream_clip_pass=()              # cg14b: per-pass gradient-clip threshold on DREAM steps (other steps stay at 1.0); cg14: dream weight x2 did not reach the
                                    #   update because every dream step is clipped to norm 1 (only the real/dream mix changed), so raise the clip instead
    dream_w_pass=()                 # cg14: per-pass multiplier on dream_weight (e.g. (1.0,2.0) doubles pass-2 dream weight; local P2 harness: x2 cut 9-old-domain BPC 0.018)
                                    #   behaviour there (it acted only once old domains existed and settled at 0.25-0.28 from the second visit)
    pass_lr=()                      # LR MULTIPLIER PER PASS (lr_sched 'warmup_once'), e.g. (1.5,1.0): first exposure faster, revisits at the base lr
    trunk_lr_mult=1.0               # Adam CONTROL for muon: the Adam update of the same 2-D block weights scaled by trunk_lr_mult (everything else unchanged)
    sim_diag=False                  # CROSS-DOMAIN SIMILARITY DIAGNOSTIC: every seen domain is evaluated after EVERY visit ('[allbpc]' lines: the measured
    sim_rows=64
    borrow_aff=0.0                  # >0: AFFINITY-WEIGHTED BORROWING - at each visit start the old heads score the first real batch; old domain d's borrowing
    film_k=0                        # >0: SIMILARITY-SHAPED FiLM CONTROLLER (needs pu_k > 0 for the row tags): each domain has a code e_d (film_k numbers);
    film_aff=True                   #   a NEW domain's code starts as the affinity-weighted mix of the old codes (softmax(-bits / film_tau) of the old heads on
    film_tau=0.25                   #   its first batch) - related domains start with shared modulation; False: random code (control)
    dp_steps=1                      # 2: DELTAPRODUCT (Siems et al. 2025) - two delta-rule writes per token (keys/values/strengths from extra projections
    pu_dream_grad=False             # True = the old behaviour (dream rows train their domain's private trunk units through every dream-row loss, BORROWING
    pu_k=0                          # >0: PRIVATE TRUNK UNITS - every block gets pu_k extra MLP units per domain, used only by that domain's rows (dreams by
    inh_rank=0                      # >0: LEARNED INHIBITORY GATES on every block's MLP hidden units: inhibition i = sigmoid(c + (LN x) A B) (rank inh_rank,
    cgate=0.0                       # >0: CONFLICT-DRIVEN, CONTEXT-KEYED GATES on every block's MLP hidden units. Each domain owns a gate vector (keyed by the
    ngpi=()                         # PRIVATE INPUT ROWS: per-domain hashed n-gram tables (these orders) added to the trunk's INPUT (next to the shared
    fz_diag=False                   # WHERE IS THE FORGETTING (diagnostic, no effect on training): (1) every old domain's BPC after every later visit
    xdg=0.0                         # >0: CONTEXT-DEPENDENT GATING (Masse et al. 2018) - each domain uses a fixed random subset (keep share xdg) of every
    side_w=0                        # >0: LADDER SIDE NETWORK of width side_w, switched on when the trunk freezes (freeze_bottom/freeze_after): it reads
    side_dream=True                 #   False: dream rows' side contribution is detached (no dream protection of the side network)
    fcp_tables=0                    # >0: FEATURE-CODED PRIVATE TABLES - fcp_tables per-domain tables of 2^fcp_bits rows, indexed by the signs of fixed random
    adam_fused=False                # SPEED: fused CUDA Adam (one kernel for all parameters; GPU only)
    bank_targets=0                  # >0: STORED TARGETS - at the bank write, the visit-end weights' top-k log-probabilities per byte are stored
    bank_tgt_cpu=False              #   keep the stored targets in CPU memory (rows moved to the device per step)
    bank_tgt_chunk_big=0            # >0: rows per target forward when the CHUNKED delta solve is active (no T x T matrices held), e.g. 128 - fewer, larger passes
    bank_tgt_chunk=32               #   rows per forward when the targets are computed (the torch solve holds B x heads x T^2 matrices)
    bank_tgt_tail='bin'             #   'bin': KL over the top-k bytes + ONE bin for the rest (the student's tail shape is free); 'uniform'
    dream_bank=0                    # >0: REPLAYED DREAM BANK instead of the sequential streams: at the end of each domain's visit, this many dream
    dream_parallel=0                # >0: PARALLEL (JACOBI) DREAMS instead of the sequential streams: each step, dream_batch rows of the CURRENT
    dream_early_mult=1.0            # EARLY-VISIT DREAM BOOST: dream weight x this at the visit start, falling linearly to x1 at dream_early_steps
    dream_early_steps=200           #   (drift peaks early in each visit: the newest old domain's dream KL 0.061 at step 200 -> 0.009 at step 800)
    kl_pos_diag=False               # DIAGNOSTIC (evaluation only): teacher -> live KL by position on REAL held-out old-domain text vs on dreams
    dream_content=0                 # >0: CONTENT-ONLY DREAMS in the top dream_content blocks: dream rows' inputs to the output ("content") projections
    dream_route=None                # PER-LAYER DREAMS (streams): multipliers on the dream-KL gradient reaching each part, (emb, b0, b1, ...), e.g.
    strand_kind=None                # CROSS-DOMAIN STRANDS (streams): 'switch' = teacher A writes, teacher B continues from a random point;
    strand_share=0.25               #   share of each dream batch (the single-domain streams supply the rest)
    strand_route=None               #   gradient multipliers for strand rows (default: dream_route), e.g. (0,0,0,1,1) = blocks 2-3 only
    strand_gap=32                   #   switch strands: no KL for this many positions after the switch (teacher B settling on A's context)
    strand_floor=0.1                #   pairs drawn by data-free HEAD CONFUSION (pairs the old heads confuse = where domains share the network),
    strand_pair_mode='confusion'    #   'confusion' (pairs the heads confuse), 'anti' (pairs they separate best) or 'uniform'
    strand_blend_w=0.5              #   blend strands: mixture weight of teacher A
    strand_anchor=0.0               #   weight per block of OLD-SUBSPACE feature anchoring on strand rows: relative error of the block output's
    strand_anchor_blocks=(2,3)      #   component inside the top directions (strand_anchor_energy of the energy) of the visit-start network's
    share_blocks=None               # DREAM ONLY THE SIMILARITIES in these blocks, e.g. (2,3): the dream gradient reaching each block's own
    share_spec=0.0                  #   multiplier for the remaining (domain-specific) component (0 = similarities only)
    share_keep=1.0                  #   multiplier for the shared component (0 with share_spec=1 = 'differences only')
    share_thr=0.6                   #   shared = eigenvalue of the mean of the domains' projectors >= this (~ used by more than half the domains)
    dream_steer=0.0                 # DRIFT-STEERED DREAMS (streams): per-domain weight on the dream KL = (EMA of that domain's KL on its FRESH
    mix_window=0                    # SPEED A1 - WINDOWED WY SOLVE: >0 = each output reads only the previous >= mix_window positions (overlapping
    samp_lowp_mm=False              # SPEED (bank write): windowed sampler's window products in bf16 straight from the bf16 window (was: the whole
    samp_window=0                   # SPEED A2 - WINDOWED SAMPLER: >0 = the recurrent sampler keeps the last samp_window rank-1 writes per head
    samp_bf16=False                 #   (u_j, k_j and the decay up to now) instead of the full state matrix; bf16 storage of u, k
    samp_rescore=True               #   with the windowed sampler, the dreams' KL TARGETS are recomputed by the EXACT teacher in one parallel pass
    gate_every=1                    # SPEED A4 - write gates recomputed every gate_every steps (cached in between; they barely move in a visit)
    shunt_lazy=False                # SPEED A5 - shunt = live weights - visit-start copy, formed only when it is needed (merges), instead of
    merge_dream_fwd=False           # SPEED A6 - real batch and dream batch in ONE forward pass (statistics from the real rows only; identical)
    cov_skip=False                  # SPEED - no running input covariances when nothing uses them (orthogonal gate off)
    lwf_metric=False                # B1 - FISHER-METRIC current-batch KL: sum over old heads of KL(teacher||live) ~ 0.5 df^T G df with G = sum_d
    lwf_every=1                     # B2 - current-batch KL (and its teacher pass) only every lwf_every steps, weight x lwf_every
    ngram_emb=()                    # C1 - HASHED BYTE N-GRAM EMBEDDINGS, e.g. (2,3,4): one table per n (ngram_size rows, zero-initialised, sparse
    ngram_size=8192                 #   updates); teachers hold a frozen copy per visit (drift is protected by the KL / dreams); an int or one per order
    ngram_hashes=1                  #   hashes per order (rows summed, as a Bloom / count-min embedding): fewer harmful collisions
    ngram_sparse=False              #   LAZY (sparse) Adam for the n-gram tables: only rows used in the step move (no momentum drift of unused rows,
    ngram_protect=0.0               #   FREQUENCY-BASED ROW PROTECTION: a row's step x 1/(1 + ngram_protect x u/mean u), u = its accumulated usage
    ngram_opt='adam'                #   table optimiser with ngram_sparse: 'adam' (lazy Adam) or 'count' (OPTIMISER-FREE COUNTING RULE: each used
    ngram_count_n0=50.0             #   uses before the counting rule's rate starts to fall
    ngram_count_p=0.5               #   decay exponent (1 = running mean, 0.5 = AdaGrad-like)
    ngram_count=()                  # NON-BACKPROP COUNT PREDICTOR, e.g. (1,2,3,4): next-byte counts after the last L bytes (count-min sketches,
    ngram_count_rows=65536          #   rows per sketch for L >= 2
    ngram_count_hashes=2            #   sketches per order for L >= 3 (count-min: the minimum over them)
    ngram_count_D=0.75              #   absolute discount
    ngram_lr_mult=1.0               #   learning-rate multiplier for the LAZY optimiser (lazy Adam's per-row bias correction gives a rarely used row
    pos_emb=True                    # False: no learned position embedding (zeroed and frozen; the n-gram tables and the recurrence carry position)
    lattn_blocks=()                 # C2 - SLIDING-WINDOW ATTENTION mixer in these blocks: softmax over the previous lattn_window positions of
    lattn_window=32                 #   scale x cos(k_t, k_j) + a learned relative-position bias per head, plus a null slot (can read nothing)
    parallel_block=False            # C3 - PARALLEL BLOCKS: the MLP reads the block input (its own LayerNorm) instead of the mixer's output
    state_spec_diag=()              # DIAGNOSTIC (evaluation only), at the end of these visits (dom ids): spectrum of each block's recurrent
    state_spec_ranks=(2,4,8,16,32)
    state_spec_rows=16
    write_gate_open=False           # ABLATION: every write gate (unit, channel, embedding thresholds) forced open - updates pass in full, but the
    cov_diag=False                  # DIAGNOSTIC: real-text drift (teacher -> live KL) by byte type vs how often each type appears in real text
    feat_drift_diag=False           # DIAGNOSTIC (evaluation only): per-block feature drift on real old-domain text (each visit; and since each
    mix_local_blocks=()             # 6. LOCAL FILTER BLOCKS: in these blocks the delta-rule memory (whose beta collapses to ~0.01 in block 0) is replaced
    mix_local_width=8               #    by a causal depthwise filter over the previous mix_local_width values (no key projection, no solve)
    block_extra=None                # CAPACITY: per-block extra width as a fraction of the base state/MLP units, e.g. (0,0,0,0.5)
    block_extra_mode='start'        # 'start' = active from initialisation (a wider block); 'fresh' = held in reserve and an equal
    uniq_rate=0.002                 # adaptation rate of each block's 'seen so far' input statistics (~350-step half-life)
    uniq_base_rate=0.005            # adaptation rate of the block's typical novelty level
    uniq_thr=1.25                   # activate when novelty / typical novelty exceeds this
    uniq_shrink=0.1                 # covariance shrinkage for the whitened distance
    uniq_min_stats=50               # steps of statistics before the detector starts
    uniq_warmup=200                 # no activation before this many steps of detector history
    recruit_every=25                # steps between activation decisions
    recruit_rate=0.5                # units per event = recruit_rate * base count * min(ratio - 1, 2)
    orth_energies=(0.5,0.8,0.9,0.99)  # old-subspace energy thresholds (trade-off curve) for the layer-level counterfactual
    orth_unit_energy=0.9            # per-unit subspace energy (protected MLP units)
    use_write_threshold=True
    theta_low=0.5; theta_high=1.5   # random base threshold range (drive is normalized to mean 1)
    theta_temp=0.2                  # softness of the threshold gate
    imp_alpha=1.0                   # threshold rise per unit of accumulated importance
    theta_floor=0.05                # min write so sub-threshold units can still wake up
    min_eligible=0.15               # grow if fraction of eligible units falls below this
    min_fresh=128           # every domain gets this many fresh units (later domains don't overwrite earlier)
    seq_len=256; batch_size=16
    lr=2e-3; steps_per_task=800
    meta_lambda=0.5      # weight on plasticity vs forgetting in the FD meta-objective
    meta_every=10        # measure the gate's forgetting effect every N steps
    meta_lr=1e-3         # MAML meta-controller optimizer LR
    buffer_batches=4                 # prior-task examples kept for the forgetting signal
_CONFIG_DEFAULTS={k:v for k,v in vars(Config).items() if not k.startswith('__')}   # driver resets class attrs per run
PILE_DOMAINS=["Gutenberg (PG-19)","Wikipedia (en)","ArXiv","PubMed Abstracts","FreeLaw"]
BASE_DOMAINS=list(PILE_DOMAINS)
# Measured intrinsic difficulty (converged-alone BPC, from measure_difficulty.py diagnostic).
# Used to PROACTIVELY widen the shared manifold for hard domains (ArXiv hardest) before
# training them - giving hard content room from the start, rather than a reactive trigger.
DOMAIN_DIFFICULTY={"Gutenberg (PG-19)":2.398,"Wikipedia (en)":2.629,"ArXiv":2.661,
                   "PubMed Abstracts":2.137,"FreeLaw":2.149}

def sample_theta(n, device=None):
    """Random base impedance per neuron. Every distribution uses the SAME n uniform draws, so the
    global RNG stream (and hence data order) is identical across distributions."""
    lo,hi,dist=Config.theta_low,Config.theta_high,Config.theta_dist
    u=torch.rand(n,device=device)
    if dist=='uniform': return u*(hi-lo)+lo
    if dist=='constant': return torch.full_like(u,(lo+hi)/2)
    if dist=='lognormal':            # median (lo+hi)/2, heavy right tail: a few very hard-to-write units
        z=torch.erfinv(2*u.clamp(1e-6,1-1e-6)-1)*math.sqrt(2)
        return (lo+hi)/2*torch.exp(Config.theta_sigma*z)
    if dist=='bimodal':              # most units easy (lo..mid); a reserve fraction hard (hi..hi+1)
        f=Config.theta_reserve
        return torch.where(u<1-f, lo+(u/(1-f))*((lo+hi)/2-lo), hi+(u-(1-f))/f)
    raise ValueError(dist)

_NOISE={'gen':None}
_RAMP={'s':1.0}                     # base-impedance strength (theta0_ramp_domains), set per domain in train_domain
def _theta(theta0, imp, cfg):
    """Effective threshold = base impedance + alpha*importance (+ per-step impedance noise)."""
    th=(theta0 if _RAMP['s']==1.0 else _RAMP['s']*theta0)+imp   # imp arrives alpha-weighted from _imp (per-group strength)
    return th

_PG_KEY={'Wk':'Wk','Wv':'Wk','Wq':'Wk','Wb':'Wk','Wa':'Wk','gate':'gate','mlp_in':'mlp_in','Wout':'Wout','mlp_out':'mlp_out'}

def _f32(fn):
    """Statistics hooks run with autocast OFF: under bf16 mixed precision their matmuls (covariances etc.) would otherwise be
    computed and stored in bf16."""
    def w(m, inp, out):
        with torch.autocast(device_type=inp[0].device.type, enabled=False): return fn(m, inp, out)
    return w
def _alignment(x, U):
    """Per-token polarisation: share of the input vector's energy along the old-domain subspace U
    (cos^2 of its angle to that subspace). x (...,d), U (d,k) -> (...)"""
    xu=x@U; return (xu*xu).sum(-1)/(x*x).sum(-1).clamp(min=1e-12)

_DSPLIT={'on':False,'keys':('Wk','Wv','gate','Wout','mlp_in','mlp_out')}
class _SplitLinear(torch.autograd.Function):
    """Linear layer whose WEIGHT gradient is split by rows: real rows [:ns] and dream rows [ns:] give two products (same FLOPs as one);
    the total is returned as usual and the dream part is stashed on the weight (W._gd) for the optimiser (dream momentum).
    Forward under the caller's autocast; backward reproduces autocast's casts (inputs and weight saved in the forward dtype)."""
    @staticmethod
    def forward(ctx, x, W, b, ns):
        y=F.linear(x,W,b); dt=y.dtype
        ctx.save_for_backward(x.to(dt),W.to(dt)); ctx.ns=ns; ctx.hasb=b is not None; ctx.xdt=x.dtype; ctx.Wdt=W.dtype; ctx.W=W
        return y
    @staticmethod
    def backward(ctx, gy):
        xc,Wc=ctx.saved_tensors; ns=ctx.ns; gy=gy.to(xc.dtype)
        gx=(gy@Wc).to(ctx.xdt) if ctx.needs_input_grad[0] else None
        o_=gy.shape[-1]; i_=xc.shape[-1]
        gr=torch.mm(gy[:ns].reshape(-1,o_).t(),xc[:ns].reshape(-1,i_)).float()
        gd=torch.mm(gy[ns:].reshape(-1,o_).t(),xc[ns:].reshape(-1,i_)).float()
        W=ctx.W; od_=W.__dict__.get('_gd'); W.__dict__['_gd']=gd if od_ is None else od_+gd
        gb=gy.reshape(-1,o_).float().sum(0) if ctx.hasb else None
        return gx,(gr+gd).to(ctx.Wdt),gb,None
def _wl(b, key, x):
    """Linear layer with POLARISATION-GATED weight writes. Forward output is unchanged; only the input
    seen by the WEIGHT gradient is modified, per token, from the angle between the token's input (the
    combined previous-layer activity) and the old domains' input subspace:
      'weight'       : tokens more old-aligned than the batch average write less (w = 2*sigmoid(-kappa*z))
      'proj'         : the old-subspace part of each token's write is removed in proportion to how
                       old-aligned the token is (s = sigmoid(kappa*z))
      'proj_uniform' : removed for every token (s = 1): gradient-space control for 'proj'
    Input gradients are exact: F.linear(xw,W) + F.linear(x-xw, W.detach()) has d/dx = W^T delta."""
    mod=getattr(b,key); pg=getattr(b,'_pg',None)
    if _DSPLIT['on'] and not pg and key in _DSPLIT['keys'] and b.training and torch.is_grad_enabled() and x.dim()==3:
        ns_=getattr(b,'_nstat',None)
        if ns_ is not None and 0<ns_<x.shape[0]:                         # DREAM / REAL weight-gradient split (dream momentum)
            y=_SplitLinear.apply(x,mod.weight,mod.bias,int(ns_))
            for h_ in mod._forward_hooks.values():                       # module hooks (Kalman statistics) as a module call would run them
                r_=h_(mod,(x,),y)
                if r_ is not None: y=r_
            return y
    if not pg or not b.training or not torch.is_grad_enabled(): return mod(x)
    U=pg.get(_PG_KEY[key])
    if U is None or U.shape[0]!=x.shape[-1]: return mod(x)
    with torch.no_grad():
        a=_alignment(x.detach().float(),U); z=(a-a.mean())/(a.std()+1e-6)
    if b._pg_mode=='proj_seqnov':
        # SEQUENCE NOVELTY: whitened (Mahalanobis) distance to the old domains' input distribution, averaged
        # over each sequence; typical sequences (below the batch average) get the old-subspace part removed
        with torch.no_grad():
            mu,V,lam=b._pg_maha[_PG_KEY[key]]
            if mu.shape[0]!=x.shape[-1]: return mod(x)
            nov=((((x.detach().float()-mu)@V)**2)/lam).sum(-1).mean(-1)          # (B,)
            zs=(nov-nov.mean())/(nov.std()+1e-6)
            sc=torch.sigmoid(-b._pg_kappa*zs).view(-1,*([1]*(x.dim()-1)))
        return F.linear(x-sc*((x@U)@U.T),mod.weight,mod.bias)+F.linear(sc*((x@U)@U.T),mod.weight.detach())
    if b._pg_mode=='weight': xw=(2*torch.sigmoid(-b._pg_kappa*z)).unsqueeze(-1)*x
    else:
        sc=1.0 if b._pg_mode=='proj_uniform' else torch.sigmoid(b._pg_kappa*z).unsqueeze(-1)
        xw=x-sc*((x@U)@U.T)
    return F.linear(xw,mod.weight,mod.bias)+F.linear(x-xw,mod.weight.detach())

@torch.no_grad()
def _track(obj, name, drive):
    """Update a neuron group's drive statistics: fast EMA (D), slow EMA, and running peak
    of the fast EMA. name in {'ssm','mlp','ch','emb'} -> buffers <name>_drive_{ema,slow,peak}."""
    a=Config.drive_ema; ema=getattr(obj,name+'_drive_ema')
    ema=torch.where(ema==0, drive, a*ema+(1-a)*drive); setattr(obj,name+'_drive_ema',ema)
    a2=Config.drive_slow; sl=getattr(obj,name+'_drive_slow')
    setattr(obj,name+'_drive_slow', torch.where(sl==0, drive, a2*sl+(1-a2)*drive))
    pk=getattr(obj,name+'_drive_peak')
    setattr(obj,name+'_drive_peak', torch.maximum(pk*Config.peak_decay, ema))

def _ref(obj, name, cfg):
    """SIGNAL 1: the reference a unit's current drive is compared against."""
    ema=getattr(obj,name+'_drive_ema')
    return ema
    raise ValueError(cfg.drive_ref)

_IMP_OF={'ssm':'unit','mlp':'mlp','ch':'ch','emb':'emb'}
def _imp(obj, name, cfg):
    """Effective importance = usage importance (+ SIGNAL 2: sens_alpha * loss sensitivity)."""
    p=_IMP_OF[name]; imp=getattr(obj,p+'_imp')
    a=_grp(cfg,'imp_alpha',name)
    v=a*imp
    fl=getattr(cfg,'imp_floor',0.0)
    if fl>0 and v.numel()>1:                                   # the least-important share gates as if fresh
        k=max(1,int(fl*v.numel())); thr=torch.kthvalue(v.flatten().float(),k).values
        v=torch.where(v<=thr,torch.zeros_like(v),v)
    return v

def _grp(cfg, attr, name):
    """Per-group override: cfg.<attr>_<group> (group in ssm/mlp/ch/emb) if set, else cfg.<attr>."""
    v=getattr(cfg, f'{attr}_{name}', None)
    return getattr(cfg, attr) if v is None else v

# ---------------- REPLAY-DRIVEN STIFFNESS (split weight-gradient multiply) ----------------
_STF={'on':False,'B':None,'ns':None,'dream_call':False}
_lin_fwd0=nn.Linear.forward
def stiff_mods(model): return [m for m in model.blocks.modules() if isinstance(m,nn.Linear) and type(m) is nn.Linear]
@torch.no_grad()
def stiff_post(sv):
    for p,p0,f in sv: p.copy_(p0+(p-p0)*f.to(p.dtype))

# ---------------- BACKPROP REDUCTION: closed-form protection, selective steps, prioritised bank rows, freezing, closed-form heads ----------------
_PJ={'on':False,'ns':None,'B':None,'dream_call':False}
def _pj_hook(mod,inp):
    if not _PJ['on'] or _PJ['ns'] is None: return
    x=inp[0]; B=_PJ['B']
    if x.dim()>=3 and x.shape[0]==B: xd=x[_PJ['ns']:]
    elif x.dim()==2 and B and x.shape[0]%B==0: xd=x[(x.shape[0]//B)*_PJ['ns']:]
    else: return
    xd=xd.detach().reshape(-1,x.shape[-1]).float()
    if xd.shape[0]==0: return
    C=xd.t().matmul(xd)/xd.shape[0]; d_=mod.__dict__; C0=d_.get('_pjC'); e=_PJ['eps']
    d_['_pjC']=C if C0 is None else C0.mul_(1-e).add_(C,alpha=e)
def proj_setup(model,cfg):
    _PJ['on']=bool(cfg.proj_lambda>0); _PJ['eps']=float(cfg.proj_eps)
    if not _PJ['on']: return
    for m in stiff_mods(model):
        if not m.__dict__.get('_pj_h'): m.register_forward_pre_hook(_pj_hook); m.__dict__['_pj_h']=True
    print(f"  [projection] closed-form protection on {len(stiff_mods(model))} trunk Linear layers, lambda {cfg.proj_lambda}"+(f" (+ dream KL every {max(1,cfg.dream_every)} steps)"))
@torch.no_grad()
def proj_post(sv,model,step):
    sh=[]
    for W,W0,P in sv:
        dW=W-W0; dP=dW.matmul(P.to(dW.dtype)); W.copy_(W0+dP)
        if step%90==0: sh.append(float(dP.norm()/dW.norm().clamp(min=1e-12)))
    if sh: print(f"      [projection] update norm kept after projection: mean {np.mean(sh):.3f} (min {np.min(sh):.3f}) over {len(sh)} layers")
def _ns5(G,steps=5):
    """Newton-Schulz orthogonalisation (Muon's quintic iteration); zero rows / columns are preserved."""
    a,b,c=3.4445,-4.7750,2.0315; X=G.to(torch.bfloat16) if G.is_cuda else G.float(); tr_=X.shape[0]>X.shape[1]   # bf16 on GPU (as in Muon)
    if tr_: X=X.t()
    X=X/(X.norm()+1e-7)
    for _ in range(steps): A=X@X.t(); X=a*X+(b*A+c*(A@A))@X
    X=X.float(); return X.t() if tr_ else X
def _ns5_batch(Zs,steps=5):
    """_ns5 for a list of SAME-SHAPE matrices in one batched call (per-matrix normalisation, identical iteration)."""
    a,b,c=3.4445,-4.7750,2.0315; X=torch.stack(Zs); X=X.to(torch.bfloat16) if X.is_cuda else X.float(); tr_=X.shape[1]>X.shape[2]
    if tr_: X=X.transpose(1,2)
    X=X/(X.norm(dim=(1,2),keepdim=True)+1e-7)
    for _ in range(steps): A=X@X.transpose(1,2); X=a*X+(b*A+c*(A@A))@X
    X=X.float(); X=X.transpose(1,2) if tr_ else X
    return list(X.unbind(0))
_MU={'gram':False}
def _ns5_gram_batch(Zs,steps=5,restart=3):
    """GRAM NEWTON-SCHULZ (Dao lab 2026): the same quintic iteration run on the small n x n Gram matrix R = X X^T (n = short side):
    Z = b R + c R^2, Q <- Q Z + a Q, R <- Z R Z (+ a terms in fp32), polar(X) = Q X; ONE RESTART before iteration `restart` (X <- Q X,
    R <- X X^T, Q <- I) keeps rounding's spurious negative eigenvalues bounded. fp16 GEMMs (values stay O(1)), fp32 accumulation and
    a-terms. Rectangular products: 4 instead of 10."""
    a,b,c=3.4445,-4.7750,2.0315; X=torch.stack(Zs).float(); tr_=X.shape[1]>X.shape[2]
    if tr_: X=X.transpose(1,2)
    dt=torch.float16 if X.is_cuda else torch.float32
    mm=lambda p,q: torch.matmul(p.to(dt),q.to(dt)).float()               # small n x n products: fp16 GEMM, fp32 result
    X=(X/(X.norm(dim=(1,2),keepdim=True)+1e-7)).to(dt)                     # the big matrix is cast ONCE and stays in fp16
    R=torch.matmul(X,X.transpose(1,2)).float(); Q=None; I=torch.eye(X.shape[1],device=X.device)
    for t in range(1,steps+1):
        if t==restart and Q is not None: X=torch.matmul(Q.to(dt),X); R=torch.matmul(X,X.transpose(1,2)).float(); Q=None
        Z=b*R+c*mm(R,R)
        Q=(Z+a*I) if Q is None else mm(Q,Z)+a*Q
        RZ=mm(R,Z)+a*R; R=mm(Z,RZ)+a*RZ
    X=torch.matmul(Q.to(dt),X).float(); X=X.transpose(1,2) if tr_ else X
    return list(X.unbind(0))
_KF={'on':False,'acc':None}
def _kf_hook(mod,inp,out):
    if not _KF['on'] or not out.requires_grad: return
    A_=_KF['acc']; k_=id(mod); nr_=_KF.get('nreal'); md_=_KF['maxdim']
    shc_=_KF.get('share')                                                 # SHARED INPUT: one covariance per input tensor (kf_share_a)
    def _cov(rows_,tag_):
        if shc_ is not None:
            e0_=shc_.get((id(inp[0]),tag_))
            if e0_ is not None and e0_[0]() is inp[0]: return e0_[1]
        a0_=rows_.detach().reshape(-1,inp[0].shape[-1]).float(); c0_=(a0_.t()@a0_) if a0_.shape[1]<=md_ else (a0_*a0_).sum(0)
        if shc_ is not None: shc_[(id(inp[0]),tag_)]=(_weakref.ref(inp[0]),c0_)
        return c0_
    if _KF.get('dream') and nr_ and inp[0].shape[0]>nr_:                   # LIVE OLD-DOMAIN KEYS: input factor of the dream rows (current network)
        ad_=inp[0][nr_:].detach().reshape(-1,inp[0].shape[-1]).float()
        sd_=_cov(inp[0][nr_:],'d')
        D_=_KF['dacc'].setdefault(k_,{'A':None,'n':0}); D_['A']=sd_ if D_['A'] is None else D_['A']+sd_; D_['n']+=ad_.shape[0]
    if _KF.get('real_off'): return
    a_=(inp[0][:nr_] if nr_ else inp[0]).detach().reshape(-1,inp[0].shape[-1]).float()
    e_=A_.setdefault(k_,{'A':None,'nA':0,'G':None,'nG':0,'sA':None})
    if not _KF.get('no_a'):                                               # v9.8 (g3): stored input factors unused at kf_dream 1 -> not collected
        s_=_cov(inp[0][:nr_] if nr_ else inp[0],'r')
        e_['A']=s_ if e_['A'] is None else e_['A']+s_
    e_['nA']+=a_.shape[0]
    if a_.shape[1]>md_ and _KF.get('sketch',0)>0:                         # LOW-RANK SKETCH of a wide input factor: Y += a' (a Omega)
        r_=min(int(_KF['sketch']),a_.shape[1]); OM_=_KF.setdefault('om',{}); ky_=(a_.shape[1],r_,str(a_.device)); Om_=OM_.get(ky_)
        if Om_ is None:
            g_=torch.Generator(device='cpu'); g_.manual_seed(7919+a_.shape[1])
            Om_=OM_[ky_]=torch.randn(a_.shape[1],r_,generator=g_).to(torch.bfloat16).float().to(a_.device)   # bf16-exact: same Omega in both precisions
        ns_=int(_KF.get('sk_tok',4096)); as_=a_[::-(-a_.shape[0]//ns_)] if a_.shape[0]>ns_ else a_
        if as_.is_cuda: ab_=as_.to(torch.bfloat16); y_=(ab_.t()@(ab_@Om_.to(torch.bfloat16))).float()
        else: y_=as_.t()@(as_@Om_)
        e_['Y']=y_ if e_.get('Y') is None else e_['Y']+y_; e_['nY']=e_.get('nY',0)+as_.shape[0]; e_['Om']=Om_
    if getattr(mod,'_kf_last',False): _KF.setdefault('lasta',{})[k_]=a_
    cs_=a_.sum(0); e_['sA']=cs_ if e_.get('sA') is None else e_['sA']+cs_
    def gh_(g):
        with torch.autocast(device_type=g.device.type, enabled=False):                 # fp32 statistics (never bf16)
            g_=(g[:nr_] if nr_ else g).detach().reshape(-1,g.shape[-1]).float(); t_=(g_.t()@g_) if g_.shape[1]<=md_ else (g_*g_).sum(0)
            e_['G']=t_ if e_['G'] is None else e_['G']+t_; e_['nG']+=g_.shape[0]
    out.register_hook(gh_)
def _kf_mods(model):
    return [m for b in model.blocks for m in b.modules() if isinstance(m,nn.Linear) and m.weight.requires_grad]
def kf_register(model):
    for m_ in _kf_mods(model):
        if not getattr(m_,'_kf_h',False): m_.register_forward_hook(_f32(_kf_hook)); m_._kf_h=True
def kf_main_on(model,cfg,nreal):
    kf_register(model); _KF['acc']=model.__dict__.setdefault('_kf_cur',{}); _KF['maxdim']=int(cfg.kf_maxdim); _KF['nreal']=int(nreal); _KF['on']=True
    _KF['no_a']=bool(float(getattr(cfg,'kf_dream',0.0))>=1.0 and getattr(cfg,'kf_drop_a',True))
    _KF['share']={} if cfg.kf_share_a else None
    _KF['sketch']=0; _KF['sk_tok']=int(cfg.kf_sketch_tok)
def kf_collect(model,cfg,x,d):
    """Real batch, labels sampled from the model: Kronecker factors of the true Fisher of every block weight (accumulated over the visit)."""
    M_=_kf_mods(model)
    for m_ in M_:
        if not getattr(m_,'_kf_h',False): m_.register_forward_hook(_f32(_kf_hook)); m_._kf_h=True
    cur_=model.__dict__.setdefault('_kf_cur',{}); _KF['acc']=cur_; _KF['maxdim']=int(cfg.kf_maxdim); _KF['nreal']=None; _KF['sketch']=0
    was_=model.training; model.eval(); fs_=model.feat_scale.clone(); _KF['on']=True
    try:
        z_=model(x,d).float().reshape(-1,256)
        with torch.no_grad(): ys_=torch.multinomial(F.softmax(z_,-1),1).squeeze(-1)
        torch.autograd.grad(F.cross_entropy(z_,ys_,reduction='sum'),[m_.weight for m_ in M_],allow_unused=True)
    finally:
        _KF['on']=False; model.feat_scale.copy_(fs_); model.train(was_)
@torch.no_grad()
def _cka(X,Y):
    return float((X*Y).sum()/(X.norm()*Y.norm()+1e-30))
def sim_diag(model,cfg,seen,tag):
    """Data-free cross-domain similarity from the dream banks (pass ends): output affinity, representation CKA, Fisher overlap."""
    BK=model.__dict__.get('_bank',{}); ds=[d for d in seen if d in BK]
    if len(ds)<2: return
    dev=model.embed.weight.device; Pp=min(int(cfg.dream_stream_prompt),8); n_=int(cfg.sim_rows); OUT={}; ST={}
    sv_=model.__dict__.pop('_kf_cur',None)
    for d in ds:
        X_=BK[d][:n_].long().to(dev); x_,y_=X_[:,:-1],X_[:,1:]
        ch_=max(1,min(x_.shape[0],int(cfg.batch_size)//2))            # chunks of half a training batch (GPU memory: the Fisher pass keeps a graph)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        with torch.no_grad():
            was_=model.training; model.eval(); fs_=model.feat_scale.clone()
            for e in ds:
                tot_=0.0
                for c0 in range(0,x_.shape[0],ch_):
                    z_=model(x_[c0:c0+ch_],e).float()[:,Pp:]; tot_+=float(F.cross_entropy(z_.reshape(-1,256),y_[c0:c0+ch_,Pp:].reshape(-1),reduction='sum'))
                OUT[(d,e)]=tot_/(x_.shape[0]*(x_.shape[1]-Pp))/LOG2
            model.feat_scale.copy_(fs_); model.train(was_)
        model.__dict__['_kf_cur']={}
        for c0 in range(0,x_.shape[0],ch_): kf_collect(model,cfg,x_[c0:c0+ch_],d)
        C_=model.__dict__.pop('_kf_cur',{}); S_={}
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for k_,e_ in C_.items():
            if e_['G'] is None or e_['nG']==0: continue
            m_=e_['sA']/e_['nA']; A_=e_['A']/e_['nA']; A_=(A_-torch.outer(m_,m_)) if A_.dim()==2 else (A_-m_*m_)
            S_[k_]=(A_,e_['G']/e_['nG'])
        ST[d]=S_
    if sv_ is not None: model.__dict__['_kf_cur']=sv_
    REP={}; FIS={}
    for i,d in enumerate(ds):
        for e in ds[i+1:]:
            ks_=[k for k in ST[d] if k in ST[e]]
            ca_=[_cka(ST[d][k][0],ST[e][k][0]) for k in ks_]; cg_=[_cka(ST[d][k][1],ST[e][k][1]) for k in ks_]
            REP[(d,e)]=float(np.mean(ca_)) if ca_ else float('nan'); FIS[(d,e)]=float(np.mean([a*b for a,b in zip(ca_,cg_)])) if ca_ else float('nan')
    import json as _js
    print("  [simjson] "+_js.dumps({'tag':tag,'doms':ds,'out':{f"{d},{e}":v for (d,e),v in OUT.items()},
                                     'rep':{f"{d},{e}":v for (d,e),v in REP.items()},'fis':{f"{d},{e}":v for (d,e),v in FIS.items()}}))
_FILM={'mask':None}
_DP2={'on':False}
@torch.no_grad()
def aff_weights(model,cfg,x,y,olds,tau):
    """Old heads' bits on a real batch -> weights n_old softmax(-bits / tau) (mean 1)."""
    was_=model.training; model.eval(); fs_=model.feat_scale.clone()
    b_=torch.tensor([float(F.cross_entropy(model(x,d).float().reshape(-1,256),y.reshape(-1)))/LOG2 for d in olds])
    model.feat_scale.copy_(fs_); model.train(was_)
    w_=torch.softmax(-(b_-b_.min())/tau,0)*len(olds); return {d:float(w_[i]) for i,d in enumerate(olds)},{d:float(b_[i]) for i,d in enumerate(olds)}
def film_mod(b,hn,ns_=None):
    C=_FILM.get('codes')
    if C is None or getattr(b,'film_g',None) is None: return hn
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return hn
    if isinstance(tag,(list,tuple)):
        RI_=_XDG.get('rows_ix')
        if RI_ is not None and RI_.shape[0]==hn.shape[0]: ix=RI_                    # SAMPLER: row index built once per sampling call
        else:
            cur=_XDG['cur'] if _XDG['cur'] is not None else 0
            ix=torch.tensor([d if d is not None else cur for d in tag],device=hn.device)
        if ix.shape[0]!=hn.shape[0]: return hn
        c_=C[ix]; g_=(c_@b.film_g.t())[:,None,:]; o_=(c_@b.film_b.t())[:,None,:]
    else:
        c_=C[tag]; g_=c_@b.film_g.t(); o_=c_@b.film_b.t()
    return hn*(1.0+g_.to(hn.dtype))+o_.to(hn.dtype)
def kf_store(model,d):
    cur_=model.__dict__.pop('_kf_cur',None)
    if not cur_: return
    D_=model.__dict__.setdefault('_kf_dom',{}).setdefault(d,{})
    for k_,e_ in cur_.items():
        if e_['G'] is None or e_['nG']==0: continue
        A_=(e_['A']/e_['nA']) if e_['A'] is not None else None; G_=e_['G']/e_['nG']
        if k_ in D_:                                                    # revisit: average with the stored factors
            G_=0.5*(D_[k_][1]+G_); A_=(0.5*(D_[k_][0]+A_)) if (A_ is not None and D_[k_][0] is not None) else A_
        D_[k_]=(A_,G_)
    nv_=model.__dict__.setdefault('_kf_nv',{}); nv_[d]=nv_.get(d,0)+1
    tr_=lambda X: float(X.diagonal().sum() if X.dim()==2 else X.sum())
    if cur_:                                                               # FLATNESS proxy: sum over weights of tr(A) tr(G) on this visit's real rows
        fl_=sum(tr_(e_['A']/e_['nA'])*tr_(e_['G']/e_['nG']) for e_ in cur_.values() if e_['G'] is not None and e_['nG']>0 and e_['A'] is not None and e_['nA']>0)
        if fl_: model.__dict__.setdefault('_flat',{})[(d,nv_[d])]=fl_; print(f"    [flatness] {_lab(model.cfg,d)}: Fisher-trace proxy sum tr(A) tr(G) {fl_:.4g} (visit {nv_[d]})")
def _proj_apply(PJ,u):
    """u P for a projector: full matrix, coordinate mask (vector) or ('lr', U): u - (u U) U'."""
    if isinstance(PJ,tuple):
        U_=PJ[1]
        if U_.dtype==torch.bfloat16: return u-((u.to(torch.bfloat16)@U_)@U_.t()).float()     # GPU: bf16 matmuls (as Newton-Schulz)
        return u-(u@U_)@U_.t()
    return (u@PJ) if PJ.dim()==2 else u*PJ[None,:]
_KFR={'r':0}
def _isqrt(X,lam):
    """(I + X / (lam mean eig X))^-1/2 for a PSD factor X (vector = diagonal). Eigenvalues of X clamped at 0, so every entry of the
    result is in (0, 1]: the whitening can only shrink a step, never enlarge it."""
    X=X.double()
    if X.dim()==1: X=X.clamp(min=0); return (1.0+X/(lam*float(X.mean())+1e-30)).rsqrt().float()
    w_,V_=torch.linalg.eigh(0.5*(X+X.t())); w_=w_.clamp(min=0)
    r_=int(_KFR['r'])
    if 0<r_<w_.shape[0]: w_=w_.clone(); w_[:-r_]=w_[:-r_].mean()             # LOW-RANK + ISOTROPIC: top r kept (ascending order), rest = its mean
    return ((V_*(1.0+w_/(lam*float(w_.mean())+1e-30)).rsqrt())@V_.t()).float()
def _isqrt_ns_batch(Xs,lams):
    """Batched (I + X / (lam mean eig X))^-1/2 for same-shape PSD factors by the COUPLED NEWTON-SCHULZ iteration (Higham): B scaled by
    1.1 x a power-iteration estimate of its top eigenvalue (all eigenvalues of B are >= 1, so the iteration count follows from that bound);
    Y -> (B/s)^1/2, Z -> (B/s)^-1/2. fp32 products with TF32 off (the factors' condition number reaches ~1e3)."""
    X=torch.stack([x.double() for x in Xs]); n=X.shape[-1]; X=0.5*(X+X.transpose(1,2)); dev=X.device
    c=torch.tensor([float(l) for l in lams],dtype=torch.float64,device=dev)[:,None,None]*(torch.diagonal(X,dim1=1,dim2=2).sum(-1)/n).clamp(min=1e-30)[:,None,None]
    B=(torch.eye(n,dtype=torch.float64,device=dev)+X/c).float(); del X
    pv_=torch.backends.cuda.matmul.allow_tf32 if dev.type=='cuda' else None
    if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=False
    try:
        g_=torch.Generator(device='cpu'); g_.manual_seed(97); v=torch.randn(B.shape[0],n,1,generator=g_).to(dev)
        for _ in range(12): v=B@v; v=v/v.norm(dim=1,keepdim=True)
        s=(1.1*(v.transpose(1,2)@B@v).squeeze(-1).squeeze(-1)).clamp(min=1.0)
        it=int(min(60,math.ceil(math.log(float(s.max()))/math.log(1.5))+6)); s=s[:,None,None]
        Y=B/s; Z=torch.eye(n,device=dev).expand_as(Y).clone(); I3=3.0*torch.eye(n,device=dev)
        for _ in range(it): T=0.5*(I3-Z@Y); Y=Y@T; Z=T@Z
    finally:
        if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=pv_
    return list((Z/s.sqrt()).unbind(0))
def kf_live_warmup(model,cfg,olds):
    """v9.8 (g3 without stored input factors): the dream rows' CURRENT input factors from one forward pass at visit start"""
    if not getattr(model,'_dsq',None): return None
    acc_=model.__dict__.setdefault('_kf_dacc',{}); was_=model.training; fs_=model.feat_scale.clone()
    for _ in range(max(1,int(getattr(cfg,'kf_warm_batches',8)))):           # several dream batches: a stable estimate of the input statistics
        sb_=stream_batch(model,cfg,olds)
        if sb_ is None: break
        xd,_,dm_,_=sb_
        kf_main_on(model,cfg,1); _KF['dream']=True; _KF['real_off']=True; _KF['dacc']=acc_
        model.eval(); model.__dict__['_s0_dom']=list(dm_)+[None]*(xd.shape[0]-len(dm_))
        try:
            with torch.enable_grad(): model.features(xd)                   # (graph discarded: the hooks only read the inputs)
        finally:
            _KF['on']=False; _KF['dream']=False; _KF['real_off']=False; model.__dict__['_s0_dom']=None; model.feat_scale.copy_(fs_); model.train(was_)
    return {k_:e_['A']/max(1,e_['n']) for k_,e_ in acc_.items() if e_['A'] is not None} or None
def kf_build(model,cfg,d,live=None,quiet=False):
    """Visit start: preconditioners from the OTHER domains' stored factors, keyed by weight id. live = {module id: current input factor of the
    dream rows} (old domains' keys as the network sees them NOW): input side (1 - kf_dream) * sum(stored A) + kf_dream * n_old * A_live."""
    _KF['wbf16']=bool(getattr(cfg,'whiten_bf16',False)); t0_=time.time(); nsj_={}; _KFR['r']=int(cfg.kf_rank); ekc_=[]; isc_=[]
    D_=model.__dict__.get('_kf_dom',{}); P_={}
    for m_ in _kf_mods(model):
        W_=model.__dict__.get('_kf_wt') or {}                           # EARLY DRIFT GUARD: flagged domains' factors weighted up
        k_=id(m_); Gs_=[D_[j][k_][1]*float(W_.get(j,1.0)) for j in D_ if j!=d and k_ in D_[j]]
        if not Gs_: continue
        nA_=len(Gs_); As_=[(D_[j][k_][0])*float(W_.get(j,1.0)) for j in D_ if j!=d and k_ in D_[j] and D_[j][k_][0] is not None]
        if not As_:                                                     # v9.8 (g3): no stored input factors -> the live dream factor alone
            if not (live and k_ in live): continue
            As_=[live[k_].float()*float(nA_)]
        elif live and k_ in live and live[k_].shape==As_[0].shape:
            mx_=float(cfg.kf_dream); As_=[a_*(1.0-mx_) for a_ in As_]+[live[k_].to(As_[0].device,As_[0].dtype)*(mx_*len(As_))]
        lam_=cfg.kf_lam
        nv_=sum(v_ for j_,v_ in model.__dict__.get('_kf_nv',{}).items() if j_!=d); nv_=min(nv_,cfg.kf_kalman_max)
        lam_=cfg.kf_lam*cfg.kf_kalman/max(1,nv_)
        PO_=model.__dict__.get('_kf_P',{}).get(id(m_.weight)) if live else None    # live-key rebuild: gradient side unchanged -> reused
        XA_=sum(As_); ZA_=None
        if cfg.kf_share_a:                                          # identical input factor (shared input) -> the same whitening
            ZA_=next((h_[2] for h_ in isc_ if h_[0].shape==XA_.shape and h_[1]==lam_ and torch.equal(h_[0],XA_)),None)
        if ZA_ is None:
            ZA_=_isqrt(XA_,lam_).float()
            if cfg.kf_share_a: isc_.append((XA_,lam_,ZA_))
        P_[id(m_.weight)]=(PO_[0] if PO_ is not None else _isqrt(sum(Gs_),lam_).float(),ZA_)
    if nsj_:
        for jb_ in nsj_.values():
            for (k_,si_,_,_),Z_ in zip(jb_,_isqrt_ns_batch([j_[2] for j_ in jb_],[j_[3] for j_ in jb_])): P_[k_][si_]=Z_.float()
        P_={k_:tuple(v_) for k_,v_ in P_.items()}
    if P_ and next(iter(P_.values()))[-1].is_cuda: torch.cuda.synchronize()
    st_=model.__dict__.setdefault('_kfbt',[0.0,0]); st_[0]+=time.time()-t0_; st_[1]+=1
    model.__dict__['_kf_P']=P_
    model.__dict__.pop('_kf_K',None)
    model.__dict__.pop('_kf_proj',None)
    if P_ and not quiet:
        sh_=[(float(v_[3].mean()) if v_[0]=='ek' else float((v_[0].diagonal() if v_[0].dim()==2 else v_[0]).mean()*(v_[1].diagonal() if v_[1].dim()==2 else v_[1]).mean())) for v_ in P_.values()]
        print(f"    [protected muon] {len(P_)} weights whitened by {len([j for j in D_ if j!=d])} other domains' Fisher factors (mean diagonal shrink {np.mean(sh_):.3f})")
def _kf_apply(P,u):
    if P[0]=='ek':                                                      # EKFAC: rotate, scale per entry, rotate back
        _,UG_,UA_,W_=P; lp_=_KF.get('wbf16') and u.is_cuda; dt_=torch.bfloat16 if lp_ else u.dtype
        x=u.to(dt_)
        if UG_ is not None: x=UG_.t().to(dt_)@x
        if UA_ is not None: x=x@UA_.to(dt_)
        x=(x.float()*W_).to(dt_)
        if UG_ is not None: x=UG_.to(dt_)@x
        if UA_ is not None: x=x@UA_.t().to(dt_)
        return x.float()
    PG_,PA_=P
    if _KF.get('wbf16') and u.is_cuda:                                  # bf16 whitening (tensor cores), fp32 result
        x=u.to(torch.bfloat16); x=(PG_.to(torch.bfloat16)@x) if PG_.dim()==2 else PG_.to(torch.bfloat16)[:,None]*x
        return ((x@PA_.to(torch.bfloat16)) if PA_.dim()==2 else x*PA_.to(torch.bfloat16)[None,:]).float()
    u=(PG_@u) if PG_.dim()==2 else PG_[:,None]*u
    return (u@PA_) if PA_.dim()==2 else u*PA_[None,:]
def muon_prep(model,cfg):
    """Before opt.step(): Muon directions for the blocks' 2-D weights; their grads are cleared so Adam skips them."""
    out=[]; pend_=[]
    if cfg.muon_precise and '_mp_priv' not in model.__dict__:               # private / per-domain matrices keep Muon (PX_PS: Muon private)
        model.__dict__['_mp_priv']={id(p_) for b in model.blocks for n_,p_ in b.named_parameters() if n_.startswith(('pu_','film_'))}
        print(f"  [precise shared] shared block matrices step along the whitened momentum (Muon-sized); {len(model.__dict__['_mp_priv'])} private / FiLM matrices keep Muon", flush=True)
    for p_ in [p for b in model.blocks for p in b.parameters() if p.ndim==2 and p.requires_grad]:
        if p_.grad is None: continue
        g_=p_.grad.float(); st_=model.__dict__.setdefault('_muB',{})
        bf_=st_.get(id(p_))
        if bf_ is None or bf_.shape!=g_.shape: bf_=st_[id(p_)]=torch.zeros_like(g_)
        mu_=cfg.muon_mom
        ud_=None
        bf_.mul_(mu_).add_(g_); u_=g_+mu_*bf_
        if ud_ is not None: u_=u_+ud_
        P_=model.__dict__.get('_kf_P',{}).get(id(p_))
        z_=_kf_apply(P_,u_) if P_ is not None else u_
        PJ_=None
        if PJ_ is not None:                                             # DISJOINT SUBSPACES: Muon inside the free input subspace only
            o_=_proj_apply(PJ_,_ns5(_proj_apply(PJ_,u_)))
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        K_=None
        if K_ is not None:                                              # KALMAN MUON: left whitening kept, input side = Kalman gain in the polar
            PG_=P_[0]; zl_=(PG_@u_) if PG_.dim()==2 else PG_[:,None]*u_
            R_,lam_=K_; s_=float(zl_.pow(2).sum())/zl_.shape[1]
            o_=_ns5(torch.cat([zl_,R_*math.sqrt(max(s_,1e-30)/lam_)],0))[:zl_.shape[0]]
            o_=(PG_@o_) if PG_.dim()==2 else PG_[:,None]*o_
            o_=o_*(zl_.abs().sum(1,keepdim=True)>0)*(zl_.abs().sum(0,keepdim=True)>0)        # zero rows / columns stay zero (private units)
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        if cfg.muon_precise and id(p_) not in model.__dict__.get('_mp_priv',set()):   # cg11 PRECISE SHARED: whitened momentum, Muon-sized
            o_=z_*(math.sqrt(min(z_.shape))/z_.norm().clamp_min(1e-12))        #   (tensor norm: no host sync)
            if cfg.muon_precise_cap>0:                                         # cg11b CAP: largest singular value <= cap x Muon's (power iteration,
                PV_=model.__dict__.setdefault('_mpV',{}); v_=PV_.get(id(p_))   #   warm-started per matrix; no host sync). cg11: Frobenius-only sizing
                if v_ is None or v_.shape[0]!=o_.shape[1]: v_=torch.randn(o_.shape[1],device=o_.device,dtype=o_.dtype)   #   let the top directions
                for _ in range(int(cfg.muon_precise_pit)): v_=o_.t()@(o_@v_); v_=v_/v_.norm().clamp_min(1e-30)          #   grow with width (GPU:
                PV_[id(p_)]=v_; sg_=(o_@v_).norm()                             #   learning far worse than Muon from the first domain)
                o_=o_*torch.clamp(float(cfg.muon_precise_cap)/sg_.clamp_min(1e-12),max=1.0)
            if P_ is not None: o_=_kf_apply(P_,o_)
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        pend_.append((p_,z_,P_)); p_.grad=None; continue
        if P_ is not None: o_=_kf_apply(P_,o_)                                       # Fisher-whitened Muon step
        out.append((p_,o_*(0.2*math.sqrt(max(p_.shape)))))
        p_.grad=None
    if pend_:                                                                         # one Newton-Schulz call per matrix shape
        grp_={}
        for e_ in pend_: grp_.setdefault(tuple(e_[1].shape),[]).append(e_)
        for G_ in grp_.values():
            ns_=_ns5_gram_batch if (cfg.muon_gram and min(G_[0][1].shape)*2<=max(G_[0][1].shape)) else _ns5_batch   # Gram NS for rectangular weights (>= 2:1)
            for (p_,z_,P_),o_ in zip(G_,ns_([e_[1] for e_ in G_],steps=int(cfg.muon_ns_steps))):   # cg12a: muon_ns_steps (5 = Muon default)
                if P_ is not None: o_=_kf_apply(P_,o_)
                out.append((p_,o_*(0.2*math.sqrt(max(p_.shape)))))
    return out
def _cnt_ctx(X):
    """Context hashes ending at each position: order 1 (exact), 2 (8192), 3 (16384)."""
    b=F.pad(X,(1,0))[:,:-1]; c=F.pad(X,(2,0))[:,:-2]; M=4294967296
    return [X,(((b*256+X)*2654435761)%M)>>19,((((c*256+b)*256+X)*2654435761)%M)>>18]
@torch.no_grad()
def _probe_lq(model,cfg,X,d):
    was=model.training; model.eval(); fs=model.feat_scale.clone(); out=[]
    for c0 in range(0,X.shape[0],16):
        Xc=X[c0:c0+16]; out.append(F.log_softmax(model.heads[d](priv(model,Xc,model.features(Xc),d)).float()/cfg.shunt_T,-1))
    model.feat_scale.copy_(fs); model.train(was); return torch.cat(out)
def _kl_topk(lq,V,I):
    """THE stored-target KL, per position: KL(top-k log-probs V at bytes I + one rest bin || student log-probs lq). Every user of the
    stored targets (dream loss, probes, sensors, noise scales, hybrid rows) calls this one function."""
    pk_=V.exp(); r_=(1-pk_.sum(-1)).clamp(min=1e-8)
    mk_=torch.ones_like(lq,dtype=torch.bool).scatter_(-1,I,False); lr_=lq.masked_fill(~mk_,float('-inf')).logsumexp(-1)   # student's log-mass outside the top-k
    return (pk_*(V-lq.gather(-1,I))).sum(-1)+r_*(r_.log()-lr_)
@torch.no_grad()
def _probe_kl(lq,V,I,Pp,k_):
    """per-row KL(stored top-k + rest bin || current) over positions >= Pp"""
    return _kl_topk(lq,V,I)[:,Pp:].mean(-1)
@torch.no_grad()
def _probe_real(model,cfg,X,d):
    """count-table log-likelihood per row (realism of a dream row for domain d)"""
    C=model.__dict__.get('_cnt',{}).get(d)
    if C is None: return None
    Xl=X.long(); hs=_cnt_ctx(Xl[:,:-1]); y=Xl[:,1:]; m_=5.0; lp=torch.zeros_like(y,dtype=torch.float32); done=torch.zeros_like(y,dtype=torch.bool)
    for o in (2,1,0):
        T_=C[o].view(-1,256)[hs[o]]; tot=T_.sum(-1); ok=(tot>=m_)&~done
        lp=torch.where(ok,(T_.gather(-1,y[...,None]).squeeze(-1)+0.5).log()-(tot+128).log(),lp); done|=ok
    return lp.mean(-1)
def probe_pre(model,cfg,olds,t,vals):
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); k_=int(cfg.bank_targets); dev=model.embed.weight.device; P={'olds':[],'t':t}
    xb_=[]; g_=vals[t]
    if hasattr(g_,'reset'): g_.reset()
    for _ in range(4): xb_.append(g_()[0])
    XB=torch.cat(xb_).long().to(dev)[:64]; P['XB']=XB
    for d in olds:
        if d not in BK or d not in TV: continue
        X=BK[d].long().to(dev); V,I=TV[d]; V=V.to(dev).float(); I=I.to(dev).long(); Pp=min(cfg.dream_stream_prompt,X.shape[1]//2)
        e={'d':d,'bpc':bpc(model,vals[d],d),'X':X,'V':V,'I':I,'Pp':Pp,'klA':_probe_kl(_probe_lq(model,cfg,X,d),V,I,Pp,k_)}
        lB=_probe_lq(model,cfg,XB,d); vB,iB=lB.topk(k_,-1); e['B']=(vB,iB)
        XC=dream_sample(model,d,64,X.shape[1],cfg.dream_temp).long(); lC=_probe_lq(model,cfg,XC,d); vC,iC=lC.topk(k_,-1); e['C']=(XC,vC,iC)
        e['real']=_probe_real(model,cfg,X,d)
        P['olds'].append(e)
    return P
def probe_post(model,cfg,P,vi,vals):
    import json as _js
    k_=int(cfg.bank_targets); rows=[]
    for e in P['olds']:
        d=e['d']; b1=bpc(model,vals[d],d); dR=(b1-e['bpc'])*math.log(2)          # real damage in nats / byte
        klA=_probe_kl(_probe_lq(model,cfg,e['X'],d),e['V'],e['I'],e['Pp'],k_); dA=klA-e['klA']
        vB,iB=e['B']; dB=float(_probe_kl(_probe_lq(model,cfg,P['XB'],d),vB,iB,0,k_).mean())
        XC,vC,iC=e['C']; dC=float(_probe_kl(_probe_lq(model,cfg,XC,d),vC,iC,e['Pp'],k_).mean())
        r_=e['real']; dD=dDm=float('nan')
        if r_ is not None:
            o_=torch.argsort(r_,descending=True); h_=len(o_)//2; dD=float(dA[o_[:h_]].mean()); dDm=float(dA[o_[h_:]].mean())
        rows.append({'vi':vi,'d':d,'real':dR,'A':float(dA.mean()),'B':dB,'C':dC,'D':dD,'Dm':dDm})
    for r in rows: print("  [probe] "+_js.dumps(r))
def muon_apply(out,lr):
    with torch.no_grad():
        for p_,u_ in out: p_.add_(u_.to(p_.dtype),alpha=-lr)
def _vp_inv(model,cfg,k_,C_):
    IC_=model.__dict__.setdefault('_vpI',{}); n_=model.__dict__.setdefault('_vpIn',{}); n_[k_]=n_.get(k_,0)+1
    if k_ not in IC_ or (n_[k_]-1)%max(1,int(cfg.varpro_inv_every))==0:
        IC_[k_]=torch.linalg.inv_ex(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device))[0]
    return IC_[k_]
@torch.no_grad()
def varpro_old_refit(model,cfg,xd,hfd,spans,Pp,vi_,tgt_d):
    """Gauss-Newton refit of every OLD head on its own dream rows (positions after the prompt) towards their targets."""
    CG_=model.__dict__.setdefault('_vpC',{}); BA_=[]; BR_=[]; BH_=[]
    k_v=max(1,int(model.__dict__.get('_vpo_k',1))); vlr_=cfg.varpro_lr if k_v==1 else 1.0-(1.0-float(cfg.varpro_lr))**k_v   # k damped steps in one (head_dream_k)
    cur_=model.__dict__.get('_live_dom')                                     # the visited domain's head is trained normally this step
    for d,a,b in spans:
        if (cfg.self_dream_nreal or cfg.self_prot_w>0) and cur_ is not None and int(d)==int(cur_): continue
        hd_=model.heads[d]
        Hd_=(priv(model,xd[a:b],hfd[a:b].detach(),d))[:,Pp:].float()
        Hd_=Hd_.reshape(-1,Hd_.shape[-1]); n_=Hd_.shape[0]
        if vi_ is not None:                                                    # stored top-k + rest bin -> dense target
            V_,I_=vi_[0][a:b,Pp:].float(),vi_[1][a:b,Pp:]; pk_=V_.exp(); r_=(1-pk_.sum(-1,keepdim=True)).clamp(min=0)
            T_=(r_/(256-I_.shape[-1])).expand(*I_.shape[:-1],256).clone(); T_.scatter_(-1,I_,pk_); T_=T_.reshape(-1,256)
        elif tgt_d is not None: T_=tgt_d[a:b,Pp:].float().exp().reshape(-1,256)
        else: continue
        Ha_=torch.cat([Hd_,torch.ones(n_,1,device=Hd_.device)],1); Cb_=Ha_.t()@Ha_/n_; k_=('dream',d)
        CG_[k_]=Cb_ if k_ not in CG_ else cfg.varpro_beta*CG_[k_]+(1.0-cfg.varpro_beta)*Cb_
        model.__dict__['_vpo_n']=model.__dict__.get('_vpo_n',0)+1
        E_=F.softmax(hd_(Hd_.to(hd_.weight.dtype)).float()/cfg.shunt_T,-1)-T_; C_=CG_[k_]
        BA_.append(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device)); BR_.append(Ha_.t()@E_/n_); BH_.append(hd_)
        continue
        Ai_=_vp_inv(model,cfg,k_,CG_[k_])
        for _ in range(max(1,cfg.varpro_k)):
            E_=F.softmax(hd_(Hd_.to(hd_.weight.dtype)).float()/cfg.shunt_T,-1)-T_
            dW_=-(Ai_@(Ha_.t()@E_)/n_).t()*vlr_
            hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
    if BA_:                                                                    # ONE batched solve for every old head of this step
        X_=torch.linalg.solve_ex(torch.stack(BA_),torch.stack(BR_))[0]          # _ex: no singularity check = no host sync
        for hd_,x_ in zip(BH_,X_):
            dW_=-x_.t()*vlr_; hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
@torch.no_grad()
def head_center_(model):
    for hd_ in model.heads:
        hd_.weight.sub_(hd_.weight.mean(0,keepdim=True))
        if hd_.bias is not None: hd_.bias.sub_(hd_.bias.mean())
def varpro_refit(model,cfg,opt,x,y,h,d):
    """VarPro (head first): refit the current head on this batch's detached features before the trunk gradient is taken."""
    hd_=model.heads[d]
    with torch.no_grad(): hv_=(priv(model,x,h.detach(),d)).reshape(-1,h.shape[-1]).float()
    yv_=y.reshape(-1); lr0_=opt.param_groups[0]['lr']
    Ha_=torch.cat([hv_,torch.ones(hv_.shape[0],1,device=hv_.device)],1); n_=Ha_.shape[0]
    CG_=model.__dict__.setdefault('_vpC',{}); Cb_=Ha_.t()@Ha_/n_
    CG_[d]=Cb_ if d not in CG_ else cfg.varpro_beta*CG_[d]+(1.0-cfg.varpro_beta)*Cb_
    Ai_=None
    with torch.no_grad():
        for _ in range(cfg.varpro_k):
            P_=F.softmax(hd_(hv_.to(hd_.weight.dtype)).float(),-1); P_[torch.arange(n_),yv_]-=1.0           # P - Y
            if Ai_ is None:                                                                                    # direct solve (LU)
                C_=CG_[d]; dW_=-torch.linalg.solve_ex(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device),Ha_.t()@P_/n_)[0].t()*cfg.varpro_lr
            else: dW_=-(Ai_@(Ha_.t()@P_)/n_).t()*cfg.varpro_lr                                               # (256, D+1)
            hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
@torch.no_grad()
def smem_reencode(model,cfg,cur):
    """SLOT MEMORY RE-ENCODING: every other slot's memory rebuilt with the current weights from slot_mem_reenc of its bank dreams (first halves)."""
    SM=model.__dict__.get('_smem',{}); BK=model.__dict__.get('_bank',{}); R=int(cfg.slot_mem_reenc); Tq_=min(cfg.dream_len,cfg.seq_len)
    dev=model.embed.weight.device; n_=0
    for d in list(SM):
        if d==cur or d not in BK: continue
        X=BK[d][:R,:Tq_].long().to(dev); was=model.training; fs=model.feat_scale.clone(); s0_=model.__dict__.get('_s0_dom')
        sv_={k_:_CARRY[k_] for k_ in ('on','nr','st','new','rows')}
        try:
            model.eval(); _CARRY.update(on=True,nr=X.shape[0],st={},new={},rows=None); model.__dict__['_s0_dom']=d
            model.features(X); new_=dict(_CARRY['new'])
        finally:
            model.train(was); model.feat_scale.copy_(fs); model.__dict__['_s0_dom']=s0_; _CARRY.update(sv_)
        for k_,v_ in new_.items():
            if isinstance(k_,tuple): continue
            SM[d][k_]=v_.reshape(X.shape[0],v_.shape[0]//X.shape[0],*v_.shape[1:]).mean(0)
        n_+=1
    if n_: print(f"  [slot memory] re-encoded {n_} other slots' memories with the current weights ({R} dreams each)")
@torch.no_grad()
def guard_sense(model,cfg,olds):
    """EARLY DRIFT GUARD sensor: KL(stored targets || current) per old domain on its first guard_rows bank rows (held out: never trained)."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); dev=model.embed.weight.device; H=min(int(cfg.guard_rows),int(cfg.bank_hold)); out={}
    for d in olds:
        if d not in BK or d not in TV or H<1: continue
        Xs=BK[d][:H].long().to(dev); V,I=TV[d]; Vs=V[:H].to(dev).float(); Is=I[:H].to(dev).long()
        sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=d
        try: lq=_probe_lq(model,cfg,Xs,d)
        finally: model.__dict__['_s0_dom']=sv_
        out[d]=float(_probe_kl(lq,Vs,Is,min(cfg.dream_stream_prompt,Xs.shape[1]//2),Vs.shape[-1]).mean())
    return out
def noise_diag(model,cfg,x,y,d,olds):
    """GRADIENT NOISE SCALES of the two parts of the step (McCandlish et al. 2018), on the trunk: two independent half batches of each part
    give per-row noise tr(Sigma) = b |g1-g2|^2 / 2 and signal |G|^2 = |g1+g2|^2/4 - tr(Sigma)/(2b); B_noise = tr(Sigma)/|G|^2 is the batch
    beyond which more rows of that part stop paying per step. NEYMAN allocation of a fixed row budget between the real loss and the
    w-weighted dream loss: n_dream / n_real = w sigma_dream / sigma_real * sqrt(c_real / c_dream). Accumulated per visit, printed at its end."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[j for j in olds if j in BK and j in TV]
    if not ol or x.shape[0]<4: return
    dev=model.embed.weight.device; P_=[p for b in model.blocks for p in b.parameters() if p.requires_grad]
    h=x.shape[0]//2; k_=int(cfg.bank_targets); was_=model.training; model.eval(); fs_=model.feat_scale.clone()
    sv_=model.__dict__.get('_s0_dom')
    def grads(loss_fn):
        with torch.enable_grad():
            l_=loss_fn(); g_=torch.autograd.grad(l_,P_,allow_unused=True)
        return [gg.detach().float() if gg is not None else torch.zeros_like(pp,dtype=torch.float32) for gg,pp in zip(g_,P_)]
    def real(i):
        model.__dict__['_s0_dom']=d
        z_=model(x[i*h:(i+1)*h],d).float(); return F.cross_entropy(z_.reshape(-1,256),y[i*h:(i+1)*h].reshape(-1))
    r_=np.random.RandomState(int(model.__dict__.get('_gstep',0))+977)
    dm=sorted(ol[i] for i in r_.randint(len(ol),size=2*h)); ri=[r_.randint(BK[j].shape[0]) for j in dm]
    X=torch.stack([BK[j][i] for j,i in zip(dm,ri)]).long().to(dev)
    V=torch.stack([TV[j][0][i] for j,i in zip(dm,ri)]).to(dev).float(); I=torch.stack([TV[j][1][i] for j,i in zip(dm,ri)]).to(dev).long()
    Pp=min(cfg.dream_stream_prompt,X.shape[1]//2)
    def dream(i):
        sl=slice(i*h,(i+1)*h); Xs,Vs,Is,ds=X[sl],V[sl],I[sl],dm[sl.start:sl.stop]
        model.__dict__['_s0_dom']=list(ds); hf=model.features(Xs); out=[]; a=0
        for j in sorted(set(ds)):
            b_=a+sum(1 for q in ds if q==j); out.append(F.log_softmax(model.heads[j](priv(model,Xs[a:b_],hf[a:b_],j)).float()/cfg.shunt_T,-1)); a=b_
        return _kl_topk(torch.cat(out),Vs,Is)[:,Pp:].mean()
    try:
        R1,R2=grads(lambda: real(0)),grads(lambda: real(1)); D1,D2=grads(lambda: dream(0)),grads(lambda: dream(1))
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs_); model.train(was_)
    dot=lambda A,B: float(sum((a*b).sum() for a,b in zip(A,B)))
    def stats(A,B):
        dd=dot(A,A)+dot(B,B)-2*dot(A,B); ss=dot(A,A)+dot(B,B)+2*dot(A,B); tr=h*dd/2; G2=ss/4-tr/(2*h); return tr,G2
    trR,GR=stats(R1,R2); trD,GD=stats(D1,D2)
    SR_=[a+b for a,b in zip(R1,R2)]; SD_=[a+b for a,b in zip(D1,D2)]                 # raw cosine of the two mean gradients (bounded; noise
    cos=dot(SR_,SD_)/max(1e-30,(dot(SR_,SR_)*dot(SD_,SD_))**0.5)                       #   biases it towards 0 - the signal estimate can be < 0)
    A_=model.__dict__.setdefault('_ndiag',[]); A_.append((trR,GR,trD,GD,cos))
@torch.no_grad()
def noise_report(model,cfg,dom_id):
    A_=model.__dict__.pop('_ndiag',None)
    if not A_: return
    trR=np.mean([a[0] for a in A_]); GR=np.mean([a[1] for a in A_]); trD=np.mean([a[2] for a in A_]); GD=np.mean([a[3] for a in A_]); cs=np.mean([a[4] for a in A_])
    P_=model.__dict__.get('_sleepP',{}); w=float(cfg.dream_weight)*(float(cfg.sleep_fixed))
    ratio=w*math.sqrt(max(trD,0)/max(trR,1e-30))*math.sqrt(1.0/1.5)                     # dream row ~1.5x the cost of a real row
    print(f"  [noise scales] {len(A_)} probes: per-row gradient noise real {trR:.3e} / dream {trD:.3e} | signal |G|^2 real {GR:.3e} / dream {GD:.3e} | "
          f"B_noise (rows) real {trR/max(GR,1e-30):.0f} / dream {trD/max(GD,1e-30):.0f} | cos(G_real, G_dream) {cs:+.2f} | "
          f"Neyman dream rows for {cfg.batch_size} real rows at weight {w:.2f}: {ratio*cfg.batch_size:.0f}")
def ngp_init(model,cfg):
    dev=model.embed.weight.device; T={}
    for d in range(len(PILE_DOMAINS)):
        T[d]=[nn.Embedding(int(cfg.ngp_rows),cfg.d_model,sparse=True).to(dev) for _ in cfg.ngp]
        for E in T[d]: E.weight.data.zero_()
    model.__dict__['_ngp']=T; model.__dict__['_ngp_opt']=torch.optim.SparseAdam([E.weight for d in T for E in T[d]],lr=cfg.lr*cfg.ngram_lr_mult)
_XDG={'on':False,'cur':None,'rows':None,'masks':{},'keep':1.0,'seed':0}
def xdg_mask(i,H,dev):
    """(rows,H) or (H,) gate for block i from the current row tags (int, list per row, or the current domain)"""
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return None
    def one(d):
        k=(i,int(d))
        if k not in _XDG['masks']:                                   # one long mask per (block, domain): widening appends units, old units keep their gate
            g=torch.Generator().manual_seed(_XDG['seed']*100003+i*1009+int(d)*17)
            _XDG['masks'][k]=(torch.rand(1<<15,generator=g)<_XDG['keep']).float()
        return _XDG['masks'][k][:H].to(dev)
    if isinstance(tag,(list,tuple)):
        cur=_XDG['cur']; return torch.stack([one(d if d is not None else (cur if cur is not None else 0)) for d in tag])
    return one(tag)
_SLP={'n':[0,0]}
_HCAP={'on':False,'h':None}                                              # cg-gate: capture of the head input (protected gate)
_BUFDET={'on':False}                                                    # v9g: per-domain parameters detached (buffered windows)
_PUOFF={'d':None}                                                        # cg19: domain whose private units are switched OFF
def pu_term(b,hn,ns_=None):
    r_=_pu_term0(b,hn,ns_); d_=_PUOFF['d']
    if d_ is None or not torch.is_tensor(r_): return r_
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if isinstance(tag,(list,tuple)):
        cur=_XDG['cur'] if _XDG['cur'] is not None else 0
        m_=torch.tensor([0.0 if (t_ if t_ is not None else cur)==d_ else 1.0 for t_ in tag],device=r_.device,dtype=r_.dtype)
        return r_*m_.view(-1,*([1]*(r_.dim()-1))) if m_.shape[0]==r_.shape[0] else r_
    return r_*0.0 if tag==d_ else r_
def _pu_term0(b,hn,ns_=None):
    """private trunk units of each row's domain. LEAK FIX (pu_dream_grad False): DREAM rows use their domain's units with the parameters
    DETACHED - an old domain's private units are then trained only by its own real text (its visits). Otherwise every loss on the dream rows
    reaches them, including BORROWING (the current domain's head fitting the old domain's dreams), which pulls the old domain's private units
    towards the current domain's objective."""
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return 0.0
    W_in,W_b,W_out=b.pu_in,b.pu_b,b.pu_out
    if _BUFDET['on']: W_in,W_b,W_out=W_in.detach(),W_b.detach(),W_out.detach()
    if isinstance(tag,(list,tuple)):
        mk_=getattr(b,'pu_mask',None); G_=_XDG.get('groups')
        if G_ is not None and G_[0]==hn.shape[0] and not torch.is_grad_enabled():     # SAMPLER: contiguous domain groups, one matmul per domain
            out_=[]
            for d_,sl_ in G_[1]:
                z=F.gelu(hn[sl_]@W_in[d_]+W_b[d_])
                if mk_ is not None: z=z*mk_[d_].to(z.dtype)
                out_.append(z@W_out[d_])
            return torch.cat(out_,0)
        cur=_XDG['cur'] if _XDG['cur'] is not None else 0
        ix=torch.tensor([d if d is not None else cur for d in tag],device=hn.device)
        if ix.shape[0]!=hn.shape[0]: return 0.0
        def f(Wi,Wb,Wo,h_,ix_):
            z=F.gelu(torch.einsum('btd,bdk->btk',h_,Wi[ix_])+Wb[ix_][:,None])
            if mk_ is not None: z=z*mk_[ix_][:,None,:].to(z.dtype)
            return torch.einsum('btk,bkd->btd',z,Wo[ix_])
        tr=torch.is_grad_enabled() and W_out.requires_grad and not getattr(b,'_pu_dream_grad',True)
        if tr and _STF.get('dream_call'): return f(W_in.detach(),W_b.detach(),W_out.detach(),hn,ix)        # dream-only forward
        if tr and ns_ is not None and 0<ns_<hn.shape[0]:                                                    # merged: real rows live, dream rows detached
            return torch.cat([f(W_in,W_b,W_out,hn[:ns_],ix[:ns_]),f(W_in.detach(),W_b.detach(),W_out.detach(),hn[ns_:],ix[ns_:])],0)
        return f(W_in,W_b,W_out,hn,ix)
    z=F.gelu(hn@W_in[tag]+W_b[tag])
    if getattr(b,'pu_mask',None) is not None: z=z*b.pu_mask[tag].to(z.dtype)
    return z@W_out[tag]
def xdg_apply(m,i):
    if not _XDG['on']: return m
    g=xdg_mask(i,m.shape[-1],m.device)
    if g is None: return m
    if g.dim()==2:
        if g.shape[0]!=m.shape[-3] if m.dim()>=3 else True: return m
        return m*g.view(*([1]*(m.dim()-3)),g.shape[0],1,g.shape[1]).to(m.dtype)
    return m*g.to(m.dtype)
def side_out(model,taps,dom,nstat):
    """ladder side network over the trunk's (detached) taps; per-row domain projection; dream rows detached when side_dream=False"""
    z=model.side_in(taps[0].float())
    for i,(t_,P_,M_) in enumerate(zip(taps[1:],model.side_tap,model.side_mlp)):
        z=z+M_(z+P_(t_.float()))
    B=z.shape[0]
    if isinstance(dom,int): o=z@model.side_out[dom]
    else:
        o=torch.zeros(*z.shape[:-1],model.side_out.shape[-1],device=z.device,dtype=z.dtype)
        for d in sorted({e for e in dom if e is not None}):
            ix=torch.tensor([j for j,e in enumerate(dom) if e==d],device=z.device); o=o.index_copy(0,ix,z[ix]@model.side_out[d])
    return o
def priv(model,x,h,d):
    """features + the domain's private tables (n-gram rows and feature-coded rows)"""
    cfg=model.cfg
    ng_=ngp_vec(model,x,d)
    if torch.is_tensor(ng_) and ng_.dim()==3 and ng_.shape[1]!=h.shape[1]: ng_=ng_[:,-h.shape[1]:]   # (incremental generation: newest positions)
    h=h+ng_
    return h
def cg_apply(b,m,ns_):
    """conflict-driven context-keyed gate (Config.cgate): gate rows by their domain's gate vector; returns (m, backward hook or None)"""
    G=b._cg_g; H_=m.shape[-1]
    if G.device!=m.device: b.__dict__['_cg_g']=G=G.to(m.device); b.__dict__['_cg_c']=b._cg_c.to(m.device)
    if G.shape[1]!=H_: return m,None
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return m,None
    cur=_XDG['cur'] if _XDG['cur'] is not None else 0
    if isinstance(tag,(list,tuple)):
        if len(tag)!=m.shape[0]: return m,None
        ix=[int(d) if d is not None else int(cur) for d in tag]; g=G[torch.tensor(ix,device=G.device)][:,None,:]
    else: ix=None; g=G[int(tag)]
    m=m*g.to(m.dtype)
    hook=None
    if b.training and torch.is_grad_enabled() and ix is not None and ns_ is not None and 0<ns_<m.shape[0]:
        b._cg_n=getattr(b,'_cg_n',0)+1
        if b._cg_n%b._cg_every==0:
            md=m.detach(); olds=sorted(set(d for d in ix[ns_:] if d!=cur))
            if olds:
                def hook(gr,md=md,ix=ix,ns_=ns_,olds=olds,cur=int(cur)):
                    with torch.no_grad():
                        gr=gr.float(); mf=md.float(); Gr=torch.einsum('btd,bth->dh',gr[:ns_],mf[:ns_]); nr=Gr.norm(dim=0)
                        rows=torch.tensor(ix,device=gr.device)
                        for d in olds:
                            sel=(rows==d).nonzero().flatten(); sel=sel[sel>=ns_]
                            if sel.numel()==0: continue
                            Gd=torch.einsum('btd,bth->dh',gr[sel],mf[sel]); c=-(Gr*Gd).sum(0)/(nr*Gd.norm(dim=0)+1e-12)
                            b._cg_c[d]=(1-b._cg_rho)*b._cg_c[d]+b._cg_rho*c
                        cm=b._cg_c.max(0).values; cl=((cm-b._cg_thr)/(b._cg_sc if b._cg_sc>0 else 1-b._cg_thr)).clamp(0,1)
                        b._cg_g[cur]=1-b._cg_k*cl
                    return None
    return m,hook
def inh_apply(b,m,hn,ns_):
    """learned inhibitory gate on the MLP hidden units m (see Config.inh_rank)"""
    A,B_,c=b.inh_A,b.inh_B,b.inh_c; H_=m.shape[-1]
    if B_.shape[1]!=H_: return m                                          # widened block: gate not resized (not used with widening)
    hd=hn.to(A.dtype); tr=b.training and torch.is_grad_enabled()
    split=tr and b._inh_mode=='dream' and ns_ is not None and 0<ns_<m.shape[0]
    if tr and b._inh_mode=='dream' and not split: z=(hd@A.detach())@B_.detach()+c.detach()        # no dream rows: the gate does not learn
    elif split: z=torch.cat([(hd[:ns_]@A.detach())@B_.detach()+c.detach(),(hd[ns_:]@A)@B_+c],0)
    else: z=(hd@A)@B_+c
    i_=torch.sigmoid(z.float())
    if tr and b._inh_open>0:                                              # openness penalty (gate parameters only)
        ip_=torch.sigmoid(((hd.detach()@A)@B_+c).float()) if b._inh_mode=='dream' else i_
        b.__dict__['_inh_pen']=b._inh_open*ip_.mean()
    if b.training and ns_ is not None and 0<ns_<m.shape[0]:
        with torch.no_grad():
            ir=i_[:ns_].mean((0,1)); idr=i_[ns_:].mean((0,1)); o_=b.__dict__.get('_inh_stat')
            st=torch.stack([ir,idr]); b.__dict__['_inh_stat']=st if o_ is None else 0.98*o_+0.02*st
    return m*(1-i_).to(m.dtype)
_NGPD={'on':False,'m':None}                                           # conduit student pass: private n-gram rows read but not trained
def _ngp_look(E,ix,m_):
    """v8b: rows looked up where m_ (B,T) is False are read DETACHED; only teaching positions' rows get a (sparse) gradient"""
    if m_ is None or tuple(m_.shape)!=tuple(ix.shape[:2]): return E(ix)
    v=F.embedding(ix,E.weight.detach()); sel=m_.nonzero(as_tuple=True)
    if len(sel[0])==0: return v
    v=v.clone(); v[sel]=E(ix[sel]); return v
def ngp_vec(model,x,d):
    T=model.__dict__.get('_ngp')
    if T is None: return 0.0
    cols=_cols(x,max(model.cfg.ngp)); out=0.0; m_=_NGPD.get('m')
    for n,E in zip(model.cfg.ngp,T[d]): out=out+_ngp_look(E,_ngram_idx(cols,n,E.num_embeddings,5),m_)
    O=model.__dict__.get('_ngo')                                      # SHARED OUTPUT-SIDE n-gram rows (every domain, before its head)
    if O is not None:
        for (n,j),E in O: out=out+_ngp_look(E,_ngram_idx(cols,n,E.num_embeddings,7+j),m_)
    return out.detach() if (_NGPD['on'] and torch.is_tensor(out)) else out
@torch.no_grad()
@torch.no_grad()
def _slot_bpc(model,xs,ys,s):
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    v=float(np.mean([float(F.cross_entropy(model(x_,s).reshape(-1,256),y_.reshape(-1))) for x_,y_ in zip(xs,ys)]))/LOG2
    model.feat_scale.copy_(fs); model.train(was); return v
class _ReplayGet:
    """Wraps a visit's batch stream: the batches already drawn to identify the context are trained on first (no text is skipped, so
    the stream is the same as with labels), and the last n batches are kept (the slot's reference BPC is measured on them)."""
    def __init__(s,get,first=(),n=4): s._g=get; s._q=list(first); s.last=__import__('collections').deque(maxlen=max(1,int(n)))
    def __call__(s):
        b=s._q.pop(0) if s._q else s._g(); s.last.append(b); return b
    def __getattr__(s,k): return getattr(s._g,k)
class _StreamSource:
    """NO VISIT BOUNDARIES: the visit schedule as one stream of batches (the true domain is known here for logging only)."""
    def __init__(s,schedule,train_of,cfg,ndom):
        s.sch=list(schedule); s.tr=train_of; s.L=int(cfg.steps_per_task); s.nd=ndom; s.pos=0; s.vi=0; s.k=0; s.bounds=[i*s.L for i in range(1,len(s.sch))]
    @property
    def done(s): return s.vi>=len(s.sch)
    @property
    def true_dom(s): return s.sch[min(s.vi,len(s.sch)-1)]
    def __call__(s):
        t=s.sch[s.vi]; b=s.tr(t,s.vi//s.nd)(); s.pos+=1; s.k+=1
        if s.k>=s.L: s.k=0; s.vi+=1
        return b
class _StreamGet:
    """The visit's batches from the stream, with change detection on the previous step's real-batch loss (bits)."""
    def __init__(s,src,model,cfg,slot,vi,first=()):
        s.src=src; s.m=model; s.cfg=cfg; s.slot=slot; s.vi=vi; s._q=list(first); s.last=__import__('collections').deque(maxlen=max(1,int(cfg.infer_batches)))
        s.pending=None; s.ref_batches=None; s.w=__import__('collections').deque(maxlen=max(4,int(cfg.stream_win))); s.cs=0.0; s.n=0; s.log=[]; s.mu=0.0
        s.rw=__import__('collections').deque(maxlen=max(8,4*int(cfg.stream_win))); s.rn=0; s.rz=0.0; s.rzmax=0.0   # cg15 representation detector
        if getattr(cfg,'stream_rep',False): model.__dict__['_srep']=None; model.__dict__.pop('_rj',None)            #   running mean restarts with the visit
    def _alarm_rep(s,r):
        s.rn+=1; c=s.cfg
        if s.rn<=int(c.stream_warm) or len(s.rw)<3: s.rw.append(r); return False                # statistics only
        w=np.array(s.rw); med=float(np.median(w)); sd=max(float(c.stream_rep_floor),1.4826*float(np.median(np.abs(w-med)))); z=(r-med)/sd
        if z>float(c.stream_rep_z): s.rz=z; return True
        s.rzmax=max(s.rzmax,z); s.rw.append(r); return False
    def _sd(s):
        w=list(s.w)
        if len(w)<3: return float(s.cfg.stream_sig_floor)
        return max(float(s.cfg.stream_sig_floor),1.4826*float(np.median(np.abs(np.diff(w))))/math.sqrt(2.0))
    def _alarm(s,l):
        s.n+=1; c=s.cfg
        if s.n<=int(c.stream_warm) or len(s.w)<3:
            s.w.append(l)
            if c.stream_trace: s.m.__dict__.setdefault('_strace',[]).append((s.src.pos,l,float('nan'),float('nan'),float('nan'),0.0))
            return False                                                                       # statistics only
        s.mu=float(np.median(list(s.w))); sd=s._sd(); z=(l-s.mu)/sd; s.cs=max(0.0,s.cs+z-float(c.stream_k))
        if c.stream_trace: s.m.__dict__.setdefault('_strace',[]).append((s.src.pos,l,s.mu,sd,z,s.cs))
        if s.cs>float(c.stream_h): s.cs=0.0; return True
        if z<float(c.stream_k)+2.0: s.w.append(l)                                              # in-control points update the statistics
        return False
    def __call__(s):
        if s._q: b=s._q.pop(0)
        else:
            l=s.m.__dict__.pop('_rl',None); al=s._alarm(float(l)/LOG2) if l is not None else False
            if getattr(s.cfg,'stream_rep',False):                                                # cg15: the representation jump decides (loss stats kept for the log)
                r_=s.m.__dict__.pop('_rj',None); ar_=s._alarm_rep(r_) if r_ is not None else False
                al=ar_ or (al and bool(getattr(s.cfg,'stream_rep_or',True)))                        #   OR the loss CUSUM (GPU cdet logs, 92 switches: jump 91,
                                                                                                 #   loss 69, either 92; the local model: loss in pass 1, jump in pass 2)
            if al and not s.src.done:
                p0=s.src.pos; th=s.mu+s._sd()
                if getattr(s.cfg,'stream_rep',False): slot_ref(s.m,s.cfg,list(s.last)[:-1] or list(s.last),s.slot)   # cg15: the usual reference (BPC on
                else: s.m.__dict__.setdefault('_slot_ref',{})[s.slot]=s.mu                       #   this visit's last batches, not the new text)
                                                                                                 # (loss mode: the current slot's reference = its recent loss level)
                                                                                                 #   (losses before each update: out of sample)
                sl,bs=infer_slot(s.m,s.cfg,s.src,s.src.true_dom,s.vi)
                s.log.append((p0,float(l)/LOG2,th,sl))
                if sl==s.slot: s._q=list(bs); b=s._q.pop(0)                                       # false alarm: same slot, keep going
                else:                                                                             # switch: one last step on this visit's own text
                    s.pending=(sl,bs); s.ref_batches=list(s.last)[:-1] or list(s.last); b=s.ref_batches[-1]; s.m.__dict__['_vend']=True
                    s.last.append(b); return b
            elif s.src.done: b=s.last[-1]; s.m.__dict__['_vend']=True; s.last.append(b); return b
            else: b=s.src()
        if s.src.done and not s._q: s.m.__dict__['_vend']=True
        s.last.append(b); return b
@torch.no_grad()
def _slot_score(model,xs,ys,s):
    """(BPC, mean top-block output) of the batches read through slot s (its codes, private units and head)"""
    was=model.training; model.eval(); fs=model.feat_scale.clone(); sv_=model.__dict__.get('_s0_dom'); v=0.0; m=0.0
    try:
        for x_,y_ in zip(xs,ys):
            model.__dict__['_s0_dom']=int(s); h_=model.features(x_); model.__dict__['_s0_dom']=sv_
            v+=float(F.cross_entropy(model.heads[s](priv(model,x_,h_,int(s))).float().reshape(-1,256),y_.reshape(-1))); m=m+h_.float().mean((0,1))
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs); model.train(was)
    return v/len(xs)/LOG2, m/len(xs)
def infer_slot_rep(model,cfg,get,t_true,vi):
    """cg9: representation candidate + loss check (see Config.infer_rep). Logged against the true domain (diagnostic only)."""
    S=model.__dict__.setdefault('_slots',[]); R=model.__dict__.setdefault('_slot_ref',{}); L=model.__dict__.setdefault('_slot_log',[])
    PR=model.__dict__.setdefault('_slot_proto',{}); TM=model.__dict__.setdefault('_slot_truth',{})
    if not S:
        S.append(0); L.append((vi,t_true,0,'new',None)); TM.setdefault(t_true,0)
        print(f"  [routing] visit {vi}: first context -> slot 0 (true {_lab(cfg,t_true)})"); return 0,[]
    b_=[get() for _ in range(cfg.infer_batches)]; xs=[a for a,_ in b_]; ys=[c for _,c in b_]
    sc={}; ds={}; BK=model.__dict__.get('_bank',{}); dev=model.embed.weight.device
    for s_ in S:
        bp_,m_=_slot_score(model,xs,ys,s_); sc[s_]=bp_
        if s_ in BK:                                                      # FRESH PROTOTYPE: the slot's own dream-bank rows read NOW (a stored
            Xb_=BK[s_][:16,:xs[0].shape[1]].long().to(dev)                 #   prototype goes stale as the trunk drifts: cg9 CPU check, the
            _,pm_=_slot_score(model,[Xb_],[torch.zeros_like(Xb_)],s_)      #   newest slot was always nearest)
            ds[s_]=float(1-F.cosine_similarity(m_,pm_,dim=0))
        else: ds[s_]=float('inf')
    cand=min(sc,key=sc.get); rnear=min(ds,key=ds.get)                     # candidate = lowest BPC; the representation must agree
    gap=sc[cand]-R.get(cand,sc[cand]); loss_ok=gap<=cfg.infer_delta; rep_ok=(rnear==cand) or not any(v_<float('inf') for v_ in ds.values())
    second=sorted(ds.values())[1] if len(ds)>1 else float('inf')
    if len(S)>=len(PILE_DOMAINS) and float(cfg.infer_gap_eps)>0 and all(s_ in R for s_ in S):   # cg13 GAP RULE (slots full): among slots within
        lo_=min(sc.values()); nr_={s_:sc[s_]-R[s_] for s_ in S if sc[s_]<=lo_+float(cfg.infer_gap_eps)}   #   infer_gap_eps of the lowest BPC, the one
        s_=min(nr_,key=nr_.get); kind='revisit (slots full, smallest gap to own reference)'             #   fitting the text best RELATIVE TO ITS OWN
        if (ds.get(rnear,float('inf'))<=cfg.infer_margin*second and (sc[rnear]-lo_)<float(cfg.infer_proto_guard) and rnear!=s_   #   reference (recency-biased slots
                and (float(cfg.infer_proto_gap)<=0 or (sc[rnear]-R.get(rnear,sc[rnear]))<=float(cfg.infer_proto_gap))):   # cg14d: the prototype's slot must fit its
            s_=rnear; kind='revisit (slots full, prototype clear)'                                     #   score low BPC on everything); clear prototype
        gap=sc[s_]-R.get(s_,sc[s_])                                           #   override kept; replayed on cg9 / cg10 routing: 10/10 and 10/10
    elif len(S)>=len(PILE_DOMAINS) and not (rep_ok and loss_ok):           # cg10 SLOTS FULL: the lowest-BPC slot is biased towards the slot trained
        lo2_=sorted(sc,key=sc.get)[:2]                                    #   most recently (cg9: Gutenberg -> FreeLaw, Github -> StackExchange,
        clear_=ds[rnear]<=cfg.infer_margin*second and (sc[rnear]-sc[cand])<cfg.infer_delta                         #   the fresh prototype was right both times): follow the prototype when
        close_=bool(cfg.infer_tiebreak) and len(lo2_)>1 and (sc[lo2_[1]]-sc[lo2_[0]])<0.1 and rnear in lo2_   #   it is clearly nearest, or when the two lowest BPCs are within
        s_=rnear if (clear_ or close_) else cand                          #   0.1 bits and it is one of them; else the lowest BPC
        # cg10 guard: a clear prototype overrides only when its slot's BPC is within infer_delta of the lowest (cg10 CPU: prototypes 0.0005 vs 0.0012
        #   apart but 0.69 bits worse -> wrong); replayed on cg9 GPU routing: 10/10 slots-full revisits right (was 8/10), CPU 5/5 (was 4/5)
        kind='revisit (slots full, '+('prototype clear' if clear_ else 'prototype tie-break' if close_ else 'lowest BPC')+')'
        if s_!=cand: gap=sc[s_]-R.get(s_,sc[s_])
    elif rep_ok and gap<=(float(cfg.infer_delta_new) if float(cfg.infer_delta_new)>0 else cfg.infer_delta): s_=cand; kind='revisit'   # cg13c free-slot threshold
    else: s_=len(S); S.append(s_); kind='new'
    exp_=TM.get(t_true); ok_=(s_==exp_) if exp_ is not None else (kind=='new')
    if exp_ is None: TM[t_true]=s_
    A=model.__dict__.setdefault('_route_acc',[0,0]); A[0]+=int(ok_); A[1]+=1
    L.append((vi,t_true,s_,kind,round(gap,3)))
    print(f"  [routing] visit {vi}: lowest-BPC slot {cand}, nearest fresh prototype slot {rnear} (agree {rep_ok}; distance {ds[rnear]:.4f}, second {second:.4f}) | BPC {sc[cand]:.3f} vs reference "
          f"{R.get(cand,float('nan')):.3f} (gap {gap:+.3f}, ok {loss_ok}) -> {kind} slot {s_} | true {_lab(cfg,t_true)}: {'CORRECT' if ok_ else 'WRONG'} "
          f"(running {A[0]}/{A[1]}) | distances "+' '.join(f"s{k}:{v:.3f}" for k,v in ds.items())+" | BPC "+' '.join(f"s{k}:{v:.2f}" for k,v in sc.items()))
    return s_,b_
def infer_slot(model,cfg,get,t_true,vi):
    if getattr(cfg,'infer_rep',False): return infer_slot_rep(model,cfg,get,t_true,vi)
    """LABEL-FREE CONTEXT: (slot, batches drawn to decide it). The batches are returned so the visit trains on them (stream unchanged)."""
    S=model.__dict__.setdefault('_slots',[]); R=model.__dict__.setdefault('_slot_ref',{}); L=model.__dict__.setdefault('_slot_log',[])
    if not S:
        S.append(0); L.append((vi,t_true,0,'new',None)); print(f"  [context] visit {vi}: first context -> slot 0 (true {_lab(cfg,t_true)})"); return 0,[]
    b_=[get() for _ in range(cfg.infer_batches)]; xs=[a for a,_ in b_]; ys=[c for _,c in b_]
    sc={s_:_slot_bpc(model,xs,ys,s_) for s_ in S}; best=min(sc,key=sc.get); gap=sc[best]-R.get(best,sc[best])
    if gap<=cfg.infer_delta or len(S)>=len(PILE_DOMAINS): s_=best; kind='revisit'
    else: s_=len(S); S.append(s_); kind='new'
    L.append((vi,t_true,s_,kind,round(gap,3)))
    print(f"  [context] visit {vi}: best slot {best} (BPC {sc[best]:.3f}, reference {R.get(best,float('nan')):.3f}, gap {gap:+.3f}) -> {kind} slot {s_} (true {_lab(cfg,t_true)}) | "
          +' '.join(f"s{k}:{v:.2f}" for k,v in sc.items()))
    return s_,b_
def slot_ref(model,cfg,b_,s):
    """The slot's reference BPC at the visit end, on the visit's last infer_batches training batches (no extra text drawn)."""
    b_=list(b_)
    if getattr(cfg,'infer_rep',False):                                    # cg9: reference BPC and representation PROTOTYPE in one pass
        bp_,m_=_slot_score(model,[a for a,_ in b_],[c for _,c in b_],s)
        model.__dict__.setdefault('_slot_ref',{})[s]=bp_; model.__dict__.setdefault('_slot_proto',{})[s]=m_.detach(); return
    model.__dict__.setdefault('_slot_ref',{})[s]=_slot_bpc(model,[a for a,_ in b_],[c for _,c in b_],s)
def _ema_params(model):
    P=list(model.parameters())
    for k_ in ('_ngp','_ngpi'):
        T=model.__dict__.get(k_)
        if T: P+=[E.weight for d in sorted(T) for E in T[d]]
    return P
def ngp_step(model,cfg,lr):
    O3=model.__dict__.get('_ngpi_opt')
    if O3 is not None:
        for g_ in O3.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        O3.step(); O3.zero_grad(set_to_none=True)
    O=model.__dict__.get('_ngp_opt')
    if O is not None:
        for g_ in O.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        O.step(); O.zero_grad(set_to_none=True)
    O4=model.__dict__.get('_ngo_opt')
    if O4 is not None:
        for g_ in O4.param_groups: g_['lr']=lr*cfg.ngram_lr_mult*cfg.ngo_lr_mult
        O4.step(); O4.zero_grad(set_to_none=True)
    O2=model.__dict__.get('_fcp_opt')
    if O2 is not None:
        for g_ in O2.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        O2.step(); O2.zero_grad(set_to_none=True)
import types as _types
@torch.no_grad()
def cfm_fold(model,cfg,d,val):
    """memory -> head: the logits K M are projected onto the head's linear features [h,1] with the visit's key statistics G
    (X = M_l + G_ll^-1 G_l,rf M_rf), added to the head, and the memory is cleared"""
    C_=model._cfm
    if d not in C_['M']: return
    on_=bpc(model,val,d); M=C_['M'].pop(d); off_=bpc(model,val,d); G=C_['G'][d].double(); dm=model.heads[d].weight.shape[1]; l=dm+1
    Gll=G[:l,:l]+1e-6*float(torch.diagonal(G[:l,:l]).mean())*torch.eye(l,device=G.device,dtype=G.dtype)
    X=(M[:l].double()+torch.linalg.solve(Gll,G[:l,l:]@M[l:].double())).float()
    model.heads[d].weight.add_(X[:dm].t()); model.heads[d].bias.add_(X[dm]); fo_=bpc(model,val,d)
    for k_ in ('G','KE','n','w'): C_[k_].pop(d,None)
    model.train()
    print(f"  [closed-form fold] {_lab(cfg,d)}: BPC memory on {on_:.4f} | off {off_:.4f} (memory effect {on_-off_:+.4f}) | folded into the head {fo_:.4f}")
class Head(nn.Linear):
    """Output head; optional logit SOFT-CAP (cap*tanh(z/cap)) bounds how extreme any prediction can get."""
    cap=0.0
    def _c(s,z): return s.cap*torch.tanh(z/s.cap) if s.cap>0 else z
    def forward(s,x): return s._c(super().forward(x))
    def detached(s,x): return s._c(F.linear(x,s.weight.detach(),s.bias.detach()))   # same head, no gradient to its weights

# ---------------- DELTA-RULE CORE: torch WY solve, or the Triton chunked kernel (flash-linear-attention) ----------------
_FLA={'fn':None,'tried':False,'err':None,'dtype':torch.float32,'checked':{}}
def _fla_get():
    """chunk_gated_delta_rule from flash-linear-attention (pip install flash-linear-attention), or None."""
    if not _FLA['tried']:
        _FLA['tried']=True
        for attempt in (0,1):
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f_; _FLA['fn']=f_; _FLA['err']=None; break
            except Exception as e:
                _FLA['err']=repr(e)[:300]
                if attempt==0 and isinstance(e,ModuleNotFoundError) and torch.cuda.is_available() and os.environ.get('FLA_AUTO_INSTALL','1')!='0':
                    import subprocess, importlib, sys as _sys    # kernels only (fla-core, no dependencies: never touches torch/triton)
                    print("  [kernel] installing the Triton kernels: pip install --no-deps fla-core einops ...", flush=True)
                    subprocess.run([_sys.executable,'-m','pip','install','-q','--no-deps','fla-core','einops'],check=False)
                    importlib.invalidate_caches()
                else: break
    return _FLA['fn']

_WIN={'W':0}                   # windowed WY solve (mix_window); set by setup_kernel
_SAMP={'W':0,'bf16':False,'rescore':True}     # windowed recurrent sampler (samp_window, samp_bf16, samp_rescore)
_ADPT={'on':False,'P':{}}                                                     # LOW-RANK MLP ADAPTER (oracle): x += B (A hn) beside each block's MLP
_PMEM={'on':False,'S':{}}                                                     # PERSISTENT MEMORY READ (oracle): out += S_b k~ at every position
_ORACLE={'on':False,'st':{}}                                                  # ORACLE MEMORY statistics collection
_BPTT={'K':1}                                                                 # segments per training row (ctx_bptt; set per visit)
_RC_GRAD={'on':True}   # replay training: the end-of-segment memory stays differentiable (False = the old detached behaviour)
class _SlotF:
    """v9.8 cg5 SLOT MEMORY of one delta block for nb = rows x heads: S = sum_s exp(Gt - Gs[s]) U[s]^T K[s] (factor slots; unused slots U = 0).
    Writes go to slot i (in place); Gt is the running log decay, Gs the log decay at each slot's write (so old slots are never rescaled)."""
    __slots__=('U','K','Gs','Gt','r0','i')
    def __init__(s,U,K,Gs,Gt,r0,i=0): s.U=U; s.K=K; s.Gs=Gs; s.Gt=Gt; s.r0=r0; s.i=i
_CARRY={'on':False,'nr':0,'st':{},'new':{},'sm':None,'slot':None,'rows':None}   # CONTINUOUS CONTEXT (ctx_carry): carried delta-rule memories of the real rows, per block
_PREC={'p':0.0,'C':0,'det':False}   # preconditioned delta-rule writes (mix_precond, mix_precond_chunk, mix_precond_detach); set by setup_kernel
def _smem_seed(slot,nr):
    """slot memory -> per-row initial states (each block's S_d repeated over the rows)"""
    SM=(_CARRY['sm'] or {}).get(slot) or {}
    return {k_:v_.unsqueeze(0).expand(int(nr),*v_.shape).reshape(int(nr)*v_.shape[0],*v_.shape[1:]) for k_,v_ in SM.items()}
def _pair_on(model,cfg,nreal):
    """PAIRED-DREAM EPISODES: the memory the current network writes while reading each paired row's first half A (no gradient) becomes the
    initial state of that row's second half B in the merged pass (alongside any real-row memories)."""
    pa_=model.__dict__.pop('_pairA',None)
    if not pa_ or not pa_[0]: return
    pj_,XA_,dA_=pa_; was=model.training; fs=model.feat_scale.clone(); s0_=model.__dict__.get('_s0_dom')
    sv_={k_:_CARRY[k_] for k_ in ('on','nr','st','new','rows')}
    try:
        model.eval(); _CARRY.update(on=True,nr=len(pj_),st={},new={},rows=None); model.__dict__['_s0_dom']=list(dA_)
        with torch.no_grad(): model.features(XA_)
        stA=dict(_CARRY['new'])                                               # memories (+ preconditioner statistics) after A
    finally:
        model.train(was); model.feat_scale.copy_(fs); model.__dict__['_s0_dom']=s0_; _CARRY.update(sv_)
    rp_=torch.tensor([int(nreal)+j for j in pj_],dtype=torch.long)
    if _CARRY['on']:                                                       # real rows carry memories too: stack (preconditioner statistics dropped)
        nr_=int(_CARRY['nr']); _CARRY['rows']=torch.cat([torch.arange(nr_),rp_]); stA={k_:v_ for k_,v_ in stA.items() if not isinstance(k_,tuple)}
        _CARRY['st']={k_:torch.cat([_CARRY['st'][k_] if (k_ in _CARRY['st'] and _CARRY['st'][k_].shape[0]==nr_*v_.shape[0]//len(pj_)) else torch.zeros(nr_*v_.shape[0]//len(pj_),*v_.shape[1:],device=v_.device),v_]) for k_,v_ in stA.items()}
    else: _CARRY.update(on=True,nr=0,new={},rows=rp_,st=stA)
    st_=model.__dict__.setdefault('_pairst',[0,0]); st_[0]+=len(pj_); st_[1]+=1
def _carry_on(cfg,nr):
    if cfg.ctx_carry or cfg.slot_mem:
        _CARRY['on']=True; _CARRY['nr']=int(nr); _CARRY['new']={}
        if cfg.slot_mem and not cfg.ctx_carry: _CARRY['st']=_smem_seed(_CARRY['slot'],nr)
def _carry_off(cfg):
    if _CARRY['on']:
        _CARRY['on']=False
        if _CARRY.get('rows') is not None:                                # paired-dream rows rode along: keep only the real rows' memories
            ns_=int(_CARRY['rows'].numel()); nr_=int(_CARRY['nr'])
            _CARRY['new']={k_:v_[:(v_.shape[0]//ns_)*nr_] for k_,v_ in _CARRY['new'].items()} if nr_>0 else {}
            _CARRY['rows']=None
        if cfg.slot_mem and _CARRY['sm'] is not None and _CARRY['slot'] is not None:      # S_d <- g S_d + (1 - g) mean of the segment-end states
            SM=_CARRY['sm'].setdefault(_CARRY['slot'],{}); g_=float(cfg.slot_mem_gamma)
            with torch.no_grad():
                for k_,v_ in _CARRY['new'].items():
                    if isinstance(k_,tuple): continue
                    nr_=int(_CARRY['nr']); m_=v_.reshape(nr_,v_.shape[0]//nr_,*v_.shape[1:]).mean(0)
                    o_=SM.get(k_); SM[k_]=m_ if (o_ is None or o_.shape!=m_.shape) else g_*o_+(1-g_)*m_
            _CARRY['st']=_CARRY['new'] if cfg.ctx_carry else {}
        else: _CARRY['st']=_CARRY['new']
        _CARRY['new']={}
def _kw_par(k):
    if _PREC['C']>0: return _kw_par_blk(k)
    """Write keys of the preconditioned delta rule for normalised keys k (BH,T,dh): causal running mean of k^2 per feature -> diagonal
    preconditioner D = (c dh + 0.1)^-p in [0.25, 4]; k_w = D k / (k . D k) (so k_w . k = 1)."""
    dh=k.shape[-1]; c=torch.cumsum(k*k,1)/torch.arange(1,k.shape[1]+1,device=k.device,dtype=k.dtype)[None,:,None]
    D=(c*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); kw=D*k
    return kw/(kw*k).sum(-1,keepdim=True).clamp(min=1e-6)
def _kw_par_blk(k):
    """Block-level preconditioner (mix_precond_chunk = C): D of block c from the mean k^2 over all positions of blocks < c (block 0: D = 1);
    normaliser n_t = sum_i D_i k_ti^2 per token. With mix_precond_detach, D and n carry no gradient (k_w = k * D / n is linear in k)."""
    BH,T,dh=k.shape; C=int(_PREC['C'])
    if T<C: C=T                                                                                        # v9c: one (partial) block - no padding (exact)
    nc=-(-T//C); pad=nc*C-T
    k2=k*k; k2p=F.pad(k2,(0,0,0,pad)) if pad else k2
    with torch.set_grad_enabled(torch.is_grad_enabled() and not _PREC['det']):
        k2b=k2p.reshape(BH,nc,C,dh); bs=k2b.sum(2)                                                     # block sums (BH,nc,dh)
        pre=torch.cumsum(bs,1)-bs                                                                      # sums over earlier blocks
        cnt=(torch.arange(nc,device=k.device,dtype=k.dtype)*C)[None,:,None].expand(BH,nc,1)
        pr_=_PREC.get('prior')                                                                         # CONTINUOUS CONTEXT: statistics of the
        if pr_ is not None:                                                                            #   earlier segments (first nb rows)
            P0_,n0_,nb_=pr_
            if torch.is_tensor(nb_): pre=pre.clone(); pre[nb_]+=P0_[:,None,:]; cnt=cnt.clone(); cnt[nb_]+=n0_      # selected rows (paired dreams)
            else: pre=torch.cat([pre[:nb_]+P0_[:,None,:],pre[nb_:]],0); cnt=torch.cat([cnt[:nb_]+n0_,cnt[nb_:]],0)
        D=(pre/cnt.clamp(min=1)*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); D=torch.where(cnt>0,D,torch.ones_like(D))   # no history: D = 1
        n=torch.bmm(k2b.reshape(BH*nc,C,dh),D.reshape(BH*nc,dh,1)).reshape(BH,nc*C)[:,:T].clamp(min=1e-6)   # k . D k per token
        if _PREC['det']: D=D.detach(); n=n.detach()
    kb=F.pad(k,(0,0,0,pad)) if pad else k
    return (kb.reshape(BH,nc,C,dh)*D[:,:,None,:]).reshape(BH,nc*C,dh)[:,:T]/n[...,None]
def _kw_rec(st,kr):
    """The same write key in the recurrent samplers (running sums kept in the sampler state)."""
    if not _PREC['p']: return kr
    if _PREC['C']>0:                                                    # block-level statistics (as _kw_par_blk)
        C=int(_PREC['C']); dh=kr.shape[-1]; n0=st.get('kn',0)
        if n0>0 and n0%C==0: st['kD']=(st['kc']/n0*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0)
        D=st.get('kD'); kw=kr if D is None else D*kr
        st['kc']=kr*kr if 'kc' not in st else st['kc']+kr*kr; st['kn']=n0+1
        return kw if D is None else kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)
    dh=kr.shape[-1]; st['kc']=kr*kr if 'kc' not in st else st['kc']+kr*kr; st['kn']=st.get('kn',0)+1
    D=(st['kc']/st['kn']*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); kw=D*kr
    return kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)

def _local_attn(k, v, mh, B, T, la_scale, la_rel, W):
    """Chunked sliding-window attention (fp32): position t reads positions t-W..t-1 (and a null slot)."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        C=k.shape[-1]; dh=C//mh; BH=B*mh
        sp=lambda z: z.float().reshape(B,T,mh,dh).transpose(1,2).reshape(BH,T,dh)
        kn=F.normalize(sp(k),dim=-1); vv=sp(v); n_=-(-T//W); Tp=n_*W
        kp=F.pad(kn,(0,0,W,Tp-T)); vp=F.pad(vv,(0,0,W,Tp-T))
        Kc=kp.unfold(1,2*W,W).permute(0,1,3,2); Vc=vp.unfold(1,2*W,W).permute(0,1,3,2)          # (BH,n,2W,dh)
        Qc=F.pad(kn,(0,0,0,Tp-T)).reshape(BH,n_,W,dh)
        sc=torch.exp(la_scale.float()).repeat(B)                                                  # (BH,)
        S=torch.einsum('bnqd,bnkd->bnqk',Qc,Kc)*sc[:,None,None,None]
        a_=torch.arange(W,device=k.device)[:,None]; b_=torch.arange(2*W,device=k.device)[None,:]; off=W+a_-b_
        ok=(off>=1)&(off<=W)
        kabs=torch.arange(n_,device=k.device)[:,None]*W-W+torch.arange(2*W,device=k.device)[None,:]       # (n,2W) absolute key position
        mask=ok[None,:,:]&(kabs>=0)[:,None,:]                                                     # (n,W,2W)
        bias=la_rel.float().repeat(B,1)[:,(off-1).clamp(0,W-1)]                                    # (BH,W,2W)
        S=(S+bias[:,None]).masked_fill(~mask[None],float('-inf'))
        Pm=F.softmax(torch.cat([S,torch.zeros_like(S[...,:1])],-1),-1)[...,:2*W]
        out=torch.einsum('bnqk,bnkd->bnqd',Pm,Vc).reshape(BH,Tp,dh)[:,:T]
        return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)

_CHUNK={'C':0,'lp':None,'Cng':0,'B':None,'Bng':None}      # Cng: chunk for no-grad passes (bank targets: large batches, forward only)
# ---- BLOCKED FULL-WY SOLVE (tensor-core GEMMs, no d x d states, no T x T tensors) ----
_BLK={'prec':'tf32','emu':False}      # blocked solve: GEMM precision ('fp32' | 'tf32' | 'bf16'); emu = TF32 rounding emulated (CPU tests)
def _tf32_round(x):
    i=x.contiguous().view(torch.int32); return (i & ~0x1FFF).view(torch.float32)
def _mm(a,b):
    """bmm in the solve's GEMM precision: fp32 | tf32 (tensor cores, fp32 in/out) | bf16 (tensor cores, fp32 out via upcast)"""
    p=_BLK['prec']
    if p=='bf16': return torch.bmm(a.bfloat16(),b.bfloat16()).float()
    if p=='tf32' and _BLK['emu']: return torch.bmm(_tf32_round(a.float()),_tf32_round(b.float()))
    if p=='tf32' and a.is_cuda:                                        # TF32 for THIS product only (the small diagonal-block products stay fp32)
        pv_=torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32=True
        try: return torch.bmm(a,b)
        finally: torch.backends.cuda.matmul.allow_tf32=pv_
    return torch.bmm(a,b)
class _DeltaBlocked(torch.autograd.Function):
    """Delta rule with decay, read before write, as the FULL WY system (I+M) r = v, out = v - r, M_ts = beta_s exp(g_t-g_s) k_t.k_s (s<t),
    solved by BLOCKED substitution: diagonal blocks inverted in fp32 (small), every off-diagonal coupling a GEMM on the fly (decays folded into
    per-block scale vectors, all <= 1). No d x d states, no T x T tensors, no explicit inverse applied to large values (the precision-sensitive
    step of the chunk form). k (BH,T,dh) normalised, v (BH,T,dh), be (BH,T), g (BH,T) cumulative log decay; T a multiple of b."""
    @staticmethod
    def forward(ctx, k, v, be, g, b, kw=None):
        BH,T,dh=k.shape; n=T//b; dev=k.device; ctx.sep=kw is not None
        kw=k if kw is None else kw                                                                   # WRITE keys (preconditioned delta rule): columns
        Kb=k.reshape(BH,n,b,dh); KWb=kw.reshape(BH,n,b,dh); gb=g.reshape(BH,n,b); bb=be.reshape(BH,n,b)
        dd=gb[...,:,None]-gb[...,None,:]; low=torch.tril(torch.ones(b,b,dtype=torch.bool,device=dev),-1)
        Md=torch.einsum('bntd,bnsd->bnts',Kb,KWb)*torch.exp(torch.where(low,dd,torch.full_like(dd,-float('inf'))))*bb[...,None,:]   # diagonal blocks (fp32)
        I=torch.eye(b,device=dev,dtype=k.dtype)
        Linv=torch.linalg.solve_triangular(I+Md.reshape(BH*n,b,b),I.expand(BH*n,b,b),upper=False).reshape(BH,n,b,b)
        r=torch.empty_like(v)
        for i in range(n):
            s0,s1=i*b,(i+1)*b; rhs=v[:,s0:s1]
            if i:
                P=_mm(k[:,s0:s1],kw[:,:s0].transpose(1,2))                                      # K_i KW_<i^T         (b x ib)
                e=torch.exp(g[:,s0:s1]-g[:,s0:s0+1]); w=torch.exp(g[:,s0:s0+1]-g[:,:s0])*be[:,:s0]   # <= 1, <= beta
                rhs=rhs-_mm(P.mul_(e[...,None]).mul_(w[:,None,:]),r[:,:s0])                     # - M_{i,<i} r_<i (scaled in place)
            r[:,s0:s1]=torch.bmm(Linv[:,i],rhs)                                                 # fp32: small b x b
        ctx.save_for_backward(k,be,g,r,Linv,kw); ctx.b=b
        return v-r
    @staticmethod
    def backward(ctx, dout):
        """lam = L^-T (-dout); dv = dout + lam; dM = -lam r^T (strictly lower) -> dA, dbeta, dg, dk. Products are formed as +lam r^T and the
        sign folded into the accumulations; decays and beta scale the (b x .) tiles in place; E = G*beta never formed (row / column sums)."""
        k,be,g,r,Linv,kw=ctx.saved_tensors; b=ctx.b; BH,T,dh=k.shape; n=T//b; dev=k.device
        dout=dout.contiguous(); lam=torch.empty_like(r)
        for i in range(n-1,-1,-1):                                                                  # backward substitution
            s0,s1=i*b,(i+1)*b; rhs=-dout[:,s0:s1]
            if i<n-1:
                gL=g[:,s1-1:s1]
                Q=_mm(kw[:,s0:s1],k[:,s1:].transpose(1,2))                                          # KW_i K_>i^T         (b x rest)
                Q.mul_((torch.exp(gL-g[:,s0:s1])*be[:,s0:s1])[...,None]).mul_(torch.exp(g[:,s1:]-gL)[:,None,:])   # = M_{>i,i}^T (decays <= 1)
                rhs=rhs-_mm(Q,lam[:,s1:])
            lam[:,s0:s1]=torch.bmm(Linv[:,i].transpose(1,2),rhs)
        dv=dout+lam
        # all diagonal blocks at once
        lb=lam.view(BH,n,b,dh); rb=r.view(BH,n,b,dh); Kb=k.view(BH,n,b,dh); KWb=kw.view(BH,n,b,dh); gb=g.view(BH,n,b); bb=be.view(BH,n,b)
        low=torch.tril(torch.ones(b,b,dtype=torch.bool,device=dev),-1); dd=gb[...,:,None]-gb[...,None,:]
        Rd=torch.einsum('bntd,bnsd->bnts',lb,rb).mul_(torch.exp(torch.where(low,dd,torch.full_like(dd,-float('inf')))))   # -dM o gamma
        Gd=torch.einsum('bntd,bnsd->bnts',Kb,KWb).mul_(Rd)                                          # -G
        Ad=Rd.mul_(bb[...,None,:])                                                                  # -dA (in place: Rd no longer needed)
        csd=Gd.sum(-2)
        dbe=(-csd).reshape(BH,T).contiguous()
        dg=((bb*csd)-(Gd@bb[...,None]).squeeze(-1)).reshape(BH,T).contiguous()
        if ctx.sep:                                                                                 # A_ts = k_t . kw_s: rows -> dk, columns -> dkw
            dk=(-(Ad@KWb)).reshape(BH,T,dh).contiguous(); dkw=(-(Ad.transpose(-1,-2)@Kb)).reshape(BH,T,dh).contiguous()
        else: dk=(-(Ad@Kb)-(Ad.transpose(-1,-2)@Kb)).reshape(BH,T,dh).contiguous(); dkw=dk
        del Rd,Gd,Ad,csd
        for i in range(1,n):                                                                        # off-diagonal row blocks (s < block start)
            s0,s1=i*b,(i+1)*b; ki=k[:,s0:s1]
            R=_mm(lam[:,s0:s1],r[:,:s0].transpose(1,2))                                             # -dM                 (b x ib)
            R.mul_(torch.exp(g[:,s0:s1]-g[:,s0:s0+1])[...,None]).mul_(torch.exp(g[:,s0:s0+1]-g[:,:s0])[:,None,:])
            P=_mm(ki,kw[:,:s0].transpose(1,2)).mul_(R)                                              # -G
            A_=R.mul_(be[:,None,:s0])                                                               # -dA
            cs=P.sum(1)
            dbe[:,:s0]-=cs; dg[:,:s0]+=be[:,:s0]*cs; dg[:,s0:s1]-=torch.bmm(P,be[:,:s0,None]).squeeze(-1)
            dk[:,s0:s1]-=_mm(A_,kw[:,:s0]); dkw[:,:s0]-=_mm(A_.transpose(1,2),ki)
        return dk,dv,dbe,dg,None,(dkw if ctx.sep else None)
_DC1={'on':True,'n':0}
def delta_blocked(k,v,be,la,b=64,kw=None):
    """k,v (BH,T,dh) (k normalised), be (BH,T), la (BH,T) log decays or None -> (BH,T,dh) = a_t S_{t-1} k_t"""
    BH,T,dh=k.shape
    if T<b: b=T                                                      # short sequences (replay segments): one block, no padding
    n=-(-T//b); pad=n*b-T
    g=torch.cumsum(la.float() if la is not None else torch.zeros(BH,T,device=k.device),-1)
    if pad: k=F.pad(k,(0,0,0,pad)); v=F.pad(v,(0,0,0,pad)); be=F.pad(be,(0,pad)); g=torch.cat([g,g[:,-1:].expand(BH,pad)],1)
    # padded positions: k=0 -> no coupling; g REPEATS the last value (zero padding made exp(g_t - g_s) overflow to inf once the decay over
    #   the block passed ~88 nats, and 0 x inf = NaN reached the real positions in the backward pass)
    if kw is not None and pad: kw=F.pad(kw,(0,0,0,pad))
    if kw is not None: return _DeltaBlocked.apply(k.float().contiguous(),v.float().contiguous(),be.float().contiguous(),g.contiguous(),b,kw.float().contiguous())[:,:T]
    return _DeltaBlocked.apply(k.float().contiguous(),v.float().contiguous(),be.float().contiguous(),g.contiguous(),b)[:,:T]

def _delta_chunked(k,v,beta,mb,la_,S0_,Cc,kw=None):
    """CHUNKED delta rule (same recurrence as the full T x T solve below, never builds T x T tensors): k, v (BH,T,dh) with k normalised;
    beta (BH,1,T) per token or a scalar tensor; la_ (BH,T) log decays or None; S0_ (BH,dh,dh) initial state (value x key) or None.
    Per chunk of Cc positions: r = (I+M_c)^-1 (v_c - decayed state readout), output v - r, state passed to the next chunk.
    Returns (BH,T,dh) = a_t S_{t-1} k_t."""
    BH,T,dh=k.shape; n=-(-T//Cc); Tp=n*Cc; pad=Tp-T
    be=(beta.reshape(BH,T) if mb else beta.reshape(1,1).expand(BH,T)).float()
    lg=(la_ if la_ is not None else torch.zeros(BH,T,device=k.device,dtype=k.dtype)).float()
    if pad:
        k=F.pad(k,(0,0,0,pad)); v=F.pad(v,(0,0,0,pad)); be=F.pad(be,(0,pad)); lg=F.pad(lg,(0,pad))
        if kw is not None: kw=F.pad(kw,(0,0,0,pad))
    Kc=k.reshape(BH,n,Cc,dh); Vc=v.reshape(BH,n,Cc,dh); bc=be.reshape(BH,n,Cc); g=torch.cumsum(lg.reshape(BH,n,Cc),-1)   # local cumulative log decay
    KWc=Kc if kw is None else kw.reshape(BH,n,Cc,dh)                     # write keys (preconditioned delta rule) or the keys
    A=torch.einsum('bncd,bnsd->bncs',Kc,KWc)
    dd=g[...,:,None]-g[...,None,:]; A=A*torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
    M=torch.tril(A,-1)*bc[...,None,:]                                    # M_ts = beta_s gamma_t/gamma_s k_t.k_s (s<t, same chunk)
    I=torch.eye(Cc,device=k.device,dtype=k.dtype).expand(BH*n,Cc,Cc)
    Tinv=torch.linalg.solve_triangular(I+M.reshape(BH*n,Cc,Cc),I,upper=False).reshape(BH,n,Cc,Cc)   # all chunks in one call
    S=S0_.float() if S0_ is not None else torch.zeros(BH,dh,dh,device=k.device,dtype=k.dtype)   # (value, key)
    gl=g[...,-1:]; eg=torch.exp(g); wdec=(bc*torch.exp(gl-g))[...,None]; egl=torch.exp(gl)[...,None]
    KL=Kc.unbind(1); KWL=KWc.unbind(1); VL=Vc.unbind(1); TL=Tinv.unbind(1); EL=eg.unbind(1); WL=wdec.unbind(1); GL=egl.unbind(1)   # unbind: ONE stack in backward
    outs=[]                                                                                                      #   (per-chunk slicing made it quadratic)
    lp_=_CHUNK.get('lp')                                                # LOW-PRECISION STATE PRODUCTS (tensor cores): the two d x d-sized products
    for c in range(n):                                                  #   per chunk in bf16, the state itself and the small solves stay fp32
        Kj=KL[c]; Vj=VL[c]
        if lp_ is not None:
            Kl=Kj.to(lp_)
            rhs=Vj-EL[c][...,None]*torch.bmm(Kl,S.to(lp_).transpose(1,2)).float()
            r=torch.bmm(TL[c],rhs); outs.append(Vj-r)
            S=GL[c]*S+torch.bmm((WL[c]*r).to(lp_).transpose(1,2),KWL[c].to(lp_)).float()
        else:
            rhs=Vj-EL[c][...,None]*torch.bmm(Kj,S.transpose(1,2))          # v_t - gamma_t/gamma_0 S k_t
            r=torch.bmm(TL[c],rhs); outs.append(Vj-r)
            S=GL[c]*S+torch.bmm((WL[c]*r).transpose(1,2),KWL[c])          # writes decayed to the chunk end
    return torch.stack(outs,1).reshape(BH,Tp,dh)[:,:T]
def _delta_core(k, v, bt, la, mh, beta_p, B, T, kern='torch', q=None, s0=None, carry=None):
    """k, v (B,T,C) projected (+ convolved) keys / values; bt (B,T,H) per-token write strengths or None (then sigmoid(beta_p));
    la (B,T,H) log decay or None; q (B,T,C) separate query or None. Returns the mixer output (B,T,C) in fp32:
    S_t = a_t S_{t-1} + beta_t (v_t - a_t S_{t-1} k_t) k_t^T, output a_t S_{t-1} k_t (read before write), or S_t q_t with a query."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        C=k.shape[-1]; dh=C//mh
        if kern=='fla' and q is None and not _FLA.get('broken') and _FLA['dtype']==torch.bfloat16 and s0 is None and not _PREC['p'] and carry is None:
            fn=_fla_get(); dt=torch.bfloat16                                    # LEAN bf16 PATH: no fp32 copies; keys L2-normalised in the kernel
            kd=k.reshape(B,T,mh,dh).to(dt); vd=v.reshape(B,T,mh,dh).to(dt)
            qs=torch.cat([kd[:,1:],kd[:,-1:]],1)                                # SHIFTED QUERIES q_t = k_{t+1} (last one unused)
            be=(bt if bt is not None else torch.sigmoid(beta_p.float()).expand(B,T,mh)).to(dt)
            g=la.float() if la is not None else torch.zeros(B,T,mh,device=k.device)
            try: o,_=fn(qs,kd,vd,g,be,scale=1.0,use_qk_l2norm_in_kernel=True)
            except Exception as e:
                _FLA['broken']=True; print(f"  [kernel] Triton kernel failed at shape {tuple(k.shape)} ({repr(e)[:150]}); torch solve from now on")
                return _delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',q,s0)
            o=o[:,:-1]
            if la is not None: o=o*torch.exp(g[:,1:,:,None]).to(dt)
            return torch.cat([torch.zeros_like(o[:,:1]),o],1).reshape(B,T,C)  # output_t = a_t S_{t-1} k_t (bf16)
        if kern=='fla' and q is None and not _FLA.get('broken') and not _PREC['p'] and carry is None:
            fn=_fla_get(); dt=_FLA['dtype']
            kf=F.normalize(k.float().reshape(B,T,mh,dh),dim=-1); vf=v.float().reshape(B,T,mh,dh)
            be=bt.float() if bt is not None else torch.sigmoid(beta_p.float()).expand(B,T,mh)
            g=la.float() if la is not None else torch.zeros(B,T,mh,device=k.device)
            qs=torch.cat([kf[:,1:],torch.zeros_like(kf[:,:1])],1)               # SHIFTED QUERIES: q_t = k_{t+1}
            h0_=None if s0 is None else s0.float().transpose(-1,-2).contiguous().to(dt)   # FLA state convention (K, V)
            try: o,_=(fn(qs.to(dt),kf.to(dt),vf.to(dt),g.to(dt),be.to(dt),scale=1.0) if h0_ is None else
                      fn(qs.to(dt),kf.to(dt),vf.to(dt),g.to(dt),be.to(dt),scale=1.0,initial_state=h0_))   # o_t = S_t k_{t+1}
            except Exception as e:                                              # any later failure: torch solve from now on
                _FLA['broken']=True; print(f"  [kernel] Triton kernel failed at shape {tuple(k.shape)} ({repr(e)[:150]}); torch solve from now on")
                return _delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',q,s0)
            o=o.float()[:,:-1]                                                  # S_{t-1} k_t for t = 1..T-1
            if la is not None: o=o*torch.exp(g[:,1:,:,None])                    # x a_t
            o0=torch.zeros_like(vf[:,:1]) if s0 is None else (torch.einsum('bhvk,bhk->bhv',s0.float(),kf[:,0])*torch.exp(g[:,0])[...,None]).unsqueeze(1)
            out=torch.cat([o0,o],1)                                             # output_t = a_t S_{t-1} k_t, output_0 = a_0 S0 k_0 (0 without S0)
            return out.reshape(B,T,C)
        sp=lambda z: z.reshape(B,T,mh,dh).transpose(1,2).to(torch.float32,memory_format=torch.contiguous_format).reshape(B*mh,T,dh)   # one fused cast + transpose
        if carry is not None and q is None and s0 is None:            # CONTINUOUS CONTEXT: the real rows start from the carried memory S0 (detached).
            rows_=carry.get('rows'); nrw_=int(carry['nr']) if rows_ is None else int(rows_.numel())   # rows: which batch rows carry a memory
            nb_=nrw_*mh; Sin=carry.get('S'); fac_=bool(carry.get('fac'))  #   Its readout c_t = gamma_t S0 k_t enters as v - c (same triangular system in
            Fin=Sin if isinstance(Sin,tuple) else None                   #   FACTORED memory (U, K): S0 = U^T K
            Slt=Sin if isinstance(Sin,_SlotF) else None                  #   v9.8 SLOT memory (fixed size, in place)
            if Fin is not None or Slt is not None: Sin=None
            ib_=None if rows_ is None else (rows_.to(k.device)[:,None]*mh+torch.arange(mh,device=k.device)[None]).reshape(-1)
            SEL=(lambda z: z[:nb_]) if ib_ is None else (lambda z: z[ib_])
            kn_=F.normalize(SEL(sp(k)),dim=-1)                          #   every solve form); out = solve(v - c) + c; the final memory
            pk_=carry.get('P') if (_PREC['p'] and _PREC['C']>0) else None   # write-key preconditioner statistics carried too (block form)
            if pk_ is not None and tuple(pk_[0].shape)!=(nb_,dh): pk_=None
            _PREC['prior']=(pk_[0],pk_[1],nb_ if ib_ is None else ib_) if pk_ is not None else None
            g_=torch.cumsum(SEL(la.float().transpose(1,2).reshape(B*mh,T)),-1) if la is not None else torch.zeros(nb_,T,device=k.device)
            if Sin is None and pk_ is not None and not fac_ and Fin is None: Sin=torch.zeros(nb_,dh,dh,device=k.device)
            if Fin is not None and Fin[0].shape[0]!=nb_: Fin=None
            if Slt is not None and (Slt.U.shape[0]!=nb_ or ib_ is not None): raise RuntimeError('slot memory: row layout mismatch')
            if Slt is not None:                                          #   S0 k = sum_s exp(Gt - Gs) U_s (K_s . k): never rescaled, never concatenated
                c_=torch.bmm(torch.bmm(kn_,Slt.K.transpose(1,2))*torch.exp(Slt.Gt[:,None]-Slt.Gs)[:,None,:],Slt.U)*torch.exp(g_)[...,None]
                c_btc=torch.zeros(B,T,C,device=k.device,dtype=torch.float32)
                c_btc[:nrw_]=c_.reshape(nrw_,mh,T,dh).transpose(1,2).reshape(nrw_,T,C)
                out=_delta_core(k,v.float()-c_btc,bt,la,mh,beta_p,B,T,'torch',None,None)+c_btc
            elif Fin is not None or (Sin is not None and tuple(Sin.shape)==(nb_,dh,dh)):   #   S_T = gamma_T S0 + sum_s gamma_T/gamma_s beta_s r_s kw_s^T is handed back
                if Fin is not None: c_=torch.bmm(torch.bmm(kn_,Fin[1].float().transpose(1,2)),Fin[0].float())*torch.exp(g_)[...,None]   # S0 k = U^T (K k)
                else: c_=torch.bmm(kn_,Sin.transpose(1,2))*torch.exp(g_)[...,None]                       # (nb,T,dh)
                c_btc=torch.zeros(B,T,C,device=k.device,dtype=torch.float32)
                cr_=c_.reshape(nrw_,mh,T,dh).transpose(1,2).reshape(nrw_,T,C)
                if rows_ is None: c_btc[:nrw_]=cr_
                else: c_btc[rows_.to(k.device)]=cr_
                try: out=_delta_core(k,v.float()-c_btc,bt,la,mh,beta_p,B,T,'torch',None,None)+c_btc
                finally: pass
            else: Sin=None; out=_delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',None,None)
            _PREC['prior']=None
            if _RDIAG['rec'] and la is not None: _RDIAG['v'].append(float(torch.exp(la.float().sum(1)).mean()))
            _gr_=bool(carry.get('grad'))                                                       # RECONSTRUCTION TRAINING: differentiable end state
            _dt_=(lambda t_: t_) if _gr_ else (lambda t_: t_.detach())
            with (contextlib.nullcontext() if _gr_ else torch.no_grad()):
                r_=(SEL(sp(v.float()))-SEL(sp(_dt_(out))))                                              # r = v - out (carried rows)
                be_=(SEL(bt.float().transpose(1,2).reshape(B*mh,T)) if bt is not None else torch.sigmoid(beta_p.float()).reshape(1,1).expand(nb_,T))
                if _PREC['p'] and pk_ is not None: _PREC['prior']=(pk_[0],pk_[1],nb_)
                kw_=_kw_par(_dt_(kn_)) if _PREC['p'] else _dt_(kn_)
                _PREC['prior']=None
                if _PREC['p'] and _PREC['C']>0:
                    k2s_=(kn_.detach()**2).sum(1); carry['Pout']=(k2s_ if pk_ is None else pk_[0]+k2s_, T if pk_ is None else pk_[1]+T)
                gd_=_dt_(g_); w_=be_*torch.exp(gd_[:,-1:]-gd_)
                if Slt is not None:                                                                  # v9.8 SLOT WRITE in place: slots i..i+T-1
                    i_=int(Slt.i); T_=r_.shape[1]
                    Slt.U[:,i_:i_+T_]=r_*be_[...,None]; Slt.K[:,i_:i_+T_]=kw_; Slt.Gs[:,i_:i_+T_]=Slt.Gt[:,None]+gd_; Slt.Gt.add_(gd_[:,-1])
                    carry['out']=Slt
                elif fac_ or Fin is not None:                                                          # FACTORED end state: the write terms unsummed
                    U_=r_*w_[...,None]; K_=kw_
                    if Fin is not None: U_=torch.cat([Fin[0].float()*torch.exp(gd_[:,-1])[:,None,None],U_],1); K_=torch.cat([Fin[1].float(),K_],1)
                    carry['out']=(U_,K_)
                else:
                    ST=torch.bmm((r_*w_[...,None]).transpose(1,2),kw_)                                  # (value, key)
                    if Sin is not None: ST=ST+torch.exp(gd_[:,-1])[:,None,None]*Sin
                    carry['out']=ST
            return out
        if T==1 and q is None and s0 is None and _DC1['on']:               # v9.8 cg8 SINGLE-BYTE PATH: read before write from an empty memory = 0 exactly
            _DC1['n']+=1; return torch.zeros(B,1,C,device=k.device,dtype=torch.float32)   #   (the solve gave v - 1*v = 0 in fp32; generation calls the inner solve with T = 1)
        k=F.normalize(sp(k),dim=-1); v=sp(v); kw=_kw_par(k) if _PREC['p'] else k      # write keys (preconditioned delta rule)
        mb=bt is not None; md=la is not None
        W_=_WIN['W']
        if W_ and q is None and T>2*W_:                                    # WINDOWED WY: overlapping chunks of 2W, keep the second half
            BH=k.shape[0]; n_=-(-T//W_); Tp=n_*W_
            be_=(bt.float().transpose(1,2).reshape(BH,T) if mb else torch.sigmoid(beta_p.float()).expand(BH,T))
            la_=(la.float().transpose(1,2).reshape(BH,T) if md else torch.zeros(BH,T,device=k.device))
            pz=lambda z,dims: F.pad(z,dims)
            kp=pz(k,(0,0,W_,Tp-T)); vp=pz(v,(0,0,W_,Tp-T)); bp=pz(be_,(W_,Tp-T)); lp=pz(la_,(W_,Tp-T))
            kc=kp.unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh); vc=vp.unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh)
            kwc=pz(kw,(0,0,W_,Tp-T)).unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh) if _PREC['p'] else kc
            bc=bp.unfold(1,2*W_,W_).reshape(BH*n_,1,2*W_); lc=lp.unfold(1,2*W_,W_).reshape(BH*n_,2*W_)
            A=torch.bmm(kc,kwc.transpose(1,2))
            if md:
                lg=torch.cumsum(lc,-1); dd=lg[:,:,None]-lg[:,None,:]; A=A*torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
            M=torch.tril(A,-1)*bc; I=torch.eye(2*W_,device=k.device).unsqueeze(0)
            r=torch.linalg.solve_triangular(I+M,vc,upper=False)
            out=(vc-r)[:,W_:].reshape(BH,n_*W_,dh)[:,:T]
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        if mb: beta=bt.float().transpose(1,2).reshape(B*mh,1,T)           # column s scaled by beta_s
        else: beta=torch.sigmoid(beta_p.float())
        SEL_=(_CHUNK.get('by') or {}).get(dh) or _CHUNK                   # solve form chosen for this head dim (start-up timing)
        ng_=(not torch.is_grad_enabled()) and bool(SEL_.get('Bng') or SEL_.get('Cng',0))   # no-grad passes have their own choice
        Bm_=SEL_.get('Bng') if ng_ else SEL_.get('B')
        if Bm_ and q is None and s0 is None:                               # BLOCKED full-WY solve (tensor-core GEMMs)
            be_=(bt.float().transpose(1,2).reshape(B*mh,T) if mb else torch.sigmoid(beta_p.float()).reshape(1,1).expand(B*mh,T))
            la_=la.float().transpose(1,2).reshape(B*mh,T) if md else None
            _BLK['prec']=Bm_[1]; out=delta_blocked(k,v,be_,la_,b=Bm_[0],kw=kw if _PREC['p'] else None)
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        Cc_=SEL_.get('Cng',0) if ng_ else SEL_.get('C',0)
        if Cc_>0 and q is None and T>Cc_:                                  # CHUNKED solve (no T x T tensors)
            la_=la.float().transpose(1,2).reshape(B*mh,T) if md else None
            S0c=s0.float().reshape(B*mh,dh,dh) if s0 is not None else None
            out=_delta_chunked(k,v,beta,mb,la_,S0c,Cc_,kw=kw if _PREC['p'] else None)
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        A=torch.bmm(k,kw.transpose(1,2))                                   # read key (row) . write key (column)
        if md:                                                             # A_ts * gamma_t/gamma_s (s<t), gamma = cumulative decay
            lg=torch.cumsum(la.float().transpose(1,2).reshape(B*mh,T),-1)
            dd=lg[:,:,None]-lg[:,None,:]; D=torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
            A=A*D
        M=torch.tril(A,-1)*beta
        I=torch.eye(T,device=k.device).unsqueeze(0)
        if s0 is not None and q is None:                                  # INITIAL STATE S0: readout of S0 through the decays,
            S0_=s0.float().reshape(B*mh,dh,dh)                             #   c_t = gamma_t S0 k_t, enters the same triangular system
            c_=torch.bmm(k,S0_.transpose(1,2))
            if md: c_=c_*torch.exp(lg)[:,:,None]
            r=torch.linalg.solve_triangular(I+M,v-c_,upper=False)
        else: r=torch.linalg.solve_triangular(I+M,v,upper=False)
        if q is not None:
            q=F.normalize(sp(q),dim=-1); u=r*(beta.transpose(1,2) if mb else beta)
            QK=torch.bmm(q,kw.transpose(1,2))
            if md: QK=QK*D
            out=torch.bmm(torch.tril(QK),u)                                  # S_t q_t (includes the current write)
        else: out=v-r
        return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)

def _kernel_check(dh, mh, T, bmax, dev, decay=True):
    """Triton kernel vs torch solve on random inputs of one block's shape: relative error of outputs and of all input gradients."""
    g_=torch.Generator(device='cpu').manual_seed(0); B=2; C=dh*mh
    mk=lambda *s_: torch.randn(*s_,generator=g_).to(dev)
    base=[mk(B,T,C),mk(B,T,C),bmax*torch.sigmoid(mk(B,T,mh)),-F.softplus(mk(B,T,mh)*2-4) if decay else None]
    w=mk(B,T,C); res={}
    for kern in ('torch','fla'):
        xs=[z.clone().requires_grad_(True) if z is not None else None for z in base]
        o=_delta_core(xs[0],xs[1],xs[2],xs[3],mh,None,B,T,kern); (o*w).sum().backward()
        res[kern]=[o.detach()]+[z.grad for z in xs if z is not None]
    rel=lambda a,b: float((a-b).norm()/(b.norm()+1e-12))
    return max(rel(a,b) for a,b in zip(res['fla'],res['torch']))

def setup_kernel(model, cfg):
    _PREC['p']=float(getattr(cfg,'mix_precond',0.0) or 0.0)                 # before the solve-form timing below (it times the real path)
    _PREC['C']=int(getattr(cfg,'mix_precond_chunk',0) or 0); _PREC['det']=bool(getattr(cfg,'mix_precond_detach',False))
    mc_=getattr(cfg,'mix_chunk',0) or 0
    _CHUNK['C']=0 if (mc_=='auto' or (isinstance(mc_,str) and mc_.startswith('b'))) else int(mc_); _CHUNK['lp']=torch.bfloat16 if (getattr(cfg,'mix_lowp',False) and torch.cuda.is_available()) else None
    _CHUNK['by']={}; _CHUNK['B']=None; _CHUNK['Bng']=None
    if isinstance(mc_,str) and mc_.startswith('b'):                    # forced blocked solve: 'b64' (TF32 GEMMs) / 'b64h' (bf16 GEMMs)
        _CHUNK['C']=0; _CHUNK['B']=(int(mc_[1:].rstrip('h')),'bf16' if mc_.endswith('h') else ('tf32' if torch.cuda.is_available() else 'fp32'))
    if mc_=='auto' and torch.cuda.is_available():                      # time the delta-rule solve forms on every distinct block shape, keep the fastest
        lp0_=_CHUNK['lp']; B=int(cfg.batch_size)+int(cfg.dream_batch); T=int(cfg.seq_len); dev=model.embed.weight.device
        Bn_=int(getattr(cfg,'bank_tgt_chunk_big',0) or 0) or int(cfg.bank_tgt_chunk)
        nm_=lambda f: 'full' if f[0]=='full' else (f"chunk {f[1]}" if f[0]=='chunk' else f"blocked {f[1]} {f[2]}")
        def setf(f,ng=False):
            k1,k2=('Cng','Bng') if ng else ('C','B')
            _CHUNK[k1]=f[1] if f[0]=='chunk' else 0; _CHUNK[k2]=(f[1],f[2]) if f[0]=='blk' else None
        cands=[('full',)]+[('chunk',c) for c in (64,128,256,512) if c<T]+[('blk',b,p) for b in (64,128,256) for p in ('tf32','bf16') if b<=T]
        shapes=sorted({(getattr(b_,'mh',1),b_.d_state) for b_ in model.blocks if not getattr(b_,'loc',False) and not getattr(b_,'mq',False)}) or [(getattr(model.blocks[0],'mh',1),model.blocks[0].d_state)]
        by_={}
        for mh,C in shapes:
            dh=C//mh; sel={'C':0,'B':None,'Cng':0,'Bng':None}
            try:
                g_=torch.Generator(device='cpu').manual_seed(0); tm={}; err_={}; outs_={}
                mk=lambda *s_: torch.randn(*s_,generator=g_).to(dev)
                kk,vv,ww=mk(B,T,C),mk(B,T,C),mk(B,T,C); bb=1.2*torch.sigmoid(mk(B,T,mh)-1.0); ll=-F.softplus(mk(B,T,mh)*1.5-3)
                _CHUNK['by']={}
                for f in cands:
                    try:
                        setf(f); _CHUNK['Cng']=0; _CHUNK['Bng']=None; xs=[z.clone().requires_grad_(True) for z in (kk,vv,bb,ll)]
                        for it in range(3):
                            if it==1: torch.cuda.synchronize(); t0=time.perf_counter()
                            o_=_delta_core(xs[0],xs[1],xs[2],xs[3],mh,None,B,T,'torch'); (o_*ww).sum().backward()
                        torch.cuda.synchronize(); tm[f]=(time.perf_counter()-t0)/2*1e3; outs_[f]=o_.detach()
                    except Exception as ex: print(f"  [delta solve] {nm_(f)} failed at {mh}x{dh} ({repr(ex)[:100]})"); torch.cuda.empty_cache()
                ref_=outs_.get(('full',))
                err_={f:(0.0 if f==('full',) or ref_ is None else float((outs_[f]-ref_).norm()/(ref_.norm()+1e-30))) for f in outs_}
                ok_=[f for f in tm if err_[f]<=5e-3]                           # every form must match the fp32 full solve
                best=min(ok_,key=tm.get); setf(best); sel['C'],sel['B']=_CHUNK['C'],_CHUNK['B']
                del kk,vv,ww,xs,outs_; torch.cuda.empty_cache()
                print(f"  [delta solve] forward+backward per block ({B}x{T}, {mh} heads x {dh}"+(", chunks with bf16 state products" if lp0_ is not None else "")+"): "
                      +" | ".join(f"{nm_(f)} {t:.0f} ms"+(f" (err {err_[f]:.0e})" if f!=('full',) else "") for f,t in tm.items())+f" -> {nm_(best)}")
                try:                                                             # NO-GRAD passes (bank targets): their own form, timed at their batch
                    kk,vv=mk(Bn_,T,C),mk(Bn_,T,C); bb=1.2*torch.sigmoid(mk(Bn_,T,mh)-1.0); ll=-F.softplus(mk(Bn_,T,mh)*1.5-3); tng={}
                    with torch.no_grad():
                        for f in [f for f in ok_ if f[0]!='full']:
                            try:
                                setf(f)
                                for it in range(3):
                                    if it==1: torch.cuda.synchronize(); t0=time.perf_counter()
                                    _delta_core(kk,vv,bb,ll,mh,None,Bn_,T,'torch')
                                torch.cuda.synchronize(); tng[f]=(time.perf_counter()-t0)/2*1e3
                            except Exception: torch.cuda.empty_cache()
                    del kk,vv; torch.cuda.empty_cache()
                    if tng:
                        bng=min(tng,key=tng.get); sel['Cng']=bng[1] if bng[0]=='chunk' else 0; sel['Bng']=(bng[1],bng[2]) if bng[0]=='blk' else None
                        print(f"  [delta solve] no-grad forward per block ({Bn_}x{T}, {mh} heads x {dh}, bank targets): "+" | ".join(f"{nm_(f)} {t:.0f} ms" for f,t in tng.items())+f" -> {nm_(bng)}")
                except Exception as ex: print(f"  [delta solve] no-grad timing failed ({repr(ex)[:120]})"); torch.cuda.empty_cache()
            except Exception as ex: print(f"  [delta solve] timing failed at {mh}x{dh} ({repr(ex)[:120]}): full solve"); torch.cuda.empty_cache()
            by_[dh]=sel
        _CHUNK['by']=by_
        S0_=_CHUNK['by'].get(model.blocks[0].d_state//getattr(model.blocks[0],'mh',1)) or next(iter(_CHUNK['by'].values()),{'C':0,'B':None,'Cng':0,'Bng':None})
        _CHUNK['C'],_CHUNK['B'],_CHUNK['Cng'],_CHUNK['Bng']=S0_['C'],S0_['B'],S0_['Cng'],S0_['Bng']    # default for any other shape
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    _PREC['p']=float(getattr(cfg,'mix_precond',0.0) or 0.0)
    _WIN['W']=int(getattr(cfg,'mix_window',0) or 0); _SAMP['W']=int(getattr(cfg,'samp_window',0) or 0); _SAMP['bf16']=bool(getattr(cfg,'samp_bf16',False)); _SAMP['mm']=bool(getattr(cfg,'samp_lowp_mm',False))
    _SAMP['graph']=bool(getattr(cfg,'samp_graph',False)); _SAMP['gwarm']=int(getattr(cfg,'samp_graph_warm',64))
    _SAMP['rescore']=bool(getattr(cfg,'samp_rescore',True))
    """Choose the delta-rule kernel per block; the Triton kernel only after it matched the torch solve (values + gradients)."""
    for b in model.blocks: b._kern='torch'
    return
    fn=_fla_get()
    if fn is None:
        print(f"  [kernel] flash-linear-attention not available ({_FLA['err']}); torch solve everywhere. Install: pip install flash-linear-attention"); return
    dev=model.embed.weight.device; msg=[]
    for li,b in enumerate(model.blocks):
        mh=getattr(b,'mh',1); dh=b.d_state//mh
        if getattr(b,'loc',False): msg.append(f"b{li} local filter (no solve)"); continue
        if getattr(b,'mq',False): msg.append(f"b{li} torch (separate query)"); continue
        if dh>256: msg.append(f"b{li} torch (head dim {dh} > 256)"); continue
        key=(dh,mh)
        if key not in _FLA['checked']:
            err=None
            dts_=(torch.float32,torch.bfloat16)
            for dt in (dts_ if dev.type=='cuda' else (torch.float32,)):
                _FLA['dtype']=dt
                try: e_=max(_kernel_check(dh,mh,T_,getattr(b,'bmax',1.0),dev,getattr(b,'md',False)) for T_ in sorted({cfg.seq_len,min(cfg.dream_len,cfg.seq_len)}))
                except Exception as ex: err=repr(ex)[:200]; continue
                tol=2e-2 if dt==torch.float32 else 5e-2
                if e_<tol: _FLA['checked'][key]=(True,dt,e_); break
                err=f"rel. error {e_:.1e} ({str(dt)[6:]})"
            else: _FLA['checked'][key]=(False,None,err)
        ok,dt,e_=_FLA['checked'][key]
        if ok: _FLA['dtype']=dt; b._kern='fla'; msg.append(f"b{li} Triton ({str(dt)[6:]}, max rel. error vs torch {e_:.1e})")
        else: msg.append(f"b{li} torch (kernel check failed: {e_})")
    print("  [kernel] delta-rule solve per block: "+" | ".join(msg))
    if dev.type=='cuda' and any(b._kern=='fla' for b in model.blocks):      # one-off micro-benchmark on the real shape
        seen_=set()
        for b in [b_ for b_ in model.blocks if b_._kern=='fla']:
            mh=b.mh; C=b.d_state; B,T=int(cfg.batch_size)+int(cfg.dream_batch),cfg.seq_len
            if (mh,C) in seen_: continue
            seen_.add((mh,C))
            z=[torch.randn(B,T,C,device=dev,requires_grad=True) for _ in range(2)]; bt=torch.rand(B,T,mh,device=dev)
            la=-torch.rand(B,T,mh,device=dev)*0.01
            tm={}
            for kern in ('torch','fla'):
                for it in range(8):
                    if it==3: torch.cuda.synchronize(); t0=time.perf_counter()
                    _delta_core(z[0],z[1],bt,la,mh,None,B,T,kern).float().sum().backward()
                torch.cuda.synchronize(); tm[kern]=(time.perf_counter()-t0)/5*1000
            del z; torch.cuda.empty_cache()
            print(f"  [kernel] forward+backward of one mixer solve ({B}x{T}, {mh} heads x {C//mh}): torch {tm['torch']:.1f} ms, Triton {tm['fla']:.1f} ms")

class DeltaBlock(nn.Module):
    def __init__(s, d_model, d_state, mlp_hidden=None):
        super().__init__(); s.d_model,s.d_state=d_model,d_state; s.mlp_hidden=mlp_hidden or d_model*2
        s.Wk=nn.Linear(d_model,d_state,bias=False); s.Wv=nn.Linear(d_model,d_state,bias=False)
        s.Wout=nn.Linear(d_state,d_model,bias=False); s.gate=nn.Linear(d_model,d_state)
        s.beta=nn.Parameter(torch.tensor(0.5)); s.ln=nn.LayerNorm(d_model); s.norm=nn.LayerNorm(d_model)
        s.mlp_in=nn.Linear(d_model,s.mlp_hidden); s.mlp_out=nn.Linear(s.mlp_hidden,d_model)
        s.register_buffer('unit_usage',torch.zeros(d_state))
        s.register_buffer('mlp_usage',torch.zeros(s.mlp_hidden))
        # home usage (frozen after each domain) - polarization feature = |current - home|
        s.register_buffer('unit_home',torch.zeros(d_state))
        s.register_buffer('mlp_home',torch.zeros(s.mlp_hidden))
        # HARD importance-freeze masks (the +0.32 floor) - meta-gate modulates ON TOP
        s.register_buffer('frozen_units',torch.zeros(d_state,dtype=torch.bool))
        s.register_buffer('mlp_frozen',torch.zeros(s.mlp_hidden,dtype=torch.bool))
        # DISTINCTNESS ROUTING: each unit has a learned distinctness-AFFINITY (which
        # part of the manifold it serves - low=prose/shared region, high=distinct region).
        # Content routes to units whose affinity matches its distinctness.
        s.unit_affinity=nn.Parameter(torch.rand(d_state)*0.1)      # learned per-unit region
        s.mlp_affinity=nn.Parameter(torch.rand(s.mlp_hidden)*0.1)
        s.cur_distinct=None   # current batch's distinctness level (set in forward)
        s.route_in_forward=True   # False when cfg.route_writes (set in run)
        # HEBBIAN AFFINITY: each unit's affinity = distinctness of the content that drives it most.
        # -1 = unassigned (accepts writes from any content until strongly driven).
        s.register_buffer('unit_aff', -torch.ones(d_state))
        s.register_buffer('mlp_aff', -torch.ones(s.mlp_hidden))
        s.calib=None              # dict of accumulators during affinity calibration
        s.register_buffer('distinct_locked', torch.zeros(1))   # per-block copy of the Model's lock flag
        # random base write thresholds + cumulative importance (persist across domains)
        s.register_buffer('unit_theta0', sample_theta(d_state))
        s.register_buffer('mlp_theta0', sample_theta(s.mlp_hidden))
        s.register_buffer('unit_imp', torch.zeros(d_state))
        s.register_buffer('mlp_imp', torch.zeros(s.mlp_hidden))
        s.register_buffer('ssm_drive', torch.zeros(d_state))    # last batch's per-unit drive
        s.register_buffer('mlp_drive', torch.zeros(s.mlp_hidden))
        s.register_buffer('ssm_drive_ema', torch.zeros(d_state))  # each unit's typical drive
        # CHANNEL NEURONS: this block's d_model residual channels, each with its own random
        # threshold, importance and drive (used when channel_thresholds=True)
        s.register_buffer('ch_theta0', sample_theta(d_model))
        s.register_buffer('ch_imp', torch.zeros(d_model))
        s.register_buffer('ch_usage', torch.zeros(d_model))
        s.register_buffer('ch_drive', torch.zeros(d_model))
        s.register_buffer('ch_drive_ema', torch.zeros(d_model))
        s.register_buffer('mlp_drive_ema', torch.zeros(s.mlp_hidden))
        # SIGNAL 1: long-memory drive references; SIGNAL 2: loss sensitivity on earlier domains
        for nm,n in (('ssm',d_state),('mlp',s.mlp_hidden),('ch',d_model)):
            s.register_buffer(nm+'_drive_slow', torch.zeros(n)); s.register_buffer(nm+'_drive_peak', torch.zeros(n))
        s.register_buffer('unit_sens', torch.zeros(d_state)); s.register_buffer('mlp_sens', torch.zeros(s.mlp_hidden))
        s.register_buffer('ch_sens', torch.zeros(d_model))
    def setup_local(s, cfg):
        """LOCAL FILTER BLOCK: causal depthwise filter over the previous mix_local_width values instead of the delta-rule memory
        (starts as their average)."""
        K=int(cfg.mix_local_width); s.loc=True
        s.conv_loc=nn.Parameter(torch.full((s.d_state,1,K),1.0/K,device=s.Wv.weight.device))
    def setup_lattn(s, cfg):
        """C2 SLIDING-WINDOW ATTENTION mixer: softmax over the previous lattn_window positions of scale_h x cos(k_t, k_j) + bias_h[t-j],
        plus a null slot (reads nothing when nothing matches); keys / values from the same projections and short convolution."""
        s.lattn=True; s.la_W=int(cfg.lattn_window); mh=getattr(s,'mh',1); dev=s.Wk.weight.device
        s.la_scale=nn.Parameter(torch.full((mh,),math.log(8.0),device=dev)); s.la_rel=nn.Parameter(torch.zeros(mh,s.la_W,device=dev))
    def setup_nov(s, cfg):
        """NOVELTY RESERVE: nov_pool closed MLP slices (zero-init output: opening one does not change the function)."""
        k=max(1,int(round(2*cfg.d_model*cfg.init_mlp_mult*cfg.nov_frac))); dev=s.Wv.weight.device
        s.nov_in=nn.ModuleList([nn.Linear(s.d_model,k).to(dev) for _ in range(int(cfg.nov_pool))])
        s.nov_out=nn.ModuleList([nn.Linear(k,s.d_model,bias=False).to(dev) for _ in range(int(cfg.nov_pool))])
        for o in s.nov_out: o.weight.data.zero_()
        s.nov_g=[0.0]*int(cfg.nov_pool)                              # python-side gates (no GPU sync; copied into the teachers)
        s.nov_tok=bool(cfg.nov_token)
    def nov_term(s, hn, rec=False):
        out=0.0; tok=getattr(s,'nov_tok',False); M=_NOV['m'] if tok else None
        for j,g in enumerate(getattr(s,'nov_g',())):
            if g<=0: continue
            if tok:                                                    # per-token gate (the sampler leaves token-gated slices out)
                if rec or M is None or j>=len(M) or M[j] is None or M[j].shape[:2]!=hn.shape[:2]: continue
                out=out+g*M[j].to(hn.dtype)*s.nov_out[j](F.gelu(s.nov_in[j](hn)))
            else: out=out+g*s.nov_out[j](F.gelu(s.nov_in[j](hn)))
        return out
    def setup_mixer(s, cfg):
        """Optional mixer upgrades; each starts out computing exactly what the plain mixer computes."""
        s.mh=max(1,int(cfg.mix_heads)); s.mk=int(cfg.mix_conv); s.mq=bool(cfg.mix_query); s.mb=bool(cfg.mix_beta_tok)
        hd_=int(getattr(cfg,'mix_head_dim',0) or 0)
        if hd_>0:                                                              # HEAD DIM CAP: per-block head count
            ds_=[d_ for d_ in range(1,min(hd_,s.d_state)+1) if s.d_state%d_==0]
            for q_ in (64,32,16,8,1):
                c_=[d_ for d_ in ds_ if d_%q_==0]
                if c_: s.mh=s.d_state//max(c_); break
        assert s.d_state%s.mh==0, 'd_state must divide into mix_heads'
        dev=s.Wk.weight.device
        if s.mq: s.Wq=nn.Linear(s.d_model,s.d_state,bias=False).to(dev); s.Wq.weight.data.copy_(s.Wk.weight.data)
        if s.mb:
            s.Wb=nn.Linear(s.d_model,s.mh).to(dev); s.Wb.weight.data.zero_()
            s.bmax=float(getattr(cfg,'mix_beta_max',1.0)); p0=float(torch.sigmoid(s.beta.data))/s.bmax      # same starting beta
            s.Wb.bias.data.fill_(math.log(p0/(1-p0)))
        s.dp2=bool(int(getattr(cfg,'dp_steps',1))>=2 and s.mb and not s.mq)
        if s.dp2:                                                              # DELTAPRODUCT: second write per token (starts ~off)
            s.Wk2=nn.Linear(s.d_model,s.d_state,bias=False).to(dev); s.Wv2=nn.Linear(s.d_model,s.d_state,bias=False).to(dev)
            s.Wb2=nn.Linear(s.d_model,s.mh).to(dev); s.Wb2.weight.data.zero_(); p2_=0.02/s.bmax; s.Wb2.bias.data.fill_(math.log(p2_/(1-p2_)))
        s.md=bool(getattr(cfg,'mix_decay',False))
        if s.md:
            s.Wa=nn.Linear(s.d_model,s.mh).to(dev); s.Wa.weight.data.zero_(); s.Wa.bias.data.fill_(float(cfg.mix_decay_bias))
        if s.mk>0:
            def idk():
                w=torch.zeros(s.d_state,1,s.mk,device=dev); w[:,0,-1]=1.0; return nn.Parameter(w)
            s.conv_k=idk(); s.conv_v=idk()
            if s.mq: s.conv_q=idk()
    def _conv(s, z, w):
        """Causal depthwise convolution along time: z (B,T,C), w (C,1,K)."""
        K=w.shape[-1]; zt=F.pad(z.transpose(1,2),(K-1,0))
        return F.conv1d(zt,w.to(zt.dtype),groups=w.shape[0]).transpose(1,2)
    def mix(s, h, x, B, T):
        """Delta-rule sequence mixer: memory S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T. Plain: output S_{t-1} k_t (the key
        reads memory before writing). With a query: output S_t q_t. WY form: one triangular solve per sequence and head
        (or the Triton chunked kernel, mix_kernel='fla'). Runs in fp32 even under autocast."""
        if getattr(s,'loc',False):                                              # LOCAL FILTER: previous values only (read before write)
            v=_wl(s,'Wv',h); K=s.conv_loc.shape[-1]
            sg_=_SEG['s']
            if sg_ is not None and tuple(sg_.shape)==(B,T):                    # v9 RUN MASKING: values before the segment start excluded
                w_=s.conv_loc.to(v.dtype)[:,0,:]; t_=torch.arange(T,device=v.device)[None]; o_=0.0
                for i in range(1,K+1): o_=o_+F.pad(v,(0,0,i,0))[:,:T]*((t_-i)>=sg_).to(v.dtype)[...,None]*w_[:,K-i]
                return o_.float()
            zt=F.pad(v.transpose(1,2),(K,0))
            return F.conv1d(zt,s.conv_loc.to(zt.dtype),groups=s.d_state)[...,:T].transpose(1,2).float()
        mh=getattr(s,'mh',1); mk=getattr(s,'mk',0); mq=getattr(s,'mq',False); mb=getattr(s,'mb',False)
        if getattr(s,'lattn',False):                                            # C2: sliding-window attention
            k=_wl(s,'Wk',h); v=_wl(s,'Wv',h)
            if mk>0: k=s._conv(k,s.conv_k); v=s._conv(v,s.conv_v)
            return _local_attn(k,v,mh,B,T,s.la_scale,s.la_rel,s.la_W)
        k=_wl(s,'Wk',h); v=_wl(s,'Wv',h); q=_wl(s,'Wq',h) if mq else None
        if mk>0:
            k=s._conv(k,s.conv_k); v=s._conv(v,s.conv_v)
            if mq: q=s._conv(q,s.conv_q)
        bt=getattr(s,'bmax',1.0)*torch.sigmoid(_wl(s,'Wb',h)) if mb else None  # (B,T,H)
        if mb and s.training and getattr(s,'bmax',1.0)>1.0:                     # usage of the reflection range (no GPU sync)
            bd=bt.detach()[:getattr(s,'_nstat',None)].float(); st=getattr(s,'_bstat',None)
            s._bstat=(bd.gt(1.0).float().mean(), bd.mean(), 1) if st is None else (st[0]+bd.gt(1.0).float().mean(), st[1]+bd.mean(), st[2]+1)
        md=getattr(s,'md',False); la=-F.softplus(_wl(s,'Wa',h)) if md else None  # log decay (B,T,H), <= 0
        if la is not None and _RESET['m'] is not None and tuple(_RESET['m'].shape)==tuple(la.shape[:2]):
            la=la.masked_fill(_RESET['m'][...,None],-30.0)                     # v7 EXACT RUNS: memory wiped at the start of a run (exp(-30) ~ 1e-13)
        s0_=None; S0p=getattr(s,'S0',None); ix_=getattr(s,'_s0_idx',None); tr_=getattr(s,'_s0_train',None)
        if S0p is not None and ix_ is not None and q is None:                    # PER-DOMAIN INITIAL STATE (only the trained domain's learns)
            g_=lambda i: S0p[i] if i==tr_ else S0p[i].detach()
            if isinstance(ix_,int): s0_=g_(ix_).unsqueeze(0).expand(B,-1,-1,-1)
            elif isinstance(ix_,(list,tuple)) and len(ix_)==B:
                z_=torch.zeros_like(S0p[0]); s0_=torch.stack([(g_(i) if i is not None and i>=0 else z_) for i in ix_])
        if getattr(s,'dp2',False) and q is None:                                 # DELTAPRODUCT: two writes per token as interleaved sub-steps
            k2=s.Wk2(h); v2=s.Wv2(h); bt2=s.bmax*torch.sigmoid(s.Wb2(h))
            K_=torch.stack([k,k2.to(k.dtype)],2).reshape(B,2*T,k.shape[-1]); V_=torch.stack([v,v2.to(v.dtype)],2).reshape(B,2*T,v.shape[-1])
            BT_=torch.stack([bt,bt2.to(bt.dtype)],2).reshape(B,2*T,bt.shape[-1])
            LA_=torch.stack([la,torch.zeros_like(la)],2).reshape(B,2*T,la.shape[-1]) if la is not None else None
            out=_delta_core(K_, V_, BT_, LA_, mh, None, B, 2*T, getattr(s,'_kern','torch'), None, s0_).reshape(B,T,2,-1)[:,:,0]
        else:
            if _ORACLE['on'] and q is None:                                     # ORACLE MEMORY: pooled least-squares statistics per head
                with torch.no_grad():
                    kn_o=F.normalize(k.float().reshape(B,T,mh,-1),dim=-1); vv_o=v.float().reshape(B,T,mh,-1)
                    be_o=(bt.float() if bt is not None else torch.sigmoid(s.beta.float()).expand(B,T,mh))[...,None]
                    A_o=_ORACLE['st'].setdefault(id(s),[0.0,0.0])
                    A_o[0]=A_o[0]+torch.einsum('bthi,bthj->hij',kn_o*be_o,kn_o); A_o[1]=A_o[1]+torch.einsum('bthi,bthj->hij',vv_o*be_o,kn_o)
            cr_={'S':_CARRY['st'].get(id(s)),'P':_CARRY['st'].get((id(s),'P')),'nr':_CARRY['nr'],'rows':_CARRY.get('rows'),'grad':bool(_CARRY.get('grad')) and _RC_GRAD['on'],'fac':bool(_CARRY.get('fac'))} if (_CARRY['on'] and q is None and s0_ is None) else None
            out=_delta_core(k, v, bt, la, mh, None if mb else s.beta, B, T, getattr(s,'_kern','torch'), q, s0_, carry=cr_)   # CONTINUOUS CONTEXT
            if cr_ is not None and 'out' in cr_:
                _CARRY['new'][id(s)]=cr_['out']
                if 'Pout' in cr_: _CARRY['new'][(id(s),'P')]=cr_['Pout']
            if _PMEM['on'] and q is None and id(s) in _PMEM['S']:                # PERSISTENT READ: a domain memory read at every position, no decay
                kn_p=F.normalize(k.float().reshape(B,T,mh,-1),dim=-1)
                out=out+torch.einsum('hvk,bthk->bthv',_PMEM['S'][id(s)],kn_p).reshape(B,T,-1).to(out.dtype)
        if s.training and getattr(s,'_surp_on',False):                          # SURPRISE: memory prediction-error ratio (no GPU sync)
            with torch.no_grad():
                n_=getattr(s,'_nstat',None); vf=v[:n_].float(); s._surp_now=((vf-out[:n_])**2).mean()/(vf.pow(2).mean()+1e-12)
        return out
    def write_route(s, aff):
        """Write-routing gate: a unit accepts writes from content whose distinctness matches its
        affinity. Unassigned units accept everything. Uniform before the lock."""
        if s.cur_distinct is None or s.distinct_locked.item()==0: return torch.ones_like(aff)
        g=torch.exp(-((s.cur_distinct-aff)**2)/Config.route_width)
        return torch.where(aff<0, torch.ones_like(g), g)
    @torch.no_grad()
    def _hebb(s, drive, aff):
        d=float(s.cur_distinct); w=drive/(drive.max()+1e-8)
        un=aff<0
        assign=un&(w>0.5); aff[assign]=d                       # strongly driven unassigned unit adopts this content's distinctness
        up=~un; aff[up]+=Config.hebb_rate*w[up]*(d-aff[up])   # assigned units drift toward content that drives them
    def route_gate(s, affinity):
        # soft routing gate: unit active for content whose distinctness ~ unit affinity.
        # BEFORE lock: route uniformly (gate=1) - accumulate reference without unstable
        # routing. AFTER lock: route by distinctness (stable reference).
        if s.cur_distinct is None or s.distinct_locked.item()==0: return torch.ones_like(affinity)
        return torch.exp(-((s.cur_distinct-torch.sigmoid(affinity))**2)/0.1)
    def forward(s,x):
        ns_=getattr(s,'_nstat',None)                                   # MERGED FORWARD: statistics from the first ns_ (real) rows only
        _lean=s.__dict__.get('_lean',False)                            # LEAN: drive statistics unused (gates open, no growth)
        if s.training and not _lean:
            cd=x.detach()[:ns_].abs().mean((0,1))
            s.ch_usage+=cd; s.ch_drive=cd
            _track(s,'ch',cd)
        h=s.ln(x); B,T,_=h.shape
        pred=s.mix(h,x,B,T)
        # DISTINCTNESS ROUTING: gate SSM units by affinity-match to content distinctness.
        # Distinct content activates its own region's units; prose the shared region.
        if s.route_in_forward: pred=pred*s.route_gate(s.unit_affinity)   # (d_state,) soft routing (read-gating)
        if s.calib is not None:
            dv=pred.detach().abs().mean((0,1)); dd=float(s.cur_distinct)
            s.calib['s_w']+=dv; s.calib['s_wd']+=dv*dd
        if s.training and getattr(s,'_ddrive_on',False) and ns_ is not None and ns_<pred.shape[0]:   # RESIDUAL GATES: activity on dreams
            dd_=pred.detach()[ns_:].abs().float().mean((0,1)); o_=getattr(s,'_dd_ssm',None)
            s._dd_ssm=dd_ if (o_ is None or o_.shape!=dd_.shape) else 0.99*o_+0.01*dd_
        if s.training and not _lean:
            d_=pred.detach()[:ns_].abs().mean((0,1))                        # computed once (usage and drive are the same statistic)
            s.unit_usage+=d_; s.ssm_drive=d_
            _track(s,'ssm',s.ssm_drive)
        gpre=_wl(s,'gate',h if getattr(s,'gate_ln',False) else x)   # gate reads the raw residual (original) or its LN
        h2=pred*F.silu(gpre); x_in=x
        _dc_=getattr(s,'_dcontent',False) and ns_ is not None and 0<ns_<h2.shape[0] and torch.is_grad_enabled()
        if _dc_: h2=torch.cat([h2[:ns_],h2[ns_:].detach()])                   # CONTENT-ONLY DREAMS: dream rows train Wout only
        o1_=_wl(s,'Wout',h2)
        x_par=x
        gp_=getattr(s,'_gproj',None)                                 # DREAM ONLY THE SIMILARITIES: projected gradient on this block's increments
        if gp_ is not None: o1_=_GProj.apply(o1_,*gp_)
        x=x+o1_
        hn=s.norm(x_par if getattr(s,'parallel',False) else x)                 # C3: parallel block
        if s.training and getattr(s,'_store_uniq',False): s._uh=h.detach()[:ns_]; s._um=hn.detach()[:ns_]   # for the uniqueness detector
        m=F.gelu(_wl(s,'mlp_in',film_mod(s,hn,ns_) if _FILM.get('codes') is not None else hn))
        if _XDG['on']: m=xdg_apply(m,getattr(s,'_xdg_i',0))
        if getattr(s,'inh_A',None) is not None: m=inh_apply(s,m,hn,ns_)
        cgm_=None
        if s.__dict__.get('_cg_g') is not None: m,cgm_=cg_apply(s,m,ns_)
        if getattr(s,'_probe',None) is not None:     # BLOW-UP PROBE: largest magnitudes inside this block
            s._probe.append((float(x_in.abs().max()),float(gpre.abs().max()),float(pred.abs().max()),float(h2.abs().max()),float(m.abs().max())))
        if s.route_in_forward: m=m*s.route_gate(s.mlp_affinity)          # route MLP units too
        if s.calib is not None:
            dv=m.detach().abs().mean((0,1)); dd=float(s.cur_distinct)
            s.calib['m_w']+=dv; s.calib['m_wd']+=dv*dd
        if (s.training and getattr(s,'write_routed',False) and s.distinct_locked.item()==1
                and s.cur_distinct is not None):
            s._hebb(s.ssm_drive, s.unit_aff)
        if s.training and getattr(s,'_ddrive_on',False) and ns_ is not None and ns_<m.shape[0]:
            dm_=m.detach()[ns_:].abs().float().mean((0,1)); o_=getattr(s,'_dd_mlp',None)
            s._dd_mlp=dm_ if (o_ is None or o_.shape!=dm_.shape) else 0.99*o_+0.01*dm_
        if s.training and not _lean:
            d_=m.detach()[:ns_].abs().mean((0,1))
            s.mlp_usage+=d_; s.mlp_drive=d_
            _track(s,'mlp',s.mlp_drive)
            if getattr(s,'write_routed',False) and s.distinct_locked.item()==1 and s.cur_distinct is not None:
                s._hebb(s.mlp_drive, s.mlp_aff)
        if _dc_: m=torch.cat([m[:ns_],m[ns_:].detach()])                        # CONTENT-ONLY DREAMS: dream rows train mlp_out only
        o2_=_wl(s,'mlp_out',m)
        if _ADPT['on'] and id(s) in _ADPT['P']:                                 # LOW-RANK ADAPTER beside the MLP (oracle)
            A_a,B_a=_ADPT['P'][id(s)]; o2_=o2_+F.linear(F.linear(hn.float(),A_a),B_a).to(o2_.dtype)
        if cgm_ is not None and o2_.requires_grad: o2_.register_hook(cgm_)
        if gp_ is not None: o2_=_GProj.apply(o2_,*gp_)
        x=x+o2_
        if getattr(s,'pu_in',None) is not None and _XDG.get('tagon'): x=x+pu_term(s,hn,ns_)
        if getattr(s,'nov_g',None): x=x+s.nov_term(hn)
        if getattr(s,'_cap',False): s._cap_out=x                   # FEATURE ANCHORING: this block's output (student, with gradient)
        return x
    @torch.no_grad()
    def widen(s, chunk, dev, rows=None):
        """Open up: add `chunk` new SSM units (zero-init Wout cols -> no disruption).
        New units are UNFROZEN (available for new content); existing frozen units stay
        frozen (protected). Widening = capacity; freeze machinery = protection.
        rows: optional {'Wk','Wv','gate','gate_b'} for the new units (activated reserve units)."""
        if getattr(s,'mq',False) or getattr(s,'mk',0) or getattr(s,'mb',False) or getattr(s,'mh',1)>1 or getattr(s,'md',False):
            raise NotImplementedError('widening is not implemented for the mixer upgrades (use block_extra_mode="start")')
        def grow_in(lin, add, new=None, newb=None):   # (out=d_state, in): add small-random (or given) rows
            w=torch.cat([lin.weight.data, new if new is not None else torch.randn(add, lin.weight.shape[1], device=dev)*0.02],0)
            nl=nn.Linear(lin.weight.shape[1], w.shape[0], bias=lin.bias is not None).to(dev)
            nl.weight.data=w
            if lin.bias is not None: nl.bias.data=torch.cat([lin.bias.data, newb if newb is not None else torch.zeros(add,device=dev)])
            return nl
        def grow_out(lin, add):  # (out, in=d_state): add ZERO cols (no disruption)
            w=torch.cat([lin.weight.data, torch.zeros(lin.weight.shape[0], add, device=dev)],1)
            nl=nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None).to(dev)
            nl.weight.data=w
            if lin.bias is not None: nl.bias.data=lin.bias.data
            return nl
        r=rows or {}
        s.Wk=grow_in(s.Wk,chunk,r.get('Wk')); s.Wv=grow_in(s.Wv,chunk,r.get('Wv')); s.gate=grow_in(s.gate,chunk,r.get('gate'),r.get('gate_b'))
        s.Wout=grow_out(s.Wout,chunk); s.d_state+=chunk
        # extend all per-unit buffers (new units: zero usage/home, UNFROZEN)
        s.unit_usage=torch.cat([s.unit_usage, torch.zeros(chunk,device=dev)])
        s.unit_home=torch.cat([s.unit_home, torch.zeros(chunk,device=dev)])
        s.frozen_units=torch.cat([s.frozen_units, torch.zeros(chunk,dtype=torch.bool,device=dev)])
        # new units get HIGH affinity (they open for DISTINCT content, which triggered widening)
        new_aff=nn.Parameter(torch.cat([s.unit_affinity.data, torch.ones(chunk,device=dev)*2.0]))
        s.unit_affinity=new_aff
        s.unit_aff=torch.cat([s.unit_aff, -torch.ones(chunk,device=dev)])
        s.unit_theta0=torch.cat([s.unit_theta0, sample_theta(chunk,dev)])
        s.unit_imp=torch.cat([s.unit_imp, torch.zeros(chunk,device=dev)])
        s.ssm_drive=torch.cat([s.ssm_drive, torch.zeros(chunk,device=dev)])
        s.ssm_drive_ema=torch.cat([s.ssm_drive_ema, torch.zeros(chunk,device=dev)])
        s.ssm_drive_slow=torch.cat([s.ssm_drive_slow, torch.zeros(chunk,device=dev)])
        s.ssm_drive_peak=torch.cat([s.ssm_drive_peak, torch.zeros(chunk,device=dev)])
        s.unit_sens=torch.cat([s.unit_sens, torch.zeros(chunk,device=dev)])
    @torch.no_grad()
    def widen_mlp(s, n, dev, w_in, b_in):
        """Activate n MLP units: given input rows/bias, ZERO output columns (no disruption), fresh buffers."""
        nl=nn.Linear(s.d_model, s.mlp_hidden+n).to(dev)
        nl.weight.data=torch.cat([s.mlp_in.weight.data, w_in],0); nl.bias.data=torch.cat([s.mlp_in.bias.data, b_in]); s.mlp_in=nl
        no=nn.Linear(s.mlp_hidden+n, s.d_model).to(dev)
        no.weight.data=torch.cat([s.mlp_out.weight.data, torch.zeros(s.d_model,n,device=dev)],1); no.bias.data=s.mlp_out.bias.data; s.mlp_out=no
        s.mlp_hidden+=n; z=lambda: torch.zeros(n,device=dev)
        for nm in ('mlp_usage','mlp_home','mlp_imp','mlp_drive','mlp_drive_ema','mlp_drive_slow','mlp_drive_peak','mlp_sens'):
            setattr(s,nm,torch.cat([getattr(s,nm),z()]))
        s.mlp_frozen=torch.cat([s.mlp_frozen, torch.zeros(n,dtype=torch.bool,device=dev)])
        s.mlp_aff=torch.cat([s.mlp_aff, -torch.ones(n,device=dev)])
        s.mlp_theta0=torch.cat([s.mlp_theta0, sample_theta(n,dev)])
        s.mlp_affinity=nn.Parameter(torch.cat([s.mlp_affinity.data, torch.ones(n,device=dev)*2.0]))

# ---------------- DORMANT RESERVE + UNIQUENESS-DRIVEN ACTIVATION ----------------
# The network is allocated at reserve_mult x its base size from the start (state units and MLP units;
# residual channels stay fixed). Only an initial set computes; the rest are DORMANT (fixed random receptive
# fields, no compute, no writes). Each block runs its own uniqueness detector: the whitened (Mahalanobis)
# distance of each sequence's mean input from everything the block has seen, relative to its typical
# novelty. When the ratio passes uniq_thr, the block activates a number of dormant units proportional to
# how unique the content is: the ones whose receptive fields respond most to what is NEW in the input
# (input minus the familiar mean). Activated units start with zero output weights (no disruption).
class Reserve:
    def __init__(s, d, n_state, n_mlp, dev):
        s.Wk=torch.randn(n_state,d,device=dev)*0.02; s.Wv=torch.randn(n_state,d,device=dev)*0.02
        s.gate=torch.randn(n_state,d,device=dev)*0.02; s.gate_b=torch.zeros(n_state,device=dev)
        bd=1/math.sqrt(d); s.min=(torch.rand(n_mlp,d,device=dev)*2-1)*bd; s.min_b=(torch.rand(n_mlp,device=dev)*2-1)*bd
    def take_state(s, idx):
        keep=torch.ones(s.Wk.shape[0],dtype=torch.bool,device=s.Wk.device); keep[idx]=False
        rows={'Wk':s.Wk[idx],'Wv':s.Wv[idx],'gate':s.gate[idx],'gate_b':s.gate_b[idx]}
        s.Wk,s.Wv,s.gate,s.gate_b=s.Wk[keep],s.Wv[keep],s.gate[keep],s.gate_b[keep]; return rows
    def take_mlp(s, idx):
        keep=torch.ones(s.min.shape[0],dtype=torch.bool,device=s.min.device); keep[idx]=False
        w,b=s.min[idx],s.min_b[idx]; s.min,s.min_b=s.min[keep],s.min_b[keep]; return w,b

class Uniq:
    """Per-block uniqueness detector on sequence-mean input features."""
    def __init__(s, d, dev):
        s.mu=torch.zeros(d,device=dev); s.S=torch.zeros(d,d,device=dev); s.mu_m=torch.zeros(d,device=dev)
        s.Sinv=None; s.n=0; s.nov_s=None; s.base=None; s.last=float('nan')
    @torch.no_grad()
    def update(s, h, hm, cfg):
        f=h.float().mean(1); B,d=f.shape
        if s.Sinv is not None:
            dl=f-s.mu; nov=float(((dl@s.Sinv)*dl).sum(-1).mean())/d; s.last=nov
            s.nov_s=nov if s.nov_s is None else 0.9*s.nov_s+0.1*nov
            s.base=s.nov_s if s.base is None else (1-cfg.uniq_base_rate)*s.base+cfg.uniq_base_rate*s.nov_s
        a=max(cfg.uniq_rate,1.0/(s.n+1))
        s.mu=(1-a)*s.mu+a*f.mean(0); dc=f-s.mu; s.S=(1-a)*s.S+a*(dc.T@dc)/B
        s.mu_m=(1-a)*s.mu_m+a*hm.float().mean((0,1)); s.n+=1
        if (s.Sinv is None and s.n>=cfg.uniq_min_stats) or (s.Sinv is not None and s.n%cfg.recruit_every==0):
            S_=torch.nan_to_num(s.S.double()); S_=0.5*(S_+S_.T)                 # double precision (the fp32 GPU solver can fail on near-degenerate S)
            try: lam,V=torch.linalg.eigh(S_)
            except Exception:
                try: lam,V=torch.linalg.eigh(S_.cpu()+1e-9*torch.eye(S_.shape[0],dtype=S_.dtype)); lam,V=lam.to(S_.device),V.to(S_.device)
                except Exception: return                                      # keep the previous inverse
            lam=lam.clamp(min=0)+cfg.uniq_shrink*lam.mean().clamp(min=1e-12)
            s.Sinv=((V/lam)@V.T).to(s.S.dtype)
    def ratio(s): return s.nov_s/s.base if (s.base is not None and s.base>0) else 1.0

@torch.no_grad()
def _select(resp_rows, feats, center, n, cfg, gen):
    """Pick n reserve units: highest mean |response| to the NOVEL part of the input (or random)."""
    X=feats.reshape(-1,feats.shape[-1]).float(); X=X[::max(1,X.shape[0]//2048)]-center
    return torch.topk((X@resp_rows.T).abs().mean(0),n).indices

@torch.no_grad()
def recruit(model, cfg, forced=None):
    """Uniqueness-driven activation (forced: {'state':n,'mlp':n} for the eligibility fallback).
    Returns list of (block, n_state, n_mlp, ratio)."""
    out=[]
    for li,(b,R,U) in enumerate(zip(model.blocks,model._reserve,model._uniq)):
        r=U.ratio()
        if forced is None:
            if U.n<cfg.uniq_warmup or r<cfg.uniq_thr: continue
            amt=min(r-1.0,2.0)                                   # proportional to how far above typical novelty
            ns=int(round(cfg.recruit_rate*cfg.d_state*amt)); nm=int(round(cfg.recruit_rate*2*cfg.d_model*amt))
        else: ns,nm=forced['state'],forced['mlp']
        ns=min(ns,R.Wk.shape[0]); nm=min(nm,R.min.shape[0])
        if ns<=0 and nm<=0: continue
        dev=b.Wk.weight.device
        if ns>0: b.widen(ns,dev,rows=R.take_state(_select(R.Wv,b._uh,U.mu,ns,cfg,model._rgen)))
        if nm>0:
            w,bb=R.take_mlp(_select(R.min,b._um,U.mu_m,nm,cfg,model._rgen)); b.widen_mlp(nm,dev,w,bb)
        out.append((li,ns,nm,r))
    return out

# ---------------- PARALLEL SLOW NETWORK / FAST-SLOW WEIGHTS (replay substitutes) ----------------
_HEAVY=('_sm','_sm_log','_tr','_hyb','_vals_share','_share_res','_bank','_reserve','_uniq','_orth','_og','_og_out','_og_common','_recruit_log','_polar_log','_slow','_slow_list','_fast','_replay_rng','_replay_buffer','_shunt','_recent','_live','_cur_cov','_old_cov','_last_cov','_dom_cov','_mid','_beta_log','_olds','_visit','_cov_handles','_tm','_tm_dream','_timing_log','_dpick','_cov_on','_topics','_klw','_klw_opt','_merge_log','_vals','_recon_stat','_recon_log','_recon_share','_recon_unew','_sleep_log','_stale_log','_mk_gen','_battr_acc','_bw_dyn','_routers','_ropt','_rver','_rcache','_bw_map','_tcombine_log','_tpeak','_tkl','_tdream','_trng','_tmerge_flag','_nb','_surprise_flag','_tswitch_log','_tdom_start','_tuse','_tpi','_tshunt_on','_tlast_merge','_tmerge_log','_tspawn_log','_gstep','_last_nll','_nll_ema','_tlast_surprise','_dworst','_dfilt','_dmax','_dlam','_dal','_snapring','_rb_fire','_rb_last','_rb_log','_spec','_cross','_xcur','_live_dom','_cross_log','_dreams','_dream_rng','_dream_gen','_dreamstat','_klstat','_klam','_bg','_fs','_last_h','_keep_h','_curve','_curve_val','_teach','_hstk','_dream_tgt','_dqueue','_dq_rng','_klacc','_mids','_dstk','_dtstk','_dscore','_prio_ix','_prio_gen','_prio_ess','_ctxacc','_surpacc','_cur_mean','_old_mean','_dom_mean','_onpol','_onpol_gen','_onpol_pick','_opstat','_op_n','_fastat','_upper_ids','_vals_diag','_fidlog','_klpos_log','_diag_gen','_dstream','_dsq','_dsr','_dstream_last','_sabasis','_hconf','_hconf_last','_strandcnt','_strandstat','_srng','_fdrift_log','_sastat','_shbasis','_shdiag','_shstat','_fsplit_log','_vlog','_vbank','_vslopes','_dsteer','_dsteer_acc','_cov_log','_dcls','_iwtab','_spec_log','_vstart','_gcache','_lwfG','_nstat','_ng_touch','_ng_state','_ng_cur','_ng_old','_ng_sc','_cnt_old','_cnt_cur','_ng_batch','_ng_lrms','_ng_lstate','_ng_dream','_ng_lwf','_lwfwacc','_relaxlog')

def _pmap(model): return {id(p_):p_ for p_ in model.parameters()}

def _lab(cfg,d): return PILE_DOMAINS[d][:6]

def domain_delta(model, d):
    """Domain d's own configuration relative to the shared main weights: its specific delta + its cross gains x
    the earlier domains' specific deltas."""
    spec=getattr(model,'_spec',None) or {}; out={}
    for k,v in spec.get(d,{}).items(): out[k]=v.clone()
    X=(getattr(model,'_cross',None) or {}).get(d)
    if X:
        for jj,j in enumerate(X['olds']):
            for kk,k in enumerate(X['keys']):
                if k in spec.get(j,{}) and X['A'][jj,kk]!=0:
                    add=X['A'][jj,kk]*spec[j][k]; out[k]=out[k]+add if k in out else add.clone()
    return out

def current_mix(model):
    """The cross-connection contribution currently in the live weights (domain being trained)."""
    X=getattr(model,'_xcur',None); spec=getattr(model,'_spec',None) or {}; out={}
    if not X: return out
    for jj,j in enumerate(X['olds']):
        for kk,k in enumerate(X['keys']):
            a=float(X['A'][jj,kk])
            if a!=0 and k in spec.get(j,{}):
                add=a*spec[j][k]; out[k]=out[k]+add if k in out else add.clone()
    return out

@torch.no_grad()
def apply_delta(model, delta, sign):
    P=_pmap(model)
    for k,v in delta.items():
        if k in P and P[k].shape==v.shape: P[k].add_(sign*v)

@torch.no_grad()
def dream_sample(model, d, n, T, temp, start=10, gen=None):
    """Old head d generates n byte sequences of length T from the CURRENT weights (callers remove the shunt first)."""
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    x=torch.full((n,1),start,dtype=torch.long,device=model.embed.weight.device)
    for _ in range(T-1):
        hd=model.heads[d]
        lg=hd(priv(model,x,model.features(x),d))[:,-1].float()
        lg=lg/temp; x=torch.cat([x,torch.multinomial(F.softmax(lg,-1),1,generator=gen)],1)
    model.feat_scale.copy_(fs); model.train(was); return x

# ---------------- PERFORMANCE: FROZEN TEACHERS, STACKED TEACHER PASS, RECURRENT DREAM SAMPLER ----------------
_TEACH_DROP=('_pg','_pg_maha','_bstat','_uh','_um','_probe')

@torch.no_grad()
def _frozen_blocks(model, subs):
    """Frozen copy of model.blocks with every {id(param): delta} dict in subs subtracted (shape-checked, as in the weight-swapping
    code). No hooks, no gradients, eval mode."""
    import copy
    saved=[{a:b.__dict__.pop(a) for a in _TEACH_DROP if a in b.__dict__} for b in model.blocks]
    try: cp=copy.deepcopy(model.blocks)
    finally:
        for b,sv in zip(model.blocks,saved): b.__dict__.update(sv)
    for m in cp.modules():
        for nm in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_forward_hooks_with_kwargs','_forward_pre_hooks_with_kwargs','_forward_hooks_always_called'):
            if isinstance(getattr(m,nm,None),dict): getattr(m,nm).clear()
    for lp,cp_ in zip(model.blocks.parameters(),cp.parameters()):
        cp_.requires_grad_(False)
        for D in subs:
            d_=D.get(id(lp)) if D else None
            if d_ is not None and d_.shape==cp_.shape: cp_.sub_(d_)
    for b in cp: b.cur_distinct=None; b.calib=None; b._s0_idx=None
    cp.eval(); return cp

def _stk_ln(x, w, b): return F.layer_norm(x,(x.shape[-1],))*w[:,None,None,:]+b[:,None,None,:]
def _stk_lin(z, W, b=None):
    """Stacked linear for M models: z (M,B,T,in), W (M,in,out) [pre-transposed], b (M,out) -> (M,B,T,out)."""
    M,B,T,_=z.shape; zf=z.reshape(M,B*T,-1)
    y=torch.baddbmm(b.unsqueeze(1),zf,W) if b is not None else torch.bmm(zf,W)
    return y.reshape(M,B,T,-1)
def _stk_conv(z, w, prev=False):
    """Causal depthwise convolution per model: z (M,B,T,C), w (M*C,1,K). prev=True: previous K values only (local filter blocks)."""
    M,B,T,C=z.shape; K=w.shape[-1]
    zt=F.pad(z.permute(1,0,3,2).reshape(B,M*C,T),(K if prev else K-1,0))
    o=F.conv1d(zt,w.to(zt.dtype),groups=M*C)
    if prev: o=o[...,:T]
    return o.reshape(B,M,C,T).permute(1,0,3,2)

def _cons_levels(model, cfg):
    """Mid-store levels from most recent to slowest: three timescales = [mid]; timescale chain = [L1, L2, ...] (non-empty prefix)."""
    L=[]
    out=[]
    for l_ in L:
        if not l_: break
        out.append(l_)
    return out

def teacher_role(model, cfg, d, olds, TT):
    """Teacher for old domain d: 'main' (recent), or 'lvl<k>' = main without the k most recent mid-store levels. Three timescales: age
    >= cons3_age. Timescale chain: the DEEPEST teacher that still contains >= cons_chain_keep of the domain's own learning (so older
    domains are held to slower memories without being pulled back from what they learned themselves)."""
    if d not in olds: return 'main'
    age=len(olds)-1-olds.index(d)
    k=min(1 if age>=cfg.cons3_age else 0, TT.depth)
    return 'main' if k==0 else f'lvl{k}'

class Teachers:
    """FROZEN TEACHERS for one visit. role 'main' = live weights minus the shunt (= the weights at the start of the visit: everything the
    visit writes into the blocks goes through the shunt); roles 'lvl1', 'lvl2', ... = main minus the 1, 2, ... most recent mid-store levels
    (three timescales: lvl1 = the slow weights). The byte and position embeddings are read LIVE unless they are in the shunt (emb_shunt),
    then each teacher has its own frozen copy."""
    def __init__(s, model, cfg, roles):
        s.model=model; s.cfg=cfg; s.blocks={r:_frozen_blocks(model,subs) for r,subs in roles.items()}; s._stk={}
        s.emb=None
        s.depth=len(roles)-1
        s.stackable=all(not getattr(b,'mq',False) and not b.route_in_forward and not getattr(b,'lattn',False) and not getattr(b,'nov_g',None) for b in model.blocks)
        if getattr(model,'ngram',None) is not None:                 # C1: frozen n-gram tables for this visit's teachers
            import copy
            s.ngram=copy.deepcopy(model.ngram)
            for p_ in s.ngram.parameters(): p_.requires_grad_(False)
    def _ng(s):                                                   # n-gram tables: the visit's frozen copy (live for ad-hoc samplers)
        return getattr(s,'ngram',None) if hasattr(s,'ngram') else getattr(s.model,'ngram',None)
    def _emb(s, x, role):
        T=x.size(1); ng=s._ng()
        if s.emb is None: e=s.model.embed(x)+s.model.pos(torch.arange(T,device=x.device))
        else: E,P_=s.emb[role]; e=F.embedding(x,E)+P_[:T]
        if getattr(s,'cnt',None) is not None: e=e+count_feature(s.model,s.cfg,_cols(x,max(s.cfg.ngram_count)),C=s.cnt,proj=s.cproj)
        return e+ngram_embed(ng,x) if ng is not None else e
    def _emb_tok(s, x, t, role, hist=None):                       # one position (recurrent sampler)
        ng=s._ng()
        if s.emb is None: e=s.model.embed(x)+s.model.pos.weight[t]
        else: E,P_=s.emb[role]; e=F.embedding(x,E)+P_[t]
        if ng is not None or getattr(s,'cnt',None) is not None:
            if hist is None: hist=torch.full((x.shape[0],_hist_len(s.model)),-1,dtype=torch.long,device=x.device)
        if ng is not None: e=e+ngram_embed_tok(ng,x,hist[:,hist.shape[1]-(max(ng.ns)-1):] if max(ng.ns)>1 else hist[:,:0])
        if getattr(s,'cnt',None) is not None:
            cols=[x]+[hist[:,-i].unsqueeze(1) for i in range(1,max(s.cfg.ngram_count))]
            e=e+count_feature(s.model,s.cfg,cols,C=s.cnt,proj=s.cproj)
        return e
    def _ac(s):
        return torch.autocast(device_type=s.model.embed.weight.device.type, dtype=torch.bfloat16) if s.cfg.amp=='bf16' else contextlib.nullcontext()
    def _out(s, h):
        h=h.float()
        return h
    @torch.no_grad()
    def features(s, x, role):
        h=s._emb(x,role)
        with s._ac():
            for b in s.blocks[role]: h=b(h)
        return s._out(h)
    @torch.no_grad()
    def features_taps(s, x, role, lis):
        """Residual stream after each block in lis (one pass)."""
        h=s._emb(x,role); out={}
        with s._ac():
            for i,b in enumerate(s.blocks[role][:max(lis)+1]):
                h=b(h)
                if i in lis: out[i]=h.float()
        return out
    @torch.no_grad()
    def features_upto(s, x, role, li):
        """Residual stream after block li of teacher `role` (feature anchoring)."""
        h=s._emb(x,role)
        with s._ac():
            for b in s.blocks[role][:li+1]: h=b(h)
        return h.float()
    @torch.no_grad()
    def _stack(s, roles):
        key=tuple(roles)
        if key in s._stk: return s._stk[key]
        lp=torch.bfloat16 if s.cfg.amp=='bf16' else None             # matmul weights pre-cast exactly as autocast would cast them
        cv=lambda t_: t_.to(lp).contiguous() if lp is not None else t_.contiguous()
        out=[]
        for li in range(len(s.model.blocks)):
            bs=[s.blocks[r][li] for r in roles]; b0=bs[0]
            st=lambda f: torch.stack([f(b) for b in bs])
            loc=bool(getattr(b0,'loc',False))
            P=dict(ln_w=st(lambda b:b.ln.weight),ln_b=st(lambda b:b.ln.bias),norm_w=st(lambda b:b.norm.weight),norm_b=st(lambda b:b.norm.bias),
                   Wv=cv(st(lambda b:b.Wv.weight.T)),gate_w=cv(st(lambda b:b.gate.weight.T)),gate_b=cv(st(lambda b:b.gate.bias)),
                   Wout=cv(st(lambda b:b.Wout.weight.T)),mi_w=cv(st(lambda b:b.mlp_in.weight.T)),mi_b=cv(st(lambda b:b.mlp_in.bias)),
                   mo_w=cv(st(lambda b:b.mlp_out.weight.T)),mo_b=cv(st(lambda b:b.mlp_out.bias)))
            mh=getattr(b0,'mh',1); mk=getattr(b0,'mk',0); mb=getattr(b0,'mb',False); md=getattr(b0,'md',False)
            if loc: P['cl']=cv(torch.cat([b.conv_loc for b in bs]))
            else:
                P['Wk']=cv(st(lambda b:b.Wk.weight.T))
                if mb: P['Wb_w']=cv(st(lambda b:b.Wb.weight.T)); P['Wb_b']=cv(st(lambda b:b.Wb.bias))
                else: P['beta']=st(lambda b:b.beta)
                if md: P['Wa_w']=cv(st(lambda b:b.Wa.weight.T)); P['Wa_b']=cv(st(lambda b:b.Wa.bias))
                if mk>0: P['ck']=cv(torch.cat([b.conv_k for b in bs])); P['cvv']=cv(torch.cat([b.conv_v for b in bs]))
            P['meta']=(mh,mk,mb,md,getattr(b0,'bmax',1.0),bool(getattr(b0,'gate_ln',False)),getattr(b0,'_kern','torch'),b0.d_state,loc)
            P['par']=bool(getattr(b0,'parallel',False))
            out.append(P)
        s._stk[key]=out; return out
    @torch.no_grad()
    def features_multi(s, x, roles):
        """Features of several teachers on the same input: ONE stacked forward pass (batched matmuls over the teachers)."""
        if len(roles)<=1 or not s.stackable: return [s.features(x,r) for r in roles]
        Ps=s._stack(roles); M=len(roles); B,T=x.shape
        if s.emb is None: h=s._emb(x,roles[0]).unsqueeze(0).expand(M,-1,-1,-1).contiguous()
        else: h=torch.stack([s._emb(x,r) for r in roles])
        with s._ac():
            for li_,P in enumerate(Ps):
                mh,mk,mb,md,bmax,gln,kern,C,loc=P['meta']
                hl=_stk_ln(h,P['ln_w'],P['ln_b'])
                v=_stk_lin(hl,P['Wv'])
                if loc: pred=_stk_conv(v,P['cl'],prev=True).float()
                else:
                    k=_stk_lin(hl,P['Wk'])
                    if mk>0: k=_stk_conv(k,P['ck']); v=_stk_conv(v,P['cvv'])
                    bt=bmax*torch.sigmoid(_stk_lin(hl,P['Wb_w'],P['Wb_b'])) if mb else torch.sigmoid(P['beta'].float())[:,None,None,None].expand(M,B,T,mh)
                    la=-F.softplus(_stk_lin(hl,P['Wa_w'],P['Wa_b'])) if md else None
                    pred=_delta_core(k.reshape(M*B,T,C),v.reshape(M*B,T,C),bt.reshape(M*B,T,mh),la.reshape(M*B,T,mh) if md else None,
                                     mh,None,M*B,T,kern).reshape(M,B,T,C)
                g_=_stk_lin(hl if gln else h,P['gate_w'],P['gate_b'])
                h0=h; h=h+_stk_lin(pred*F.silu(g_),P['Wout'])
                hn=_stk_ln(h0 if P.get('par') else h,P['norm_w'],P['norm_b'])
                mm_=F.gelu(_stk_lin(hn,P['mi_w'],P['mi_b']))
                if _XDG['on']: mm_=xdg_apply(mm_,li_)
                h=h+_stk_lin(mm_,P['mo_w'],P['mo_b'])
        return [s._out(h[m]) for m in range(M)]

@torch.no_grad()
def _rec_step(b, x, st):
    """One token through block b in RECURRENT form (eval): x (B,1,d) residual; st = this block's state (memory S per head, the last
    conv-width keys/values). Same arithmetic as the parallel WY solve: S <- a S; out = S k; S <- S + beta (v - out) k^T."""
    hl=b.ln(x); Bn=x.shape[0]; C=b.d_state; x0=x
    if getattr(b,'lattn',False):                                  # C2: sliding-window attention over a ring of the last W keys / values
        mh=getattr(b,'mh',1); mk=getattr(b,'mk',0); dh=C//mh; Wn=b.la_W; R=Bn*mh
        k=b.Wk(hl); v=b.Wv(hl)
        if mk>0:
            if 'bk' not in st: st['bk']=torch.zeros(Bn,mk,C,dtype=k.dtype,device=k.device); st['bv']=torch.zeros(Bn,mk,C,dtype=v.dtype,device=v.device)
            st['bk']=torch.cat([st['bk'][:,1:],k],1); st['bv']=torch.cat([st['bv'][:,1:],v],1)
            k=F.conv1d(st['bk'].transpose(1,2),b.conv_k.to(k.dtype),groups=C).transpose(1,2)
            v=F.conv1d(st['bv'].transpose(1,2),b.conv_v.to(v.dtype),groups=C).transpose(1,2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            kr=F.normalize(k.float().reshape(R,dh),dim=-1); vr=v.float().reshape(R,dh)
            if 'K' not in st: st['K']=torch.zeros(R,Wn,dh,device=x.device); st['V']=torch.zeros(R,Wn,dh,device=x.device); st['cnt']=0
            sc=torch.bmm(st['K'],kr.unsqueeze(-1)).squeeze(-1)*torch.exp(b.la_scale.float()).repeat(Bn)[:,None]+b.la_rel.float().flip(-1).repeat(Bn,1)
            valid=torch.arange(Wn,device=x.device)>=Wn-st['cnt']
            sc=sc.masked_fill(~valid[None],float('-inf'))
            Pm=F.softmax(torch.cat([sc,torch.zeros_like(sc[:,:1])],-1),-1)[:,:Wn]
            out=torch.bmm(Pm.unsqueeze(1),st['V']).squeeze(1).reshape(Bn,1,C)
            st['K']=torch.cat([st['K'][:,1:],kr.unsqueeze(1)],1); st['V']=torch.cat([st['V'][:,1:],vr.unsqueeze(1)],1); st['cnt']+=1
    elif getattr(b,'loc',False):                                  # LOCAL FILTER: the previous K values
        v=b.Wv(hl); K=b.conv_loc.shape[-1]
        if 'bl' not in st: st['bl']=torch.zeros(Bn,K,C,dtype=v.dtype,device=v.device)
        out=F.conv1d(st['bl'].transpose(1,2),b.conv_loc.to(v.dtype),groups=C).transpose(1,2).float()
        st['bl']=torch.cat([st['bl'][:,1:],v],1)
    else:
        mh=getattr(b,'mh',1); mk=getattr(b,'mk',0); mb=getattr(b,'mb',False); md=getattr(b,'md',False)
        k=b.Wk(hl); v=b.Wv(hl); dh=C//mh
        if mk>0:
            if 'bk' not in st: st['bk']=torch.zeros(Bn,mk,C,dtype=k.dtype,device=k.device); st['bv']=torch.zeros(Bn,mk,C,dtype=v.dtype,device=v.device)
            st['bk']=torch.cat([st['bk'][:,1:],k],1); st['bv']=torch.cat([st['bv'][:,1:],v],1)
            k=F.conv1d(st['bk'].transpose(1,2),b.conv_k.to(k.dtype),groups=C).transpose(1,2)
            v=F.conv1d(st['bv'].transpose(1,2),b.conv_v.to(v.dtype),groups=C).transpose(1,2)
        bt=b.bmax*torch.sigmoid(b.Wb(hl)) if mb else None
        la=-F.softplus(b.Wa(hl)) if md else None
        dp2_=bool(getattr(b,'dp2',False))
        if dp2_: k2_=b.Wk2(hl); v2_=b.Wv2(hl); bt2_=b.bmax*torch.sigmoid(b.Wb2(hl))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            if dp2_: k2f=F.normalize(k2_.float().reshape(Bn*mh,dh),dim=-1); v2f=v2_.float().reshape(Bn*mh,dh); be2=bt2_.float().reshape(Bn*mh,1)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(b.beta.float()).expand(Bn,mh)
            R=Bn*mh
            if _SAMP['W']>0:                                               # WINDOWED SAMPLER: S = sum_j exp(L_now - L_j) u_j k_j^T, last W
                Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
                if 'U' not in st:
                    st['U']=torch.zeros(R,Wn,dh,dtype=dt_,device=x.device); st['K']=torch.zeros(R,Wn,dh,dtype=dt_,device=x.device)
                    st['Lw']=torch.full((R,Wn),1e9,device=x.device); st['Lc']=torch.zeros(R,device=x.device); st['p']=0
                if md: st['Lc']=st['Lc']+la.float().reshape(R)
                kr=kf.reshape(R,dh); w_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))              # decay since each write
                if _SAMP.get('mm') and dt_!=torch.float32:                                                # window products in bf16 (no fp32
                    sc=torch.bmm(st['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_                 #   copies of the whole window per step)
                    pred=torch.bmm(sc.to(dt_).unsqueeze(1),st['U']).squeeze(1).float()
                else:
                    sc=torch.bmm(st['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_                        # (R, W)
                    pred=torch.bmm(sc.unsqueeze(1),st['U'].float()).squeeze(1)                           # a_t S_{t-1} k_t
                sl=st['p']%Wn; st['U'][:,sl]=((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_); st['K'][:,sl]=_kw_rec(st,kr).to(dt_)
                st['Lw'][:,sl]=st['Lc']; st['p']+=1
                if dp2_:                                                       # DELTAPRODUCT second write (no decay in between)
                    w2_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))
                    p2_=torch.bmm((torch.bmm(st['K'].float(),k2f.unsqueeze(-1)).squeeze(-1)*w2_).unsqueeze(1),st['U'].float()).squeeze(1)
                    sl=st['p']%Wn; st['U'][:,sl]=((v2f-p2_)*be2).to(dt_); st['K'][:,sl]=k2f.to(dt_); st['Lw'][:,sl]=st['Lc']; st['p']+=1
            else:
                S=st.get('S')
                if S is None: S=st['S']=torch.zeros(R,dh,dh,device=x.device)       # (rows x heads, value dim, key dim), updated in place
                if md: S.mul_(torch.exp(la.float().reshape(R))[:,None,None])
                kr=kf.reshape(R,dh)
                pred=torch.bmm(S,kr.unsqueeze(-1)).squeeze(-1)                     # a S_{t-1} k_t
                S.baddbmm_(((vf.reshape(R,dh)-pred)*be.reshape(R,1)).unsqueeze(-1),_kw_rec(st,kr).unsqueeze(1))   # + beta (v - S k) k_w^T
                if dp2_:                                                       # DELTAPRODUCT second write
                    p2_=torch.bmm(S,k2f.unsqueeze(-1)).squeeze(-1); S.baddbmm_(((v2f-p2_)*be2).unsqueeze(-1),k2f.unsqueeze(1))
            out=pred.reshape(Bn,1,C)
    g_=b.gate(hl if getattr(b,'gate_ln',False) else x)
    x=x+b.Wout(out*F.silu(g_)); hn=b.norm(x0 if getattr(b,'parallel',False) else x)   # C3: parallel block reads the block input
    x=x+b.mlp_out(F.gelu(b.mlp_in(film_mod(b,hn) if _FILM.get('codes') is not None else hn)))
    if getattr(b,'pu_in',None) is not None and _XDG.get('tagon') and isinstance(_XDG.get('rows'),(list,tuple)) and len(_XDG['rows'])==x.shape[0]:
        x=x+pu_term(b,hn).to(x.dtype)                                 # PRIVATE TRUNK UNITS in the recurrent sampler (rows tagged by their domain)
    return x+b.nov_term(hn,rec=True) if getattr(b,'nov_g',None) else x

def _lowrank_(S, r):
    """In place: each (dv x dk) state matrix replaced by its best rank-r approximation."""
    if r>=min(S.shape[-2:]): return
    with torch.autocast(device_type=S.device.type, enabled=False):              # fp32 (the sampler runs inside bf16 autocast; QR has no bf16)
        U,sv,V=torch.svd_lowrank(S.float(),q=min(r+6,min(S.shape[-2:])),niter=2)   # randomised (the states are close to low rank)
        S.copy_(((U[...,:r]*sv[...,None,:r])@V[...,:r].transpose(-1,-2)).to(S.dtype))

@torch.no_grad()
def state_spectrum_diag(model, cfg, olds, n_rows=None):
    """STATE SPECTRUM while dreaming (diagnostic; own random stream). 1) Exact recurrent sampler from the visit-start teacher, rows spread
    over the old domains: at the end, per block, the singular-value spectrum of every head's state matrix (rank holding 90% / 99% of the
    energy, participation ratio), mean decay (memory timescale 1/(1-a)) and mean beta. 2) The same dreams re-run teacher-forced with a
    CHEAPER state - truncated to rank r after every update (all blocks, or one block at a time at the smallest rank), or stored in bf16 - and
    the KL of its next-byte predictions to the exact ones (and top-byte agreement)."""
    TT=getattr(model,'_teach',None)
    if TT is None or not olds: return
    n_rows=n_rows or cfg.state_spec_rows
    dev=model.embed.weight.device; T=min(cfg.dream_len,cfg.seq_len); blocks=TT.blocks['main']
    doms=sorted([olds[i%len(olds)] for i in range(n_rows)])
    g_=torch.Generator(device=dev); g_.manual_seed(cfg.seed+53)
    stat=[{'a':0.0,'b':0.0,'n':0} for _ in blocks]
    def run(x_forced, comp):
        sts=[{} for _ in blocks]; out=[]; x=x_forced[:,:1] if x_forced is not None else torch.full((n_rows,1),10,dtype=torch.long,device=dev); seq=[x]
        groups=[]; i0=0
        for i in range(1,n_rows+1):
            if i==n_rows or doms[i]!=doms[i0]: groups.append((doms[i0],slice(i0,i))); i0=i
        hist=None
        for t in range(T):
            h=TT._emb_tok(x,t,'main',hist); hist=_hist_push(TT,hist,x)
            with TT._ac():
                for li,(b,st) in enumerate(zip(blocks,sts)):
                    if comp is None and x_forced is None and getattr(b,'md',False):      # exact pass: decay and beta statistics
                        hl=b.ln(h); a_=torch.exp(-F.softplus(b.Wa(hl).float())); stat[li]['a']+=float(a_.mean())
                        if getattr(b,'mb',False): stat[li]['b']+=float((b.bmax*torch.sigmoid(b.Wb(hl).float())).mean())
                        stat[li]['n']+=1
                    h=_rec_step(b,h,st)
                    if comp is not None and 'S' in st:
                        kind,r,which=comp
                        if which is None or which==li:
                            if kind=='rank': _lowrank_(st['S'],r)
                            elif kind=='bf16': st['S'].copy_(st['S'].bfloat16().float())
            hh=TT._out(h)[:,0]
            lg=torch.cat([model.heads[d](hh[sl]) for d,sl in groups]).float(); out.append(F.log_softmax(lg,-1))
            if t<T-1:
                x=x_forced[:,t+1:t+2] if x_forced is not None else torch.multinomial(F.softmax(lg/cfg.dream_temp,-1),1,generator=g_)
                seq.append(x)
        return torch.cat(seq,1), torch.stack(out,1), sts
    xs,lp0,sts=run(None,None)
    lines=[]
    for li,(b,st) in enumerate(zip(blocks,sts)):
        if 'S' not in st: lines.append(f"b{li} local filter"); continue
        sv=torch.linalg.svdvals(st['S']).pow(2); sv=sv/sv.sum(-1,keepdim=True).clamp(min=1e-20)   # (rows*heads, d)
        cs=torch.cumsum(sv,-1); r90=float(((cs<0.90).sum(-1)+1).float().mean()); r99=float(((cs<0.99).sum(-1)+1).float().mean())
        pr=float((1.0/sv.pow(2).sum(-1).clamp(min=1e-20)).mean()); dh=sv.shape[-1]
        a_=stat[li]['a']/max(stat[li]['n'],1); b_=stat[li]['b']/max(stat[li]['n'],1)
        lines.append(f"b{li} rank90 {r90:.1f} rank99 {r99:.1f} PR {pr:.1f} of {dh} | decay {a_:.3f} (memory ~{1/max(1-a_,1e-6):.0f} bytes) beta {b_:.2f}")
    print("  [state spectrum] while dreaming, per block, each head's state matrix at the end of a dream: "+" || ".join(lines))
    burn=min(cfg.dream_burnin,T//2)
    def kl(lp):
        k=(lp0.exp()*(lp0-lp)).sum(-1)[:,burn:]; ag=(lp0.argmax(-1)==lp.argmax(-1)).float()[:,burn:]
        return float(k.mean()), float(ag.mean())
    res=[]
    for r in cfg.state_spec_ranks: _,lp,_=run(xs,('rank',r,None)); res.append((f"rank {r}",)+kl(lp))
    _,lp,_=run(xs,('bf16',0,None)); res.append(("bf16",)+kl(lp))
    r0=min(cfg.state_spec_ranks); per=[]
    for li in range(len(blocks)): _,lp,_=run(xs,('rank',r0,li)); per.append((li,)+kl(lp))
    print("  [state spectrum] cheaper sampler vs exact on the same dreams, KL of next-byte predictions / top-byte agreement: "
          +" | ".join(f"{n} {k:.4f}/{a:.3f}" for n,k,a in res)
          +f" || rank {r0} in ONE block only: "+" ".join(f"b{li} {k:.4f}/{a:.3f}" for li,k,a in per))
    model.__dict__.setdefault('_spec_log',{})[getattr(model,'_visit',0)]={'lines':lines,'res':res,'per':per}

@torch.no_grad()
def rec_sample(TT, role, doms, T, temp, gen, tgt_T=None, start=10, prompt=None):
    sv_=(_XDG.get('rows'),_XDG.get('groups'),_XDG.get('rows_ix'))
    if _XDG.get('tagon'):
        _XDG['rows']=list(doms)                                          # per-row domain tags for private trunk units (pu_k)
        gr_=[]; i0=0; n_=len(doms)
        for i in range(1,n_+1):
            if i==n_ or doms[i]!=doms[i0]: gr_.append((int(doms[i0]),slice(i0,i))); i0=i
        _XDG['groups']=(n_,gr_); _XDG['rows_ix']=torch.tensor([int(d) for d in doms],device=TT.model.embed.weight.device)
    try: return _rec_sample(TT, role, doms, T, temp, gen, tgt_T, start, prompt)
    finally: _XDG['rows'],_XDG['groups'],_XDG['rows_ix']=sv_
def _rec_sample(TT, role, doms, T, temp, gen, tgt_T=None, start=10, prompt=None):
    """RECURRENT DREAM SAMPLER: rows = old domains (one head per row, rows grouped by domain), all generated together from teacher
    `role`, one token at a time with the recurrent mixer state (O(T) instead of O(T^2)). Same sampling as dream_sample (one multinomial
    draw per step over all rows). tgt_T: also return the teacher's log-probabilities log_softmax(logits / tgt_T) at every position."""
    model=TT.model; blocks=TT.blocks[role]; n=len(doms); dev=model.embed.weight.device
    groups=[]; i0=0
    for i in range(1,n+1):
        if i==n or doms[i]!=doms[i0]: groups.append((doms[i0],slice(i0,i))); i0=i
    P=0 if prompt is None else prompt.shape[1]                      # PROMPTED: the first P bytes are given (teacher-forced), the rest sampled
    x=prompt[:,:1] if P else torch.full((n,1),start,dtype=torch.long,device=dev); seq=[x]; sts=[{} for _ in blocks]
    tg=torch.empty(n,T,model.cfg.vocab,device=dev) if tgt_T else None
    hist=None; xw_=None; t=0
    while t<T:
        if _SAMP.get('graph') and P==0 and tg is None and t==int(_SAMP.get('gwarm',64)) and T-t>4:   # v9.8 cg8: the remaining bytes as graph replays
            x,t,hist,xw_=_rsg_run(TT,role,blocks,sts,groups,x,t,hist,xw_,T,temp,gen,seq)
            if t>=T: break
        lg,hist,xw_=_rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_)
        if tg is not None: tg[:,t]=F.log_softmax(lg/tgt_T,-1)
        if t<T-1:
            x=prompt[:,t+1:t+2] if t+1<P else torch.multinomial(F.softmax(lg/temp,-1),1,generator=gen)
            seq.append(x)
        t+=1
    xs=torch.cat(seq,1)
    if tg is not None and _SAMP['W']>0 and _SAMP['rescore']:          # EXACT targets: parallel teacher pass over the generated text
        tg=_rescore_targets(TT,role,groups,xs,tgt_T,tg)
    return xs, tg

def _rsg_msg():
    if not _SAMP.get('graph'): return ""
    g_=_SAMP.get('gstat') or {}
    if not g_.get('calls'): return " | sampler graph: not used"
    return (f" | sampler graph: {g_['ok']}/{g_['calls']} sampling calls replayed ({g_['bytes']} bytes), self-check bit-identical" if g_['ok']==g_['calls']
            else f" | sampler graph: {g_['ok']}/{g_['calls']} calls replayed, eager otherwise ({g_.get('why')})")

def _rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_):
    """One byte of the recurrent sampler (eager): logits for every row, state advanced in place / rebound in sts, hist, xw_."""
    model=TT.model
    h=TT._emb_tok(x,t,role,hist); hist=_hist_push(TT,hist,x)
    with TT._ac():
        for b,st in zip(blocks,sts): h=_rec_step(b,h,st)
    h=TT._out(h)[:,0]
    if model.cfg.ngp and model.__dict__.get('_ngp') is not None:   # PRIVATE N-GRAM ROWS in the recurrent sampler (last max(ngp) bytes)
        xw_=x if xw_ is None else torch.cat([xw_,x],1)[:,-max(model.cfg.ngp):]   # rolling window (was: whole history re-joined per byte)
        lg=torch.cat([model.heads[d](h[sl]+ngp_vec(model,xw_[sl],d)[:,-1]) for d,sl in groups]).float()
    else: lg=torch.cat([model.heads[d](h[sl]) for d,sl in groups]).float()
    return lg,hist,xw_

def _rec_tail(b,x,x0,hl,out):
    """Block output after the mixer (same arithmetic as _rec_step)."""
    g_=b.gate(hl if getattr(b,'gate_ln',False) else x)
    x=x+b.Wout(out*F.silu(g_)); hn=b.norm(x0 if getattr(b,'parallel',False) else x)
    x=x+b.mlp_out(F.gelu(b.mlp_in(film_mod(b,hn) if _FILM.get('codes') is not None else hn)))
    if getattr(b,'pu_in',None) is not None and _XDG.get('tagon') and isinstance(_XDG.get('rows'),(list,tuple)) and len(_XDG['rows'])==x.shape[0]:
        x=x+pu_term(b,hn).to(x.dtype)
    return x+b.nov_term(hn,rec=True) if getattr(b,'nov_g',None) else x

def _rec_step_ip(b,x,g,tb):
    """_rec_step with every state write IN PLACE at a device-side position tb (graph-capturable): local-filter blocks and windowed
    delta-rule blocks (decay, per-token beta, block-level write-key preconditioner whose D is refreshed on the host between replays)."""
    hl=b.ln(x); Bn=x.shape[0]; C=b.d_state; x0=x
    if getattr(b,'loc',False):
        v=b.Wv(hl)
        out=F.conv1d(g['bl'].transpose(1,2),b.conv_loc.to(v.dtype),groups=C).transpose(1,2).float()
        g['bl'].copy_(torch.cat([g['bl'][:,1:],v],1))
    else:
        mh=getattr(b,'mh',1); mb=getattr(b,'mb',False); md=getattr(b,'md',False)
        k=b.Wk(hl); v=b.Wv(hl); dh=C//mh
        bt=b.bmax*torch.sigmoid(b.Wb(hl)) if mb else None
        la=-F.softplus(b.Wa(hl)) if md else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(b.beta.float()).expand(Bn,mh)
            R=Bn*mh; Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
            if md: g['Lc'].add_(la.float().reshape(R))
            kr=kf.reshape(R,dh); w_=torch.exp((g['Lc'][:,None]-g['Lw']).clamp(max=0.0))
            if _SAMP.get('mm') and dt_!=torch.float32:
                sc=torch.bmm(g['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_
                pred=torch.bmm(sc.to(dt_).unsqueeze(1),g['U']).squeeze(1).float()
            else:
                sc=torch.bmm(g['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_
                pred=torch.bmm(sc.unsqueeze(1),g['U'].float()).squeeze(1)
            sl=torch.remainder(tb,Wn)
            g['U'].index_copy_(1,sl,((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_).unsqueeze(1))
            if _PREC['p']:
                kw=g['kD']*kr; g['kc'].add_(kr*kr); kw=kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)
            else: kw=kr
            g['K'].index_copy_(1,sl,kw.to(dt_).unsqueeze(1))
            g['Lw'].index_copy_(1,sl,g['Lc'].unsqueeze(1))
        out=pred.reshape(Bn,1,C)
    return _rec_tail(b,x,x0,hl,out)

def _rsg_supported(TT,blocks,sts,t):
    if getattr(TT,'cnt',None) is not None or _SAMP['W']<=0: return 'count features or no window'
    if _PREC['p'] and _PREC['C']<=0: return 'per-token preconditioner'
    for b,st in zip(blocks,sts):
        if getattr(b,'lattn',False) or getattr(b,'mk',0) or getattr(b,'dp2',False) or getattr(b,'mq',False): return 'block type'
        if getattr(b,'loc',False):
            if 'bl' not in st: return 'local state missing'
        else:
            if 'U' not in st: return 'window state missing'
            if _PREC['p'] and ('kc' not in st or ('kD' not in st and t%int(_PREC['C'])!=0)): return 'preconditioner state'
    return None

def _rsg_kd(gst,t):
    """Host-side refresh of the block-level write-key preconditioner (as _kw_rec at n0 = t writes)."""
    if not (_PREC['p'] and _PREC['C']>0 and t>0 and t%int(_PREC['C'])==0): return
    for g in gst:
        if 'kc' in g:
            dh=g['kc'].shape[-1]
            with torch.autocast(device_type=g['kc'].device.type, enabled=False): g['kD'].copy_((g['kc']/t*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0))

@torch.no_grad()
def _rsg_run(TT,role,blocks,sts,groups,x,t,hist,xw_,T,temp,gen,seq):
    """CUDA-GRAPH SAMPLER from byte t on. The graph runs on CLONES of the eager state; the first two bytes are run both ways (eager advances
    the real state and draws the bytes) and must agree bit for bit (logits and states), then the graph takes over. Returns (x, t, hist, xw_):
    t = T when the graph finished, else the eager loop continues from t with the real state."""
    model=TT.model; dev=x.device; real_=dev.type=='cuda' and not _SAMP.get('gfake'); st_=_SAMP.setdefault('gstat',{'calls':0,'ok':0,'bytes':0,'why':None,'diff':0.0})
    st_['calls']+=1
    why=_rsg_supported(TT,blocks,sts,t)
    if why is not None: st_['why']=why; return x,t,hist,xw_
    ngw_=bool(model.cfg.ngp and model.__dict__.get('_ngp') is not None)
    if ngw_ and (xw_ is None or xw_.shape[1]!=max(model.cfg.ngp)): st_['why']='n-gram window'; return x,t,hist,xw_
    gst=[]
    for b,st in zip(blocks,sts):
        if getattr(b,'loc',False): gst.append({'bl':st['bl'].clone()})
        else:
            g={k_:st[k_].clone() for k_ in ('U','K','Lw','Lc')}
            if _PREC['p']: g['kc']=st['kc'].clone(); g['kD']=st['kD'].clone() if 'kD' in st else torch.zeros_like(st['kc'])
            gst.append(g)
    xb=x.clone(); hb=hist.clone() if hist is not None else None; xwb=xw_.clone() if ngw_ else None
    tb=torch.full((1,),int(t),dtype=torch.long,device=dev); ng=TT._ng(); lgb=torch.empty(x.shape[0],model.cfg.vocab,device=dev)
    amp_=(TT.cfg.amp=='bf16')
    def gstep():
        if TT.emb is None: e=model.embed(xb)+model.pos.weight.index_select(0,tb)
        else: E,P_=TT.emb[role]; e=F.embedding(xb,E)+P_.index_select(0,tb)
        if ng is not None:
            if hb is None: raise RuntimeError('no history buffer')
            e=e+ngram_embed_tok(ng,xb,hb[:,hb.shape[1]-(max(ng.ns)-1):] if max(ng.ns)>1 else hb[:,:0])
        if hb is not None and hb.shape[1]>0: hb.copy_(torch.cat([hb[:,1:],xb],1))
        h=e
        with (torch.autocast(device_type=dev.type,dtype=torch.bfloat16,cache_enabled=False) if amp_ else contextlib.nullcontext()):
            for b,g in zip(blocks,gst): h=_rec_step_ip(b,h,g,tb)
        h=h.float()[:,0]
        if ngw_:
            xwb.copy_(torch.cat([xwb[:,1:],xb],1))
            lg=torch.cat([model.heads[d](h[sl]+ngp_vec(model,xwb[sl],d)[:,-1]) for d,sl in groups]).float()
        else: lg=torch.cat([model.heads[d](h[sl]) for d,sl in groups]).float()
        lgb.copy_(lg); tb.add_(1)
    G=None; took_=False
    try:
        if real_:
            bufs=[xb,tb]+([hb] if hb is not None else [])+([xwb] if xwb is not None else [])+[v_ for g in gst for v_ in g.values()]
            snap=[v_.clone() for v_ in bufs]
            s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(2): gstep()
            torch.cuda.current_stream().wait_stream(s_)
            for v_,o_ in zip(bufs,snap): v_.copy_(o_)
            del snap
            G=torch.cuda.CUDAGraph()
            with torch.cuda.graph(G): gstep()
        rep=(G.replay if G is not None else gstep)
        ok=True; md_=0.0
        for c_ in range(2):                                           # SELF-CHECK: two bytes both ways
            _rsg_kd(gst,t); rep()
            lg,hist,xw_=_rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_)
            if not torch.equal(lgb,lg): ok=False; md_=max(md_,float((lgb-lg).abs().max()))
            if t<T-1:
                x=torch.multinomial(F.softmax(lg/temp,-1),1,generator=gen); seq.append(x); xb.copy_(x)
            t+=1
        if ok:
            for b,st,g in zip(blocks,sts,gst):
                for k_,v_ in g.items():
                    if k_ in st and torch.is_tensor(st[k_]) and not torch.equal(st[k_],v_): ok=False
        if not ok:
            st_['why']=f'self-check mismatch (max logit diff {md_:.1e})'; st_['diff']=max(st_['diff'],md_); return x,t,hist,xw_
        st_['ok']+=1; took_=True
        while t<T:
            _rsg_kd(gst,t); rep()
            if t<T-1:
                x=torch.multinomial(F.softmax(lgb/temp,-1),1,generator=gen); seq.append(x); xb.copy_(x)
            t+=1; st_['bytes']+=1
        return x,t,hist,xw_
    except Exception as ex:
        st_['why']=f'error {repr(ex)[:160]}'
        if took_: raise                                               # the graph owned the state: no eager continuation possible
        return x,t,hist,xw_                                           # before the hand-over the real state is intact: eager continues
    finally:
        del G

def _stack_ok(TT):
    """The stacked sampler covers the configurations in use: stackable teachers, shared embeddings, no key/value convolution."""
    m=TT.model
    return bool(TT.stackable and TT.emb is None and all(getattr(b,'mk',0)==0 and not getattr(b,'mq',False) for b in m.blocks))

@torch.no_grad()
def rec_sample_multi(TT, specs, T, temp, gen, tgt_T, start=10):
    """ONE recurrent sampling loop for rows from several teachers and domains. specs: [(role, domain, prompt or None)]. Teachers are stacked
    (batched matmuls over roles; rows padded per role with copies of that role's first row), every row's logits come from one matmul over all
    the old heads (each row then takes its own head), warm rows are teacher-forced through their prompt, cold rows start from `start`.
    Returns xs (n,T) and targets (n,T,V) in spec order: re-scored by the exact parallel teacher when the sampler is windowed, else the sampler's."""
    model=TT.model; dev=model.embed.weight.device; V=model.cfg.vocab
    roles=sorted(set(r for r,_,_ in specs)); M=len(roles)
    by_role={r:[i for i,sp in enumerate(specs) if sp[0]==r] for r in roles}
    N=max(len(v) for v in by_role.values())
    order=[]                                                                       # flat row -> spec index (padding: the role's first row)
    for r in roles: ix=by_role[r]; order+=ix+[ix[0]]*(N-len(ix))
    order_t=torch.tensor(order,device=dev); R0=M*N
    doms=sorted(set(sp[1] for sp in specs)); dpos={d:i for i,d in enumerate(doms)}
    dix=torch.tensor([dpos[specs[i][1]] for i in order],device=dev)
    Wh=torch.cat([model.heads[d].weight.detach().float() for d in doms]); bh=torch.cat([model.heads[d].bias.detach().float() for d in doms])
    cap=model.heads[doms[0]].cap; ar=torch.arange(R0,device=dev)
    P=max([sp[2].shape[0] for sp in specs if sp[2] is not None] or [0])
    forced=torch.tensor([specs[i][2] is not None for i in order],device=dev)
    pr=torch.full((R0,max(P,1)),start,dtype=torch.long,device=dev)
    for j,i in enumerate(order):
        if specs[i][2] is not None: pr[j,:P]=specs[i][2][:P]
    x=pr[:,:1].clone(); seq=[x]; Ps=TT._stack(roles); sts=[{} for _ in Ps]; hist=None
    resc=_SAMP['W']>0 and _SAMP['rescore']
    tg=None if (resc or not tgt_T) else torch.empty(R0,T,V,device=dev)
    for t in range(T):
        e=TT._emb_tok(x,t,roles[0],hist); hist=_hist_push(TT,hist,x)
        h=e.reshape(M,N,1,-1)
        with TT._ac():
            for P_,st in zip(Ps,sts): h=_rec_step_stk(P_,h,st,M,N)
        hf=TT._out(h).reshape(R0,-1)
        z=(hf@Wh.T+bh).view(R0,len(doms),V)[ar,dix]
        if cap>0: z=cap*torch.tanh(z/cap)
        if tg is not None: tg[:,t]=F.log_softmax(z/tgt_T,-1)
        if t<T-1:
            smp=torch.multinomial(F.softmax(z/temp,-1),1,generator=gen)
            x=torch.where(forced[:,None]&(t+1<P),pr[:,min(t+1,max(P,1)-1):min(t+1,max(P,1)-1)+1],smp)
            seq.append(x)
    xs_all=torch.cat(seq,1); n=len(specs); first={}
    for j,i in enumerate(order): first.setdefault(i,j)
    pick=torch.tensor([first[i] for i in range(n)],device=dev); xs=xs_all[pick]
    if tg is not None: return xs, tg[pick]
    if not tgt_T: return xs, None
    out=torch.empty(n,T,V,device=dev); ch=max(4,int(128*(256/max(T,1))**2))
    for r in roles:                                                                # exact targets, per teacher, rows grouped by domain
        ix=sorted(by_role[r],key=lambda i:specs[i][1]); xr=xs[ix]; tr=torch.empty(len(ix),T,V,device=dev)
        gr=[]; i0=0
        for i in range(1,len(ix)+1):
            if i==len(ix) or specs[ix[i]][1]!=specs[ix[i0]][1]: gr.append((specs[ix[i0]][1],slice(i0,i))); i0=i
        tr=_rescore_targets(TT,r,gr,xr,tgt_T,tr,chunk=ch); out[torch.tensor(ix,device=dev)]=tr
    return xs, out

@torch.no_grad()
def _rec_step_stk(P, h, st, M, N):
    assert not _DP2['on'], 'dp_steps 2 is not implemented in the stacked sampler (use dream_bank / dream_stack False)'
    """One token through one block for M stacked teachers: h (M,N,1,d); same arithmetic as _rec_step (windowed or full-state memory)."""
    mh,mk,mb,md,bmax,gln,kern,C,loc=P['meta']; dev=h.device
    hl=_stk_ln(h,P['ln_w'],P['ln_b']); v=_stk_lin(hl,P['Wv'])
    if loc:                                                                        # local filter: the previous K values
        K=P['cl'].shape[-1]; cl=P['cl'].reshape(M,C,K)
        if 'bl' not in st: st['bl']=torch.zeros(M,N,K,C,dtype=v.dtype,device=dev)
        out=torch.einsum('mnkc,mck->mnc',st['bl'].float(),cl.float()).unsqueeze(2)
        st['bl']=torch.cat([st['bl'][:,:,1:],v.to(st['bl'].dtype)],2)
    else:
        k=_stk_lin(hl,P['Wk'])
        bt=bmax*torch.sigmoid(_stk_lin(hl,P['Wb_w'],P['Wb_b'])) if mb else None
        la=-F.softplus(_stk_lin(hl,P['Wa_w'],P['Wa_b'])) if md else None
        with torch.autocast(device_type=dev.type, enabled=False):
            Bn=M*N; dh=C//mh; R=Bn*mh
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(P['beta'].float())[:,None,None].expand(M,N,mh).reshape(Bn,mh)
            kr=kf.reshape(R,dh)
            if _SAMP['W']>0:
                Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
                if 'U' not in st:
                    st['U']=torch.zeros(R,Wn,dh,dtype=dt_,device=dev); st['K']=torch.zeros(R,Wn,dh,dtype=dt_,device=dev)
                    st['Lw']=torch.full((R,Wn),1e9,device=dev); st['Lc']=torch.zeros(R,device=dev); st['p']=0
                if md: st['Lc']=st['Lc']+la.float().reshape(R)
                w_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))
                if _SAMP.get('mm') and dt_!=torch.float32:
                    sc=torch.bmm(st['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_
                    pred=torch.bmm(sc.to(dt_).unsqueeze(1),st['U']).squeeze(1).float()
                else:
                    sc=torch.bmm(st['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_
                    pred=torch.bmm(sc.unsqueeze(1),st['U'].float()).squeeze(1)
                sl=st['p']%Wn; st['U'][:,sl]=((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_); st['K'][:,sl]=_kw_rec(st,kr).to(dt_)
                st['Lw'][:,sl]=st['Lc']; st['p']+=1
            else:
                S=st.get('S')
                if S is None: S=st['S']=torch.zeros(R,dh,dh,device=dev)
                if md: S.mul_(torch.exp(la.float().reshape(R))[:,None,None])
                pred=torch.bmm(S,kr.unsqueeze(-1)).squeeze(-1)
                S.baddbmm_(((vf.reshape(R,dh)-pred)*be.reshape(R,1)).unsqueeze(-1),_kw_rec(st,kr).unsqueeze(1))
            out=pred.reshape(M,N,1,C)
    g_=_stk_lin(hl if gln else h,P['gate_w'],P['gate_b'])
    h0=h; h=h+_stk_lin(out*F.silu(g_),P['Wout'])
    hn=_stk_ln(h0 if P['par'] else h,P['norm_w'],P['norm_b'])
    return h+_stk_lin(F.gelu(_stk_lin(hn,P['mi_w'],P['mi_b'])),P['mo_w'],P['mo_b'])

@torch.no_grad()
def _rescore_targets(TT, role, groups, xs, tgt_T, tg, chunk=128):
    """Teacher log-probabilities log_softmax(head_d(teacher(x)) / tgt_T) for every row, from the exact (parallel) teacher."""
    model=TT.model
    for c0 in range(0,xs.shape[0],chunk):
        c1=min(xs.shape[0],c0+chunk); h=TT.features(xs[c0:c1],role)
        for d,sl in groups:
            a,b=max(sl.start,c0),min(sl.stop,c1)
            if a<b: tg[a:b]=F.log_softmax(model.heads[d](h[a-c0:b-c0]).float()/tgt_T,-1)
    return tg

def build_teachers(model, cfg):
    """At the start of a visit: frozen main teacher and one teacher per (non-empty) mid-store level."""
    S=getattr(model,'_shunt',None) or {}
    roles={'main':[S]}; subs=[S]
    for k,L in enumerate(_cons_levels(model,cfg)):
        subs=subs+[L]; roles[f'lvl{k+1}']=list(subs)
    object.__setattr__(model,'_teach',Teachers(model,cfg,roles))

def stack_heads(model, olds):
    """Old heads (fixed during a visit: no gradient reaches them) as one (n_old,V,d) matrix for the fused KL."""
    W=torch.stack([model.heads[d].weight.detach() for d in olds]).clone(); b=torch.stack([model.heads[d].bias.detach() for d in olds]).clone()
    object.__setattr__(model,'_hstk',{'W':W,'b':b,'cap':model.heads[olds[0]].cap,'sub':{}})

def heads_logp(model, h, idx, T):
    """log_softmax(head_d(h)/T) for the old domains at positions idx in olds (stacked): (len(idx),B,T,V)."""
    H=model._hstk; key=tuple(idx)
    if key not in H['sub']: H['sub'][key]=(H['W'][list(idx)].contiguous(),H['b'][list(idx)].contiguous())
    W,b=H['sub'][key]
    z=torch.einsum('btd,nvd->nbtv',h,W)+b[:,None,None,:]
    if H['cap']>0: z=H['cap']*torch.tanh(z/H['cap'])
    return F.log_softmax(z/T,-1)

def _pool_add(model, cfg, d, xs, tgs):
    """Add dreams (and cached targets) to old domain d's pool; keep the newest dream_cap of them (as dream_refresh).
    Prioritised dreams: pools are also kept as stacked GPU tensors with one priority score per dream (new dreams: the pool maximum)."""
    pool=model.__dict__.setdefault('_dreams',{}).setdefault(d,[]); pool.extend(list(xs))
    _ol=list(getattr(model,'_olds',None) or [])
    cap=cfg.dream_n
    del pool[:-cap]
    if tgs is not None:
        tp=model.__dict__.setdefault('_dream_tgt',{}).setdefault(d,[]); tp.extend(list(tgs)); del tp[:-cap]

@torch.no_grad()
def dream_fidelity(model, cfg, x, d, seen):
    """Posterior of domain d for each dream (rows of x) under the stored per-domain byte-pair statistics, whole sequence (the
    label-free 'bytes' inference): how much each dream looks like its own domain rather than another one."""
    bg=getattr(model,'_bg',None)
    if bg is None or d not in seen: return None
    cnt=(bg[seen]+cfg.hs_alpha).view(len(seen),256,256); logp=torch.log(cnt/cnt.sum(-1,keepdim=True)).view(len(seen),-1)   # log p_d(next|prev)
    pair=(x[:,:-1]*256+x[:,1:])                                                       # (n,T-1)
    ll=logp[:,pair].sum(-1)                                                           # (D,n)
    return F.softmax(ll*cfg.hs_temp,0)[seen.index(d)]

@torch.no_grad()
def fast_dream_visit(model, cfg, dom_id, olds, pools_only=False):
    """All of a visit's dreams at its start: the initial pools AND every scheduled refresh (dream_batch sequences every dream_refresh
    steps, domains drawn by the same recency shares with their own random stream), generated by the recurrent sampler in a few large
    batches per teacher (graduated domains dream from their own, older teacher). Refreshes enter the pools at their scheduled steps."""
    TT=model._teach; n_old=len(olds); sh=dream_shares(cfg,n_old)
    model._dreams={}
    for k_ in ('_dream_tgt','_dstk','_dtstk','_dscore')+(() if pools_only else ('_dqueue',)): model.__dict__[k_]={}
    req=[]
    for i_,d in enumerate(olds):
        if sh is None: req.append((d,cfg.dream_n,'pool'))
        elif sh[i_]>0: req.append((d,dream_cap(cfg,n_old,i_),'pool'))
    over=1.0
    r_=model.__dict__.get('_dq_rng')
    if r_ is None: r_=np.random.RandomState(cfg.seed+17); model.__dict__['_dq_rng']=r_
    for s_ in (range(cfg.dream_refresh,cfg.steps_per_task,cfg.dream_refresh) if not pools_only else ()):
        d=olds[(s_//cfg.dream_refresh)%n_old] if sh is None else olds[int(r_.choice(n_old,p=sh))]
        req.append((d,cfg.dream_batch,s_))
    if not hasattr(model,'_dream_gen'):
        g_=torch.Generator(device=model.embed.weight.device); g_.manual_seed(cfg.seed+11); object.__setattr__(model,'_dream_gen',g_)
    T=min(cfg.dream_len,cfg.seq_len); out={j:([],[]) for j in range(len(req))}
    _t0=time.perf_counter()
    for role in TT.blocks:
        rows=sorted([(d,j) for j,(d,n,_) in enumerate(req) if teacher_role(model,cfg,d,olds,TT)==role for _ in range(int(math.ceil(n*over)))])
        for c0 in range(0,len(rows),cfg.fast_dream_chunk):
            ch=rows[c0:c0+cfg.fast_dream_chunk]
            xs,tg=rec_sample(TT,role,[d for d,_ in ch],T,cfg.dream_temp,model._dream_gen,cfg.shunt_T)
            for i,(d,j) in enumerate(ch):
                out[j][0].append(xs[i])
                if tg is not None: out[j][1].append(tg[i])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    model.__dict__['_tm_dream']=model.__dict__.get('_tm_dream',0.0)+time.perf_counter()-_t0
    bg=getattr(model,'_bg',None); seen=[k for k in range(bg.shape[0]) if float(bg[k].sum())>0] if bg is not None else []
    fl={}
    for j,(d,n,dest) in enumerate(req):
        xs,tg=out[j]; tg=tg
        fid=dream_fidelity(model,cfg,torch.stack(xs),d,seen) if seen else None
        if fid is not None:
            a_=fl.setdefault(d,[0.0,0,0.0,0]); a_[0]+=float(fid.sum()); a_[1]+=len(xs)
            a_[2]+=float(fid.sum()); a_[3]+=len(xs)
        if dest=='pool': _pool_add(model,cfg,d,xs,tg)
        else: model._dqueue[dest]=(d,xs,tg)
    if fl and not pools_only: print("    [dream fidelity] share of each old domain's dreams attributed to that domain by the byte-pair statistics"
                 +("")+": "
                 +" ".join(f"{_lab(cfg,d)} {a_[0]/max(a_[1],1):.2f}"+("") for d,a_ in fl.items()))

@torch.no_grad()
def stream_generate(model, cfg, olds, n_steps):
    """CONTINUOUS DREAM STREAMS: chunks for the next n_steps steps (dream_batch rows each, one use per chunk), split over the old domains by
    their dream shares (+10% slack). Each domain keeps as many parallel streams as it needs chunks per block; a stream's next chunk is prompted
    with the last dream_stream_prompt bytes of its previous chunk (cold streams start from a newline). Chunks queue per domain with their cached
    teacher targets."""
    TT=model._teach; n_old=len(olds); sh=dream_shares(cfg,n_old)
    w=np.array(sh if sh is not None else [1.0/n_old]*n_old,dtype=float)
    ss_=0.0             # CROSS-DOMAIN STRANDS take this share of the batch
    need=np.ceil(w*n_steps*cfg.dream_batch*(1.0-ss_)*1.1).astype(int)+1
    T=min(cfg.dream_len,cfg.seq_len); P=min(cfg.dream_stream_prompt,T//2)
    ST=model.__dict__.setdefault('_dstream',{}); Q=model.__dict__.setdefault('_dsq',{})
    if not hasattr(model,'_dream_gen'):
        g_=torch.Generator(device=model.embed.weight.device); g_.manual_seed(cfg.seed+11); object.__setattr__(model,'_dream_gen',g_)
    _t0=time.perf_counter(); fl={}
    def _post(out):                                                                 # queue the chunks, keep the tails as next prompts
        for d,lst in out.items():
            xs=torch.stack([a for a,_ in lst]); tg=torch.stack([b for _,b in lst])
            ST[d]=xs[:,T-P:].clone()                                                  # tails = next prompts
            q=Q.setdefault(d,[]); q.extend(zip(list(xs),list(tg)))
            del q[:-int(need[olds.index(d)])*2]                                       # bounded queue
            bg=getattr(model,'_bg',None)
            if bg is not None and not fl.get('done'):
                seen=[k_ for k_ in range(bg.shape[0]) if float(bg[k_].sum())>0]
                f_=dream_fidelity(model,cfg,xs,d,seen)
                if f_ is not None: fl[d]=float(f_.mean())
    if cfg.dream_stack and _stack_ok(TT):                                       # ONE loop for every row of the refill (1-3)
        specs=[]
        for i_,d in enumerate(olds):
            if need[i_]<=0: continue
            r_=teacher_role(model,cfg,d,olds,TT); tails=ST.get(d); k=int(need[i_]); nw=min(k,0 if tails is None else tails.shape[0])
            specs+=[(r_,d,tails[j]) for j in range(nw)]+[(r_,d,None) for _ in range(k-nw)]
        specs.sort(key=lambda z:(z[0],z[1]))
        out={}
        for c0 in range(0,len(specs),cfg.fast_dream_chunk):
            sp=specs[c0:c0+cfg.fast_dream_chunk]
            xs,tg=rec_sample_multi(TT,sp,T,cfg.dream_temp,model._dream_gen,cfg.shunt_T)
            for i,(_,d,_) in enumerate(sp): out.setdefault(d,[]).append((xs[i],tg[i]))
        _post(out); roles_iter=[]
    else: roles_iter=list(TT.blocks)
    for role in roles_iter:
        ds=[(i_,d) for i_,d in enumerate(olds) if teacher_role(model,cfg,d,olds,TT)==role and need[i_]>0]
        if not ds: continue
        warm=[];cold=[]
        for i_,d in ds:
            tails=ST.get(d); k=int(need[i_]); nw=min(k,0 if tails is None else tails.shape[0])
            warm+=[(d,tails[j]) for j in range(nw)]; cold+=[d]*(k-nw)
        out={}
        for grp,is_warm in ((warm,True),(cold,False)):
            rows=sorted(range(len(grp)),key=lambda j:(grp[j][0] if is_warm else grp[j]))
            for c0 in range(0,len(rows),cfg.fast_dream_chunk):
                ch=rows[c0:c0+cfg.fast_dream_chunk]; doms=[(grp[j][0] if is_warm else grp[j]) for j in ch]
                pr=torch.stack([grp[j][1] for j in ch]) if is_warm else None
                xs,tg=rec_sample(TT,role,doms,T,cfg.dream_temp,model._dream_gen,cfg.shunt_T,prompt=pr)
                for i,d in enumerate(doms): out.setdefault(d,[]).append((xs[i],tg[i]))
        _post(out)
    if ss_>0:
        ch_={d:torch.stack([a for a,_ in Q[d][-8:]]) for d in olds if Q.get(d)}
        C=head_confusion(model,cfg,olds,ch_); model.__dict__['_hconf']=(list(olds),C)
        ns=int(math.ceil(n_steps*cfg.dream_batch*ss_*1.1))+1
        q=Q.setdefault('__strand__',[]); q.extend(strand_generate(model,cfg,olds,ns,C)); del q[:-2*ns]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    model.__dict__['_tm_dream']=model.__dict__.get('_tm_dream',0.0)+time.perf_counter()-_t0
    return fl

@torch.no_grad()
def rec_sample_mix(TT, ra, rb, da, db, T, temp, gen, w, start=10, prompt=None, mode='mix', tgt_T=None):
    """BLENDED STRANDS: every byte sampled from w p_A + (1-w) p_B, two teachers (roles ra / rb, heads da[i] / db[i] per row) each with its
    own recurrent state (one shared state when both rows use the same teacher weights: only the heads differ)."""
    model=TT.model; n=len(da); dev=model.embed.weight.device; cap=model.heads[da[0]].cap
    Wa=torch.stack([model.heads[d].weight.detach() for d in da]); ba=torch.stack([model.heads[d].bias.detach() for d in da])
    Wb=torch.stack([model.heads[d].weight.detach() for d in db]); bb=torch.stack([model.heads[d].bias.detach() for d in db])
    same=(ra==rb); P=0 if prompt is None else prompt.shape[1]
    x=prompt[:,:1] if P else torch.full((n,1),start,dtype=torch.long,device=dev); seq=[x]
    sa=[{} for _ in TT.blocks[ra]]; sb=sa if same else [{} for _ in TT.blocks[rb]]
    hist=[None]
    def run(role,sts,t):
        h=TT._emb_tok(x,t,role,hist[0])
        with TT._ac():
            for b,st in zip(TT.blocks[role],sts): h=_rec_step(b,h,st)
        return TT._out(h)[:,0]
    tg=torch.empty(n,T,model.cfg.vocab,device=dev) if tgt_T else None
    for t in range(T):
        ha=run(ra,sa,t); hb=ha if same else run(rb,sb,t); hist[0]=_hist_push(TT,hist[0],x)
        if tg is None and t+1<P: x=prompt[:,t+1:t+2]; seq.append(x); continue
        za=torch.einsum('nd,nvd->nv',ha,Wa)+ba; zb=torch.einsum('nd,nvd->nv',hb,Wb)+bb
        if cap>0: za=cap*torch.tanh(za/cap); zb=cap*torch.tanh(zb/cap)
        if tg is not None:                                           # AGREEMENT targets: log of the normalised product of the two teachers
            tg[:,t]=F.log_softmax(w*F.log_softmax(za/tgt_T,-1)+(1-w)*F.log_softmax(zb/tgt_T,-1),-1)
        if t==T-1: break
        if t+1<P: x=prompt[:,t+1:t+2]
        elif mode=='poe': x=torch.multinomial(F.softmax(w*F.log_softmax(za/temp,-1)+(1-w)*F.log_softmax(zb/temp,-1),-1),1,generator=gen)
        else: x=torch.multinomial(w*F.softmax(za/temp,-1)+(1-w)*F.softmax(zb/temp,-1),1,generator=gen)
        seq.append(x)
    xs=torch.cat(seq,1)
    if tg is not None and _SAMP['W']>0 and _SAMP['rescore']:          # EXACT agreement targets from the parallel teachers
        for c0 in range(0,n,128):
            c1=min(n,c0+128); ha_=TT.features(xs[c0:c1],ra); hb_=ha_ if same else TT.features(xs[c0:c1],rb)
            za_=torch.einsum('ntd,nvd->ntv',ha_,Wa[c0:c1])+ba[c0:c1,None]; zb_=torch.einsum('ntd,nvd->ntv',hb_,Wb[c0:c1])+bb[c0:c1,None]
            if cap>0: za_=cap*torch.tanh(za_/cap); zb_=cap*torch.tanh(zb_/cap)
            tg[c0:c1]=F.log_softmax(w*F.log_softmax(za_/tgt_T,-1)+(1-w)*F.log_softmax(zb_/tgt_T,-1),-1)
    return (xs,tg) if tgt_T else xs

@torch.no_grad()
def head_confusion(model, cfg, olds, chunks):
    """DATA-FREE HEAD CONFUSION: chunks {d: (n,T) dreams of old domain d}. C[a,b] = mean per-byte posterior of old head b (softmax over the old
    heads of each byte's log-likelihood, live network, bytes after the prompt) on domain a's dreams. High C[a,b]+C[b,a] = the two domains
    are predicted by shared machinery (where interference between them happens)."""
    n=len(olds); C=np.eye(n); P=min(cfg.dream_stream_prompt,min(cfg.dream_len,cfg.seq_len)//2)
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    W=torch.stack([model.heads[d].weight.detach() for d in olds]); b=torch.stack([model.heads[d].bias.detach() for d in olds]); cap=model.heads[olds[0]].cap
    for i,d in enumerate(olds):
        x=chunks.get(d)
        if x is None or len(x)==0: continue
        h=model.features(x)[:,P:-1]
        z=torch.einsum('btd,nvd->nbtv',h,W)+b[:,None,None,:]
        if cap>0: z=cap*torch.tanh(z/cap)
        ll=F.log_softmax(z.float(),-1).gather(-1,x[:,P+1:].unsqueeze(0).unsqueeze(-1).expand(n,-1,-1,1)).squeeze(-1)   # (n_heads,B,T')
        C[i]=F.softmax(ll,0).mean((1,2)).cpu().numpy()
    model.feat_scale.copy_(fs); model.train(was)
    return C

def strand_pairs(C, floor, mode='confusion'):
    """Unordered pairs of old-domain positions and their draw probabilities: symmetrised confusion (1-floor) + uniform (floor).
    mode 'anti': weights from 1/confusion (pairs the heads separate best); 'uniform': every pair alike."""
    n=C.shape[0]; S=C+C.T; iu=[(a,b_) for a in range(n) for b_ in range(a+1,n)]
    w=np.array([S[a,b_] for a,b_ in iu],dtype=float)
    if mode=='anti': w=1.0/np.maximum(w,1e-6)
    elif mode=='uniform': w=np.ones(len(iu))
    w=w/w.sum() if w.sum()>0 else np.ones(len(iu))/len(iu)
    return iu,(1-floor)*w+floor/len(iu)

@torch.no_grad()
def strand_generate(model, cfg, olds, n_rows, C):
    """CROSS-DOMAIN STRANDS for the next block: pairs (A,B) of old domains drawn by head confusion, random order. 'switch': teacher A
    (prompted by A's stream tail) writes up to a random switch point, teacher B (fed A's bytes) continues; KL targets from A's head before
    the switch and B's after it (none on the prompt or the first strand_gap bytes after the switch). 'blend': bytes from a mixture of the two
    teachers (no output KL; used for old-subspace anchoring). Returns per-row records."""
    TT=model._teach; T=min(cfg.dream_len,cfg.seq_len); P=min(cfg.dream_stream_prompt,T//2); dev=model.embed.weight.device
    r_=model.__dict__.get('_srng')
    if r_ is None: r_=np.random.RandomState(cfg.seed+43); model.__dict__['_srng']=r_
    iu,w=strand_pairs(C,cfg.strand_floor,cfg.strand_pair_mode)
    pk=r_.choice(len(iu),size=n_rows,p=w); fl=r_.rand(n_rows)<0.5
    pairs=[(olds[iu[k][1]],olds[iu[k][0]]) if f else (olds[iu[k][0]],olds[iu[k][1]]) for k,f in zip(pk,fl)]
    cnt=model.__dict__.setdefault('_strandcnt',{})
    for pr in pairs: cnt[pr]=cnt.get(pr,0)+1
    ST=model.__dict__.get('_dstream',{})
    def tail(d):
        t_=ST.get(d); return t_[r_.randint(t_.shape[0])] if (t_ is not None and t_.shape[0]) else None
    kl_on=(cfg.strand_kind=='switch')
    grp={}
    for A,B in pairs: grp.setdefault((teacher_role(model,cfg,A,olds,TT),teacher_role(model,cfg,B,olds,TT)),[]).append((A,B))
    out=[]
    for (ra,rb),lst in grp.items():
        for c0 in range(0,len(lst),cfg.fast_dream_chunk):
            ch=lst[c0:c0+cfg.fast_dream_chunk]; n=len(ch)
            prs=[tail(A) for A,_ in ch]; usep=all(p_ is not None for p_ in prs)
            z0=P if usep else min(cfg.dream_burnin,T//2)                  # no target on the prompt / context-free start
            pr=torch.stack(prs) if usep else None
            aw=torch.zeros(n,T,device=dev); aw[:,z0:]=1.0
            x=rec_sample_mix(TT,ra,rb,[A for A,_ in ch],[B for _,B in ch],T,cfg.dream_temp,model._dream_gen,cfg.strand_blend_w,prompt=pr)
            for i in range(n): out.append({'x':x[i],'tg':None,'hd':None,'w':None,'aw':aw[i],'seg':None,'kind':2})
    order=r_.permutation(len(out)); return [out[i] for i in order]

_RR_KINDS=('live','rekey','ngram')
def rr_keys(model, kind, x=None, h=None):
    RR=model._rr
    if kind=='ngram':
        B,T=x.shape; xs=x.long()+1; zs=[]
        for n in range(1,5):
            hh=torch.zeros(B,T,dtype=torch.long,device=x.device)
            for i in range(n): sh=F.pad(xs,(i,0))[:,:T]; hh=(hh*1000003+sh+97*n)%2147483629
            zs.append(RR['T'][n-1][hh%16384])
        z=torch.cat(zs,-1); return torch.cat([torch.ones_like(z[...,:1]),F.relu(z@RR['Rn']+RR['bn'])],-1)
    h=h.float(); return torch.cat([h,torch.ones_like(h[...,:1]),F.relu(h@RR['Rl']+RR['bl'])],-1)

_SM_LET=b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'; _SM_DIG=b'0123456789'

@torch.no_grad()
def bank_refresh(model, cfg, d):
    """DREAM BANK: dream_bank rows for domain d sampled once from the current (visit-end) weights with head d, stored as bytes."""
    if not hasattr(model,'_dream_gen'):
        g_=torch.Generator(device=model.embed.weight.device); g_.manual_seed(cfg.seed+11); object.__setattr__(model,'_dream_gen',g_)
    _t0bank=time.time(); _SAMP['gstat']={'calls':0,'ok':0,'bytes':0,'why':None,'diff':0.0}
    T=min(cfg.dream_len,cfg.seq_len)*(2 if cfg.dream_pair>0 else 1); rows=[]; NP=cfg.dream_bank*(1)   # paired: 2 x length
    NA_=int(float(cfg.bank_anchor)*NP) if float(cfg.bank_anchor)>0 else 0; AN_=model.__dict__.setdefault('_banc',{})
    if NA_>0 and d in AN_ and AN_[d].shape[1]==T: rows.append(AN_[d].to(model.embed.weight.device).long()); NP=NP-AN_[d].shape[0]   # cg16 anchor kept
    model.__dict__['_s0_dom']=d
    TP_=model.__dict__.setdefault('_dtemp',{})
    if d not in TP_:                                                     # cg10: temperature per domain, decided at its first bank write
        tv_=cfg.dream_temp
        if float(cfg.dream_temp_bpc)>0 and model.__dict__.get('_live'):
            lv_=model._live[-4:]; b_=_slot_bpc(model,[a for a,_ in lv_],[c for _,c in lv_],d); model.__dict__['_s0_dom']=d
            if b_<float(cfg.dream_temp_bpc): tv_=float(cfg.dream_temp_low)
            print(f"  [dream temperature] {_lab(cfg,d)}: BPC {b_:.3f} on its last batches -> temperature {tv_:g}")
        TP_[d]=tv_
    _dT=TP_[d]
    if cfg.bank_rec:                                                     # recurrent sampler from a frozen copy of the visit-end weights
        build_teachers(model, cfg); TT=model._teach
        ch_b=cfg.bank_chunk if _SAMP['W']>0 else min(cfg.bank_chunk,512)    # full-state sampler: ~50 MB per row -> cap the chunk
        for c0 in range(0,NP,ch_b):
            n=min(ch_b,NP-c0); xs,_=rec_sample(TT,'main',[d]*n,T,_dT,model._dream_gen,None); rows.append(xs)
        object.__setattr__(model,'_teach',None)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        for c0 in range(0,NP,cfg.fast_dream_chunk):
          n=min(cfg.fast_dream_chunk,NP-c0); rows.append(dream_sample(model,d,n,T,cfg.dream_temp,gen=model._dream_gen))
    X=torch.cat(rows); W=None
    if NA_>0 and d not in AN_: AN_[d]=X[:NA_].to(torch.uint8).clone()        # cg16: the first bank's rows become the permanent anchor
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _t1bank=time.time()
    model.__dict__.setdefault('_bank',{})[d]=X.to(torch.uint8); model.__dict__.get('_bprio',{}).pop(d,None)
    model.__dict__.get('_rowkl',{}).pop(d,None); model.__dict__.get('_rowkl_h',{}).pop(d,None)   # cg16: fresh rows, fresh priorities
    k_=int(cfg.bank_targets); was=model.training; model.eval(); fs=model.feat_scale.clone(); Vs=[]; Is=[]; tail=0.0
    ch_=max(1,min(cfg.bank_tgt_chunk,cfg.eval_batch or cfg.bank_tgt_chunk))
    if cfg.bank_tgt_chunk_big>0 and _CHUNK['C']>0: ch_=int(cfg.bank_tgt_chunk_big)      # chunked solve: no T x T matrices, larger passes
    for c0 in range(0,X.shape[0],ch_):
        Xc=X[c0:c0+ch_].long()
        zb_=model.heads[d](priv(model,Xc,model.features(Xc),d)).float()
        lp=F.log_softmax(zb_/cfg.shunt_T,-1)
        v_,i_=lp.topk(k_,-1); tail+=float((1-v_.exp().sum(-1)).clamp(min=0).sum())
        dev_='cpu' if cfg.bank_tgt_cpu else v_.device
        vv_=v_.to(torch.float16).to(dev_); ii_=i_.to(torch.uint8).to(dev_)
        if dev_=='cpu' and torch.cuda.is_available(): vv_=vv_.pin_memory(); ii_=ii_.pin_memory()   # pinned: async copies per step
        Vs.append(vv_); Is.append(ii_)
    model.feat_scale.copy_(fs); model.train(was)
    model.__dict__.setdefault('_bank_tv',{})[d]=(torch.cat(Vs),torch.cat(Is))
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  [bank targets] {_lab(cfg,d)}: top-{k_} log-probabilities stored per byte, mean mass outside the top-{k_} {tail/(X.shape[0]*X.shape[1]):.4f}")
    if NA_>0: print(f"  [anchored bank] {_lab(cfg,d)}: {min(NA_,X.shape[0])} anchor rows (the first bank's) + {X.shape[0]-min(NA_,X.shape[0])} re-sampled, all with fresh targets")
    print(f"  [dream bank] {_lab(cfg,d)}: {X.shape[0]} rows x {X.shape[1]} bytes written ({X.numel()/1e6:.2f} MB, {time.time()-_t0bank:.0f}s: sampling {_t1bank-_t0bank:.0f}s, targets {time.time()-_t1bank:.0f}s)"+_rsg_msg())
    model.__dict__.setdefault('_bankw',{})[d]=W
    model.__dict__['_s0_dom']=None

@torch.no_grad()
def bank_roll(model, cfg, cur):
    """ROLLING DREAM REFRESH (visit end of `cur`): bank_roll of every other old domain's bank rows, oldest first (cyclic pointer), are re-sampled
    from the CURRENT weights with that domain's head (recurrent sampler) and get fresh stored top-k targets."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); olds=[j for j in sorted(BK) if j!=cur and j in TV]
    if not olds: return
    t0=time.time(); dev=model.embed.weight.device; k_=int(cfg.bank_targets); ptr=model.__dict__.setdefault('_roll_ptr',{}); nt_=0
    tag_=cfg.state_prior or cfg.lsmem or cfg.side_w>0 or cfg.xdg>0 or cfg.pu_k>0 or cfg.film_k>0 or cfg.cgate>0 or bool(cfg.ngpi)
    if cfg.bank_rec: build_teachers(model, cfg); TT=model._teach
    was=model.training; fs=model.feat_scale.clone()
    try:
        plan=[]
        for j in olds:
            K0=min(int(cfg.bank_keep),BK[j].shape[0]-1); N=BK[j].shape[0]-K0; n=max(1,int(round(cfg.bank_roll*N))); p0=ptr.get(j,0)
            plan.append((j,K0+(torch.arange(n)+p0)%N,n)); ptr[j]=(p0+n)%N              # rows < bank_keep: persistent anchor
        T=BK[olds[0]].shape[1]; XS={}
        if cfg.bank_rec:                                                 # ONE sampling stream for ALL old domains (rows grouped by domain):
            doms=[j for j,_,n in plan for _ in range(n)]                 #   the sampler is latency-bound, so big mixed chunks cost ~ one small one
            ch_b=cfg.bank_chunk if _SAMP['W']>0 else min(cfg.bank_chunk,512); out=[]
            for c0 in range(0,len(doms),ch_b): xs,_=rec_sample(TT,'main',doms[c0:c0+ch_b],T,cfg.dream_temp,model._dream_gen,None); out.append(xs)
            Xall=torch.cat(out); o0=0
            for j,_,n in plan: XS[j]=Xall[o0:o0+n]; o0+=n
        for j,ix,n in plan:
            if tag_: model.__dict__['_s0_dom']=j
            if j in XS: X=XS[j]
            else:
                rows=[]
                for c0 in range(0,n,cfg.fast_dream_chunk): rows.append(dream_sample(model,j,min(cfg.fast_dream_chunk,n-c0),T,cfg.dream_temp,gen=model._dream_gen))
                X=torch.cat(rows)
            model.eval(); Vs=[]; Is=[]
            ch_=max(1,min(cfg.bank_tgt_chunk,cfg.eval_batch or cfg.bank_tgt_chunk))
            for c0 in range(0,n,ch_):
                Xc=X[c0:c0+ch_].long(); zb_=model.heads[j](priv(model,Xc,model.features(Xc),j)).float()
                v_,i_=F.log_softmax(zb_/cfg.shunt_T,-1).topk(k_,-1); Vs.append(v_); Is.append(i_)
            model.feat_scale.copy_(fs); model.train(was)
            V0,I0=TV[j]; ixv=ix.to(V0.device)
            V0[ixv]=torch.cat(Vs).to(V0.device,torch.float16); I0[ixv]=torch.cat(Is).to(V0.device,torch.uint8)
            BK[j][ix.to(BK[j].device)]=X.to(BK[j].device,torch.uint8); model.__dict__.get('_bprio',{}).pop(j,None); nt_+=n
    finally:
        model.__dict__['_s0_dom']=None; model.feat_scale.copy_(fs); model.train(was)
        if cfg.bank_rec: object.__setattr__(model,'_teach',None)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  [bank roll] {len(olds)} old domains: {nt_} rows ({cfg.bank_roll:.0%} each) re-dreamed from the current weights with fresh targets ({time.time()-t0:.0f}s)")

@torch.no_grad()
def bank_dream_batch(model, cfg, olds):
    """dream_batch rows from the stored banks (old domains by dream share), targets from the CURRENT frozen teacher (one parallel pass)."""
    TT=model._teach; BK=model.__dict__.get('_bank',{}); ol=[d for d in olds if d in BK]
    SN=model.__dict__.get('_anc_snap',{}); snrep=(cfg.anchor_mode=='snap' and cfg.anchor_replace)
    TV=model.__dict__.get('_bank_tv',{})
    if (TT is None and not snrep and not TV) or not ol: return None
    sh=dream_shares(cfg,len(olds)); w=np.array([(sh[olds.index(d)] if sh is not None else 1.0) for d in ol],dtype=float)
    se_=model.__dict__.get('_share_sig',{})
    if cfg.share_mode=='kl' and se_:                                     # v9.8 cg10 KL-WEIGHTED SHARES: smoothed dream KL per old domain (a domain
        mk_=float(np.mean(list(se_.values()))); se_={d:se_.get(d,mk_) for d in ol}   #   without a value yet gets the mean)
    if cfg.share_mode=='sim' and model.__dict__.get('_borrow_r') is not None:
        R_=model._borrow_r; se_={d:float(R_[d]) for d in ol}
    if se_ and all(d in se_ for d in ol):                                # ADAPTIVE SHARES: equal floor + signal-proportional part
        sg=np.array([max(0.0,se_[d]) for d in ol],dtype=float)
        if sg.sum()>0: w=cfg.share_floor/len(ol)+(1-cfg.share_floor)*sg/sg.sum()
    D_=model.__dict__.get('_dctl')
    ss_=model.__dict__.get('_self_slot')                                 # cg15b SELF-PROTECTION: the revisited slot's own share x self_prot_w
    if ss_ is not None and ss_ in ol and cfg.self_prot_w>0: w=w.copy(); w[ol.index(ss_)]*=float(cfg.self_prot_w)
    gb_=model.__dict__.get('_guard_boost')                               # EARLY DRIFT GUARD: flagged domains get more of the rows
    if gb_: w=w*np.array([float(gb_.get(d,1.0)) for d in ol])
    r_=model.__dict__.get('_dsr')
    if r_ is None: r_=np.random.RandomState(cfg.seed+41); model.__dict__['_dsr']=r_
    dm=sorted(ol[i] for i in r_.choice(len(ol),size=cfg.dream_batch,p=w/w.sum()))
    dr_=model.__dict__.setdefault('_drows',[0,0]); dr_[0]+=len(dm); dr_[1]+=1; model.__dict__['_dcount']=model.__dict__.get('_dcount',0)+1
    pw=w/w.sum(); model.__dict__['_share_last']={d:float(pw[i]) for i,d in enumerate(ol)}
    hold_=max(int(cfg.bank_hold),int(cfg.dstep_hold) if cfg.dstep else 0)     # (v9.8 cg8: direct-step held-out rows)
    BW=model.__dict__.get('_bankw',{}); H0_=lambda d: min(hold_,BK[d].shape[0]-1) if hold_>0 else 0   # held-out sensor rows never trained on
    pf_=float(cfg.dream_prio); PH_=model.__dict__.get('_rowkl_h',{})
    def _pick(d):                                                        # cg16 PRIORITISED ROWS (stale KL; never-used rows at the domain's mean)
        lo_=H0_(d); n_=BK[d].shape[0]
        if pf_>0 and d in PH_ and r_.rand()<pf_:
            p_=PH_[d][lo_:n_].astype(np.float64); ok_=np.isfinite(p_)
            if ok_.any():
                p_=np.where(ok_,p_,p_[ok_].mean()); p_=np.maximum(p_,1e-6); return lo_+int(r_.choice(n_-lo_,p=p_/p_.sum()))
        return r_.randint(lo_,n_)
    ri=[_pick(d) for d in dm]
    if cfg.dream_sim: ri=_sim_rows(model,cfg,dm,ri,_pick,BK,TV,r_)              # SWIL: similarity-weighted rows within each domain
    SEM_=model.__dict__.get('_sem',{}) if float(cfg.sem_share)>0 else {}
    sj_={}                                                                      # SEMANTIC ROWS: slot j -> semantic row index (its domain's bank)
    for j,d in enumerate(dm):
        if d in SEM_ and SEM_[d]['x'] and SEM_[d]['x'][0].shape[0]==BK[d].shape[1] and r_.rand()<float(cfg.sem_share): sj_[j]=r_.randint(len(SEM_[d]['x']))
    if sj_:
        GS_=model.__dict__.setdefault('_semstat',[0,0]); GS_[0]+=len(sj_); GS_[1]+=len(dm)
    X=torch.stack([(SEM_[d]['x'][sj_[j]] if j in sj_ else BK[d][i]) for j,(d,i) in enumerate(zip(dm,ri))]).long().to(model.embed.weight.device)
    Tq_=min(cfg.dream_len,cfg.seq_len); of_=None
    if X.shape[1]>=2*Tq_ and TV and all(d in TV for d in dm) and not cfg.dream_pair_full:   # PAIRED-DREAM EPISODES: 2 x length rows; B halves for the paired share
        pr_=model.__dict__.get('_dpr')
        if pr_ is None: pr_=np.random.RandomState(cfg.seed+57); model.__dict__['_dpr']=pr_
        if cfg.dream_pair_both and len(dm)>=2:                              # both halves of dream_batch/2 pairs (same row budget)
            h_=len(dm)//2; dm=list(dm[:h_])*2; ri=list(ri[:h_])*2
            X=torch.stack([BK[d][i] for d,i in zip(dm,ri)]).long().to(model.embed.weight.device)
            pj_=list(range(h_,2*h_))
        else: pj_=[j for j in range(len(dm)) if cfg.dream_pair>0 and pr_.rand()<cfg.dream_pair]
        of_=[Tq_ if j in pj_ else 0 for j in range(len(dm))]
        model.__dict__['_pairA']=(pj_,X[pj_,:Tq_].contiguous(),[dm[j] for j in pj_]) if pj_ else None
        X=torch.stack([X[j,o:o+Tq_] for j,o in enumerate(of_)])
    model.__dict__['_bank_ri']=(list(dm),list(ri)) if pf_>0 else None          # cg16: the rows drawn (after pairing), for their priorities
    if float(cfg.prot_decay)<1.0:                                              # PROTECTION DECAYS WITH DISUSE: rows weighted decay ** (visits since last)
        SH_=model.__dict__.get('_slot_hist',[])
        def _age(d):
            idx_=[i_ for i_,s_ in enumerate(SH_) if s_==d]; return (len(SH_)-1-idx_[-1]) if idx_ else 0
        model.__dict__['_bank_w_pd']=torch.tensor([float(cfg.prot_decay)**_age(d) for d in dm],device=model.embed.weight.device)
    if any(BW.get(d) is not None for d in dm) or (cfg.share_mode and cfg.share_reweight):
        rw={d:(1.0) for d in dm}
        model.__dict__['_bank_w']=torch.stack([(BW[d][i] if BW.get(d) is not None else torch.tensor(1.0,device=X.device))*rw[d] for d,i in zip(dm,ri)]).to(X.device)
    if float(cfg.prot_decay)<1.0 and model.__dict__.get('_bank_w_pd') is not None:
        pdw_=model.__dict__.pop('_bank_w_pd'); bw0_=model.__dict__.get('_bank_w')
        model.__dict__['_bank_w']=pdw_ if (bw0_ is None or bw0_.shape[0]!=pdw_.shape[0]) else bw0_*pdw_
    if TV and all(d in TV for d in dm):                                   # STORED TARGETS: no teacher / snapshot pass
        k_=int(cfg.bank_targets)
        if of_ is None:
            V=torch.stack([(SEM_[d]['v'][sj_[j]].to(TV[d][0].device) if j in sj_ else TV[d][0][i]) for j,(d,i) in enumerate(zip(dm,ri))])
            I=torch.stack([(SEM_[d]['i'][sj_[j]].to(TV[d][1].device) if j in sj_ else TV[d][1][i]) for j,(d,i) in enumerate(zip(dm,ri))])
        else: V=torch.stack([TV[d][0][i][o:o+Tq_] for d,i,o in zip(dm,ri,of_)]); I=torch.stack([TV[d][1][i][o:o+Tq_] for d,i,o in zip(dm,ri,of_)])
        if V.device.type=='cpu' and X.device.type=='cuda': V=V.pin_memory(); I=I.pin_memory()      # pinned: a truly asynchronous copy (a
        V=V.to(X.device,non_blocking=True).float(); I=I.to(X.device,non_blocking=True).long()        #   stacked tensor is pageable: copy waited)
        rest=(1-V.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-k_)
        TG=rest.log().expand(X.shape[0],X.shape[1],256).clone(); TG.scatter_(-1,I,V)
        model.__dict__['_bank_vi']=(V,I)
        Q=model._dsq
        for j,d in enumerate(dm):
            q=Q.setdefault(d,[]); q.append((X[j],TG[j])); del q[:-8]
        return X, TG, dm, None
    if snrep and all(d in SN for d in dm): return X, None, dm, None    # SNAPSHOT TARGETS replace the teacher: no rescoring pass
    TG=torch.empty(X.shape[0],X.shape[1],256,device=X.device); groups={}
    for j,d in enumerate(dm): groups.setdefault((teacher_role(model,cfg,d,olds,TT),d),[]).append(j)
    for (role,d),ix in groups.items():
        ix_=torch.tensor(ix,device=X.device); TG[ix_]=F.log_softmax(model.heads[d](TT.features(X[ix_],role)).float()/cfg.shunt_T,-1)
    Q=model._dsq
    for j,d in enumerate(dm):
        q=Q.setdefault(d,[]); q.append((X[j],TG[j])); del q[:-8]
    return X, TG, dm, None

def stream_batch(model, cfg, olds):
    """One dream batch from the streams: rows drawn over the old domains by dream share (own RNG), each chunk used once."""
    if cfg.dream_bank: return bank_dream_batch(model, cfg, olds)
    n_old=len(olds); sh=dream_shares(cfg,n_old); w=np.array(sh if sh is not None else [1.0/n_old]*n_old,dtype=float)
    r_=model.__dict__.get('_dsr')
    if r_ is None: r_=np.random.RandomState(cfg.seed+41); model.__dict__['_dsr']=r_
    Q=model._dsq; SQ=None
    n_s=min(len(SQ),int(round(cfg.dream_batch*cfg.strand_share))) if SQ else 0
    picks=r_.choice(n_old,size=cfg.dream_batch-n_s,p=w/w.sum()); xs=[];tg=[];dm=[]
    for i_ in picks:
        d=olds[i_]
        if not Q.get(d):                                                              # (slack exhausted: any domain with chunks left)
            d=next((e for e in olds if Q.get(e)),None)
            if d is None: break
        a,b=Q[d].pop(0); xs.append(a); tg.append(b); dm.append(d)
        dl=model.__dict__.setdefault('_dstream_last',{}).setdefault(d,[]); dl.append(a); del dl[:-16]   # last chunks (diagnostic only)
    if not xs: return None
    order=sorted(range(len(dm)),key=lambda j:dm[j])
    X=torch.stack([xs[j] for j in order]); TG=torch.stack([tg[j] for j in order]); DM=[dm[j] for j in order]
    if not n_s: return X, TG, DM, None
    rs=[SQ.pop(0) for _ in range(n_s)]; T=X.shape[1]; z=torch.zeros_like(TG[0])
    zl=torch.full((T,),-1,dtype=torch.long,device=X.device); zw=torch.zeros(T,device=X.device); zs=torch.zeros(T,dtype=torch.int8,device=X.device)
    ex={'n':n_s,'kind':[r['kind'] for r in rs],'pr':[r.get('pr') for r in rs],
        'hd':torch.stack([r['hd'] if r['hd'] is not None else zl for r in rs]),'w':torch.stack([r['w'] if r['w'] is not None else zw for r in rs]),
        'aw':torch.stack([r['aw'] for r in rs]),'seg':torch.stack([r['seg'] if r['seg'] is not None else zs for r in rs])}
    return torch.cat([X,torch.stack([r['x'] for r in rs])]), torch.cat([TG,torch.stack([r['tg'] if r['tg'] is not None else z for r in rs])]), DM, ex

_BYTE_NAMES=['letters','space','layout','digits','punct','symbols','other']
_BYTE_CLS=torch.full((256,),6,dtype=torch.long)
for _c in range(256):
    _ch=chr(_c)
    if _ch.isascii() and _ch.isalpha(): _BYTE_CLS[_c]=0
    elif _c==32: _BYTE_CLS[_c]=1
    elif _c in (9,10,13): _BYTE_CLS[_c]=2
    elif _ch.isdigit() and _c<128: _BYTE_CLS[_c]=3
    elif _ch in ".,;:'\"!?-": _BYTE_CLS[_c]=4
    elif _ch in "()[]{}<>=+*/\\|_&^%$#@~`": _BYTE_CLS[_c]=5

def _cons3_grad(model, cfg, d):
    """THREE TIMESCALES: is old domain d graduated (visited >= cons3_age domains ago)?"""
    return False
    ol=list(getattr(model,'_olds',None) or [])
    return d in ol and (len(ol)-1-ol.index(d))>=cfg.cons3_age

def dream_shares(cfg, dom_id):
    """Share of the dream budget per old domain (recency-weighted, graduated domains at the floor). None = equal rotation."""
    return None
    w=np.array([cfg.dream_recency**(dom_id-1-d) for d in range(dom_id)],dtype=float)
    for d in range(dom_id):
        if dom_id-1-d>=cfg.dream_graduate: w[d]=cfg.dream_grad_floor
    if w.sum()<=0: w[-1]=1.0
    return w/w.sum()

def dream_pick(model, cfg, dom_id, sh, salt):
    """Domain for this dream batch / refresh, drawn by share with its own RNG (training stream untouched)."""
    r=model.__dict__.get('_dpick')
    if r is None: r=np.random.RandomState(cfg.seed+13); object.__setattr__(model,'_dpick',r)
    return int(r.choice(dom_id,p=sh))

def dream_cap(cfg, dom_id, d):
    """Pool size for old domain d: the same total budget (dream_n per old domain) redistributed by share."""
    sh=dream_shares(cfg, dom_id); m_=float(getattr(cfg,'dream_pool_mult',1.0))
    if sh is None: return int(round(cfg.dream_n*m_))
    if sh[d]<=0: return 0
    return int(round(m_*min(4*cfg.dream_n, max(cfg.dream_batch, round(cfg.dream_n*dom_id*sh[d])))))

@torch.no_grad()
def dream_refresh(model, cfg, dom_id, d, n):
    """Add n fresh dreams for old domain d, sampled from the MAIN weights (weights minus shunt)."""
    Pm={id(p_):p_ for p_ in model.parameters()}; Sd={k:v for k,v in (getattr(model,'_shunt',None) or {}).items() if k in Pm and Pm[k].shape==v.shape}
    if _cons3_grad(model,cfg,d):                   # graduated: dream from the SLOW weights (main minus mid store)
        for k,v in (getattr(model,'_mid',None) or {}).items():
            if k in Pm and Pm[k].shape==v.shape: Sd[k]=(Sd[k]+v) if k in Sd else v
    for k,v in Sd.items(): Pm[k].sub_(v)
    mx=current_mix(model); dd=domain_delta(model,d) if getattr(model,'_spec',None) else {}
    apply_delta(model,mx,-1); apply_delta(model,dd,+1)             # domain d's own configuration
    if not hasattr(model,'_dream_gen'):          # own RNG: dreaming must not shift the training random stream
        g_=torch.Generator(device=model.embed.weight.device); g_.manual_seed(cfg.seed+11); object.__setattr__(model,'_dream_gen',g_)
    nn_=n
    _t0=time.perf_counter()
    x=dream_sample(model, d, nn_, min(cfg.dream_len,cfg.seq_len), cfg.dream_temp, gen=model._dream_gen)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    model.__dict__['_tm_dream']=model.__dict__.get('_tm_dream',0.0)+time.perf_counter()-_t0
    apply_delta(model,dd,-1); apply_delta(model,mx,+1)
    for k,v in Sd.items(): Pm[k].add_(v)
    pool=model.__dict__.setdefault('_dreams',{}).setdefault(d,[])
    _ol=list(getattr(model,'_olds',None) if getattr(model,'_olds',None) is not None else range(dom_id))
    pool.extend(list(x)); del pool[:-(cfg.dream_n)]

def adam_capture(opt, model): return {n:dict(opt.state.get(p,{})) for n,p in model.named_parameters()}
def adam_rebuild(model, lr, before):
    """New Adam that KEEPS moment estimates of every existing weight (zero for newly activated rows/cols)."""
    opt=torch.optim.Adam(_opt_params(model),lr=lr)
    for n,p in model.named_parameters():
        st=before.get(n)
        if not st or not p.requires_grad: continue
        ns={}
        for k,v in st.items():
            if torch.is_tensor(v) and v.dim()>0 and v.shape!=p.shape:
                z=torch.zeros_like(p); z[tuple(slice(0,m) for m in v.shape)]=v; ns[k]=z
            else: ns[k]=v.clone() if torch.is_tensor(v) else v
        opt.state[p]=ns
    return opt

class _GScale(torch.autograd.Function):
    """Identity forward; the backward pass multiplies the gradient by s (per row: s of shape (B,1,1))."""
    @staticmethod
    def forward(ctx, x, s): ctx.save_for_backward(s); return x.view_as(x)
    @staticmethod
    def backward(ctx, g): (s,)=ctx.saved_tensors; return g*s.to(g.dtype), None

class _GProj(torch.autograd.Function):
    """Identity forward; backward: a x the gradient's component in span(U) + c x the rest: g -> a g U U^T + c (g - g U U^T)."""
    @staticmethod
    def forward(ctx, x, U, c, a=1.0): ctx.save_for_backward(U); ctx.c=c; ctx.a=a; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        (U,)=ctx.saved_tensors; gf=g.float(); gs=(gf@U)@U.T
        return (ctx.a*gs+ctx.c*(gf-gs)).to(g.dtype), None, None, None

_NG_P=2147483647                                        # Mersenne prime 2^31 - 1
_NG_A=[1103515245,  987654323, 1789773571,  612481199]   # per-hash multipliers (< P) of the universal ((A h + C) mod P) mod size map
_NG_C=[  12345,     40503,    7919,     104729]
def _ngram_idx(cols, n, size, salt=0):
    """Hash of n bytes (cols: list of (B,T) or (B,1) long tensors, newest first; -1 = before the start); salt = which hash. A polynomial
    fingerprint mod P (base depends on the hash) followed by a universal map ((A h + C) mod P) mod size, so that different hashes
    collide INDEPENDENTLY (a multiplicative map mod a power of two would make hash 2 a function of hash 1)."""
    base=1000003+7777*salt; h=torch.full_like(cols[0],n)
    for i in range(n): h=(h*base+cols[i]+1)%_NG_P
    return ((h*_NG_A[salt%4]+_NG_C[salt%4]+salt//4)%_NG_P)%size

def ngram_embed(tables, x, touched=None, counts=None, ncount=None):
    """C1: sum over n (and over the hashes) of the hashed n-gram embedding ending at each position (x (B,T) bytes). touched: rows used
    (lazy optimiser); counts: per-row usage of the first ncount rows (real text) for the row protection."""
    out=0.0; nmax=max(tables.ns); nh=getattr(tables,'nh',1)
    cols=_seg_cols([x]+[F.pad(x,(i,0),value=-1)[:,:x.shape[1]] for i in range(1,nmax)])
    for i_,(n,E) in enumerate(zip(tables.ns,tables)):
        for j in range(nh):
            idx=_ngram_idx(cols,n,E.num_embeddings,j); out=out+E(idx)
            if touched is not None: touched.setdefault(i_,[]).append(idx.flatten())
            if counts is not None: ic_=idx[:ncount].flatten(); counts[i_].index_add_(0,ic_,torch.ones(ic_.shape,device=ic_.device,dtype=counts[i_].dtype))   # (no CPU wait)
    return out

def ngram_embed_tok(tables, x, hist):
    """One position: x (B,1) current byte, hist (B,nmax-1) previous bytes, newest LAST (-1 = before the start)."""
    nmax=max(tables.ns); nh=getattr(tables,'nh',1); cols=[x]+[hist[:,-i].unsqueeze(1) for i in range(1,nmax)]
    out=0.0
    for n,E in zip(tables.ns,tables):
        for j in range(nh): out=out+E(_ngram_idx(cols,n,E.num_embeddings,j))
    return out

def _new_opt(model, cfg, lr=None):
    """Main optimiser: Adam over _opt_params; fused CUDA kernel when adam_fused and on GPU."""
    kw={'fused':True} if (getattr(cfg,'adam_fused',False) and torch.cuda.is_available()) else {}
    return torch.optim.Adam(_opt_params(model),lr=cfg.lr if lr is None else lr,**kw)

_COMPILED={}
def setup_compile(model, cfg):
    """torch.compile of Model._features_eager (registered by id: frozen copies and teachers stay eager)."""
    return
    try:
        import torch._dynamo as _dyn; _dyn.config.cache_size_limit=max(64,_dyn.config.cache_size_limit)
        _COMPILED[id(model)]=torch.compile(model._features_eager, dynamic=True); print("  [compile] model forward compiled (torch.compile, dynamic shapes)")
    except Exception as e: print(f"  [compile] not available ({repr(e)[:120]}); eager")

def _opt_params(model):
    """Parameters of the main optimiser (the n-gram tables have their own lazy optimiser when ngram_sparse)."""
    ex={id(p) for p in model.ngram.parameters()} if (getattr(model,'ngram',None) is not None and model.cfg.ngram_sparse) else set()
    return [p for p in model.parameters() if p.requires_grad and id(p) not in ex]

def _ng_prot_scale(model, cfg, i_):
    """Row-protection step scale 1/(1 + gamma u/mean u) of table i_ (None when off or before any old usage)."""
    old=model.__dict__.get('_ng_old')
    if cfg.ngram_protect<=0 or old is None: return None
    C=model.__dict__.setdefault('_ng_sc',{})                        # cached per visit (no GPU sync per step)
    if i_ not in C:
        u=old[i_]; C[i_]=None if float(u.sum())<=0 else 1.0/(1.0+cfg.ngram_protect*(u/u[u>0].mean()))
    return C[i_]

_SEG={'s':None}                                                     # v9 RUN MASKING: (B,T) start of the clean segment of each position
def _seg_cols(cols):
    s_=_SEG['s']
    if s_ is None or tuple(s_.shape)!=tuple(cols[0].shape): return cols
    t_=torch.arange(cols[0].shape[1],device=cols[0].device)[None]
    return [cols[0]]+[c_.masked_fill((t_-i)<s_,-1) for i,c_ in enumerate(cols[1:],1)]
def _cols(x, nmax):
    """Newest-first byte columns of x (B,T): cols[i] = the byte i positions back (-1 before the start)."""
    return _seg_cols([x]+[F.pad(x,(i,0),value=-1)[:,:x.shape[1]] for i in range(1,nmax)])

def _cnt_idx(cfg, L, cols, j):
    """Row of the order-L count sketch j for the context ending at each position (L = 1: the byte itself)."""
    if L==1: return cols[0].clamp(min=0)
    return _ngram_idx(cols,L,cfg.ngram_count_rows,j+2)

def _cnt_tables(cfg):
    """(order, sketch, rows) of every count sketch."""
    out=[]
    for L in cfg.ngram_count:
        nh=1 if L<=2 else max(1,int(cfg.ngram_count_hashes))
        for j in range(nh): out.append((int(L),j,256 if L==1 else int(cfg.ngram_count_rows)))
    return out

@torch.no_grad()
def count_logratio(cfg, C, cols):
    """Interpolated absolute-discount backoff over the orders, from the count sketches C (list aligned with _cnt_tables); returns
    log(p * 256) (B,T,256): zero for an uninformative context."""
    tabs=_cnt_tables(cfg); D=cfg.ngram_count_D; sh=cols[0].shape
    p=torch.full((*sh,256),1.0/256,device=cols[0].device)
    for L in sorted(set(cfg.ngram_count)):
        c=None
        for (L_,j,_),T_ in zip(tabs,C):
            if L_!=L: continue
            ix_=_cnt_idx(cfg,L,cols,j)
            g=(T_[0][ix_]+T_[1][ix_]) if isinstance(T_,tuple) else T_[ix_]
            c=g if c is None else torch.minimum(c,g)
        tot=c.sum(-1,keepdim=True); nz=(c>0).float().sum(-1,keepdim=True)
        pn=((c-D).clamp(min=0)+D*nz*p)/tot.clamp(min=1e-9)
        p=torch.where(tot>0,pn,p)
    return torch.log(p.clamp(min=1e-7)*256.0)

def count_feature(model, cfg, cols, C=None, proj=None):
    """Count-predictor feature for the residual stream (live counts = old + current unless C is given)."""
    if C is None: C=list(zip(model._cnt_old,model._cnt_cur))            # live = old + current (summed after the gather)
    lr_=count_logratio(cfg,C,cols)
    return (proj or model.cproj)(lr_.to((proj or model.cproj).weight.dtype))

@torch.no_grad()
def ngram_lazy_step(model, cfg, lr, b1=0.9, b2=0.999, eps=1e-8):
    """LAZY ADAM for the n-gram tables: only the rows used this step are updated (per-row step counts and moments); optional frequency-
    based protection of rows the earlier domains used heavily."""
    T_=model.__dict__.pop('_ng_touch',{}) ; ST=model.__dict__.setdefault('_ng_state',{})
    for i_,E in enumerate(model.ngram):
        W=E.weight; G=W.grad
        if G is None: continue
        if i_ in T_: r=torch.unique(torch.cat(T_[i_]))
        else: r=torch.nonzero(G.abs().sum(-1)>0).squeeze(1)
        cnt_=cfg.ngram_opt=='count'
        if i_ not in ST: ST[i_]=({} if cnt_ else {'m':torch.zeros_like(W),'v':torch.zeros_like(W)})|{'t':torch.zeros(W.shape[0],device=W.device)}
        st=ST[i_]; g=G[r]
        if cnt_:                                                    # COUNTING RULE: no moments - the row's gradient direction (RMS-
            n_=st['t'][r][:,None]                                   # normalised over the row) x (n0 / (n0 + n))^p, n = the row's lifetime uses
            up=g/(g.pow(2).mean(-1,keepdim=True).sqrt()+1e-12)*(cfg.ngram_count_n0/(cfg.ngram_count_n0+n_))**cfg.ngram_count_p
            st['t'][r]+=1
        else:
            st['t'][r]+=1; tr=st['t'][r][:,None]
            st['m'][r]=b1*st['m'][r]+(1-b1)*g; st['v'][r]=b2*st['v'][r]+(1-b2)*g*g
            up=(st['m'][r]/(1-b1**tr))/((st['v'][r]/(1-b2**tr)).sqrt()+eps)
        sc=_ng_prot_scale(model,cfg,i_)
        if sc is not None: up=up*sc[r][:,None]
        W[r]-=lr*cfg.ngram_lr_mult*cfg.ngram_in_mult*up
        W.grad=None

def _hist_push(TT, hist, x):
    """Sampler byte history for the n-gram embeddings (None when the model has none)."""
    ng=getattr(TT.model,'ngram',None)
    if ng is None and not TT.model.cfg.ngram_count: return None
    if hist is None: hist=torch.full((x.shape[0],_hist_len(TT.model)),-1,dtype=torch.long,device=x.device)
    return torch.cat([hist[:,1:],x],1) if hist.shape[1]>0 else hist

def _hist_len(model):
    ng=getattr(model,'ngram',None); a=max(ng.ns) if ng is not None else 1
    return max(a,1)-1

def _dream_route_rows(cfg, n1, ns, dev, nb):
    """Per-row part multipliers for a dream batch (single-domain rows then strand rows), or None when nothing is routed."""
    return None
    r1=torch.tensor([1.0]*(nb+1),device=dev,dtype=torch.float32)
    r2=r1
    return torch.cat([r1.expand(n1,-1),r2.expand(ns,-1)])

def route_scales(M):
    """Per-row part multipliers M (B, n_blocks+1) = (emb, b0, b1, ...) -> gradient scales at each block's input and after the last block,
    such that the gradient reaching part j's parameters is M[:,j] x the unrouted gradient (a part below a zero-weight part gets 0)."""
    L=M.shape[1]-1; sc=[]
    for j in range(L):
        lo,up=M[:,j],M[:,j+1]
        sc.append(torch.where(up>0,lo/up.clamp(min=1e-12),torch.zeros_like(lo)).view(-1,1,1))
    return sc, M[:,L].reshape(-1,1,1)

class Model(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg
        s.embed=nn.Embedding(cfg.vocab,cfg.d_model); s.pos=nn.Embedding(cfg.seq_len,cfg.d_model)
        s.pos.weight.data.zero_(); s.pos.weight.requires_grad_(False)
        ns0=int(round(cfg.d_state*cfg.init_state_mult)); nm0=int(round(2*cfg.d_model*cfg.init_mlp_mult))
        ex=lambda li: (cfg.block_extra[li] if (cfg.block_extra and li<len(cfg.block_extra) and cfg.block_extra_mode=='start') else 0.0)
        s.blocks=nn.ModuleList([DeltaBlock(cfg.d_model,int(round(ns0*(1+ex(li)))),int(round(nm0*(1+ex(li))))) for li in range(cfg.n_layers)])
        for li_,b in enumerate(s.blocks):
            b.gate_ln=cfg.gate_ln; b.setup_mixer(cfg)
            if li_ in tuple(cfg.mix_local_blocks or ()): b.setup_local(cfg)
            if li_ in tuple(cfg.lattn_blocks or ()): b.setup_lattn(cfg)
            b.parallel=bool(cfg.parallel_block); b._ddrive_on=cfg.gate_dream_relax>0
        sz=cfg.ngram_size if isinstance(cfg.ngram_size,(tuple,list)) else [cfg.ngram_size]*len(cfg.ngram_emb)
        s.ngram=nn.ModuleList([nn.Embedding(int(z),cfg.d_model) for z in sz]); s.ngram.ns=tuple(int(n) for n in cfg.ngram_emb)
        s.ngram.nh=max(1,int(cfg.ngram_hashes))
        for E in s.ngram: E.weight.data.zero_()
        s.__dict__['_ng_cur']=[torch.zeros(int(z)) for z in sz]; s.__dict__['_ng_old']=[torch.zeros(int(z)) for z in sz]
        if cfg.pu_k>0:                                                  # PRIVATE TRUNK UNITS per domain
            k_=int(cfg.pu_k); ND_=len(PILE_DOMAINS)
            for b in s.blocks:
                b.pu_in=nn.Parameter(torch.randn(ND_,cfg.d_model,k_)*(1.0/math.sqrt(cfg.d_model))); b.pu_b=nn.Parameter(torch.zeros(ND_,k_))
                b.pu_out=nn.Parameter(torch.zeros(ND_,k_,cfg.d_model)); b._pu_dream_grad=bool(cfg.pu_dream_grad)
                if 0<cfg.pu_k_base<k_:
                    mk_=torch.zeros(ND_,k_); mk_[:,:int(cfg.pu_k_base)]=1.0; b.register_buffer('pu_mask',mk_)
        k_=int(cfg.film_k); s.film_codes=nn.Parameter(torch.randn(len(PILE_DOMAINS),k_)*0.1)
        for b in s.blocks:
            b.film_g=nn.Parameter(torch.zeros(cfg.d_model,k_)); b.film_b=nn.Parameter(torch.zeros(cfg.d_model,k_))
        s.heads=nn.ModuleList([Head(cfg.d_model,cfg.vocab) for _ in range(len(PILE_DOMAINS))])
        for h_ in s.heads: h_.cap=cfg.logit_cap
        # EMBEDDING CHANNEL NEURONS: shared byte/position entry point, gated like any layer
        s.register_buffer('emb_theta0', sample_theta(cfg.d_model))
        s.register_buffer('emb_imp', torch.zeros(cfg.d_model))
        s.register_buffer('emb_usage', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive_ema', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive_slow', torch.zeros(cfg.d_model)); s.register_buffer('emb_drive_peak', torch.zeros(cfg.d_model))
        s.register_buffer('emb_sens', torch.zeros(cfg.d_model))
        # DISTINCTNESS: running feature-centroid of content seen so far (the "typical"
        # content, dominated by the majority=prose). distinctness = distance from it.
        s.register_buffer('feat_centroid', torch.zeros(cfg.d_model))
        s.register_buffer('feat_scale', torch.ones(1))         # running distance scale (for [0,1] normalization)
        s.register_buffer('centroid_init', torch.zeros(1))
        s.register_buffer('distinct_locked', torch.zeros(1))   # once locked, centroid/scale frozen
    def lock_distinctness(s):
        s.distinct_locked.fill_(1)   # freeze the reference -> stable routing for all content
        for b in s.blocks: b.distinct_locked.fill_(1)   # blocks read their own copy in route_gate
    def _distinctness(s, h):
        # h: (B,T,d_model). distinctness = normalized distance from the running centroid.
        feat=h.detach().mean((0,1))
        if s.centroid_init.item()==0:
            s.feat_centroid.copy_(feat); s.centroid_init.fill_(1); return torch.tensor(0.0,device=h.device)
        dist=(feat-s.feat_centroid).norm()
        if s.distinct_locked.item()==0:               # only update reference BEFORE lock
            s.feat_scale.mul_(0.99).add_(0.01*dist)
        d=torch.clamp(dist/(s.feat_scale*2+1e-6),0,1)
        # centroid drifts ONLY before lock; after lock it's frozen -> STABLE routing
        # (a given content always gets the same distinctness -> reads its own region)
        if s.training and s.distinct_locked.item()==0:
            s.feat_centroid.mul_(0.999).add_(0.001*feat)
        return d
    def forward(s,x,domain=0):
        if s.cfg.slot_mem and not _CARRY['on'] and int(domain) in s.__dict__.get('_smem',{}):     # SLOT MEMORY: scoring slot d starts from S_d
            sv_=(_CARRY['st'],_CARRY['slot'],_CARRY['sm']); _CARRY['sm']=s.__dict__['_smem']; _CARRY['slot']=int(domain)
            _CARRY['on']=True; _CARRY['nr']=x.shape[0]; _CARRY['new']={}; _CARRY['st']=_smem_seed(int(domain),x.shape[0])
            try: return s._forward0(x,domain)
            finally: _CARRY['on']=False; _CARRY['new']={}; _CARRY['st'],_CARRY['slot'],_CARRY['sm']=sv_
        return s._forward0(x,domain)
    def _forward0(s,x,domain=0):
        if (s.cfg.ngp and s.__dict__.get('_ngp') is not None) or (s.cfg.fcp_tables and s.__dict__.get('_fcp') is not None):   # + private tables
            sv_=s.__dict__.get('_s0_dom')
            s.__dict__['_s0_dom']=int(domain)
            try: h_=s.features(x)
            finally: s.__dict__['_s0_dom']=sv_
            hd_=s.heads[domain]; z_=priv(s,x,h_,int(domain))
            if _HCAP['on']: _HCAP['h']=z_.detach()
            return hd_.detached(z_) if _BUFDET['on'] else hd_(z_)
        sv_=s.__dict__.get('_s0_dom'); s.__dict__['_s0_dom']=int(domain)
        try:
            z_=s.features(x)
            if _HCAP['on']: _HCAP['h']=z_.detach()
            return s.heads[domain].detached(z_) if _BUFDET['on'] else s.heads[domain](z_)
        finally: s.__dict__['_s0_dom']=sv_
        return s.heads[domain](s.features(x))
    def features(s,x,route=None,nstat=None):
        f_=_COMPILED.get(id(s))
        if f_ is not None and not _COMPILED.get('failed'):
            try: return f_(x,route=route,nstat=nstat)
            except Exception as e:
                _COMPILED['failed']=True; print(f"  [compile] compiled forward failed ({repr(e)[:160]}); eager from now on")
        return s._features_eager(x,route,nstat)
    def _features_eager(s,x,route=None,nstat=None):
        T=x.size(1); h=s.embed(x)+s.pos(torch.arange(T,device=x.device)%s.pos.num_embeddings)
        if _STF['on']:                                                                  # STIFFNESS: which rows are dreams in this pass
            if torch.is_grad_enabled() and (s.training or _STF['dream_call']):
                ns0_=0 if _STF['dream_call'] else nstat; _STF['B']=x.shape[0]; _STF['ns']=ns0_ if (ns0_ is not None and ns0_<x.shape[0]) else None
            else: _STF['ns']=None
        if _XDG['on'] or _XDG.get('tagon'): _XDG['rows']=s.__dict__.get('_s0_dom')
        if _PJ['on']:                                                                   # PROJECTION: which rows are dreams in this pass
            _PJ['B']=x.shape[0]; _PJ['ns']=0 if _PJ['dream_call'] else (nstat if (s.training and nstat is not None and nstat<x.shape[0]) else None)
        if getattr(s,'ngram',None) is not None:                                         # C1: hashed byte n-grams
            rec_=s.training and torch.is_grad_enabled()
            h=h+ngram_embed(s.ngram,x,touched=s.__dict__.setdefault('_ng_touch',{}) if (rec_ and s.cfg.ngram_sparse) else None,
                            counts=s.__dict__.get('_ng_cur') if (rec_ and s.cfg.ngram_protect>0) else None,ncount=nstat)
        s._nstat=nstat
        for b in s.blocks: b._nstat=nstat
        if s.training:
            cd=h.detach()[:nstat].abs().mean((0,1))
            s.emb_usage+=cd; s.emb_drive=cd
            _track(s,'emb',cd)
        if s.cfg.route_mode=='none' and all(b.calib is None for b in s.blocks):   # unused: skip (it forced 2 GPU syncs per pass)
            distinct=None
        else: distinct=s._distinctness(h)                      # online distinctness of this content
        for b in s.blocks: b.cur_distinct=distinct
        ac=(torch.autocast(device_type=h.device.type, dtype=torch.bfloat16) if s.cfg.amp=='bf16' else contextlib.nullcontext())
        rs_=route_scales(route) if route is not None else None      # PER-LAYER DREAMS: gradient multipliers at the block boundaries
        with ac:
            side_=s.cfg.side_w>0 and s.__dict__.get('_side_on') and s.__dict__.get('_s0_dom') is not None
            taps_=[h.detach()] if side_ else None
            for i_,b in enumerate(s.blocks):
                if _INC['on'] and i_==_INC['k']: h=h[:,-1:]                       # INCREMENTAL GENERATION: newest byte only from here on
                b._dcontent=bool(s.cfg.dream_content>0 and i_>=len(s.blocks)-int(s.cfg.dream_content))
                if rs_ is not None: h=_GScale.apply(h,rs_[0][i_])
                h=b(h)
                if side_: taps_.append(h.detach())
                if _ENC_STOP['k'] is not None and i_>=_ENC_STOP['k']: break          # v9e FIRST-BLOCK CHECK: the re-encode stops here
            if rs_ is not None: h=_GScale.apply(h,rs_[1])
        h=h.float()
        if side_: h=h+side_out(s,taps_,s.__dict__.get('_s0_dom'),nstat)
        if s.training and getattr(s,'_keep_h',False): s._last_h=h.detach()[:nstat]
        if nstat is not None:
            s._nstat=None
            for b in s.blocks: b._nstat=None
        _STF['ns']=None; _PJ['ns']=None; _XDG['rows']=None
        return h
    def widen_all(s, chunk):
        for b in s.blocks:
            if b.d_state < s.cfg.d_state_max: b.widen(chunk, s.embed.weight.device)

# ---------------- learned meta-controller ----------------
class MetaController(nn.Module):
    """Takes per-unit features -> per-unit write-gate in [0,1]. Shared across all
    units/blocks (small, generalizes). Features: [usage, grad_mag, flow, frozen_frac]."""
    def __init__(s, n_feat=6, hid=16):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(n_feat,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1))
    def forward(s, feats):  # (N_units, n_feat) -> (N_units,) gate
        return torch.sigmoid(s.net(feats)).squeeze(-1)

@torch.no_grad()
def flow_importance(b):
    fi=b.Wk.weight.abs().sum(1)+b.Wv.weight.abs().sum(1); fo=b.Wout.weight.abs().sum(0)
    ssm=(fi/(fi.mean()+1e-8))*(fo/(fo.mean()+1e-8))
    mfi=b.mlp_in.weight.abs().sum(1); mfo=b.mlp_out.weight.abs().sum(0)
    mlp=(mfi/(mfi.mean()+1e-8))*(mfo/(mfo.mean()+1e-8))
    return ssm, mlp

def unit_features(b, prog):
    # per-unit features for SSM units and MLP units (prog = fraction of training done)
    ssm_flow, mlp_flow = flow_importance(b)
    def norm(x): return x/(x.mean()+1e-8)
    ssm_gm = norm(b.Wv.weight.grad.abs().mean(1)) if b.Wv.weight.grad is not None else torch.zeros_like(b.unit_usage)
    mlp_gm = norm(b.mlp_in.weight.grad.abs().mean(1)) if b.mlp_in.weight.grad is not None else torch.zeros_like(b.mlp_usage)
    ssm_imp=norm(b.unit_usage)*ssm_flow; mlp_imp=norm(b.mlp_usage)*mlp_flow
    ssm_pol=(norm(b.unit_usage)-norm(b.unit_home)).abs(); mlp_pol=(norm(b.mlp_usage)-norm(b.mlp_home)).abs()
    ssm=torch.stack([norm(b.unit_usage), ssm_gm, ssm_flow, ssm_imp, ssm_pol, torch.full_like(b.unit_usage,prog)],1)
    mlp=torch.stack([norm(b.mlp_usage), mlp_gm, mlp_flow, mlp_imp, mlp_pol, torch.full_like(b.mlp_usage,prog)],1)
    return ssm, mlp

# ---------------- MAML-style differentiable meta helpers ----------------
def _diff_features(b, grads_by_param, prog):
    """Differentiable per-unit features (uses graph-connected grads, not .grad buffers)."""
    ssm_flow, mlp_flow = flow_importance(b)   # no_grad (weight-based), fine as features
    def norm(x): return x/(x.mean()+1e-8)
    gWv=grads_by_param.get(id(b.Wv.weight)); gmlp=grads_by_param.get(id(b.mlp_in.weight))
    ssm_gm = norm(gWv.abs().mean(1)) if gWv is not None else torch.zeros_like(b.unit_usage)
    mlp_gm = norm(gmlp.abs().mean(1)) if gmlp is not None else torch.zeros_like(b.mlp_usage)
    # IMPORTANCE score (usage*flow - what plain importance-freezing ranks by; the floor)
    ssm_imp = norm(b.unit_usage)*ssm_flow; mlp_imp = norm(b.mlp_usage)*mlp_flow
    # POLARIZATION: |current usage - home usage| (far from home region = high)
    ssm_pol = (norm(b.unit_usage)-norm(b.unit_home)).abs(); mlp_pol=(norm(b.mlp_usage)-norm(b.mlp_home)).abs()
    ssm=torch.stack([norm(b.unit_usage), ssm_gm, ssm_flow, ssm_imp, ssm_pol, torch.full_like(b.unit_usage,prog)],1)
    mlp=torch.stack([norm(b.mlp_usage), mlp_gm, mlp_flow, mlp_imp, mlp_pol, torch.full_like(b.mlp_usage,prog)],1)
    return ssm, mlp

def _virtual_gated_step(model, meta, block_params, grads, prog, lr):
    """Form virtual updated params p' = p - lr*gate*grad, differentiable w.r.t. meta.
    Returns a dict {param_name: virtual_tensor} for functional_call."""
    grads_by_param={id(p): g for p,g in zip(block_params, grads)}
    # per-block gates (graph-connected through meta)
    gates={}
    for b in model.blocks:
        sf,mf=_diff_features(b, grads_by_param, prog)
        gs=meta(sf); gm=meta(mf)
        # apply the HARD freeze in the virtual step too (meta-training sees the floor)
        gs=gs*(~b.frozen_units).float(); gm=gm*(~b.mlp_frozen).float()
        gates[id(b.Wv.weight)]=gs.unsqueeze(1); gates[id(b.Wk.weight)]=gs.unsqueeze(1)
        gates[id(b.Wout.weight)]=gs.unsqueeze(0)
        gates[id(b.mlp_in.weight)]=gm.unsqueeze(1); gates[id(b.mlp_in.bias)]=gm
        gates[id(b.mlp_out.weight)]=gm.unsqueeze(0)
    # build full param dict for functional_call: virtual for gated params, real for rest
    virtual={}
    name_by_id={id(p):n for n,p in model.named_parameters()}
    gp_ids=set(id(p) for p in block_params)
    for n,p in model.named_parameters():
        if id(p) in gp_ids:
            g=grads_by_param[id(p)]
            if g is None: virtual[n]=p; continue
            gate=gates.get(id(p), 1.0)
            virtual[n]=p - lr*gate*g       # differentiable virtual update
        else:
            virtual[n]=p
    return virtual

def _functional_buffer_loss(model, virtual_params, x, y, domain):
    """Forward the model with the VIRTUAL params (differentiable) and return CE loss."""
    from torch.func import functional_call
    out=functional_call(model, virtual_params, (x, domain))
    return F.cross_entropy(out.reshape(-1,256), y.reshape(-1))

def _gate(drive, ema, theta0, imp, cfg, floor=None):
    d=drive/(ema+1e-8)
    return torch.clamp(torch.sigmoid((d-_theta(theta0,imp,cfg))/cfg.theta_temp), min=cfg.theta_floor if floor is None else floor)

def _comp_gate(drive, ema, theta0, imp, cfg, budget=None, count=None):
    """IMPEDANCE COMPETITION: impedance R = random base + alpha*importance. Each step, units are
    ranked by (relative drive - R); the write flows through the lowest-impedance path: the top
    write_budget fraction gets ~full write, the rest fall off sharply to comp_floor."""
    d=drive/(ema+1e-8)
    s=d-(theta0+imp)
    if count is not None: k=max(1,min(int(count),s.numel()))
    else: k=max(1,int(round((cfg.write_budget if budget is None else budget)*s.numel())))
    cut=torch.topk(s,k).values[-1]
    return torch.clamp(torch.sigmoid((s-cut)/cfg.comp_temp), min=cfg.comp_floor)

def _any_gate(drive, ema, theta0, imp, cfg, floor=None):
    # channels & embeddings: their own rule and budget
    return (_comp_gate(drive,ema,theta0,imp,cfg,cfg.channel_budget) if cfg.channel_gate_mode=='competitive'
            else _gate(drive,ema,theta0,imp,cfg,floor))

def threshold_gates(b, cfg):
    """Per-unit write gate: sigmoid((drive - threshold)/temp), threshold = theta0 + alpha*imp.
    Drive is normalized to mean 1 per layer. Floor keeps a trickle so dead units can wake."""
    def g(drive, ema, theta0, imp):
        # drive relative to THIS unit's typical drive: similar content that drives an
        # important unit at its usual (high) level no longer clears its raised threshold
        d=drive/(ema+1e-8)
        theta=_theta(theta0,imp,cfg)
        return torch.clamp(torch.sigmoid((d-theta)/cfg.theta_temp), min=cfg.theta_floor)
    sr,mr=_ref(b,'ssm',cfg),_ref(b,'mlp',cfg)          # SIGNAL 1
    si,mi=_imp(b,'ssm',cfg),_imp(b,'mlp',cfg)          # usage importance (+ SIGNAL 2)
    return (g(b.ssm_drive,sr,b.unit_theta0,si),
            g(b.mlp_drive,mr,b.mlp_theta0,mi))

def _mixer_snaps(b, gs, gc):
    """Write gates for the mixer-upgrade parameters: query like the keys (unit x channel), per-head beta projection by the mean
    unit gate of its head, convolutions by their unit's gate. (All joined to the shunt and block write strengths via snaps.)"""
    out=[]
    if getattr(b,'Wq',None) is not None: out.append((b.Wq.weight,gs[:,None]*gc[None,:],'full'))
    if getattr(b,'Wb',None) is not None:
        gh=gs.reshape(b.mh,-1).mean(1); out+= [(b.Wb.weight,gh[:,None]*gc[None,:],'full'),(b.Wb.bias,gh,'full')]
    if getattr(b,'Wa',None) is not None:
        gh=gs.reshape(b.mh,-1).mean(1); out+= [(b.Wa.weight,gh[:,None]*gc[None,:],'full'),(b.Wa.bias,gh,'full')]
    for nm in ('conv_k','conv_v','conv_q','conv_loc'):
        w=getattr(b,nm,None)
        if w is not None: out.append((w,gs[:,None,None],'full'))
    return out

@torch.no_grad()
def snapshot_threshold(model, cfg, meta_gates=None):
    """UPDATE-SPACE write gating. Adam normalizes gradient scale away, so scaling grads
    does ~nothing (verified: a 0.05 grad gate reduced Adam's update by 0.0%). Instead:
    snapshot gated weights BEFORE opt.step, then after the step move each unit only
    gate*(new-old). Exact write strength, independent of the optimizer."""
    snaps=[]; elig=[]
    GC=model.__dict__.setdefault('_gcache',{}); gs_step=getattr(model,'_gstep',0)
    fresh=cfg.gate_every<=1 or GC.get('step') is None or gs_step-GC['step']>=cfg.gate_every
    if fresh: GC.clear(); GC['step']=gs_step                              # A4: gates recomputed every gate_every steps
    for i,b in enumerate(model.blocks):
        if fresh or ('u',i) not in GC or GC[('u',i)][0].shape[0]!=b.d_state:
            gs,gm=threshold_gates(b,cfg)
            GC[('u',i)]=(gs,gm,(gs>0.5).float().mean())
        gs,gm,el_=GC[('u',i)]; elig.append(el_)                        # tensor: converted only when printed
        gs=torch.ones_like(gs); gm=torch.ones_like(gm)
        if cfg.route_writes:                  # WRITE ROUTING: content writes to units matching its distinctness
            rs=b.write_route(b.unit_aff); rm=b.write_route(b.mlp_aff)
            gs=gs*rs; gm=gm*rm
            if i==0: model._route_mean=float(rs.mean())
        if meta_gates is not None:            # combined write strength = threshold x meta
            gs=gs*meta_gates[i][0]; gm=gm*meta_gates[i][1]
        # ALL LAYERS: each weight's write strength = (its unit's gate) x (its channel's gate)
        if fresh or ('c',i) not in GC: GC[('c',i)]=_any_gate(b.ch_drive,_ref(b,'ch',cfg),b.ch_theta0,_imp(b,'ch',cfg),cfg,_grp(cfg,'theta_floor','ch'))
        gc=GC[('c',i)]
        gc=torch.ones_like(gc)
        if i==0: model._ch_elig=(gc>0.5).float().mean()
        full=[(b.Wv.weight,gs[:,None]*gc[None,:]),(b.Wk.weight,gs[:,None]*gc[None,:]),
              (b.gate.weight,gs[:,None]*gc[None,:]),(b.gate.bias,gs),
              (b.Wout.weight,gc[:,None]*gs[None,:]),
              (b.mlp_in.weight,gm[:,None]*gc[None,:]),(b.mlp_in.bias,gm),
              (b.mlp_out.weight,gc[:,None]*gm[None,:]),(b.mlp_out.bias,gc),
              (b.ln.weight,gc),(b.ln.bias,gc),(b.norm.weight,gc),(b.norm.bias,gc),
              (b.beta,gc.mean()),(b.unit_affinity,gs),(b.mlp_affinity,gm)]
        full+=[(p,gt) for p,gt,_ in _mixer_snaps(b,gs,gc)]
        for p,gt in full:
            if p is None: continue
            snaps.append((p,p.data.clone(),gt,'full'))
    cm=getattr(model,'_og_common',None)
    if cm and id(model.pos.weight) in cm and not (cfg.channel_thresholds and cfg.emb_thresholds):
        snaps.append((model.pos.weight, model.pos.weight.data.clone(), torch.ones(1,device=model.pos.weight.device), 'full'))
    return snaps, torch.stack([e.float() for e in elig]).mean()

@torch.no_grad()
def apply_update_gate(snaps, og=None, og_in=1.0, og_orth='gate', stats=None, orth_boost=1.0, common=None, common_scale=1.0, og_out=None, shunt=None):
    """og: {id(weight): U} old-domain input subspaces (ORTHOGONAL-GATED UPDATE). Each matrix update d
    is split into din = d U U^T (inside the old subspace: moves old-domain outputs) and dorth = d - din.
      din   is written at gate * og_in        (og_in=0: interfering part removed; 1: gated as usual)
      dorth is written at gate (og_orth='gate') or at a fixed strength og_orth (e.g. 1.0 = free).
    Free orthogonal writes are only safe when U holds ~all old energy (og_energy ~0.99): with a 50%
    subspace the complement still carries half the old-domain input energy."""
    for p,old,gt,kind in snaps:
        g = gt.unsqueeze(1) if kind=='row' else (gt.unsqueeze(0) if kind=='col' else gt)   # 'vec'/'full': as given
        if common and id(p) in common: g=g*common_scale          # common-mode params (biases, LayerNorm, beta)
        U = og.get(id(p)) if og else None
        if U is None or U.shape[0]!=p.shape[-1]:
            p.data.copy_(old + g*(p.data-old)); continue
        d=p.data-old; din=(d@U)@U.T; dorth=d-din
        so=(g if orth_boost==1.0 else torch.clamp(g*orth_boost,max=1.0)) if og_orth=='gate' else og_orth
        if stats is not None:
            stats[0]+=(din*din).sum(); stats[1]+=((g*og_in*din)**2).sum(); stats[2]+=(d*d).sum()   # tensors: no GPU sync
        if shunt is not None:                        # SHUNT: interfering part written, but tracked separately
            sh=g*din; prev=shunt.get(id(p))
            shunt[id(p)]=sh if (prev is None or prev.shape!=sh.shape) else prev+sh
            upd=sh + so*dorth
        else: upd=(g*og_in)*din + so*dorth
        Uo=og_out.get(id(p)) if og_out else None
        if Uo is not None and Uo.shape[0]==p.shape[0]: upd=upd-Uo@(Uo.T@upd)     # two-sided: no change along old dominant OUTPUT directions
        p.data.copy_(old + upd)

# ---------------- TOPIC IDENTIFICATION FROM BYTE CONTEXT (diagnostic: no effect on training) ----------------
def ctx_profile(x, H, W):
    """(B,T) bytes -> (B,T,H): hashed single-byte, byte-pair and byte-triple counts over the last W bytes (causal:
    token t uses bytes <= t), square-rooted and L2-normalised."""
    B,T=x.shape; u=x.long(); p1=torch.cat([torch.zeros_like(u[:,:1]),u[:,:-1]],1); p2=torch.cat([torch.zeros_like(u[:,:2]),u[:,:-2]],1)
    oh=torch.zeros(B,T,H,device=x.device); one=torch.ones(B,T,1,device=x.device)
    for h in ((u*2654435761)%H, ((p1*256+u)*40503+17)%H, ((p2*65536+p1*256+u)*97531+7)%H):
        oh.scatter_add_(2,h.unsqueeze(-1),one)
    cs=oh.cumsum(1); win=cs.clone()
    if T>W: win[:,W:]-=cs[:,:-W]
    f=win.sqrt(); return f/f.norm(dim=-1,keepdim=True).clamp(min=1e-6)

_RECON_LAYERS=('Wk','Wv','gate','Wout','mlp_in','mlp_out')

@torch.no_grad()
def _recon2_once(model, cfg, d, xd):
    """One v2 reconsolidation step on old domain d's dreams xd. Returns per-layer (weighted error ratio) and the share of
    dream tokens corrected."""
    mods=[(li,nm,getattr(model.blocks[li],nm)) for li in cfg.recon_blocks if li<len(model.blocks) for nm in cfg.recon_layers]
    cc=getattr(model,'_cur_cov',None) or {}; uc=model.__dict__.setdefault('_recon_unew',{})
    for li,nm,m in mods:
        if id(m.weight) not in uc:
            c_=cc.get((li,_PG_KEY[nm]))
            uc[id(m.weight)]=None if (c_ is None or c_[1]==0) else _subspaces(c_[0]/c_[1],[cfg.recon_new_energy])[cfg.recon_new_energy][0]
    if not mods: return None
    rec={}
    def mk(tag,li,nm):
        def hk(m,inp,out): rec[(tag,li,nm)]=(inp[0].reshape(-1,inp[0].shape[-1]).float(),out.reshape(-1,out.shape[-1]).float())
        return _f32(hk)
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    P=_pmap(model); S=getattr(model,'_shunt',None) or {}; Sd={k:v for k,v in S.items() if k in P and P[k].shape==v.shape}
    for k,v in Sd.items(): P[k].sub_(v)                                        # main weights
    hooks=[m.register_forward_hook(mk('main',li,nm)) for li,nm,m in mods]; lm=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for h in hooks: h.remove()
    for k,v in Sd.items(): P[k].add_(v)                                        # live weights
    hooks=[m.register_forward_hook(mk('live',li,nm)) for li,nm,m in mods]; ll=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for h in hooks: h.remove()
    model.feat_scale.copy_(fs); model.train(was)
    B,T=xd.shape; w=torch.ones(B,T,device=xd.device)
    kl=(lm.exp()*(lm-ll)).sum(-1)                                              # per-token KL main -> live
    w=w/(1.0+kl/cfg.recon_cap)
    tgt=xd[:,1:]
    dn=-(ll[:,:-1].gather(-1,tgt[...,None])[...,0]-lm[:,:-1].gather(-1,tgt[...,None])[...,0])   # nll_live - nll_main
    worse=torch.zeros(B,T,device=xd.device); worse[:,:-1]=(dn>0).float(); w=w*worse
    wf=w.reshape(-1); W_=float(wf.sum())
    if W_<1e-6: return {'share':0.0,'err':{}}
    out={}
    for li,nm,m in mods:
        _,v_old=rec[('main',li,nm)]; k_,y=rec[('live',li,nm)]; E=(v_old-y)*wf[:,None]
        U=model._recon_unew.get(id(m.weight))
        if U is not None: U=U.to(k_.dtype); k_=k_-(k_@U)@U.T
        nrm=float(((k_*k_).sum(1)*wf).sum()/W_)+1e-6
        dW=cfg.recon_lr*(E.T@k_)/(W_*nrm); m.weight.add_(dW.to(m.weight.dtype))
        if id(m.weight) in S: S[id(m.weight)].add_(dW.to(m.weight.dtype))
        out[(li,nm)]=(float(((v_old-y)**2*wf[:,None]).sum()/(W_*v_old.shape[1])),float((v_old*v_old).mean()))
    return {'share':float((wf>0).float().mean()),'err':out}

def _recon2_stat(model, r):
    if not r: return
    st=model.__dict__.setdefault('_recon_stat',{})
    for (li,nm),(e,v) in r['err'].items():
        a=st.setdefault(li,[0.0,0.0,0]); a[0]+=e; a[1]+=v; a[2]+=1
    sh=model.__dict__.setdefault('_recon_share',[0.0,0]); sh[0]+=r['share']; sh[1]+=1

@torch.no_grad()
def _dream_kl_now(model, cfg, d, xd):
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    P=_pmap(model); S=getattr(model,'_shunt',None) or {}; Sd={k:v for k,v in S.items() if k in P and P[k].shape==v.shape}
    for k,v in Sd.items(): P[k].sub_(v)
    lm=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for k,v in Sd.items(): P[k].add_(v)
    ll=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    model.feat_scale.copy_(fs); model.train(was)
    return float((lm.exp()*(lm-ll)).sum(-1).mean())

@torch.no_grad()
def recon_sleep(model, cfg, x, y, dom_id):
    """v2 SLEEP phase at domain end, before the shunt merge: recon_sleep correction steps over the old domains' dreams
    (cycled, random batches from each pool), nothing else learned. Reports dream KL (old heads, live vs main weights) and the
    current domain's loss on this batch before and after."""
    pools=getattr(model,'_dreams',None)
    if not pools: return
    keys=[d for d in sorted(pools) if pools[d]]
    if not keys: return
    rng=np.random.RandomState(cfg.seed*1000+dom_id+77)
    model.__dict__.pop('_recon_unew',None)
    probe={d:torch.stack(pools[d][:cfg.dream_batch]) for d in keys}
    def cur():
        was=model.training; model.eval(); fs=model.feat_scale.clone()
        v=float(F.cross_entropy(model.heads[dom_id](model.features(x)).reshape(-1,256).float(),y.reshape(-1)))/math.log(2)
        model.feat_scale.copy_(fs); model.train(was); return v
    kb={d:_dream_kl_now(model,cfg,d,probe[d]) for d in keys}; cb=cur()
    for i in range(cfg.recon_sleep):
        d=keys[i%len(keys)]; pl=pools[d]
        xd=torch.stack([pl[j] for j in rng.randint(len(pl),size=cfg.dream_batch)])
        _recon2_stat(model,_recon2_once(model,cfg,d,xd))
    ka={d:_dream_kl_now(model,cfg,d,probe[d]) for d in keys}; ca=cur()
    model.__dict__.setdefault('_sleep_log',{})[dom_id]={'kl_before':kb,'kl_after':ka,'cur_before':cb,'cur_after':ca}
    print(f"  [sleep] {cfg.recon_sleep} reconsolidation steps before the merge | old heads' dream KL to main weights before -> after: "
          +" ".join(f"{_lab(cfg,d)} {kb[d]:.4f}->{ka[d]:.4f}" for d in keys)
          +f" | current-domain BPC on this batch {cb:.3f} -> {ca:.3f} ({ca-cb:+.3f})")

@torch.no_grad()
def stale_diag(model, cfg, t):
    """Do the protected subspaces match where old domains' inputs REALLY are now? Overlap (fraction of the real subspace's
    energy directions captured) of the stored and of the dream-refreshed covariances with real validation text."""
    vals=getattr(model,'_vals',None)
    if vals is None: return
    bt=[]
    for d in range(t):
        g=vals[d]
        if hasattr(g,'reset'): g.reset()
        x,y=g(); bt.append((x,y,d))
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    real=_input_cov(model,bt); model.feat_scale.copy_(fs); model.train(was)
    old=getattr(model,'_old_cov',None) or {}; stored={k:v[0]/v[1] for k,v in old.items()}; dream=_dream_cov(model,cfg)
    e=cfg.og_energy; res={'stored':{},'dream':{}}
    for k,C in real.items():
        Ur,kr=_subspaces(C,[e])[e]
        for tag,src in (('stored',stored),('dream',dream)):
            if k in src:
                Ue,_=_subspaces(src[k],[e])[e]; ov=float((Ur.T@Ue).pow(2).sum())/max(kr,1)
                res[tag].setdefault(k[0],[]).append(ov)
    nb_=len(model.blocks)
    row={tag:[float(np.mean(res[tag].get(i,[float('nan')]))) for i in range(nb_)] for tag in res}
    model.__dict__.setdefault('_stale_log',{})[t]=row
    print('    [staleness] protected old-input subspaces vs the REAL ones (old domains\' validation text through the current '
          'network), overlap per block: stored '+' '.join(f'b{i} {v:.2f}' for i,v in enumerate(row['stored']))
          +' | dream-refreshed '+' '.join(f'b{i} {v:.2f}' for i,v in enumerate(row['dream'])))

# ---------------- FORGETTING DIAGNOSTICS: which part of the model lost an earlier domain ----------------
_ROUTE_KEYS=('mu','used','n','gm','W')
def _fd_group(name):
    if name.startswith('heads.') or name.startswith('theads.'): return 'heads'
    if name.startswith('blocks.'): return 'block'+name.split('.')[1]
    return 'emb'                                       # embeddings, positions, model-level buffers

@torch.no_grad()
def _fd_swap(model, snap, groups, routing):
    """Put the named groups (and optionally the routing) back to the snapshot; returns what is needed to undo it."""
    cur={}; named=dict(list(model.named_parameters())+list(model.named_buffers()))
    existed=None                                       # topic heads that did not exist yet at that time keep their current state
    for n,t in named.items():
        if existed is not None and n.startswith('theads.') and int(n.split('.')[1]) not in existed: continue
        if _fd_group(n.replace('module.','')) in groups or ('backbone' in groups and _fd_group(n)!='heads'):
            o=snap['p'].get(n)
            if o is not None and o.dim()==t.dim() and o.shape!=t.shape and all(a<=b_ for a,b_ in zip(o.shape,t.shape)):
                z=torch.zeros_like(t); z[tuple(slice(0,a) for a in o.shape)]=o.to(t.device,t.dtype); o=z   # units added later: absent then
            if o is not None and o.shape==t.shape: cur[n]=t.detach().clone(); t.copy_(o.to(t.device))
            elif o is None and n.startswith('blocks.') and n.split('.')[-2] in ('Wout','mlp_out'):   # block added later: absent then
                cur[n]=t.detach().clone(); t.zero_()
    rs=None
    if routing:
        tr=getattr(model,'_topics',None)
        if tr is not None and 'tr' in snap:
            rs=('tr',{k:(getattr(tr,k).clone() if torch.is_tensor(getattr(tr,k)) else getattr(tr,k)) for k in _ROUTE_KEYS})
            for k,v in snap['tr'].items(): setattr(tr,k,v.clone() if torch.is_tensor(v) else v)
        nb=getattr(model,'_nb',None)
        if nb is not None and 'nb' in snap:
            rs=('nb',{'C':nb['C'].clone(),'used':nb['used'].clone()}); nb['C']=snap['nb']['C'].clone(); nb['used']=snap['nb']['used'].clone(); nb['lp']=None
            R=getattr(model,'_routers',None)
            if R is not None and 'routers' in snap:
                rs[1]['routers']={k:v.clone() for k,v in R.state_dict().items()}; R.load_state_dict(snap['routers']); object.__setattr__(model,'_rver',model._rver+1)
    return cur,rs

@torch.no_grad()
def _fd_undo(model, cur, rs):
    named=dict(list(model.named_parameters())+list(model.named_buffers()))
    for n,t in cur.items(): named[n].copy_(t)
    if rs is not None:
        if rs[0]=='tr':
            for k,v in rs[1].items(): setattr(model._topics,k,v)
        else:
            model._nb['C']=rs[1]['C']; model._nb['used']=rs[1]['used']; model._nb['lp']=None
            if 'routers' in rs[1]: model._routers.load_state_dict(rs[1]['routers']); object.__setattr__(model,'_rver',model._rver+1)

def _cka(X, Y):
    """Linear CKA between two (N,d) representations of the same tokens (1 = identical up to rotation / isotropic scale)."""
    X=X-X.mean(0); Y=Y-Y.mean(0)
    return float((Y.T@X).pow(2).sum()/((X.T@X).norm()*(Y.T@Y).norm()).clamp(min=1e-12))

def forget_diag(model, cfg, vals, t, snaps, post, doms=None):
    """For every earlier domain d: BPC now, and the BPC change when one part is restored to its state at the end of d.
    Negative 'undo X' = restoring X recovers that much = X's later changes caused that much forgetting."""
    n=cfg.fd_batches; out={}; nb_=len(model.blocks)
    for d in (range(t) if doms is None else doms):
        sd=snaps[d]; now=bpc(model,vals[d],d,n=n); row={'now':now,'drift':now-sd.get('post',post[d])}   # same batches both times
        tests=[('backbone',{'backbone'},False),('heads',{'heads'},False)]
        tests+=[('emb',{'emb'},False)]+[(f'b{i}',{f'block{i}'},False) for i in range(nb_)]
        for lab,g,rt in tests:
            cur,rs=_fd_swap(model,sd,g,rt); row[lab]=bpc(model,vals[d],d,n=n)-now; _fd_undo(model,cur,rs)
        out[d]=row
    print(f"  [forgetting diagnostics] BPC change when one part is put back to its state at the end of that domain (n={n} batches; negative = that part's later change caused forgetting):")
    for d,r in out.items():
        line=(f"     {PILE_DOMAINS[d][:8]:8} drift {r['drift']:+.3f} | undo backbone {r['backbone']:+.3f} heads {r['heads']:+.3f}"
              +(f" routing {r['routing']:+.3f}" if 'routing' in r else '')
              +" | per part: emb "+f"{r['emb']:+.3f} "+' '.join(f"b{i} {r[f'b{i}']:+.3f}" for i in range(nb_)))
        xk_=[k for k in r if isinstance(k,str) and k.startswith('X ')]
        if xk_: line+=" | WITH PRIVATE ROWS: "+" ".join(f"{k[2:]} {r[k]:+.3f}" for k in xk_)
        if 'overlap' in r:
            line+=f" | routing overlap {r['overlap']:.2f} | its topics (mass then->now, head change, protection): "+' '.join(f"t{k} {a:.2f}->{b:.2f} {rel:.0%} {pi:.1f}" for k,a,b,rel,pi in r['topics'])
        print(line)
    if any('fdrift' in r for r in out.values()):
        print("  [feature drift since each domain was learned] its real text, each block's output now vs at the end of that domain: relative squared change / linear CKA (1 = same up to rotation and scale):")
        for d,r in out.items():
            if 'fdrift' in r: print(f"     {PILE_DOMAINS[d][:8]:8} "+" ".join(f"b{i} {a:.3f}/{c:.3f}" for i,(a,c) in enumerate(r['fdrift']))
                                    +(" | shared/specific rel. change (shared share): "+" ".join(f"b{i} {a:.3f}/{b_:.3f} ({c:.0%})" for i,(a,b_,c) in enumerate(r['fsplit'])) if 'fsplit' in r else ""))
    return out

class TopicTracker:
    """Online topic identification from byte context (up to topic_k topics).
    - profiles are CENTRED on a running global mean profile (topics form around what is distinctive);
    - topics are updated WINNER-TAKE-ALL, at rate topic_lr/(1+use/topic_mature) (young topics adapt, mature ones settle);
    - a new topic opens on a NOVELTY JUMP: the short-term share of unmatched contexts (best cosine < topic_new) rises
      topic_jump above its long-term level (at most once per topic_gap steps)."""
    def __init__(s, cfg, dev):
        s.cfg=cfg; s.K,s.H,s.W=cfg.topic_k,cfg.topic_h,cfg.topic_window
        s.mu=torch.zeros(s.K,s.H,device=dev); s.used=torch.zeros(s.K,dtype=torch.bool,device=dev); s.n=torch.zeros(s.K,device=dev)
        s.gm=None; s.u=0.0; s.us=None; s.since=0; s.own={}; s.T={}; s.events=[]
    def feats(s, x, stride=None):
        f=ctx_profile(x,s.H,s.W)[:,s.W-1:]
        if stride: f=f[:,::stride]
        f=f.reshape(-1,s.H)
        if s.gm is None: s.gm=f.mean(0)
        g=f-s.gm; return g/g.norm(dim=-1,keepdim=True).clamp(min=1e-6), f
    def cen(s):
        return s.mu
        c=s.mu-s.gm; return c/c.norm(dim=-1,keepdim=True).clamp(min=1e-6)   # raw centroids centred with the CURRENT mean
    def sims(s, g): return (g@s.cen().T).masked_fill(~s.used,-1e9)
    def choose(s, sm):
        """'argmax': nearest topic. 'art' (adaptive-resonance style): among topics whose similarity passes the vigilance
        level, the OLDEST wins, so a newer topic only takes contexts that no older topic already resonates with."""
        best,arg=sm.max(-1)
        return best,arg
    def post(s, g):
        sm=s.sims(g)
        p=F.softmax(sm/s.cfg.topic_temp,-1)
        return p
    @torch.no_grad()
    def update(s, x, dom=None):
        if s.W>x.shape[1]: s.W=x.shape[1]    # window can't exceed the sequence (local tests use short sequences)
        g,f=s.feats(x,stride=max(1,s.W//4)); s.nup=getattr(s,'nup',0)+1
        s.gm=0.999*s.gm+0.001*f.mean(0)
        raw=s.cfg.topic_center=='raw'
        if not s.used.any():
            m=f.mean(0) if raw else g.mean(0); s.mu[0]=m if raw else m/m.norm().clamp(min=1e-6); s.used[0]=True; return
        best,arg=s.choose(s.sims(g)); un=best<s.cfg.topic_new; fr=float(un.float().mean())
        s.u=0.9*s.u+0.1*fr; s.us=fr if s.us is None else 0.995*s.us+0.005*fr; s.since+=1
        if s.u-s.us>s.cfg.topic_jump and s.since>s.cfg.topic_gap and (~s.used).any() and un.any():
            j=int(torch.nonzero(~s.used)[0]); parent=int(s.sims(g[un]).mean(0).argmax())   # nearest existing topic
            s.events.append((j,parent))
            if raw: s.mu[j]=f[un].mean(0)
            else: m=g[un].mean(0); s.mu[j]=m/m.norm().clamp(min=1e-6)
            s.used[j]=True
            s.u=s.us; s.since=0; best,arg=s.choose(s.sims(g))
        if dom is not None:                   # which topics this domain's TRAINING contexts went to (for head-coverage)
            s.T.setdefault(dom,torch.zeros(s.K,device=s.mu.device)).add_(torch.bincount(arg,minlength=s.K).float())
        for k in arg.unique():
            kk=int(k); sel=(f if raw else g)[arg==kk]; eta=s.cfg.topic_lr/(1+float(s.n[kk])/s.cfg.topic_mature)
            m=s.mu[kk]+eta*(sel.mean(0)-s.mu[kk]); s.mu[kk]=m if raw else m/m.norm().clamp(min=1e-6); s.n[kk]+=sel.shape[0]
    @torch.no_grad()
    def force_spawn(s, x):
        """SURPRISE-driven birth: a new topic centred on the current batch's contexts (parent = nearest topic)."""
        if not bool((~s.used).any()) or s.gm is None: return None
        g,f=s.feats(x,stride=max(1,s.W//4)); j=int(torch.nonzero(~s.used)[0]); parent=int(s.sims(g).mean(0).argmax())
        m=g.mean(0); s.mu[j]=m/m.norm().clamp(min=1e-6)
        s.used[j]=True; s.events.append((j,parent)); s.since=0; s.u=s.us if s.us is not None else 0.0
        return j
    @torch.no_grad()
    def token_post(s, x):
        """Causal topic posterior for every position (B,T,K): position t uses bytes <= t."""
        B,T=x.shape; f=ctx_profile(x,s.H,min(s.W,T)); g=f-s.gm
        g=g/g.norm(dim=-1,keepdim=True).clamp(min=1e-6)
        return s.post(g.reshape(-1,s.H)).reshape(B,T,s.K)
    @torch.no_grad()
    def assign(s, get, n=6):
        if hasattr(get,'reset'): get.reset()
        A=[];C=[]
        for _ in range(n):
            x,_=get()
            if x.shape[1]<s.W: continue
            p=s.post(s.feats(x)[0]); A.append(p.argmax(-1)); C.append(p.max(-1).values)
        if not A: z=torch.zeros(0,device=s.mu.device); return z.long(), z
        return torch.cat(A), torch.cat(C)
    def report(s, vals, t):
        """Topic/domain contingency over trained domains, purity, NMI, confidence, stability of each domain's topics."""
        K=s.K; M=np.zeros((t+1,K)); conf=[]; stab={}; cover={}
        for d in range(t+1):
            a,c=s.assign(vals[d]); M[d]=np.bincount(a.cpu().numpy(),minlength=K); conf.append(float(c.mean()) if c.numel() else float('nan'))
            if d==t: s.own[d]=a.clone()
            elif d in s.own: stab[d]=float((a==s.own[d]).float().mean())
            if d in s.T and a.numel():         # HEAD-COVERAGE: share of this domain's contexts now routed to a topic that
                sh=s.T[d]/s.T[d].sum().clamp(min=1)      # received >= topic_cover of its training contexts (a head that saw it)
                cover[d]=float((sh[a]>=s.cfg.topic_cover).float().mean())
        tot=max(M.sum(),1); purity=float(M.max(0).sum()/tot)
        pj=M/tot; pd_=pj.sum(1,keepdims=True); pk=pj.sum(0,keepdims=True); nz=pj>0
        mi=float((pj[nz]*np.log(pj[nz]/(pd_@pk)[nz])).sum()); hd=-float((pd_[pd_>0]*np.log(pd_[pd_>0])).sum()); hk=-float((pk[pk>0]*np.log(pk[pk>0])).sum())
        return dict(used=int(s.used.sum()),purity=purity,nmi=2*mi/max(hd+hk,1e-9),conf=conf,stab=stab,cover=cover,M=M.tolist())

# ---- DATA-FREE running input covariances (orthogonal gating without stored samples)
def install_cov_hooks(model, cfg):
    """While training, every cov_every steps, a token subsample of each projectable layer's input is added to a running
    covariance for the CURRENT domain. At domain end it is folded (equal weight per domain) into the old-domain covariance."""
    object.__setattr__(model,'_cur_cov',{}); object.__setattr__(model,'_old_cov',{}); object.__setattr__(model,'_cov_on',False)
    object.__setattr__(model,'_cov_handles',{})
    for li,b in enumerate(model.blocks): cov_hook_block(model, cfg, li)

def cov_hook_block(model, cfg, li):
    """(Re-)register the running-covariance hooks of block li (needed after a widening replaces its layers)."""
    b=model.blocks[li]; H=model._cov_handles
    for key in ORTH_KEYS:
        if (li,key) in H: H[(li,key)].remove()
        def hk(m,inp,out,k=(li,key)):
            if not (model._cov_on and m.training): return
            with torch.no_grad():
                x_=inp[0][:model.__dict__.get('_nstat')].reshape(-1,inp[0].shape[-1]).float(); x_=x_[::max(1,x_.shape[0]//cfg.cov_tokens)]
                c,n=model._cur_cov.get(k,(0.0,0)); model._cur_cov[k]=(c+x_.T@x_, n+x_.shape[0])
        H[(li,key)]=getattr(b,key).register_forward_hook(_f32(hk))

@torch.no_grad()
def fold_domain_covariance(model, dom=None):
    last={}
    for k,(c,n) in model._cur_cov.items():
        C=c/max(n,1); oc,od=model._old_cov.get(k,(0.0,0))
        model._old_cov[k]=(oc+C, od+1)              # stored as (sum of per-domain covariances, number of domains)
        last[k]=C
    lastm={}
    for k,(ms,mn) in (model.__dict__.get('_cur_mean') or {}).items():   # centring: per-domain means, pooled like the covariances
        m_=ms/max(mn,1); om,odn=model.__dict__.setdefault('_old_mean',{}).get(k,(0.0,0)); model._old_mean[k]=(om+m_, odn+1); lastm[k]=m_
    if dom is not None:                             # REPEATED PASSES: each domain's latest covariance kept (CPU) so that a revisit
        model.__dict__.setdefault('_dom_cov',{})[dom]={k:C.detach().cpu() for k,C in last.items()}   # protects only the OTHERS
        if lastm: model.__dict__.setdefault('_dom_mean',{})[dom]={k:m_.detach().cpu() for k,m_ in lastm.items()}
    object.__setattr__(model,'_last_cov',last)      # the newest old domain's own covariance (light protection option)
    model._cur_cov={}; model.__dict__['_cur_mean']={}

@torch.no_grad()
def rebuild_old_cov(model, olds):
    """Old-input statistics = the latest covariance of every OTHER domain seen so far (equal weight per domain)."""
    DC=getattr(model,'_dom_cov',None) or {}
    if not DC: return
    dev=model.embed.weight.device; acc={}
    for d in olds:
        for k,C in DC.get(d,{}).items():
            c,n=acc.get(k,(0.0,0)); acc[k]=(c+C.to(dev), n+1)
    model._old_cov=acc
    DM=getattr(model,'_dom_mean',None) or {}
    if DM:
        am={}
        for d in olds:
            for k,m_ in DM.get(d,{}).items():
                c,n=am.get(k,(0.0,0)); am[k]=(c+m_.to(dev), n+1)
        model.__dict__['_old_mean']=am

def _imp_T(v, cfg):
    """Importance transform of a per-neuron score (normalised to mean 1)."""
    n=v/(v.mean()+1e-8)
    t=cfg.imp_transform
    if t=='relu': return torch.relu(n-1.0)            # D: only above-average neurons gain importance
    if t=='raw':  return n                            # every neuron gains, proportional to its score
    if t=='sqrt': return torch.sqrt(torch.relu(n-1.0))
    if t=='log':  return torch.log1p(torch.relu(n-1.0))
    if t=='rank':                                     # top imp_top fraction get +1, rest 0
        k=max(1,int(round(cfg.imp_top*v.numel()))); out=torch.zeros_like(v); out[torch.topk(v,k).indices]=1.0; return out
    raise ValueError(t)

@torch.no_grad()
def importance_increments(model, cfg):
    """This domain's importance increment per neuron group (no mutation)."""
    inc=[]
    for b in model.blocks:
        ssm_flow, mlp_flow = flow_importance(b)
        su=(b.unit_usage/(b.unit_usage.mean()+1e-8))*ssm_flow; mu=(b.mlp_usage/(b.mlp_usage.mean()+1e-8))*mlp_flow
        if su is None: inc.append((torch.zeros_like(b.unit_imp),torch.zeros_like(b.mlp_imp),torch.zeros_like(b.ch_imp)))
        else: inc.append((_imp_T(su,cfg),_imp_T(mu,cfg),_imp_T(b.ch_usage,cfg)))
    emb=_imp_T(model.emb_usage,cfg)
    return inc, emb

@torch.no_grad()
def update_importance(model, cfg):
    """After a domain: decay old importance (imp_decay) and add this domain's increment, raising
    those neurons' write thresholds for all later domains."""
    inc,emb=importance_increments(model,cfg)
    for b,(u,m,c) in zip(model.blocks,inc):
        b.unit_imp.mul_(cfg.imp_decay).add_(u); b.mlp_imp.mul_(cfg.imp_decay).add_(m)
        b.ch_imp.mul_(cfg.imp_decay).add_(c); b.ch_usage.zero_()
    model.emb_imp.mul_(cfg.imp_decay).add_(emb); model.emb_usage.zero_()

_REV_BUFS=('unit_imp','mlp_imp','ch_imp','unit_sens','mlp_sens','ch_sens')
def _rev_items(model):
    return [(b,n) for b in model.blocks for n in _REV_BUFS]+[(model,'emb_imp'),(model,'emb_sens')]

@torch.no_grad()
def revisit_begin(model, cfg, t):
    """Visit start: take domain t's own stored importance / sensitivity out of the totals; remember the totals without it."""
    D=model.__dict__.setdefault('_imp_dom',{}); own=D.get(t)
    Lf=model.__dict__.setdefault('_rev_left',{}); vi_=getattr(model,'_visit',0); tau=cfg.imp_recency_tau; c0=cfg.imp_recency_floor
    def w_(d):
        if tau<=0: return 1.0
        age=max(0,vi_-Lf.get(d,vi_-1)-1); return c0+(1.0-c0)*math.exp(-age/tau)
    oth=[(w_(d),c) for d,c in D.items() if d!=t]
    for j,(o,n) in enumerate(_rev_items(model)):
        cur=getattr(o,n); m=torch.zeros_like(cur)
        for w,c in oth: m=torch.maximum(m,w*c[j])
        setattr(o,n,m)
    if tau>0 and oth: print('  [recency] protection weight per old domain: '+' '.join(f'{PILE_DOMAINS[d][:6]} {w_(d):.2f}' for d in sorted(D) if d!=t))
    model.__dict__['_imp_base']=[getattr(o,n).clone() for o,n in _rev_items(model)]
    return own is not None

@torch.no_grad()
def revisit_end(model, cfg, t):
    """Visit end (after importance and sensitivity are updated): this visit's contribution becomes domain t's stored one."""
    D=model.__dict__.setdefault('_imp_dom',{}); base=model.__dict__.pop('_imp_base',None)
    if base is None: return
    new=[getattr(o,n)-b0 for (o,n),b0 in zip(_rev_items(model),base)]
    D[t]=[c.clamp_min(0) for c in new]; model.__dict__.setdefault('_rev_left',{})[t]=getattr(model,'_visit',0)
    for j,(o,n) in enumerate(_rev_items(model)):
        m=torch.zeros_like(getattr(o,n))
        for c in D.values(): m=torch.maximum(m,c[j])
        setattr(o,n,m)
    return
    D[t]=new

def _imp_bufs(model):
    return [(b,'unit_imp') for b in model.blocks]+[(b,'mlp_imp') for b in model.blocks]+\
           [(b,'ch_imp') for b in model.blocks]+[(model,'emb_imp')]

@torch.no_grad()
def online_importance(model, cfg, base, frac):
    """ONLINE importance: during a domain, imp = importance before the domain + frac * the domain's
    increment so far (the current domain's key neurons start protecting themselves mid-domain)."""
    inc,emb=importance_increments(model,cfg)
    flat=[x[0] for x in inc]+[x[1] for x in inc]+[x[2] for x in inc]+[emb]
    for (o,nm),b0,d in zip(_imp_bufs(model),base,flat):
        pad=torch.cat([b0,torch.zeros(d.numel()-b0.numel(),device=d.device)]) if d.numel()>b0.numel() else b0
        setattr(o,nm,pad+frac*d)

def apply_meta_gate(model, meta, prog):
    """HARD importance-freeze (the +0.32 floor: frozen units get gate 0) THEN soft
    meta-gate modulates the non-frozen units. This guarantees the controller's floor
    IS importance-freezing (hard freeze always applied) - it can only improve by
    soft-protecting the long tail. (Fix: the previous version had no hard freeze and
    a sigmoid gate that couldn't reproduce it, so it couldn't even match importance.)"""
    for b in model.blocks:
        ssm_f, mlp_f = unit_features(b, prog)
        gs=meta(ssm_f); gm=meta(mlp_f)                       # soft gate in (0,1)
        gs=gs.clone(); gm=gm.clone()
        gs[b.frozen_units]=0.0; gm[b.mlp_frozen]=0.0         # HARD freeze on top (floor)
        if b.Wv.weight.grad is not None: b.Wv.weight.grad*=gs.unsqueeze(1)
        if b.Wk.weight.grad is not None: b.Wk.weight.grad*=gs.unsqueeze(1)
        if b.Wout.weight.grad is not None: b.Wout.weight.grad*=gs.unsqueeze(0)
        if b.mlp_in.weight.grad is not None: b.mlp_in.weight.grad*=gm.unsqueeze(1)
        if b.mlp_in.bias.grad is not None: b.mlp_in.bias.grad*=gm
        if b.mlp_out.weight.grad is not None: b.mlp_out.weight.grad*=gm.unsqueeze(0)

# ---------------- data ----------------
def _rank(v): return np.argsort(np.argsort(v)).astype(float)
def _spearman(a,b):
    ra,rb=_rank(a),_rank(b); ra-=ra.mean(); rb-=rb.mean()
    return float((ra*rb).sum()/(np.sqrt((ra*ra).sum()*(rb*rb).sum())+1e-12))

def locality_metrics(W0_in,W1_in,W0_out,W1_out,dom_usage,prior_imp,k):
    """Orientation = each MLP unit's input weight direction (row of mlp_in).
    (1) clustering: are this domain's top-10% units closer in orientation than random units?
    (2) damage locality: among units important for EARLIER domains, does orientation proximity
        to this domain's most active units predict how much they were rewritten?
    (3) neighbour effect: does neighbours' prior importance predict damage beyond own importance?
    Uses its own RNG so measurement never perturbs training."""
    rng=np.random.RandomState(0)
    O=W1_in/(np.linalg.norm(W1_in,axis=1,keepdims=True)+1e-8)
    C=O@O.T; n=len(O); np.fill_diagonal(C,np.nan)
    top=np.argsort(-dom_usage)[:max(4,n//10)]
    rnd=rng.choice(n,len(top),replace=False)
    clust=float(np.nanmean(C[np.ix_(top,top)])-np.nanmean(C[np.ix_(rnd,rnd)]))
    damage=np.linalg.norm(W1_in-W0_in,axis=1)+np.linalg.norm(W1_out-W0_out,axis=0)
    prox=np.nanmax(C[:,top],axis=1)                          # closeness to this domain's active units
    old=np.argsort(-prior_imp)[:max(4,n//4)]                  # units important for earlier domains
    loc=_spearman(prox[old],damage[old])
    Cn=np.nan_to_num(C,nan=-2.0); nn_idx=np.argsort(-Cn,axis=1)[:,:k]
    neigh=prior_imp[nn_idx].mean(1)
    z=lambda v:(v-v.mean())/(v.std()+1e-8)
    X1=np.stack([np.ones(n),z(prior_imp)],1); X2=np.stack([np.ones(n),z(prior_imp),z(neigh)],1); y=z(damage)
    b1=np.linalg.lstsq(X1,y,rcond=None)[0]; b2=np.linalg.lstsq(X2,y,rcond=None)[0]
    r2=lambda X,b:1-((y-X@b)**2).mean()/(y.var()+1e-12)
    own_neigh=float(np.corrcoef(prior_imp,neigh)[0,1]) if prior_imp.std()>0 else float('nan')
    # neigh_r2_gain is the reliable statistic; if own_neigh_corr is near 1 the test can't
    # separate a unit's own importance from its neighbours' (report it so this is visible)
    return {'cluster':clust,'damage_locality':loc,'neigh_r2_gain':float(r2(X2,b2)-r2(X1,b1)),'own_neigh_corr':own_neigh}

_BYTES_CACHE={}
_RUN_T0=time.time()                                                     # v9h: wall-clock origin for the visit headers
PILE_CACHE_DIR=os.environ.get('PILE_CACHE','pile_cache')      # streamed domain text is also cached on disk (re-runs, seeds)
def _cache_path(key):
    if _stream_bytes is not _hf_stream_bytes: return None       # local stand-in data: no disk cache
    d,seed,n=key; return os.path.join(PILE_CACHE_DIR,f"{''.join(c if c.isalnum() else '_' for c in d)}_{seed}_{n}.bin")
def _cache_put(key, data):
    fp=_cache_path(key)
    if fp:
        os.makedirs(PILE_CACHE_DIR,exist_ok=True)
        with open(fp,'wb') as f: f.write(data)
def stream_bytes(domain,seed,n):
    key=(domain,seed,n)
    if key not in _BYTES_CACHE:
        fp=_cache_path(key)
        if fp and os.path.exists(fp): _BYTES_CACHE[key]=open(fp,'rb').read()
        else: _BYTES_CACHE[key]=_stream_bytes(domain,seed,n); _cache_put(key,_BYTES_CACHE[key])
    return _BYTES_CACHE[key]
def prefetch_pile(reqs, seed=2026, max_docs=8_000_000):
    """ONE streaming pass collecting several (new) domains at once - a separate scan per domain and split would read the Pile many
    times over for the rarer subsets. reqs: [(domain, val_key, train_key)]; per domain, whole documents go to validation until it is
    full, then to training (document-level disjoint)."""
    need={}
    for d,vk,tk in reqs:
        if all(k in _BYTES_CACHE or (_cache_path(k) and os.path.exists(_cache_path(k))) for k in (vk,tk)): continue
        need[d]={'val':(vk,bytearray()),'train':(tk,bytearray())}
    if not need: return
    from datasets import load_dataset
    print(f"  [data] one streaming pass for {len(need)} new domains: "+", ".join(need))
    ds=load_dataset("monology/pile-uncopyrighted",split="train",streaming=True).shuffle(seed=seed,buffer_size=10000)
    seen_names={}; n_docs=0; t0=time.time()
    for ex in ds:
        n_docs+=1; nm=(ex.get("meta") or {}).get("pile_set_name"); seen_names[nm]=seen_names.get(nm,0)+1
        if nm in need and ex.get("text"):
            slot=need[nm]; tb=ex["text"].encode("utf-8","ignore")+b"\n"
            (vk,vb),(tk,tb_)=slot['val'],slot['train']
            if len(vb)<vk[2]: vb.extend(tb)
            elif len(tb_)<tk[2]: tb_.extend(tb)
            if len(vb)>=vk[2] and len(tb_)>=tk[2]:
                for k,b in ((vk,vb),(tk,tb_)): _BYTES_CACHE[k]=bytes(b[:k[2]]); _cache_put(k,_BYTES_CACHE[k])
                del need[nm]; print(f"    [data] {nm} complete after {n_docs} documents ({time.time()-t0:.0f}s)")
                if not need: return
        if n_docs==1_000_000:                                          # a requested name never seen: fail early with the names that exist
            miss_=[d for d in need if seen_names.get(d,0)==0]
            if miss_: raise RuntimeError(f"no documents named {miss_} in the first 1M; pile_set_names seen: {sorted(k for k in seen_names if k)}")
        if n_docs%500000==0: print(f"    [data] {n_docs} documents scanned ({time.time()-t0:.0f}s); still filling: "
                                   +", ".join(f"{d} {len(v['val'][1])+len(v['train'][1])}" for d,v in need.items()))
        if n_docs>=max_docs:
            raise RuntimeError(f"domains not filled after {n_docs} documents: {list(need)}; pile_set_names seen: {seen_names}")
def _stream_bytes(domain,seed,n):
    from datasets import load_dataset
    ds=load_dataset("monology/pile-uncopyrighted",split="train",streaming=True).shuffle(seed=seed,buffer_size=10000)
    buf=bytearray()
    for ex in ds:
        if (ex.get("meta") or {}).get("pile_set_name")==domain and ex.get("text"):
            buf.extend(ex["text"].encode("utf-8","ignore")+b"\n")
            if len(buf)>=n: break
    return bytes(buf[:n])
_hf_stream_bytes=_stream_bytes
def _facts(cfg):
    r_=__import__('random').Random(int(cfg.fact_seed)); C='bcdfghjklmnprstvz'; V='aeiou'
    ks=[]; 
    while len(ks)<int(cfg.fact_n):
        k_=(r_.choice(C).upper()+r_.choice(V)+r_.choice(C)+r_.choice(V)+r_.choice(C))
        if k_ not in ks: ks.append(k_)
    va=[r_.randint(1000,9999) for _ in ks]; vb=[]
    for a_ in va:
        b_=r_.randint(1000,9999)
        while b_==a_ or str(b_)[0]==str(a_)[0]: b_=r_.randint(1000,9999)        # B differs from A from the first digit
        vb.append(b_)
    return ks,va,vb
def _inject_facts(raw,cfg,d,p):
    """UPDATE-OR-PROTECT TEST: ' Name:1234;' sentences inserted every `every` bytes (random fact each time) into domain d's pass-p shard."""
    sp=[e for e in cfg.fact_spec if int(e[0])==int(d) and int(e[1])==int(p)]
    if not sp: return raw
    ks,va,vb=_facts(cfg); out=bytearray(); b_=bytes(raw); r_=__import__('random').Random(1000*int(d)+int(p)+int(cfg.fact_seed)); n_=0
    for _,_,ver,ev in sp:
        ev=int(ev); i0=0
        while i0<len(b_):
            out+=b_[i0:i0+ev]; i0+=ev; j_=r_.randrange(len(ks)); out+=f" {ks[j_]}:{(va if int(ver)==0 else vb)[j_]};".encode(); n_+=1
        b_=bytes(out); out=bytearray()
    print(f"  [facts] domain {d} pass {p}: {n_} fact sentences inserted ({len(ks)} facts, version {'/'.join(str(int(e[2])) for e in sp)}, every {'/'.join(str(int(e[3])) for e in sp)} bytes)")
    return b_
@torch.no_grad()
def fact_eval(model,cfg,vi,slots):
    """log2 p of each fact's original (A) and changed (B) value given ' Name:' through each fact domain's slot."""
    ks,va,vb=_facts(cfg); was=model.training; model.eval(); fs=model.feat_scale.clone(); dev=model.embed.weight.device; out=[]
    try:
        for nm_,s_ in slots:
            la=lb=0.0; nb=0
            for k_,a_,b_ in zip(ks,va,vb):
                lp_=[]
                for v_ in (a_,b_):
                    t_=f"; {k_}:{v_};".encode(); x_=torch.tensor(list(t_),device=dev)[None]
                    sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=int(s_)
                    try: lg_=F.log_softmax(model(x_[:,:-1],int(s_)).float(),-1)
                    finally: model.__dict__['_s0_dom']=sv_
                    P0=len(f"; {k_}:")-1; tg_=x_[0,P0+1:P0+5]
                    lp_.append(float(lg_[0,P0:P0+4].gather(-1,tg_[:,None]).sum())/LOG2)
                la+=lp_[0]; lb+=lp_[1]; nb+=int(lp_[1]>lp_[0])
            n_=len(ks); out.append(f"{nm_} slot {s_}: original {la/n_:.2f} bits, changed {lb/n_:.2f} bits, changed preferred {nb}/{n_}")
    finally: model.feat_scale.copy_(fs); model.train(was)
    print(f"  [facts] visit {vi}: log2 p of the 4-digit value | "+" || ".join(out)+" | (chance -13.3)")
def make_loader(raw,cfg,fixed_seed=None,no_revisit=False,lanes=False,span=1,jump=0,keep=False):
    """fixed_seed: eval loader that resets to the SAME batches on every bpc() call,
    so evaluation noise is identical across configs and seeds. no_revisit: non-overlapping windows in random order, each at most once."""
    src=torch.frombuffer(bytearray(raw),dtype=torch.uint8).long().to(DEVICE)
    if lanes:                                                          # CONTINUOUS CONTEXT: lanes, each reading its own part in order
        Tq=int(cfg.seq_len); st={'j':0,'ep':0,'K':None}
        def _lay(K):                                                   # K segments per row (ctx_bptt): batch_size // K lanes of K x seq_len bytes
            Bn=max(1,int(cfg.batch_size)//K); Ln=(len(src)-1)//Bn
            st.update(K=K,Bn=Bn,nst=max(1,(Ln-1)//(Tq*K)),base=(torch.arange(Bn,device=DEVICE)*Ln)[:,None],tx=torch.arange(Tq*K,device=DEVICE)[None],j=0,k=0,b=None)
        if int(jump)>0:                                                # EPISODES: every `jump` segments, fresh random places
            gj_=torch.Generator(device='cpu'); gj_.manual_seed(len(src)%100003+7); J=int(jump)
            def get():
                K=int(_BPTT['K'])
                if st['K']!=K: _lay(K)
                Jf=max(1,J//K)                                         # fetches per episode
                if st['k']%Jf==0: st['b']=torch.randint(len(src)-Jf*K*Tq-2,(st['Bn'],1),generator=gj_).to(DEVICE)
                get.fresh=(st['k']==0) if keep else (st['k']%Jf==0); ix=st['b']+(st['k']%Jf)*K*Tq+st['tx']; st['k']+=1
                return src[ix], src[ix+1]
            get.reset=lambda: None; get.fresh=True; get.windows=max(1,(len(src)-1)//Tq)
            return get
        def get():
            K=int(_BPTT['K'])
            if st['K']!=K: _lay(K)
            if st['j']>=st['nst']:
                st['j']=0; st['ep']+=1
                if st['ep']==1: print(f"    [data] lanes: shard read through ({st['nst']} rows of {K} segments per lane) - lanes start again (memory reset)")
            get.fresh=(st['j']==0); ix=st['base']+st['j']*K*Tq+st['tx']; st['j']+=1
            return src[ix], src[ix+1]
        get.reset=lambda: None; get.fresh=True; get.windows=max(1,(len(src)-1)//Tq)
        return get
    if no_revisit:
        W=cfg.seq_len+1; nw=max(1,(len(src)-1)//W); st={'perm':torch.randperm(nw).to(DEVICE)*W,'pos':0,'ep':0}
        tx=torch.arange(cfg.seq_len,device=DEVICE)
        def get():
            if st['pos']+cfg.batch_size>nw:                          # shard exhausted: reshuffle (reported - should not happen when sized right)
                st['perm']=torch.randperm(nw).to(DEVICE)*W; st['pos']=0; st['ep']+=1
                print(f"    [data] WARNING: training shard exhausted ({nw} windows) - text now repeats (epoch {st['ep']+1})")
            ix=st['perm'][st['pos']:st['pos']+cfg.batch_size]; st['pos']+=cfg.batch_size
            return src[ix[:,None]+tx[None]], src[ix[:,None]+tx[None]+1]
        get.reset=lambda: None; get.windows=nw
        return get
    Tsp_=int(cfg.seq_len)*max(1,int(span))                            # span > 1: longer evaluation windows (eval_ctx)
    N=max(1,len(src)-Tsp_-1); tx=torch.arange(Tsp_,device=DEVICE)
    gen=torch.Generator(device='cpu')
    def reset():
        if fixed_seed is not None: gen.manual_seed(fixed_seed)
    def get():
        if fixed_seed is not None: ix=torch.randint(N,(cfg.batch_size,),generator=gen).to(DEVICE)
        else: ix=torch.randint(N,(cfg.batch_size,),device=DEVICE)
        return src[ix[:,None]+tx[None]], src[ix[:,None]+tx[None]+1]
    get.reset=reset; reset()
    return get
# ---------------- INFERRED HEADS (domain identity not given at test time) ----------------
@torch.no_grad()
def head_stats_update(model, x, dom_id, step, cfg):
    """Per-domain summary statistics gathered while training: byte-pair counts (whole domain) and the mean /
    variance of the final representation (second half of the domain, when it has settled)."""
    dev=x.device
    if not hasattr(model,'_bg'):
        object.__setattr__(model,'_bg',torch.zeros(len(PILE_DOMAINS),256*256,device=dev))
        object.__setattr__(model,'_fs',{})
    pair=(x[:,:-1]*256+x[:,1:]).reshape(-1)
    model._bg[dom_id]+=torch.bincount(pair,minlength=256*256).float()
    if step>=cfg.steps_per_task//2 and getattr(model,'_last_h',None) is not None:
        h=model._last_h.reshape(-1,model._last_h.shape[-1]).float()
        s1,s2,n=model._fs.get(dom_id,(0.0,0.0,0))
        model._fs[dom_id]=(s1+h.sum(0), s2+(h*h).sum(0), n+h.shape[0])

@torch.no_grad()
def inferred_logp(model, x, ncand, mode, cfg):
    return inferred_logp_multi(model, x, ncand, (mode,), cfg)[mode]

@torch.no_grad()
def inferred_logp_multi(model, x, ncand, modes, cfg):
    """Mixture of the trained heads weighted by a CAUSAL posterior over domains (token t uses bytes <= t), for several posterior
    modes from ONE forward pass. mode 'bytes': naive-Bayes on byte pairs; 'heads': the heads' own likelihood; 'both';
    'features': diagonal Gaussian on the final representation."""
    h=model.features(x); D=ncand
    if getattr(model,'_spec',None):                     # domain-specific shunts: each head reads its own configuration
        LPl=[]
        for d in range(D):
            dd=domain_delta(model,d) if d!=getattr(model,'_live_dom',None) else {}
            apply_delta(model,dd,+1); LPl.append(F.log_softmax(model.heads[d](model.features(x)),-1)); apply_delta(model,dd,-1)
        LP=torch.stack(LPl)
    else:
        LP=torch.stack([F.log_softmax(model(x,d),-1) for d in range(D)])
    def _bytes_ll():
        cnt=model._bg[:D]+cfg.hs_alpha; cnt=cnt.view(D,256,256)
        logp=torch.log(cnt/cnt.sum(-1,keepdim=True))                                  # log p_d(next | prev)
        prev,nxt=x[:,:-1],x[:,1:]
        l_=logp[:,prev,nxt]                                                            # (D,B,T-1)
        return torch.cat([torch.zeros_like(l_[...,:1]),l_],-1)
    def _heads_ll():                       # HEAD LIKELIHOOD: how well each head predicted the bytes seen so far (Bayesian
        nxt=x[:,1:]                        # model averaging over the heads; causal: position t uses bytes <= t)
        l_=LP[:,:,:-1].gather(-1,nxt[None,...,None].expand(D,-1,-1,1))[...,0]        # (D,B,T-1)
        return torch.cat([torch.zeros_like(l_[...,:1]),l_],-1)
    res={}
    for mode in modes:
        if mode=='bytes': ll=_bytes_ll()
        elif mode=='heads': ll=_heads_ll()
        elif mode=='both': ll=_bytes_ll()+_heads_ll()
        else:
            hs=[]
            for d in range(D):
                s1,s2,n=model._fs[d]; mu=s1/n; var=(s2/n-mu*mu).clamp(min=1e-4)
                hs.append((-0.5*((h-mu)**2/var+torch.log(var))).sum(-1))                    # (B,T)
            ll=torch.stack(hs)
        post=F.log_softmax(torch.cumsum(ll,-1)*cfg.hs_temp,0)                              # (D,B,T)
        res[mode]=(torch.logsumexp(post.unsqueeze(-1)+LP,0), post)
    return res

def bpc_inferred_modes(model, get, domain, ncand, modes, cfg, n=15):
    """bpc_inferred for several modes with one forward pass per batch (identical values)."""
    if hasattr(get,'reset'): get.reset()
    model.eval(); L={m:[] for m in modes}; A={m:[] for m in modes}
    with torch.no_grad():
        for _ in range(n):
            x,y=get()
            Ts_=int(cfg.seq_len)
            for m,(lp,post) in inferred_logp_multi(model,x,ncand,modes,cfg).items():
                L[m].append(float(-lp[:,-Ts_:].gather(-1,y[:,-Ts_:].unsqueeze(-1)).mean())); A[m].append(float((post[...,-Ts_:].argmax(0)==domain).float().mean()))
    model.train(); return {m:(float(np.mean(L[m]))/LOG2, float(np.mean(A[m]))) for m in modes}

def bpc_inferred(model, get, domain, ncand, mode, cfg, n=15):
    if hasattr(get,'reset'): get.reset()
    model.eval(); L=[]; A=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get(); lp,post=inferred_logp(model,x,ncand,mode,cfg)
            L.append(float(-lp.gather(-1,y.unsqueeze(-1)).mean())); A.append(float((post.argmax(0)==domain).float().mean()))
    model.train(); return float(np.mean(L))/LOG2, float(np.mean(A))

def bpc(model,get,domain,n=15):
    if hasattr(get,'reset'): get.reset()
    dd=domain_delta(model,domain) if (getattr(model,'_spec',None) and domain!=getattr(model,'_live_dom',None)) else {}
    apply_delta(model,dd,+1)
    model.eval(); L=[]
    eb=int(getattr(getattr(model,'cfg',None),'eval_batch',0) or 0)
    Ts_=int(model.cfg.seq_len)                                        # eval_ctx: longer windows, scored on their last seq_len bytes
    with torch.no_grad():
        if eb>0:                                                     # same batches, stacked: one forward per eb sequences
            xs,ys=zip(*[get() for _ in range(n)]); X=torch.cat(xs); Y=torch.cat(ys); tot=0.0
            for i in range(0,X.shape[0],eb):
                tot+=float(F.cross_entropy(model(X[i:i+eb],domain)[:,-Ts_:].reshape(-1,256),Y[i:i+eb,-Ts_:].reshape(-1),reduction='sum'))
            L=[tot/Y[:,-Ts_:].numel()]
        else:
            for _ in range(n): x,y=get(); L.append(F.cross_entropy(model(x,domain)[:,-Ts_:].reshape(-1,256),y[:,-Ts_:].reshape(-1)).item())
    apply_delta(model,dd,-1)
    model.train(); return float(np.mean(L))/LOG2

def _tmark(model, cfg, step, name):
    """Per-phase wall-clock (GPU synchronised) on every timing_every-th step."""
    e_=max(2,int(cfg.timing_every))
    if not cfg.timing or step%e_ not in (3,(3+e_//2+(1 if (e_//2)%2==0 else 0))%e_): return   # regular steps of BOTH parities (dream steps can be every 2nd step)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    T=model.__dict__.setdefault('_tm',{'last':None,'acc':{},'n':0})
    for k_,v_ in (('sd',[0.0,0]),('sn',[0.0,0]),('t0',None),('d0',0)): T.setdefault(k_,v_)
    now=time.perf_counter()
    if name=='start': T['last']=now; T['n']+=1; T['t0']=now; T['d0']=model.__dict__.get('_dcount',0); return
    if T['last'] is not None: T['acc'][name]=T['acc'].get(name,0.0)+now-T['last']; T['last']=now
    if name=='rest' and T['t0'] is not None:                         # whole-step time, split by whether the step dreamed
        a_=T['sd'] if model.__dict__.get('_dcount',0)>T['d0'] else T['sn']; a_[0]+=now-T['t0']; a_[1]+=1

_NOV={'m':None}

def borrow_align(model, cfg, olds, dom_id, x, y, n_rows=8):
    """BORROW ONLY IF IT HELPS: per old domain d, cos(grad of the borrowing KL on d's queued dreams, grad of the current domain's loss on the
    current batch) over the blocks and the current head; stores max(0, cos) as d's borrowing weight."""
    ps=[p_ for p_ in model.blocks.parameters() if p_.requires_grad]+list(model.heads[dom_id].parameters())
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    flat=lambda g: torch.cat([(gi if gi is not None else torch.zeros_like(p_)).flatten().float() for gi,p_ in zip(g,ps)])
    gc=flat(torch.autograd.grad(F.cross_entropy(model.heads[dom_id](model.features(x)).reshape(-1,256),y.reshape(-1)),ps,allow_unused=True))
    G=model.__dict__.get('_borrow_g')
    if G is None: G=model.__dict__['_borrow_g']=torch.zeros(len(PILE_DOMAINS),device=y.device)
    Pp=min(cfg.dream_stream_prompt,min(cfg.dream_len,cfg.seq_len)//2); Q=model._dsq; out={}
    for d in olds:
        q=Q.get(d) or []
        if not q: G[d]=0.0; continue
        xd=torch.stack([a for a,_ in q[:n_rows]]); tg=torch.stack([b for _,b in q[:n_rows]])
        lc=F.log_softmax(model.heads[dom_id](model.features(xd)).float()/cfg.shunt_T,-1)
        kb=(tg.exp()*(tg-lc)).sum(-1)[:,Pp:].mean()
        gb=flat(torch.autograd.grad(kb,ps,allow_unused=True))
        c=(gb@gc)/(gb.norm()*gc.norm()+1e-30); G[d]=c.clamp(min=0.0); out[d]=c
    model.feat_scale.copy_(fs); model.train(was)
    A=model.__dict__.setdefault('_borrow_acc',{})
    for d,c in out.items(): a_=A.setdefault(d,[0.0,0]); a_[0]+=c; a_[1]+=1

def dctl_state(model):
    return model.__dict__.setdefault('_dctl',{'v':0,'ref':{},'cal':{},'D':{},'debt':{},'u':1.0,'acc':0.0,'s0':None,'s':None,'rff':1.0,
                                               'c':{},'rema':1.0,'nd':0,'calib':False,'last':None,'ns':0})
def dctl_rate(model,cfg,step,olds):
    """Visit bookkeeping at step 0; returns the dream interval for this step (1 = dream now, step+2 = skip) - weight stays x1."""
    S=dctl_state(model)
    if S.get('vtok')!=model.__dict__.get('_dctl_vtok'):                  # first dream decision of a new visit
        S['vtok']=model.__dict__.get('_dctl_vtok')
        if S['last'] is not None:                                      # close the previous visit: calibration means, debts
            for d,(sm,n) in S['cal'].items():
                if n>0 and d not in S['ref']: S['ref'][d]=sm/n
            if not S['calib']:
                for d,Dv in S['D'].items():
                    if d in S['ref']: S['debt'][d]=S['debt'].get(d,0.0)+Dv-cfg.dctl_rel*S['ref'][d]
        S['v']+=1; S['cal']={}; S['D']={}; S['s0']=None; S['rff']=1.0; S['nd']=0; S['ns']=0; S['acc']=0.0
        S['calib']=any(d not in S['ref'] for d in olds) and not cfg.dctl_calwin>0 and cfg.dctl!='rate'; S['last']=list(olds); S['H']={}; S['vref']={}
    S['ns']+=1
    if S['calib']: S['nd']+=1; return 1                                 # calibration visit: dream every step
    r=min(1.0,max(cfg.dctl_rmin,(1.0)*S['u'])); S['rema']=0.98*S['rema']+0.02*r
    S['acc']+=r
    if S['acc']>=1.0 or S['ns']==1: S['acc']=max(0.0,S['acc']-1.0); S['nd']+=1; return 1
    return step+2
def guard_step(model,cfg,olds,dom_id,step):
    """EARLY DRIFT GUARD: sense (held-out bank KL) early in the visit, flag fast-rising old domains, set their row boost / Kalman weight."""
    Lv=int(cfg.steps_per_task); ev_=max(1,int(round(cfg.guard_every*Lv))); span_=int(round(cfg.guard_span*Lv)); dur_=max(1,int(round(cfg.guard_dur*Lv)))
    G=model.__dict__.setdefault('_guard',{})
    if step==0: G.clear(); G.update(base={},until={},kfw={},log=[],n=0,last={},atr={},na={})
    atr_=cfg.guard_mode=='atr'
    if step<=span_ and step%ev_==0:
        kl_=guard_sense(model,cfg,olds); G['n']+=1
        if step==0: G['base']=dict(kl_); G['last']=dict(kl_)
        else:
            for d,v in kl_.items():
                b_=G['base'].get(d,v)
                if atr_:                                                                    # AVERAGE TRUE RANGE of the held-out KL
                    tr_=abs(v-G['last'].get(d,v)); G['last'][d]=v; n_=G['na'].get(d,0)+1; G['na'][d]=n_
                    a_=2.0/(int(cfg.guard_atr_n)+1); E_=G.setdefault('ema',{}); e0_=E_.get(d,b_)
                    hit_=n_>=3 and v>e0_+cfg.guard_atr_k*G['atr'].get(d,tr_) and (v-b_)>0.5*cfg.guard_min   # breakout against the channel so far
                    G['atr'][d]=tr_ if n_==1 else (1-a_)*G['atr'][d]+a_*tr_; E_[d]=(1-a_)*e0_+a_*v
                else: hit_=v>max(cfg.guard_ratio*b_,b_+cfg.guard_min)
                if hit_:
                    if G['until'].get(d,-1)<=step: G['log'].append((step,d,round(v,4),round(b_,4)))
                    G['until'][d]=step+dur_
    act={d for d,u in G['until'].items() if u>step}
    bst_={d:float(cfg.guard_boost) for d in act} if (act and cfg.guard_boost!=1.0) else {}
    if atr_ and cfg.guard_atr_share and len(G['atr'])==len(olds) and len(olds)>1:              # shares follow volatility
        m_=float(np.mean(list(G['atr'].values())))
        if m_>0: bst_={d:max(bst_.get(d,1.0),float(np.clip(G['atr'][d]/m_,0.5,cfg.guard_boost))) for d in olds}
    model.__dict__['_guard_boost']=bst_ or None
    if atr_ and cfg.guard_atr_rows and G.get('atr'):                                          # ATR-CONTROLLED DREAM ROWS
        A_=float(np.mean(list(G['atr'].values()))); GA_=model.__dict__.setdefault('_atr_glob',[0.0,0])
        if step<=span_ and step%ev_==0 and G['na'] and min(G['na'].values())>=2: GA_[0]+=A_; GA_[1]+=1
        if GA_[1]>=3 and GA_[0]>0:
            R0_=int(cfg.dream_batch); R_=int(np.clip(round(R0_*A_/(GA_[0]/GA_[1])),max(1,round(R0_*cfg.guard_rows_min)),max(1,round(R0_*cfg.guard_rows_max))))
            model.__dict__['_sldb']=R0_; cfg.dream_batch=R_
            st_=G.setdefault('rows',[0,0]); st_[0]+=R_; st_[1]+=1
    if cfg.guard_kf!=1.0 and cfg.pmuon:
        new={d:float(cfg.guard_kf) for d in act}
        if new!=G['kfw']:
            G['kfw']=new; model.__dict__['_kf_wt']=new or None; kf_build(model,cfg,dom_id,quiet=True)
    if not G.get('rep') and (step==int(cfg.steps_per_task)-1 or step>span_+dur_):
        G['rep']=True
        print(f"    [drift guard] {G['n']} checks, {len(G['log'])} flags: "+(' '.join(f"s{s_} {_lab(cfg,d)} {v:.3f}>{b:.3f}" for s_,d,v,b in G['log']) or 'none')
              +(" | ATR "+' '.join(f"{_lab(cfg,d)} {a:.4f}" for d,a in sorted(G['atr'].items())) if G.get('atr') else "")
              +(f" | ATR rows: mean {G['rows'][0]/max(1,G['rows'][1]):.1f} per step so far" if G.get('rows') else ""))
def _dstep_obj(model,cfg,dom_id,olds,seed):
    """DIRECT STEP held-out objective J: CE of the new domain's head on held-out text rows + KL(stored top-k targets || old heads) on
    held-out bank rows (rows < dstep_hold of each old domain's bank, never drawn for training), prompt positions excluded."""
    xh,yh=model.__dict__['_dsH']; dev=xh.device
    rs_=np.random.RandomState(int(seed)%(2**31-1))
    it_=torch.as_tensor(rs_.choice(xh.shape[0],size=min(int(cfg.dstep_rows),xh.shape[0]),replace=False),device=dev)
    xt,yt=xh[it_],yh[it_]; nt=xt.shape[0]
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[d for d in olds if d in BK and d in TV]
    dm=sorted(int(ol[i]) for i in rs_.choice(len(ol),size=int(cfg.dstep_drows))) if (ol and int(cfg.dstep_drows)>0) else []
    T_=xt.shape[1]; parts=[xt]; V=I=None
    if dm:
        ri=[int(rs_.randint(0,max(1,min(int(cfg.dstep_hold),BK[d].shape[0])))) for d in dm]
        Xd=torch.stack([BK[d][i][:T_] for d,i in zip(dm,ri)]).long().to(dev); parts.append(Xd)
        V=torch.stack([TV[d][0][i][:T_] for d,i in zip(dm,ri)]).to(dev).float(); I=torch.stack([TV[d][1][i][:T_] for d,i in zip(dm,ri)]).to(dev).long()
    was=model.training; fs=model.feat_scale.clone(); sv_=model.__dict__.get('_s0_dom')
    try:
        model.eval(); model.__dict__['_s0_dom']=[dom_id]*nt+dm
        rt_=_dream_route_rows(cfg,len(dm),0,dev,len(model.blocks)) if dm else None
        if rt_ is not None: rt_=torch.cat([torch.ones(nt,rt_.shape[1],device=dev),rt_])
        H=model.features(torch.cat(parts),route=rt_)
        J=F.cross_entropy(model.heads[dom_id](priv(model,xt,H[:nt],dom_id)).float().reshape(-1,256),yt.reshape(-1))
        if dm:
            Pp=min(int(cfg.dream_stream_prompt),T_//2); spans=[]; i0=0
            for j in range(1,len(dm)+1):
                if j==len(dm) or dm[j]!=dm[i0]: spans.append((dm[i0],i0,j)); i0=j
            ld=torch.cat([F.log_softmax(model.heads[d](priv(model,Xd[a:b],H[nt+a:nt+b],d)).float()/cfg.shunt_T,-1) for d,a,b in spans])
            J=J+_kl_topk(ld,V,I)[:,Pp:].mean()
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs); model.train(was)
    return J

def _dstep_grad(model,cfg,dom_id,olds,seed,P_):
    with torch.enable_grad():
        J=_dstep_obj(model,cfg,dom_id,olds,seed)
        gs=torch.autograd.grad(J,P_,allow_unused=True)
    return [torch.zeros_like(p_) if g_ is None else g_ for p_,g_ in zip(P_,gs)], J.detach()

@torch.no_grad()
def _srep_note(model,h):
    """cg15 stream_rep: this step's representation jump (before the running mean takes the batch in), read by _StreamGet."""
    with torch.no_grad():
        m=h.detach().float().mean((0,1)); e=model.__dict__.get('_srep')
        if e is None: model.__dict__['_srep']=m.clone(); return
        model.__dict__['_rj']=float(1.0-F.cosine_similarity(m,e,dim=0)); model.__dict__['_srep']=0.9*e+0.1*m
@torch.no_grad()
def _cdet_step(model,cfg,x,y,h,dom_id,step):
    """CHANGE-DETECTION LOG (no training effect): representation jump 1 - cos(batch-mean top-block output, running mean) and loss jump
    (real-text CE before the head refit minus its running mean), kept on the device; reported at the visit end."""
    D=model.__dict__.setdefault('_cdet',{'ema':None,'el':None,'cur':[],'hist':[]})
    m=h.detach().float().mean((0,1))
    ce=F.cross_entropy(model.heads[dom_id](priv(model,x,h.detach(),dom_id)).float().reshape(-1,256),y.reshape(-1))
    vt_=model.__dict__.get('_dctl_vtok',0)
    if D.get('vt')!=vt_: D['vt']=vt_; D['cur']=[]; D['sw']=D['ema'] is not None      # first measured step of a visit (step 0 may take the
                                                                                    #   plain path); first measured visit: no switch to score
    if D['ema'] is not None:
        jr=1.0-F.cosine_similarity(m,D['ema'],dim=0); jl=ce-D['el']; D['cur'].append(torch.stack([jr,jl]))
        D['ema']=0.9*D['ema']+0.1*m; D['el']=0.9*D['el']+0.1*ce
    else: D['ema']=m.clone(); D['el']=ce.clone()

def _cdet_report(model,cfg):
    D=model.__dict__.get('_cdet')
    if not D or len(D['cur'])<20 or not D.get('sw'):
        if D: D['cur']=[]
        return
    A=torch.stack(D['cur']).cpu().numpy(); D['cur']=[]
    sw=A[0]; w=A[5:]; mu=w.mean(0); sd=w.std(0)+1e-12; z=(sw-mu)/sd; sh=(w>=sw[None]).mean(0)
    D['hist'].append((float(z[0]),float(z[1]),float(sh[0]),float(sh[1])))
    H_=np.array(D['hist'])
    print(f"  [change detection] switch into this visit: representation jump {sw[0]:.4f} (z {z[0]:.1f}, {sh[0]:.1%} of the visit's steps at or above it) | "
          f"loss jump {sw[1]/LOG2:+.3f} bits (z {z[1]:.1f}, {sh[1]:.1%}) || {len(H_)} switches so far, flagged with no false alarm in the visit "
          f"(no step at or above): representation {int((H_[:,2]==0).sum())}, loss {int((H_[:,3]==0).sum())} | median z: representation {np.median(H_[:,0]):.1f}, loss {np.median(H_[:,1]):.1f}")

def train_domain(model, meta, meta_opt, opt, get, dom_id, prior_buffer, cfg):
    _t_dom=time.perf_counter(); model.__dict__['_tm']={'last':None,'acc':{},'n':0,'dc0':model.__dict__.get('_dcount',0)}; model.__dict__['_tm_dream']=0.0
    # OLD DOMAINS = every domain seen so far except the current one, ordered from least to most recently visited (single pass:
    # range(dom_id), exactly as before; repeated passes: revisited domains are 'current', all others 'old')
    olds=list(getattr(model,'_olds',None) if getattr(model,'_olds',None) is not None else range(dom_id)); n_old=len(olds)
    model.train()
    _RAMP['s']=1.0   # weak base impedance early
    imp_base=None
    _L=cfg.steps_per_task; _ce_hist=[]                       # this visit's length (VISIT LENGTH BY PROGRESS may change it at visit_decide)
    if cfg.stream: _L=int(cfg.steps_per_task*cfg.stream_cap)                # NO VISIT BOUNDARIES: visits end by detection
    _lr_replan=None
    _nov_j=None
    model.__dict__['_dctl_vtok']=model.__dict__.get('_dctl_vtok',0)+1            # visit token (drift-budget controller bookkeeping)
    if cfg.ctx_carry and int(cfg.ctx_bptt)>1:                                       # WRITE-PATH TRAINING: segments per row this visit (curriculum)
        nv_=model.__dict__['_dctl_vtok']; f_=int(cfg.ctx_bptt_from)
        _BPTT['K']=1 if nv_<=f_ else min(int(cfg.ctx_bptt),2**(nv_-f_))
        print(f"    [write path] visit {nv_}: {_BPTT['K']} segment(s) per training row ({max(1,cfg.batch_size//_BPTT['K'])} rows of {_BPTT['K']*cfg.seq_len} bytes)")
    else: _BPTT['K']=1
    _prof_=None
    _nsl_=int(round(float(cfg.sleep_phase)*cfg.steps_per_task)) if (float(cfg.sleep_phase)>0 and not cfg.stream and float(model.__dict__.get('_real_w',1.0))==0.0) else 0
    if _nsl_>0: _L=cfg.steps_per_task+_nsl_
    _SLP['n']=[0,0]; model.__dict__['_ctrl_bytes']=False
    for step in range(max(_L if (cfg.stream or _nsl_>0) else cfg.steps_per_task,0)):
        if _nsl_>0:                                                                  # OFFLINE SLEEP PHASE (after the awake steps)
            j_=step-cfg.steps_per_task
            ph_=None if (j_<0 or cfg.sleep_ctrl) else ('nrem' if j_<int(round(_nsl_*float(cfg.sleep_nrem))) else 'rem')
            model.__dict__['_sleep_phase']=ph_
            model.__dict__['_ctrl_bytes']=bool(cfg.sleep_ctrl and cfg.sleep_ctrl_bytes and j_>=0)
            if ph_: _SLP['n'][0 if ph_=='nrem' else 1]+=1
        if cfg.profile_visit and model.__dict__.get('_dctl_vtok')==cfg.profile_visit and getattr(cfg,'profile_cpu',False):   # v9.8: CPU COPY-SOURCE PROFILE
            if step==20:
                _prof_=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_stack=True); _prof_.__enter__()
            elif step==24 and _prof_ is not None:
                _prof_.__exit__(None,None,None); cnt_={}
                for ev_ in _prof_.events():
                    if ev_.name in ('aten::copy_','aten::_to_copy','aten::clone','aten::contiguous'):
                        fr_=[f_ for f_ in (ev_.stack or []) if 'threshold_lab' in f_ or '_defs' in f_]
                        k_=(ev_.name,fr_[0] if fr_ else '?'); cnt_[k_]=cnt_.get(k_,0)+1
                print("  [copy sources] 4 steps, copies by code line:\n"+"\n".join(f"    {c_:6d}  {n_:16} {l_[:150]}" for (n_,l_),c_ in sorted(cnt_.items(),key=lambda z:-z[1])[:30]),flush=True)
                _prof_=None
        elif cfg.profile_visit and model.__dict__.get('_dctl_vtok')==cfg.profile_visit and torch.cuda.is_available():   # ONE-OFF GPU PROFILE
            if step==100:
                _prof_=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]); _prof_.__enter__(); _PRF['on']=True
            elif step==110 and _prof_ is not None:
                torch.cuda.synchronize(); _prof_.__exit__(None,None,None); ka_=_prof_.key_averages()
                print("  [profile] 10 steps of visit "+str(cfg.profile_visit)+", top operations by SELF GPU time:\n"+ka_.table(sort_by="self_cuda_time_total",row_limit=30))
                print("  [profile] top operations by TOTAL GPU time (incl. children):\n"+ka_.table(sort_by="cuda_time_total",row_limit=25))
                try:
                    rg_=[e for e in ka_ if e.key.startswith('R:')]; rg_.sort(key=lambda e:-e.device_time_total if hasattr(e,'device_time_total') else -e.cuda_time_total)
                    print("  [profile] NAMED REGIONS (GPU time incl. everything inside, per 10 steps):\n"+"\n".join(f"    {e.key[2:]:20} {((e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total)/1e3):9.1f} ms  ({e.count} calls)" for e in rg_))
                except Exception as ex: print(f"  [profile] region table unavailable ({repr(ex)[:100]})")
                _PRF['on']=False; _prof_=None
        if step>=_L: break
        if model.__dict__.get('_sldb') is not None: cfg.dream_batch=model.__dict__.pop('_sldb')   # (safety: never carried past a step)
        if model.__dict__.get('_slmul') is not None:
            mu_=model.__dict__.pop('_slmul')
            for g_ in opt.param_groups: g_['lr']/=mu_
        gs=getattr(model,'_gstep',0); object.__setattr__(model,'_gstep',gs+1)
        _tmark(model,cfg,step,'start')
        if imp_base is not None and step>0 and step%cfg.imp_online_every==0:
            online_importance(model, cfg, imp_base, step/_L)
        if step==0:
            _XDG['cur']=dom_id
            model.__dict__['_dmom']={}
            P_=_ema_params(model); model.__dict__['_ema']=(P_,[p_.detach().clone() for p_ in P_],1.0-1.0/max(2.0,cfg.ema_tail*(cfg.steps_per_task if cfg.stream else _L)))
            model.__dict__['_selh']=[]; model.__dict__['_selk']=0
            model.__dict__.pop('_gcache',None); model.__dict__['_vstart']=None; model.__dict__.pop('_lwfG',None)   # fresh per visit
            object.__setattr__(model,'_live_dom',dom_id)
            object.__setattr__(model,'_xcur',None)
        if step==0:                                     # PERFORMANCE: frozen teachers + stacked old heads for this visit
            object.__setattr__(model,'_teach',None); object.__setattr__(model,'_hstk',None); model.__dict__.pop('_klacc',None)
            if n_old>0 and not cfg.topic_mode and cfg.shunt and (cfg.fast_teacher or cfg.fast_dreams):
                if cfg.grow_blocks_at or (cfg.block_extra and cfg.block_extra_mode=='fresh') or getattr(model,'_spec',None) or cfg.dream_filter:
                    print("    [fast] frozen teachers off for this config (growing / fresh units / specific shunts / dream filter)")
                else: build_teachers(model, cfg)
            if n_old>0 and not cfg.topic_mode and cfg.fast_kl: stack_heads(model, olds)
        fdv=bool(cfg.fast_dreams and getattr(model,'_teach',None) is not None)
        if cfg.dream and n_old>0 and not cfg.topic_mode and fdv and cfg.dream_stream:   # CONTINUOUS DREAM STREAMS
            if step==0: model._dreams={}; model.__dict__['_dsq']={}; model.__dict__['_dstream_last']={}
            if step%cfg.dream_stream_block==0 and (cfg.dream_parallel or cfg.dream_bank):
                model._dsq.setdefault('__par__',[1]); fl_=None                         # parallel dreams: generated per step in stream_batch
            elif step%cfg.dream_stream_block==0:
                fl_=stream_generate(model, cfg, olds, min(cfg.dream_stream_block, _L-step))
                if step==0 and fl_: print("    [dream fidelity] streams, share of each old domain's chunks attributed to that domain: "+" ".join(f"{_lab(cfg,d)} {v:.2f}" for d,v in fl_.items()))
        elif cfg.dream and n_old>0 and not cfg.topic_mode and fdv:      # RECURRENT SAMPLER: the whole visit's dreams at its start
            if step==0: fast_dream_visit(model, cfg, dom_id, olds); model.__dict__['_onpol']={}
            elif step%cfg.dream_refresh==0:
                q_=model.__dict__.get('_dqueue',{}).pop(step,None)
                if q_ is not None: _pool_add(model, cfg, q_[0], q_[1], q_[2])
        elif cfg.dream and n_old>0 and not cfg.topic_mode:
            if step==0:
                model._dreams={}
                sh=dream_shares(cfg, n_old)                     # shares by POSITION in olds (last = most recently visited)
                for i_,d in enumerate(olds):
                    if sh is None: dream_refresh(model, cfg, dom_id, d, cfg.dream_n)
                    elif sh[i_]>0: dream_refresh(model, cfg, dom_id, d, dream_cap(cfg, n_old, i_))   # graduated: no pool
                if sh is not None:
                    print("    [dream budget] share per old domain (recency-weighted"+(f", graduated at age {cfg.dream_graduate}")+"): "
                          +" ".join(f"{_lab(cfg,d)} {sh[i_]:.2f} ({dream_cap(cfg,n_old,i_)})" for i_,d in enumerate(olds)))
            elif step%cfg.dream_refresh==0:
                sh=dream_shares(cfg, n_old)
                dr=olds[(step//cfg.dream_refresh)%n_old] if sh is None else olds[dream_pick(model, cfg, n_old, sh, 1)]
                dream_refresh(model, cfg, dom_id, dr, cfg.dream_batch)
        if step==0:                                                                     # v9.8 cg8 DIRECT LEARNED STEP: fresh multiplier, held-out rows
            _ds_on=bool(cfg.dstep and n_old>0 and cfg.dream_bank and (not cfg.dstep_pass1 or float(model.__dict__.get('_real_w',1.0))>0))
            model.__dict__['_dsm']=1.0; _ds_t=0.0
            if _ds_on: xh_,yh_=get(); model.__dict__['_dsH']=(xh_,yh_); model.__dict__['_dsS']=[]; model.__dict__['_dsM']=[]; model.__dict__['_dsJ']=[]
        x,y=get()
        if cfg.inf_on and float(model.__dict__.get('_real_w',1.0))>0: inf_store_add(model,cfg,x,int(model.__dict__.get('_cur_dom',dom_id)))   # cg20 store
        if cfg.nrem_real and float(model.__dict__.get('_real_w',1.0))>0:            # nrem_real DIAGNOSTIC: keep random pass-1 rows per domain
            with torch.no_grad():
                P1_=model.__dict__.setdefault('_p1win',{}); td1_=int(model.__dict__.get('_cur_dom',dom_id)); E1_=P1_.setdefault(td1_,{'r':[],'seen':0})
                rs1_=model.__dict__.setdefault('_p1_rs',np.random.RandomState(int(cfg.seed)+515))
                for r_ in range(x.shape[0]):
                    E1_['seen']+=1; j1_=len(E1_['r']) if len(E1_['r'])<int(cfg.nrem_real_cap) else rs1_.randint(E1_['seen'])
                    if j1_<int(cfg.nrem_real_cap):
                        rw_=torch.cat([x[r_],y[r_,-1:]]).detach().clone()
                        if j1_<len(E1_['r']): E1_['r'][j1_]=rw_
                        else: E1_['r'].append(rw_)
        if step==0 or (cfg.ctx_carry and getattr(get,'fresh',False)): _CARRY['st']={}          # CONTINUOUS CONTEXT: empty memory at a visit start / lane restart
        if cfg.slot_mem and step==0: _CARRY['slot']=dom_id; _CARRY['sm']=model.__dict__.setdefault('_smem',{})   # SLOT MEMORY of this visit's slot
        model.__dict__.pop('_pairA',None)
        if cfg.stream and model.__dict__.pop('_vend',False): _L=step+1                    # NO VISIT BOUNDARIES: the detector ended this visit (this is its last step)
        if cfg.guard and n_old>0: guard_step(model,cfg,olds,dom_id,step)                   # EARLY DRIFT GUARD
        if step==0: model.__dict__.pop('_film_aff',None)
        if step==0 and n_old>0 and (cfg.borrow_aff>0 or (cfg.film_k>0 and cfg.film_aff and model.__dict__.get('_first_visit',True))):
            w_,b_=aff_weights(model,cfg,x,y,olds,cfg.film_tau)
            if cfg.film_k>0 and cfg.film_aff and model.__dict__.get('_first_visit',True):
                wf_=w_
                with torch.no_grad():
                    C_=model.film_codes; C_[dom_id]=sum(wf_[d_]/len(olds)*C_[d_] for d_ in olds)+0.02*torch.randn_like(C_[dom_id])
            print("    [affinity] old heads' bits on the first batch: "+" ".join(f"{_lab(cfg,d_)} {b_[d_]:.2f}" for d_ in olds)
                  +" | weights: "+" ".join(f"{_lab(cfg,d_)} {w_[d_]:.2f}" for d_ in olds))
        if step==0 and n_old>0 and cfg.pu_k_base>0 and cfg.pu_k>cfg.pu_k_base and model.__dict__.get('_first_visit',True):   # SURPRISE-TRIGGERED GROWTH
            bg_=aff_weights(model,cfg,x,y,olds,1.0)[1]; mb_=min(bg_.values()); hg_=model.__dict__.setdefault('_grow_hist',[])
            med_=float(np.median(hg_)) if hg_ else None; grow_=med_ is not None and mb_>cfg.pu_grow_rel*med_
            if grow_:
                for b_ in model.blocks:
                    if getattr(b_,'pu_mask',None) is not None: b_.pu_mask[dom_id]=1.0
            hg_.append(mb_)
            print(f"    [growth] {_lab(cfg,dom_id)}: best old head {mb_:.2f} bits on the first batch (median of earlier new domains "
                  +(f"{med_:.2f}" if med_ is not None else "-")+f") -> {'ALL '+str(cfg.pu_k) if grow_ else str(cfg.pu_k_base)} private units per block")
        if step==0 and cfg.film_k>0: model.__dict__['_film_frozen']=model.film_codes.detach().clone()
        xf_,yf_=x,y                                                                         # full real batch (counts see every row)
        if cfg.sleep_keep>0 and not (cfg.sleep_revisit and model.__dict__.get('_first_visit',True)):   # WAKE / SLEEP: queue the unexplained rows, train them merged
            SQ_=model.__dict__.setdefault('_slq',[])
            if step==0: SQ_.clear(); model.__dict__['_slst']=[0,0,0]                    # training steps, rows trained, rows left to the fast parts
            else:
                B_=x.shape[0]; k_=max(1,int(round(B_*cfg.sleep_keep)))
                if k_<B_:
                    hs0_=None
                    if cfg.sleep_random:
                        rs_=model.__dict__.setdefault('_slrng',np.random.RandomState(cfg.seed+505))
                        ixk_=torch.as_tensor(np.sort(rs_.choice(B_,k_,replace=False)),device=x.device)
                    else:
                        with torch.no_grad():
                            was_=model.training; model.eval(); fs_=model.feat_scale.clone()
                            hs0_=model.features(x)
                            zs_=model.heads[dom_id](priv(model,x,hs0_,dom_id))
                            lr_row=F.cross_entropy(zs_.reshape(-1,256).float(),y.reshape(-1),reduction='none').view(B_,-1).mean(1)
                            if cfg.sleep_score=='mem':                          # MEMORY GAIN on the second half (context vs none)
                                Th_=x.shape[1]//2; x2_=x[:,Th_:]; h2_=model.features(x2_)
                                z2_=model.heads[dom_id](priv(model,x2_,h2_,dom_id))
                                c0_=F.cross_entropy(z2_.reshape(-1,256).float(),y[:,Th_:].reshape(-1),reduction='none').view(B_,-1).mean(1)
                                c1_=F.cross_entropy(zs_[:,Th_:].reshape(-1,256).float(),y[:,Th_:].reshape(-1),reduction='none').view(B_,-1).mean(1)
                                lr_row=c0_-c1_
                            model.feat_scale.copy_(fs_); model.train(was_)
                        ixk_=torch.sort(torch.topk(lr_row,k_).indices).values
                    msk_=torch.ones(B_,dtype=torch.bool,device=x.device); msk_[ixk_]=False
                    if cfg.sleep_wake and (cfg.ngp or cfg.fcp_tables):                  # rows left to the fast parts: private rows learn now
                        xw_,yw_=x[msk_],y[msk_]
                        if hs0_ is None:
                            with torch.no_grad():
                                was_=model.training; model.eval(); fs_=model.feat_scale.clone()
                                hw_=model.features(xw_); model.feat_scale.copy_(fs_); model.train(was_)
                        else: hw_=hs0_[msk_]
                        opt.zero_grad()
                        with torch.enable_grad():
                            lk_=F.cross_entropy(model.heads[dom_id].detached(priv(model,xw_,hw_.detach(),dom_id)).reshape(-1,256),yw_.reshape(-1)); lk_.backward()
                        ngp_step(model,cfg,opt.param_groups[0]['lr'])
                    model._slst[2]+=int(msk_.sum())
                    x,y=x[ixk_],y[ixk_]
                SQ_.append((x,y))
                nq_=sum(a_.shape[0] for a_,_ in SQ_); need_=int(cfg.sleep_merge*cfg.batch_size)
                if nq_<need_ and step<_L-1: continue                                    # still awake: no trunk step
                xs_=torch.cat([a_ for a_,_ in SQ_]); ys_=torch.cat([b_ for _,b_ in SQ_]); SQ_.clear()
                if step<_L-1 and xs_.shape[0]>need_: SQ_.append((xs_[need_:],ys_[need_:])); xs_,ys_=xs_[:need_],ys_[:need_]
                x,y=xs_,ys_; G_=x.shape[0]/cfg.batch_size
                mu_=cfg.sleep_lr if cfg.sleep_lr>0 else math.sqrt(G_)
                if abs(mu_-1.0)>1e-9: model.__dict__['_slmul']=mu_
                if cfg.sleep_dream and abs(G_-1.0)>1e-9:
                    model.__dict__['_sldb']=cfg.dream_batch; cfg.dream_batch=max(1,int(round(cfg.dream_batch*G_)))
                model._slst[0]+=1; model._slst[1]+=x.shape[0]
            if step==_L-1: print(f"    [sleep] {model._slst[0]} trunk steps on {model._slst[1]} rows (merged {cfg.sleep_merge} x {cfg.batch_size}); "
                                 f"{model._slst[2]} rows left to the fast parts")
        if cfg.pmuon and step==0:
            model.__dict__['_kf_dacc']={}
            if float(cfg.kf_dream)>=1.0 and getattr(cfg,'kf_drop_a',True) and n_old>0:   # v9.8 (g3): live factors from the first step
                with torch.no_grad(): lv0_=kf_live_warmup(model,cfg,olds)
                kf_build(model,cfg,dom_id,live=lv0_); model.__dict__['_kf_dacc']={}
            else: kf_build(model,cfg,dom_id)
        if cfg.pmuon and cfg.kf_dream>0 and n_old>0 and step>0 and step%max(1,int(cfg.kf_dream_every))==0 and model.__dict__.get('_kf_dacc'):
            with torch.no_grad():                                                     # LIVE KEYS: whitening rebuilt from the dream rows' current inputs
                lv_={k_:e_['A']/max(1,e_['n']) for k_,e_ in model._kf_dacc.items() if e_['A'] is not None}
                kf_build(model,cfg,dom_id,live=lv_,quiet=True)
                model.__dict__['_kf_drb']=model.__dict__.get('_kf_drb',0)+1
        _kfm_=bool(cfg.pmuon and cfg.kf_source=='main' and step%cfg.kf_every==cfg.kf_every//2)
        if cfg.noise_diag>0 and n_old>0 and step%cfg.noise_diag==cfg.noise_diag//2: noise_diag(model,cfg,x,y,dom_id,olds)
        if cfg.head_select_eval or cfg.fwt_zeroshot or cfg.block_attr or cfg.block_auto: head_stats_update(model, x, dom_id, step, cfg)
        lv=model.__dict__.setdefault('_live',[]); lv.append((x,y)); del lv[:-cfg.buffer_batches]   # transient, current domain only
        model._cov_on=(step%cfg.cov_every==0) and not (cfg.cov_skip and not (cfg.orth_gate or cfg.pg_mode or cfg.og_common_always or cfg.polar_diag))
        if cfg.topic_diag: model._topics.update(x, dom_id)
        rb=model.__dict__.setdefault('_recent',[]); rb.append((x,y)); del rb[:-cfg.shunt_batches]
        orth=getattr(model,'_orth',None)
        model._keep_h=bool(cfg.head_select_eval or cfg.fwt_zeroshot)
        if orth is not None and step==0: orth.begin(model, prior_buffer, (x,y), dom_id)
        prog=step/_L
        f_=min(1.0,(step+1)/max(1,cfg.lr_warmup)) if (n_old==0 and model.__dict__.get('_first_visit',True)) else 1.0
        for g_ in opt.param_groups: g_['lr']=cfg.lr*f_*model.__dict__.get('_pass_lr',1.0)*model.__dict__.get('_dsm',1.0)
        if model.__dict__.get('_slmul') is not None:                               # WAKE / SLEEP: merged step's lr (undone after the step)
            for g_ in opt.param_groups: g_['lr']*=model._slmul
        _snrm_=1.0
        if n_old>0 and meta is not None and step % cfg.meta_every == 0 and len(prior_buffer)>0:
            # PROPER UNROLLED META-LEARNING (MAML-style): take a VIRTUAL gated step
            # that stays in the computation graph (differentiable w.r.t. the meta
            # params), measure buffer-loss (forgetting) AFTER the step, and backprop
            # THROUGH the step to the meta-controller. This gives a CLEAN gradient
            # about the gate's effect on forgetting (unlike the noisy FD / invalid
            # proxy that failed). meta learns: "gate so that a step here doesn't
            # raise loss on prior tasks, while keeping current loss low."
            meta_opt.zero_grad()
            # 1. compute current-task grads w.r.t. model params (graph-connected)
            cur_loss=F.cross_entropy(model(x,dom_id).reshape(-1,256),y.reshape(-1))
            block_params=[p for b in model.blocks for p in b.parameters() if p.requires_grad]
            grads=torch.autograd.grad(cur_loss, block_params, create_graph=True, allow_unused=True)
            # 2. build per-unit gates from meta-controller (differentiable), scale grads,
            #    form VIRTUAL updated params: p' = p - lr * gate * grad
            gated=_virtual_gated_step(model, meta, block_params, grads, prog, cfg.lr)
            # 3. measure forgetting: buffer-loss under the VIRTUAL params (functional fwd)
            bx,by,bd=prior_buffer[np.random.randint(len(prior_buffer))]
            buf_loss=_functional_buffer_loss(model, gated, bx, by, bd)
            # 4. plasticity: current-task loss under virtual params (should stay low)
            plast_loss=_functional_buffer_loss(model, gated, x, y, dom_id)
            # 5. meta-objective: low forgetting (buf) + low current loss (plasticity)
            meta_loss = buf_loss + cfg.meta_lambda*plast_loss
            meta_loss.backward()
            meta_opt.step()
        # REAL step: recompute grads, gate with the (updated) meta, step
        if (cfg.borrow_w>0 and cfg.borrow_mode=='grad' and n_old>0 and step%cfg.borrow_check==0 and getattr(model,'_dsq',None)
                and getattr(model,'_teach',None) is not None and (not cfg.borrow_first or model.__dict__.get('_first_visit',True))):
            borrow_align(model,cfg,olds,dom_id,x,y)
        opt.zero_grad()
        model.__dict__.pop('_ng_touch',None)                          # lazy n-gram optimiser: rows touched by THIS step only
        _tmark(model,cfg,step,'prep')
        if cfg.shunt and cfg.shunt_consol=='continuous' and n_old>0 and (getattr(model,'_shunt',None) or (cfg.shunt_lazy and model.__dict__.get('_vstart'))):
            FT=getattr(model,'_teach',None)
            if FT is not None:      # FROZEN TEACHERS: same targets as below without weight swaps; both teachers in ONE stacked pass
                with torch.no_grad():
                    rl_=[teacher_role(model,cfg,d,olds,FT) for d in olds]; grp={}
                    for i_,r_ in enumerate(rl_): grp.setdefault(r_,[]).append(i_)
                    roles=list(grp)
                    lwf_on=(step%max(1,cfg.lwf_every)==0) and cfg.shunt_old_weight>0   # B2: every lwf_every steps; weight 0: never (no teacher pass)
                    hh=dict(zip(roles,FT.features_multi(x,roles))) if lwf_on else {}
                    if not lwf_on or cfg.lwf_metric: tgt=None                   # B1: no per-head targets (Fisher metric instead)
                    elif getattr(model,'_hstk',None) is not None:               # FUSED: all old heads in one matmul per teacher
                        tgt=torch.empty(n_old,*x.shape,cfg.vocab,device=x.device)
                        for r_,ix in grp.items(): tgt[ix]=heads_logp(model,hh[r_],ix,cfg.shunt_T)
                    else: tgt=[F.log_softmax(model.heads[d](hh[rl_[i_]])/cfg.shunt_T,-1) for i_,d in enumerate(olds)]
                    if cfg.dream and getattr(model,'_dreams',None):               # a dream batch from one old domain
                        sh_=dream_shares(cfg, n_old)
                        dd_=olds[step%n_old] if sh_ is None else olds[dream_pick(model, cfg, n_old, sh_, 0)]
                        pool=model._dreams[dd_]
                        if not hasattr(model,'_dream_rng'): object.__setattr__(model,'_dream_rng',np.random.RandomState(cfg.seed+7))
                        ix_=model._dream_rng.randint(len(pool),size=cfg.dream_batch)
                        xd=torch.stack([pool[i] for i in ix_])
                        tp_=model.__dict__.get('_dream_tgt',{}).get(dd_)
                        tgt_d=torch.stack([tp_[i] for i in ix_]) if (tp_ and len(tp_)==len(pool)) else None   # CACHED dream targets
                        if tgt_d is None: tgt_d=F.log_softmax(model.heads[dd_](FT.features(xd,teacher_role(model,cfg,dd_,olds,FT)))/cfg.shunt_T,-1)
                        op_=None
                        model.__dict__['_op_n']=0
                        if op_ is not None and len(op_[0]):                    # ON-POLICY share of the batch (live samples, teacher targets)
                            n_on=max(1,min(cfg.dream_batch-1,int(round(cfg.dream_onpolicy*cfg.dream_batch))))
                            if not hasattr(model,'_onpol_pick'):
                                g_=torch.Generator(device=x.device); g_.manual_seed(cfg.seed+31); object.__setattr__(model,'_onpol_pick',g_)
                            io_=torch.randint(len(op_[0]),(n_on,),generator=model._onpol_pick,device=x.device)
                            xd=torch.cat([xd[:cfg.dream_batch-n_on],op_[0][io_]]); tgt_d=torch.cat([tgt_d[:cfg.dream_batch-n_on],op_[1][io_]])
                            model.__dict__['_op_n']=n_on; model.__dict__.pop('_prio_ix',None)
            else:
                # CONTINUOUS CONSOLIDATION: old memories read live from the MAIN weights (weights minus shunt);
                # the full network's old-domain outputs on this batch are kept matching them
                Pm={id(p_):p_ for p_ in model.parameters()}; Sd={k:v for k,v in model._shunt.items() if k in Pm and Pm[k].shape==v.shape}
                with torch.no_grad():
                    wasm=model.training; model.eval(); fsm=model.feat_scale.clone()
                    _Pl=[Pm[k] for k in Sd]; _Sl=list(Sd.values())
                    if _Pl: torch._foreach_sub_(_Pl,_Sl)                                  # main weights (fused)
                    hm=model.features(x); tgt=[F.log_softmax(model.heads[d](hm)/cfg.shunt_T,-1) for d in olds]
                    if cfg.dream and getattr(model,'_dreams',None):               # a dream batch from one old domain
                        sh_=dream_shares(cfg, n_old)
                        dd_=olds[step%n_old] if sh_ is None else olds[dream_pick(model, cfg, n_old, sh_, 0)]
                        pool=model._dreams[dd_]
                        if not hasattr(model,'_dream_rng'): object.__setattr__(model,'_dream_rng',np.random.RandomState(cfg.seed+7))
                        xd=torch.stack([pool[i] for i in model._dream_rng.randint(len(pool),size=cfg.dream_batch)])
                        tgt_d=F.log_softmax(model.heads[dd_](model.features(xd))/cfg.shunt_T,-1)
                    if _Pl: torch._foreach_add_(_Pl,_Sl)
                    model.feat_scale.copy_(fsm); model.train(wasm)
            pre_=None
            _dk0_=None
            _dk0_=int(model.__dict__.get('_dream_k',max(1,cfg.dream_every)))
            if model.__dict__.get('_sleep_phase')=='rem': _dk0_=1                         # REM: dreams on every step
            if step==0 and (cfg.sleep_eps>0 or cfg.sleep_rel>0): model.__dict__['_dream_k']=1; _dk0_=1   #   (they used to ride along in the
            if (cfg.merge_dream_fwd and cfg.dream_stream and getattr(model,'_dsq',None) and not cfg.share_blocks and cfg.strand_anchor<=0    #   merged pass
                    and step%max(1,cfg.dream_every)==0 and not cfg.dream_reuse and not cfg.proj_nodream          #   and its backward with no loss)
                    and (_dk0_ is None or step%_dk0_==0)):
                sbp_=stream_batch(model,cfg,olds)                                     # A6: real + dream rows in ONE forward pass
                if sbp_ is not None and sbp_[0].shape[1]==x.shape[1]:
                    ns_p=sbp_[3]['n'] if sbp_[3] is not None else 0
                    rtp_=_dream_route_rows(cfg,sbp_[0].shape[0]-ns_p,ns_p,x.device,len(model.blocks))
                    if rtp_ is not None: rtp_=torch.cat([torch.ones(x.shape[0],rtp_.shape[1],device=x.device),rtp_])
                    model.__dict__['_s0_dom']=[dom_id]*x.shape[0]+list(sbp_[2])+[None]*(sbp_[0].shape[0]-len(sbp_[2]))
                    kfd_=bool(cfg.kf_dream>0 and cfg.pmuon and model.__dict__.get('_dcount',0)%max(1,int(cfg.kf_dream_stride))==0)
                    if _kfm_ or kfd_:
                        kf_main_on(model,cfg,x.shape[0]); _KF['dream']=kfd_; _KF['real_off']=not _kfm_; _KF['dacc']=model.__dict__.setdefault('_kf_dacc',{})
                    _carry_on(cfg,x.shape[0])
                    if cfg.dream_pair>0: _pair_on(model,cfg,x.shape[0])                         # PAIRED-DREAM EPISODES
                    try: Hall=model.features(torch.cat([x,sbp_[0]]),route=rtp_,nstat=x.shape[0])
                    finally: _carry_off(cfg)
                    _KF['on']=False; _KF['dream']=False; _KF['real_off']=False
                    model.__dict__['_s0_dom']=None
                    h_=Hall[:x.shape[0]]; pre_=(sbp_,Hall[x.shape[0]:])
                elif sbp_ is not None: pre_=(sbp_,None)
            if pre_ is None or pre_[1] is None:
                model.__dict__['_s0_dom']=dom_id
                if _kfm_: kf_main_on(model,cfg,x.shape[0])
                _carry_on(cfg,x.shape[0])
                _nr_=bool(cfg.cd_skip_real and cfg.cd_w>0 and float(model.__dict__.get('_real_w',1.0))==0.0 and not _kfm_)
                try:
                    with _rf('real_forward'):
                        if _nr_:
                            with torch.no_grad(): h_=model.features(x)                # (its loss is weighted 0: no trunk backward)
                        else: h_=model.features(x)
                finally: _carry_off(cfg)
                model.__dict__['_s0_dom']=None; _KF['on']=False
            if cfg.stream:                                                                  # change detection: this batch's loss BEFORE the head refit
                with torch.no_grad(): model.__dict__['_rl']=F.cross_entropy(model.heads[dom_id](priv(model,x,h_.detach(),dom_id)).float().reshape(-1,256),y.reshape(-1))
                if cfg.stream_rep: _srep_note(model,h_)
            if cfg.cdet_log: _cdet_step(model,cfg,x,y,h_,dom_id,step)                       # v9.8 cg8 change-detection log (no training effect)
            if cfg.vp_nreal or float(model.__dict__.get('_real_w',1.0))>0: varpro_refit(model,cfg,opt,x,y,h_,dom_id)   # (vp_nreal False: no label leak)
            zc_=model.heads[dom_id](priv(model,x,h_,dom_id))
            loss=F.cross_entropy(zc_.reshape(-1,256),y.reshape(-1))
            _rw_=float(model.__dict__.get('_real_w',1.0))
            if _rw_!=1.0: loss=loss*_rw_                                                     # REAL-TEXT LOSS WEIGHT (real_w_pass)
            klm=0.0
            fk_=bool(cfg.fast_kl and getattr(model,'_hstk',None) is not None and cfg.shunt_kl_mode=='fixed')
            skip_lwf_=False
            if FT is not None and (not lwf_on or cfg.lwf_metric):
                skip_lwf_=True; fk_=False
                if lwf_on:                                                           # B1: FISHER-METRIC KL (all old heads in one metric)
                    Gd_,Gr_=model._lwfG
                    for r_,ix in grp.items():
                        dl_=(h_-hh[r_]).float(); klm=klm+0.5*((dl_@Gr_[r_])*dl_).sum(-1).mean()
                    if step%cfg.stat_every==0:                                       # per-domain values for the log (sampled)
                        with torch.no_grad():
                            kd_=torch.stack([0.5*(((h_-hh[rl_[i_]]).float()@Gd_[i_])*(h_-hh[rl_[i_]]).float()).sum(-1).mean() for i_ in range(n_old)])
                            ka_=model.__dict__.get('_klacc'); model.__dict__['_klacc']=(kd_ if ka_ is None else ka_[0]+kd_, 1 if ka_ is None else ka_[1]+1)
            pw_=None
            if cfg.lwf_posterior and n_old>0 and not skip_lwf_:                       # POSTERIOR WEIGHTS for the current-batch KL
                with torch.no_grad():
                    tgs_=tgt if torch.is_tensor(tgt) else torch.stack(list(tgt))
                    llo_=tgs_.float().gather(-1,y.unsqueeze(0).unsqueeze(-1).expand(tgs_.shape[0],-1,-1,1)).squeeze(-1)   # (n_old,B,T)
                    lln_=F.log_softmax(model.heads[dom_id].detached(h_.detach()).float(),-1).gather(-1,y.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
                    LL_=torch.cat([llo_,lln_]); po_=F.softmax((torch.cumsum(LL_,-1)-LL_)*cfg.hs_temp,0)              # causal: bytes before t
                    pw_=cfg.lwf_post_floor+(1-cfg.lwf_post_floor)*(n_old+1)*po_[:n_old]
                    wa_=model.__dict__.get('_lwfwacc'); m_=pw_.mean((1,2)); pn_=po_[n_old].mean()
                    model.__dict__['_lwfwacc']=(m_,pn_,1) if wa_ is None else (wa_[0]+m_,wa_[1]+pn_,wa_[2]+1)
                    rel_=(po_[:n_old].mean((1,2))*(n_old+1)).clamp(max=1.0)
                    R_=model.__dict__.get('_borrow_r'); oi_=torch.tensor(olds,device=y.device)
                    if R_ is None or step<=1:
                        R_=model.__dict__['_borrow_r']=torch.zeros(len(PILE_DOMAINS),device=y.device); R_[oi_]=rel_
                    else: R_[oi_]=0.9*R_[oi_]+0.1*rel_
            if fk_:                                                                  # FUSED KL over every old head at once
                tg_=tgt if torch.is_tensor(tgt) else torch.stack(tgt)
                lt=heads_logp(model,h_,tuple(range(n_old)),cfg.shunt_T)
                kt_=(tg_.exp()*(tg_-lt)).sum(-1)                                     # (n_old,B,T)
                if pw_ is not None: kt_=kt_*pw_
                kd_=kt_.mean((1,2)); klm=kd_.sum()
                ka_=model.__dict__.get('_klacc'); model.__dict__['_klacc']=(kd_.detach() if ka_ is None else ka_[0]+kd_.detach(), 1 if ka_ is None else ka_[1]+1)
            for i_,d in (enumerate(olds) if not (fk_ or skip_lwf_) else []):
                hd=model.heads[d]; lt=F.log_softmax(hd.detached(h_)/cfg.shunt_T,-1)
                kt=(tgt[i_].exp()*(tgt[i_]-lt)).sum(-1)                                # per-token KL
                if pw_ is not None: kt=kt*pw_[i_]
                kd=kt.mean()
                kst=model.__dict__.setdefault('_klstat',{}).setdefault(d,[0.0,0.0,0])
                kst[1]+=cfg.shunt_old_weight
                kst[0]+=kt.mean().detach(); kst[2]+=1                    # tensor: no GPU sync per step
                klm=klm+kd
            loss=loss+cfg.shunt_old_weight*klm*(cfg.shunt_T**2)/n_old*(max(1,cfg.lwf_every) if FT is not None else 1)
            eb_=1.0+(cfg.dream_early_mult-1.0)*max(0.0,1.0-step/max(1,cfg.dream_early_steps))   # EARLY-VISIT dream boost
            _dterm=0.0
            if _dk0_ is not None: _dk_=_dk0_                                         # (decided before the merged forward)
            else:
                _dk_=int(model.__dict__.get('_dream_k',max(1,cfg.dream_every)))
                if step==0 and (cfg.sleep_eps>0 or cfg.sleep_rel>0): model.__dict__['_dream_k']=1; _dk_=1
                if model.__dict__.get('_sleep_phase')=='rem': _dk_=1
            if cfg.dream_stream and getattr(model,'_dsq',None) and step%_dk_==0 and not cfg.proj_nodream:   # CONTINUOUS STREAMS (every dream_every steps)
                sb_=pre_[0] if pre_ is not None else stream_batch(model,cfg,olds)
                if sb_ is not None:
                    xd,tgt_d,dm_,ex_=sb_
                    Bd=xd.shape[0]; ns_=ex_['n'] if ex_ is not None else 0; n1_=Bd-ns_; Pp=min(cfg.dream_stream_prompt,xd.shape[1]//2)
                    rt_=_dream_route_rows(cfg,n1_,ns_,xd.device,len(model.blocks))
                    shb_=None
                    if shb_:                                                                  # DREAM ONLY THE SIMILARITIES in these blocks
                        for li_,(U_,_,_) in shb_.items(): model.blocks[li_]._gproj=(U_,float(cfg.share_spec),float(cfg.share_keep))
                    anc_=bool(ns_ and cfg.strand_anchor>0 and getattr(model,'_sabasis',None))
                    if anc_:
                        for li_ in cfg.strand_anchor_blocks: model.blocks[li_]._cap=True
                    if pre_ is not None and pre_[1] is not None: hfd=pre_[1]              # A6: already computed with the real batch
                    else:
                        wasd=model.training; model.eval(); fsd=model.feat_scale.clone()
                        model.__dict__['_s0_dom']=list(dm_)+[None]*(xd.shape[0]-len(dm_))
                        _STF['dream_call']=True
                        with _rf('dream_forward'): hfd=model.features(xd,route=rt_)
                        _STF['dream_call']=False; model.__dict__['_s0_dom']=None
                        model.feat_scale.copy_(fsd); model.train(wasd)
                    if shb_:
                        for li_ in shb_: model.blocks[li_]._gproj=None
                    spans=[]; i0=0
                    for j in range(1,n1_+1):
                        if j==n1_ or dm_[j]!=dm_[i0]: spans.append((dm_[i0],i0,j)); i0=j
                    if ex_ is None:
                        hh_=hfd                 # OLD HEADS learn from their own dreams (option)
                        def _dz(d,a,b):
                            hq_=priv(model,xd[a:b],hh_[a:b],d)
                            z_=model.heads[d](hq_)
                            return z_
                        vv_=model.__dict__.get('_bank_vi'); vv_=vv_ if (vv_ is not None and spans and vv_[0].shape[0]==spans[-1][2]) else None
                        model.__dict__['_vpo_k']=int(_dk_)
                        varpro_old_refit(model,cfg,xd,hfd,spans,Pp,vv_,tgt_d)
                        ld=torch.cat([F.log_softmax(_dz(d,a,b)/cfg.shunt_T,-1) for d,a,b in spans])
                        R_b=model.__dict__.get('_borrow_g')
                        if cfg.borrow_w>0 and R_b is not None and (not cfg.borrow_first or model.__dict__.get('_first_visit',True)):   # FORWARD BORROWING
                            rr_=R_b[torch.tensor(dm_,device=xd.device)]
                            lc_=F.log_softmax(model.heads[dom_id](hfd).float()/cfg.shunt_T,-1)
                            kb_=(tgt_d.exp()*(tgt_d-lc_)).sum(-1)[:,Pp:].mean(-1)
                            loss=loss+cfg.borrow_w*_dk_*(kb_*rr_).mean()
                        vi_=model.__dict__.pop('_bank_vi',None)
                        if vi_ is not None and vi_[0].shape[0]==ld.shape[0]:                  # STORED TOP-k TARGETS: top-k bytes + one rest bin
                            ktk=_kl_topk(ld,vi_[0],vi_[1])[:,Pp:]
                        else: ktk=(tgt_d.exp()*(tgt_d-ld)).sum(-1)[:,Pp:]                     # no KL on the prompt
                        bw_=model.__dict__.pop('_bank_w',None)                               # importance weights of the bank rows (bank_select='is')
                        if bw_ is not None and bw_.shape[0]==ktk.shape[0]: ktk=ktk*bw_[:,None]
                        kr_=ktk.detach().mean(-1)
                        bri_=model.__dict__.pop('_bank_ri',None)
                        if bri_ is not None and len(bri_[1])==kr_.shape[0] and float(cfg.dream_prio)>0:   # cg16: each row's KL (no sync) for its priority
                            RK_=model.__dict__.setdefault('_rowkl',{}); BKp_=model.__dict__.get('_bank',{})
                            for d_p,a_p,b_p in spans:
                                if d_p not in BKp_: continue
                                if d_p not in RK_: RK_[d_p]=torch.full((BKp_[d_p].shape[0],),float('nan'),device=kr_.device)
                                RK_[d_p][torch.as_tensor(bri_[1][a_p:b_p],device=kr_.device)]=kr_[a_p:b_p].float()
                            pc_=model.__dict__.get('_rowkl_n',0)+1; model.__dict__['_rowkl_n']=pc_
                            if pc_%max(1,int(cfg.dream_prio_sync))==0:
                                model.__dict__['_rowkl_h']={k_p:v_p.cpu().numpy() for k_p,v_p in RK_.items()}
                        ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                        tab_=None
                        if tab_:                                                             # BYTE-TYPE IMPORTANCE WEIGHTS per dream token
                            wt_=torch.ones_like(ktk); cn_=_BYTE_CLS.to(xd.device)[xd[:,Pp+1:]]   # KL at t predicts byte t+1
                            for d,a,b in spans:
                                if d in tab_: wt_[a:b,:-1]=tab_[d][cn_[a:b]]
                            wt_=wt_/wt_.mean(); ktk=ktk*wt_
                        kdr=ktk.mean()*float(cfg.sleep_fixed)                            #   (no host sync, deterministic schedule)
                        km_=int(cfg.sleep_kmax)
                        if cfg.sleep_kmax_pass: km_=int(cfg.sleep_kmax_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.sleep_kmax_pass)-1)])
                        model.__dict__['_dream_k']=max(1,km_)
                        sc_=model.__dict__.setdefault('_sleepC',[0,0]); sc_[0]+=1
                    else:                                                                    # STRANDS in the batch: per-position heads and masks
                        T_=xd.shape[1]
                        hd_=torch.cat([torch.tensor(dm_,device=xd.device).view(-1,1).expand(-1,T_),ex_['hd']])
                        w1_=torch.ones(n1_,T_,device=xd.device); w1_[:,:Pp]=0.0; wm_=torch.cat([w1_,ex_['w']])
                        ktk=torch.zeros(Bd,T_,device=xd.device)
                        for d in sorted(set(hd_.unique().tolist())-{-1}):
                            m_=(hd_==d); rw_=m_.any(1).nonzero().squeeze(1)
                            ld_=F.log_softmax(model.heads[d].detached(hfd[rw_])/cfg.shunt_T,-1)
                            ktk=ktk.index_add(0,rw_,(tgt_d[rw_].exp()*(tgt_d[rw_]-ld_)).sum(-1)*m_[rw_])
                        for j_,pr_ in enumerate(ex_['pr']):                                  # AGREEMENT rows: KL(teacher product || live product)
                            if pr_ is None: continue
                            i_=n1_+j_; hh_=hfd[i_:i_+1]
                            lp_=F.log_softmax(0.5*F.log_softmax(model.heads[pr_[0]].detached(hh_)/cfg.shunt_T,-1)+0.5*F.log_softmax(model.heads[pr_[1]].detached(hh_)/cfg.shunt_T,-1),-1)
                            ka_=(tgt_d[i_:i_+1].exp()*(tgt_d[i_:i_+1]-lp_)).sum(-1)
                            ktk=ktk.index_add(0,torch.tensor([i_],device=xd.device),ka_)
                        kraw_=ktk.detach()
                        if any(k_==3 for k_ in ex_['kind']):
                            st0_=model.__dict__.setdefault('_strandstat',{'a':0.0,'b':0.0,'n':0,'anc':{}})
                            ag_=kraw_[n1_:]; wa_=ex_['w']; st0_['ag']=st0_.get('ag',0.0)+(ag_*wa_).sum()/wa_.sum().clamp(min=1); st0_['agn']=st0_.get('agn',0)+1
                        kr_=(kraw_*wm_).sum(-1)/wm_.sum(-1).clamp(min=1.0)
                        ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                        kdr=(ktk*wm_).sum()/wm_.sum().clamp(min=1.0)
                        st_=model.__dict__.setdefault('_strandstat',{'a':0.0,'b':0.0,'n':0,'anc':{}})
                        sg_=ex_['seg']
                        if bool((sg_==1).any()):
                            ks_=kraw_[n1_:]
                            st_['a']+=(ks_*(sg_==1)).sum()/(sg_==1).sum().clamp(min=1); st_['b']+=(ks_*(sg_==2)).sum()/(sg_==2).sum().clamp(min=1); st_['n']+=1
                        if anc_:                                                             # OLD-SUBSPACE ANCHORING on the strand rows
                            hts_=model._teach.features_taps(xd[n1_:],'main',tuple(cfg.strand_anchor_blocks)); aw_=ex_['aw']
                            for li_ in cfg.strand_anchor_blocks:
                                bk_=model.blocks[li_]; bk_._cap=False; hs_=bk_.__dict__.pop('_cap_out')[n1_:].float(); U_=model._sabasis[li_]
                                ds_=(hs_-hts_[li_])@U_; dt_=hts_[li_]@U_
                                fe_=((ds_**2).sum(-1)*aw_).sum()/((dt_**2).sum(-1)*aw_).sum().clamp(min=1e-8)
                                loss=loss+cfg.strand_anchor*fe_
                                a_=st_['anc'].setdefault(li_,[0.0,0,U_.shape[1]]); a_[0]+=fe_.detach(); a_[1]+=1; a_[2]=U_.shape[1]
                    for li_ in (cfg.strand_anchor_blocks if anc_ else ()):
                        model.blocks[li_]._cap=False; model.blocks[li_].__dict__.pop('_cap_out',None)
                    wdr=(cfg.dream_weight*model.__dict__.get('_dwp',1.0))*(1.0-cfg.dream_decay*step/_L)*eb_*(_dk_)
                    model.__dict__['_dk_last']=_dk_
                    model.__dict__['_dstep_k']=int(_dk_)            # this step carries dreams standing for k steps
                    loss=loss+wdr*kdr*(cfg.shunt_T**2)
                    for d,a,b in spans:
                        kd_=kr_[a:b].mean(); ds=model.__dict__.setdefault('_dreamstat',{}).setdefault(d,[0.0,0,0.0,0.0,0])
                        ds[0]+=kd_; ds[1]+=1; ds[2]+=wdr; ds[3]+=kd_; ds[4]+=(kd_>1.0).float()
                        dm=model.__dict__.setdefault('_dmax',{}); dm[d]=torch.maximum(dm[d],kd_) if d in dm else kd_
            if cfg.dream and getattr(model,'_dreams',None):
                wasd=model.training; model.eval(); fsd=model.feat_scale.clone()      # eval mode: dreams leave no statistics
                fa_=cfg.feat_anchor>0 and getattr(model,'_teach',None) is not None
                fab_=(cfg.feat_anchor_block,)
                if fa_:
                    for li_ in fab_: model.blocks[li_]._cap=True
                hdd=model.heads[dd_]; ld=F.log_softmax(hdd.detached(model.features(xd))/cfg.shunt_T,-1)
                if fa_:
                    hts_=model._teach.features_taps(xd,teacher_role(model,cfg,dd_,olds,model._teach),fab_)
                    bi_=min(cfg.dream_burnin,xd.shape[1]//2) if cfg.dream_burnin>0 else 0
                    fa_s=model.__dict__.setdefault('_fastat',{})
                    for li_ in fab_:
                        bk_=model.blocks[li_]; bk_._cap=False; hs_=bk_.__dict__.pop('_cap_out').float()[:,bi_:]; ht_=hts_[li_][:,bi_:]
                        fe_=((hs_-ht_)**2).mean()/(ht_.pow(2).mean()+1e-8)             # relative error of the block's output on dreams
                        loss=loss+cfg.feat_anchor*fe_
                        a_=fa_s.setdefault(li_,[0.0,0]); a_[0]+=fe_.detach(); a_[1]+=1
                model.feat_scale.copy_(fsd); model.train(wasd)
                ktk=(tgt_d.exp()*(tgt_d-ld)).sum(-1)                              # per-token KL (B,T)
                if cfg.dream_burnin>0: ktk=ktk[:,min(cfg.dream_burnin,ktk.shape[1]//2):]   # BURN-IN: no KL on the context-free first positions
                kraw_t=ktk.detach().mean()
                on_=model.__dict__.get('_op_n',0)
                if on_:                                                      # on-policy vs teacher-sampled dream KL (no sync)
                    os_=model.__dict__.setdefault('_opstat',[0.0,0.0,0]); os_[0]+=ktk.detach()[-on_:].mean(); os_[1]+=ktk.detach()[:-on_].mean(); os_[2]+=1
                pix_=model.__dict__.pop('_prio_ix',None)
                if pix_ is not None and pix_[0]==dd_: model._dscore[dd_][pix_[1]]=ktk.detach().mean(-1)   # new priorities (GPU)
                ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                kdr=ktk.mean()
                _sync=(step%cfg.stat_every==0) or cfg.dream_adapt or cfg.dream_rollback   # host values only when needed
                kraw=float(kraw_t) if _sync else 0.0; kv=0.0
                wd_=model.__dict__.setdefault('_dworst',{})                         # worst dream batch this domain (diagnostic)
                if _sync and kraw>wd_.get(dd_,(0.0,))[0]:
                    with torch.no_grad():
                        sq=ktk.detach().mean(-1); i_=int(sq.argmax())
                        pr=[]
                        if kraw>1.0:
                            wasp=model.training; model.eval(); fsp=model.feat_scale.clone()
                            for b in model.blocks: b._probe=pr
                            hp=model.features(xd[i_:i_+1]); lg=model.heads[dd_](hp)
                            for b in model.blocks: b._probe=None
                            model.feat_scale.copy_(fsp); model.train(wasp)
                            pr.append((float(hp.abs().max()),float(lg.max()-lg.min())))
                    wd_[dd_]=(kraw,bytes(xd[i_].tolist()).decode('latin-1'),float(ld[i_].detach().min()),pr)
                ds=model.__dict__.setdefault('_dreamstat',{}).setdefault(dd_,[0.0,0,0.0,0.0,0]); ds[0]+=kraw_t; ds[1]+=1
                ds[3]+=kdr.detach(); ds[4]+=(kraw_t>1.0).float()
                dm=model.__dict__.setdefault('_dmax',{}); dm[dd_]=torch.maximum(dm[dd_],kraw_t) if dd_ in dm else kraw_t
                wdr=(cfg.dream_weight*model.__dict__.get('_dwp',1.0))*(1.0-cfg.dream_decay*step/_L)*eb_
                ds[2]+=wdr
                loss=loss+wdr*kdr*(cfg.shunt_T**2)
        else:
            if _kfm_: kf_main_on(model,cfg,x.shape[0])                         # Fisher factors of the first domain too (observation only)
            _carry_on(cfg,x.shape[0])                                          # CONTINUOUS CONTEXT on the first domain too
            try: loss=F.cross_entropy(model(x,dom_id).reshape(-1,256),y.reshape(-1))
            finally: _carry_off(cfg)
            _KF['on']=False
            if cfg.stream: model.__dict__['_rl']=loss.detach()                  # change detection (first domain: no head refit before the loss)
            if cfg.stream and cfg.stream_rep:                                  # cg15: the first domain's representation (one no-grad trunk pass)
                model.__dict__['_s0_dom']=dom_id
                try:
                    with torch.no_grad(): _srep_note(model,model.features(x))
                finally: model.__dict__['_s0_dom']=None
        if cfg.recon_w>0 and float(model.__dict__.get('_real_w',1.0))>0:             # REPLAY TRAINING (only with real-text gradient on)
            loss=loss+cfg.recon_w*(recon_loss_mark(model,cfg,x,dom_id) if cfg.recon_mark else recon_loss(model,cfg,x,y,dom_id))
            if step==_L-1 and model.__dict__.get('_rc_stat'):
                st_=model.__dict__.pop('_rc_stat'); nb_=model.__dict__.pop('_rc_bad',0)
                print(f"    [replay training] mean loss of a window regenerated from its own memory {st_[0]/max(1,st_[1])/LOG2:.4f} BPC"+(f" | {nb_} non-finite steps skipped" if nb_ else "")+_rs_msg())
        if cfg.cd_w>0 and float(model.__dict__.get('_real_w',1.0))==0.0:            # CONDUIT REPLAY (Stage C)
            if cfg.cd_teacher=='live': model.__dict__.pop('_cd_P',None)              # LIVE TEACHER: no copy - the current weights read the traces
            elif (step==0 and not cfg.cd_fixed) or '_cd_P' not in model.__dict__: model.__dict__['_cd_P']={k_:p_.detach().clone() for k_,p_ in model.named_parameters()}
            _cdf_=_cd_lrf(cfg,x); model.__dict__['_cdf']=_cdf_                          # (cd_lr: conduit update scale)
            loss=loss+cfg.cd_w*(_cdf_ if model.__dict__.get('_dstep_k') is not None else 1.0)*cd_loss(model,cfg,x,dom_id)
            if step==_L-1:
                st_=model.__dict__.pop('_cd_stat',None)
                if not cfg.cd_fixed: model.__dict__.pop('_cd_P',None)
                model.__dict__.pop('_cd_cache',None)
                if _CDR[1] and (cfg.cd_reuse>1 or cfg.cd_lr_teach): print(f"    [v9] replays reused on {_CDR[0]} of {_CDR[1]} conduit steps"+(f" | mean step scale (teaching share) {_CDR[2]/max(1,_CDR[3]):.3f}" if _CDR[3] else ""))
                _CDR[:]=[0,0,0.0,0]
                if _CDT['n']:
                    a_=_CDT['acc']; print("    [conduit timing] ms per conduit step (sampled, synchronised): "+" | ".join(f"{k_} {1000*v_/_CDT['n']:.0f}" for k_,v_ in a_.items() if k_!='_start')+f" | total {1000*sum(v_ for k_,v_ in a_.items() if k_!='_start')/_CDT['n']:.0f}"+_cdg_msg())
                    _CDT.update(acc={},n=0)
                RVp_=model.__dict__.get('_nres')
                if RVp_ and RVp_.get('seen') and int(cfg.nrem_res)>0:
                    print(f"    [NREM v1] reservoir {len(RVp_['e'])}/{cfg.nrem_res} windows of {RVp_['seen']} seen | sleep: {_SLP['n'][0]} NREM steps, {RVp_['rep']} re-teachings, {RVp_['ret']} retired as consolidated | "
                          f"mean remaining signal {np.mean([e_['v'] for e_ in RVp_['e']]) if RVp_['e'] else 0:.3f} (at storage {np.mean([e_['v0'] for e_ in RVp_['e']]) if RVp_['e'] else 0:.3f})"
                          +(f" || cross-domain: {RVp_.get('rep_o',0)} re-teachings of other domains' windows, {RVp_.get('ret_o',0)} retired; kept samples "+", ".join(f"domain {d_}: {len(l_)}" for d_,l_ in sorted(model.__dict__.get('_nres_bank',{}).items())) if float(cfg.nrem_mix)>0 else ""))
                SBp_=model.__dict__.get('_sbuf')
                if SBp_ and SBp_['st'].get('seen'):
                    S_=SBp_['st']; E_=SBp_['e']; bd_={}
                    for e_ in E_: bd_[e_['dom']]=bd_.get(e_['dom'],0)+1
                    print(f"    [long-term store] size {len(E_)}/{cfg.cd_store_cap} | admitted {S_.get('adm',0)} by signal + {S_.get('admr',0)} at random of {S_.get('seen',0)} windows | "
                          f"re-checks {S_.get('chk',0)} ({S_.get('chk_old',0)} of other domains' windows) | retired as consolidated {S_.get('ret',0)}"
                          f" (after {S_.get('ret_n',0)/max(1,S_.get('ret',0)):.1f} re-checks) | evicted {S_.get('evict',0)} | mean signal of stored {np.mean([e_['v'] for e_ in E_]) if E_ else 0:.3f}"
                          f" | by domain "+" ".join(f"{_lab(cfg,d_)}:{c_}" for d_,c_ in sorted(bd_.items()))
                          +(f" || sleep: NREM {_SLP['n'][0]} steps ({S_.get('nrem',0)} re-replays, {S_.get('nrem_ret',0)} retired), REM {_SLP['n'][1]} steps (dreams only)" if sum(_SLP['n']) else ""))
                    SMs_=model.__dict__.pop('_semstat',None); SMb_=model.__dict__.get('_sem',{})
                    if float(cfg.sem_share)>0: print(f"    [semantic rows] written this visit {S_.get('sem',0)} | per domain "+" ".join(f"{_lab(cfg,d_)}:{len(v_['x'])}" for d_,v_ in sorted(SMb_.items()))
                                                     +(f" | dream rows that were semantic {SMs_[0]}/{SMs_[1]}" if SMs_ else " | none used yet"))
                    SBp_['st']={}
                LBp_=model.__dict__.get('_lbuf')
                if LBp_ and LBp_.get('st') and LBp_['st'][0]>0:
                    S_=LBp_['st']; print(f"    [maintenance loop] {S_[0]} re-checks of stored windows | easy (interval x2) {S_[1]}, degrading (re-taught, interval halved) {S_[2]}, "
                                         f"dropped after two failures {S_[3]}, dropped as consolidated {S_[4]} | mean degradation {S_[5]/S_[0]:.4f} | stored now {len(LBp_['e'])}")
                    LBp_['st']=[0,0,0,0,0,0.0]
                RUp_=model.__dict__.pop('_rustat',None)
                if RUp_: print(f"    [replay reuse] {RUp_[0]} windows reused ({cfg.cd_reuse_mode}, lag {cfg.cd_reuse_lag}) | fragile-piece share: reused {float(RUp_[1])/max(1,RUp_[0]):.3f}, eligible pool {float(RUp_[2])/max(1,RUp_[0]):.3f}")
                RSp_=model.__dict__.pop('_rsig',None)
                if RSp_:
                    A_=torch.cat(RSp_,1).float(); kq_,cq_,nq_=A_[0],A_[1],A_[2]; sig_=kq_*nq_; tot_=float(sig_.sum()); N_=kq_.numel()
                    qs_=[float(v_) for v_ in torch.quantile(kq_[:min(N_,200000)],torch.tensor([0.1,0.5,0.9],device=kq_.device))]
                    srt_=torch.sort(sig_).values; cum_=srt_.cumsum(0)
                    bot50_=float(cum_[N_//2-1])/max(tot_,1e-12) if N_>1 else 0.0; top20_=1.0-float(cum_[int(N_*0.8)-1])/max(tot_,1e-12) if N_>4 else 1.0
                    und_=' '.join(f"<{t_}:{float((kq_<t_).float().mean()):.2f}" for t_ in (0.02,0.05,0.1,0.2))
                    gs_=[]
                    for f_ in (0.5,0.25):                                       # keep only the most surprising f_ of pieces: share of the signal kept
                        k_=max(1,int(N_*f_)); ix_=torch.topk(cq_,k_).indices; gs_.append(float(sig_[ix_].sum())/max(tot_,1e-12))
                    km_=kq_-kq_.mean(); cm_=cq_-cq_.mean(); r_=float((km_*cm_).sum()/((km_.norm()*cm_.norm()).clamp_min(1e-12)))
                    print(f"    [replay signal] {N_} pieces sampled | KL per piece p10 {qs_[0]:.3f} p50 {qs_[1]:.3f} p90 {qs_[2]:.3f} | share of pieces {und_} | "
                          f"signal share: bottom 50% of pieces {bot50_:.3f}, top 20% {top20_:.3f} | keep the most surprising 50% / 25% of pieces -> {gs_[0]:.2f} / {gs_[1]:.2f} of the signal | "
                          f"corr(signal, surprise at experience) {r_:+.2f}")
                GSp_=model.__dict__.pop('_gsel',None)
                GKp_=model.__dict__.pop('_gkl',None)
                GPp_=model.__dict__.pop('_gprot',None)
                GRp_=model.__dict__.pop('_grho',None)
                SMp_=model.__dict__.pop('_simstat',None)
                if SMp_ and SMp_[0]>0: print(f"    [similar dreams] {cfg.dream_sim}: {SMp_[0]} rows chosen among {cfg.dream_sim_k} candidates | share of candidates the current step would harm {SMp_[1]/SMp_[0]:.2f} | chosen row changed {SMp_[3]/SMp_[0]:.2f}")
                if GRp_ and GRp_[0]>0: print(f"    [reducible gate] {GRp_[0]} calls | overlap with the surprise selection {float(GRp_[1])/GRp_[0]:.2f} | share of pieces the trace predicts WORSE than the weights (stale teacher) {float(GRp_[2])/GRp_[0]:.2f}")
                if GPp_ and GPp_[0]>0: print(f"    [protected gate] {GPp_[0]} calls | overlap with the KL selection {GPp_[1]/GPp_[0]:.2f}, with the surprise selection {GPp_[2]/GPp_[0]:.2f} | context novelty of the kept pieces x{GPp_[3]/GPp_[0]:.2f} the mean")
                if GKp_ and GKp_[0]>0: print(f"    [KL gate] {GKp_[0]} calls | overlap with the surprise selection {GKp_[1]/GKp_[0]:.2f} | share of the KL signal the surprise selection would keep {GKp_[2]/GKp_[0]:.2f}")
                if GSp_ and GSp_[0]>0: print(f"    [diversity gate] {GSp_[0]} calls | overlap with the surprise selection {GSp_[1]/GSp_[0]:.2f} | gradient energy kept: diversity {GSp_[2]/GSp_[0]:.2f}, surprise would keep {GSp_[3]/GSp_[0]:.2f}")
                EVp_=model.__dict__.pop('_evid',None)
                if EVp_ is not None: print(f"    [evidence targets] teacher's top byte differed from the verified replayed byte at {float(EVp_[0])/max(1.0,float(EVp_[1])):.4f} of teaching positions (blend w {cfg.cd_evid_w}{', all positions' if cfg.cd_evid_all else ''})")
                if st_: k_=max(1,st_[1]); print(f"    [conduit replay] KL {st_[0]/k_:.4f} | exact segments {st_[2]/k_:.2f} | tries/segment {st_[3]/k_:.2f} | true byte fidelity {st_[4]/k_:.2f} (diagnostic) | exactness test agrees with truth {st_[6]/k_:.3f} | eps {st_[5]/k_:.1e} over {st_[1]} steps"
                               +(f" | replay training on exact replays {st_[7]/max(1,st_[8])/LOG2:.4f} BPC ({int(st_[8])} steps)" if len(st_)>8 and st_[8] else "")+_rs_msg())
                if _HINT['calls']:                                                 # cg12b hints
                    tb_=max(1,_HINT['tot']); wr_=float(_HINT['wrong'])
                    print(f"    [hints] {float(_HINT['bytes'])/tb_:.3f} of replayed bytes hinted ({float(_HINT['pieces'])/(tb_/max(1,int(cfg.cd_seg))):.2f} of pieces) | "
                          f"teacher-forced wrong bytes {wr_/tb_:.4f}, of them hinted {float(_HINT['wcov'])/max(1.0,wr_):.2f} | {_HINT['calls']} calls"
                          +(" | REPLAY FIDELITY ALARM: over 5% of bytes wrong with the trace (cg14c Gutenberg 16.9%)" if wr_/tb_>0.05 else ""))
                    _HINT.update(calls=0,bytes=0,tot=0,wrong=0,wcov=0,pieces=0)
                if FOKS['skip']:
                    fz_=_fok_dom(dom_id)
                    es_=(_fsum(FOKS['estn'])/max(1.0,_fsum(FOKS['estd']))) if FOKS.get('estd') else (_fm(FOKS['est']) if FOKS['est'] else 0.0)
                    tr_=(_fsum(FOKS['trn'])/max(1.0,_fsum(FOKS['trd']))) if FOKS.get('trd') else (_fm(FOKS['true']) if FOKS['true'] else 0.0)
                    print(f"    [confidence gate] checks skipped {_fm(FOKS['skip']):.2f} | wrongly accepted: estimated {es_:.4f} "
                          f"(random checks), true {tr_:.4f} (diagnostic) | tries per piece {_fm(FOKS['k']):.2f} | threshold {float(fz_['thr']):.2f} | running exact share {float(fz_['exr'] if fz_.get('exrT') is None else fz_['exrT']):.2f}")
                if FOKS['runs']: print(f"    [exact runs] share of positions that teach {float(torch.stack([torch.as_tensor(v_,dtype=torch.float32).cpu() for v_ in FOKS['runs']]).mean()):.3f} (exact share {st_[2]/k_:.2f})")
                if cfg.cd_repair and _V97['calls']:                                 # v9e report
                    pr_=_fsum(_V97['pn'])/max(1.0,_fsum(_V97['pd'])); ag_=_fsum(_V97['ag'])/max(1.0,_fsum(_V97['agd'])); nchk_=max(1,sum(_V97['chk']))
                    fz2_=_fok_dom(dom_id)
                    print(f"    [v9.7 recall] positions that teach (verified) {_fm(_V97['vp']) if _V97['vp'] else 0:.3f} | repaired {_V97['rep']/nchk_:.2f} of checked pieces, "
                          f"{_fsum(_V97['imp'])/max(1,_V97['rep']):.2f} of them improved | verified positions truly correct {pr_:.4f} | verified prefix = true prefix {ag_:.3f} (checked pieces)"
                          +(f" | Gram/key disagreements {int(_fsum(_V97['dis']))}" if _V97['dis'] else "")+f" | key tolerance {float(model.__dict__.get('_cd_ktol',cfg.cd_keytol)):.3f}"
                          +(f" | floor weight mean {np.mean(_V97['fw']):.2f}, skipped calls {_V97['skip']} of {_V97['calls']} (running verified share {float(fz2_['vps']) if fz2_.get('vps') is not None else float('nan'):.2f})" if float(cfg.cd_floor_hi)>float(cfg.cd_floor_lo) else ""))
                    for k3_ in _V97: _V97[k3_]=[] if isinstance(_V97[k3_],list) else 0
                if float(cfg.cd_buf)>0 and _CBUF['calls']:
                    print(f"    [v9.7 buffer] {_CBUF['mix']/_CBUF['calls']:.1f} buffered windows per conduit step | from: "+(" ".join(f"{dn_(d_) if callable(globals().get('dn_')) else d_}:{c_}" for d_,c_ in sorted(_CBUF['src'].items())) or "-")
                          +" | buffer now: "+" ".join(f"{d_}:{len(e_['x'])}" for d_,e_ in sorted(_CBUF['d'].items())))
                    _CBUF.update(mix=0,calls=0,src={})
                for k2_ in list(FOKS): FOKS[k2_]=[]
        if cfg.mk_dream>0 and (cfg.mk_always or float(model.__dict__.get('_real_w',1.0))==0.0):   # MEMORY-KEYED DREAMS
            if step%max(1,int(cfg.mk_every))==0: mk_refresh(model,cfg,x,dom_id)
            loss=loss+cfg.mk_w*mk_loss(model,cfg,dom_id)
            if step==_L-1 and model.__dict__.get('_mk_stat'):
                st_=model.__dict__.pop('_mk_stat'); print(f"    [memory-keyed dreams] mean KL(teacher || student) {st_[0]/max(1,st_[1]):.4f} over {st_[1]} steps, pool {len(model.__dict__.get('_mk_pool',[]))}")
            if step==_L-1: model.__dict__['_mk_pool']=[]
        if hasattr(model,'_klw_opt'): model._klw_opt.zero_grad()
        _dt=locals().get('_dterm',0.0)
        _pt=locals().get('_pterm',None); _pterm=None; _pg=None
        with _rf('backward'): loss.backward()
        if _pg is not None:
            with torch.no_grad():
                R_=model.__dict__.setdefault('_decR',{}); dl_=[]
                for p_,g_ in _pg:
                    if g_ is None: continue
                    if cfg.dec_mode=='move' and p_.grad is not None: p_.grad.sub_(g_)
                    r_=float(g_.float().pow(2).mean()); R_[id(p_)]=r_ if id(p_) not in R_ else 0.99*R_[id(p_)]+0.01*r_
                    dl_.append((p_,g_.float()/math.sqrt(max(R_[id(p_)],1e-30))))
                model.__dict__['_decstep']=dl_
        _tmark(model,cfg,step,'forward+backward')
        if hasattr(model,'_klw_opt') and model._klw.grad is not None: model._klw_opt.step()
        xg=None
        if getattr(model,'_xcur',None):                  # d loss / d gain = <grad of weight tensor, earlier domain's delta>
            P_=_pmap(model); X=model._xcur; xg=torch.zeros_like(X['A'])
            with torch.no_grad():
                for jj,j in enumerate(X['olds']):
                    for kk,k in enumerate(X['keys']):
                        v_=model._spec[j].get(k)
                        if v_ is not None and P_[k].grad is not None: xg[jj,kk]=float((P_[k].grad*v_).sum())
        _dh_ids=set()
        _dsk_=model.__dict__.pop('_dstep_k',None)                                       # dream step (the k steps it stands for) or None
        _cth_=1.0
        if _dsk_ is not None and cfg.dream_clip_pass:                                   # cg14b: DREAM-STEP CLIP THRESHOLD per pass (GPU: dream steps sat at
            _cth_=float(cfg.dream_clip_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.dream_clip_pass)-1)])   #   norm 1.7-6.2, clipped 100%)
        with _rf('grad_clip'): _gn_=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and id(p) not in _dh_ids],_cth_)
        if n_old>0:                                                                     # CLIP STATISTICS, dream vs other steps (on device: no sync)
            cs_=model.__dict__.setdefault('_clipst',{}); e_=cs_.setdefault('d' if _dsk_ is not None else 'r',[0.0,0.0,0])
            e_[0]=e_[0]+_gn_.detach().float(); e_[1]=e_[1]+(_gn_.detach().float()>_cth_).float(); e_[2]+=1
        if _dh_ids and hasattr(model,'draft_heads'): torch.nn.utils.clip_grad_norm_(list(model.draft_heads.parameters()),1.0)   # draft heads clipped on their own
        mg=None
        if n_old>0 and meta is not None:
            with torch.no_grad(): apply_meta_gate(model, meta, prog)
        _gopen_=(cfg.write_gate_open and mg is None and not cfg.route_writes and not cfg.shared_head and not cfg.block_write and not cfg.block_auto
                 and not cfg.surprise_mod and not cfg.dual_rate and not cfg.dream_rollback and not (cfg.shunt and cfg.shunt_mode=='subspace')
                 and not (cfg.shunt and cfg.shunt_mode=='all' and not cfg.shunt_lazy))
        _gneed_=not _gopen_ or (cfg.shunt and cfg.shunt_mode=='all' and cfg.shunt_lazy and model.__dict__.get('_vstart') is None)
        if (cfg.use_write_threshold or mg is not None) and n_old>=cfg.gate_start_domain and _gneed_:   # gates all open: the per-step snapshot /
                                                                                                        #   blend is a no-op - taken only when the lazy fast store needs its segment start
            with _rf('gate_snapshot'): snaps,elig=snapshot_threshold(model, cfg, mg)
        else:
            snaps,elig=[],1.0
        ngp_=None
        stv_=None
        pjv_=None
        mu_out_=muon_prep(model,cfg)
        tl_snap_=None
        hk_=None
        if cfg.head_dream_k and _dsk_ is not None and _dsk_>1:                           # OLD HEADS: Adam step x k on a dream step standing for k
            hk_=[(p_,p_.detach().clone()) for d_ in olds if d_!=dom_id for p_ in model.heads[d_].parameters() if p_.grad is not None]
        _cdf_=model.__dict__.pop('_cdf',None); cds_=None
        if _cdf_ is not None and _cdf_<1.0 and _dsk_ is None:                             # CONDUIT-ONLY step: the whole update scaled by cd_lr
            cds_=[(p_,p_.detach().clone()) for p_ in model.parameters() if p_.requires_grad]
        _dsc_=bool(_ds_on and step%max(1,int(cfg.dstep_every))==int(cfg.dstep_every)-1)
        if _dsc_:                                                                       # DIRECT STEP: held-out gradient BEFORE the update (same rows after)
            _t_ds=time.perf_counter(); _dsP=[p_ for p_ in model.parameters() if p_.requires_grad]
            _dsseed=1000003*step+17+97*int(dom_id); _dsg0,_dsJ0=_dstep_grad(model,cfg,dom_id,olds,_dsseed,_dsP); _dsth=[p_.detach().clone() for p_ in _dsP]
            _ds_t+=time.perf_counter()-_t_ds
        with _rf('adam_step'): opt.step()
        if hk_:
            with torch.no_grad():
                for p_,o_ in hk_: p_.add_(p_-o_,alpha=float(_dsk_)-1.0)
        if mu_out_: muon_apply(mu_out_,opt.param_groups[0]['lr']*cfg.muon_lr)
        if model.__dict__.get('_decstep'):                                             # DECOUPLED PROTECTION step (linear restoring force)
            with torch.no_grad():
                a_=opt.param_groups[0]['lr']*cfg.muon_lr*cfg.dec_prot*0.2
                for p_,d_ in model.__dict__.pop('_decstep'): p_.add_(d_.to(p_.dtype),alpha=-a_)
        if cfg.pmuon and step==_L-1: kf_store(model,dom_id)
        head_center_(model)
        if cfg.film_k>0 and model.__dict__.get('_film_frozen') is not None:             # only the current domain's code learns
            with torch.no_grad():
                fz_=model._film_frozen; keep_=model.film_codes[dom_id].clone(); model.film_codes.copy_(fz_); model.film_codes[dom_id]=keep_
        if cfg.varpro_old and step==_L-1 and n_old>0: print(f"    [varpro old] {model.__dict__.pop('_vpo_n',0)} old-head refits this visit")
        if (cfg.sleep_eps>0 or cfg.sleep_rel>0) and step==_L-1 and n_old>0:
            sc_=model.__dict__.pop('_sleepC',[0,0]); P_=model.__dict__.get('_sleepP',{}); SE_=model.__dict__.get('_sleepE',{})
            dr_=model.__dict__.pop('_drows',[0,0])
            print(f"    [sleep pressure] dreams on {sc_[0]}/{_L} steps, {dr_[0]/max(1,dr_[1]):.1f} rows per dream step | pressure / smoothed KL per old domain: "+(f"fixed {cfg.sleep_fixed} (open loop)"))
        if cfg.noise_diag>0 and step==_L-1 and n_old>0: noise_report(model,cfg,dom_id)
        if step==_L-1 and n_old>0 and model.__dict__.get('_clipst'):
            cs_=model.__dict__.pop('_clipst'); f_=lambda k_: (lambda e_: f"{float(e_[0])/max(1,e_[2]):.3f} mean norm, clipped {float(e_[1])/max(1,e_[2]):.0%} of {e_[2]}" if e_ else "none")(cs_.get(k_))
            print(f"    [grad clip] dream steps: {f_('d')} | other steps: {f_('r')} (threshold 1.0"+(f", dream steps {float(cfg.dream_clip_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.dream_clip_pass)-1)]):.1f}" if cfg.dream_clip_pass else '')+")")
        if cfg.kf_dream>0 and step==_L-1 and n_old>0:
            bt_=model.__dict__.pop('_kfbt',[0.0,0]); dc_=model.__dict__.get('_kf_dacc',{}); print(f"    [live keys] {bt_[1]} whitening builds took {bt_[0]:.1f}s ({'eigh'}) | rebuilt {model.__dict__.pop('_kf_drb',0)} times from the dream rows' current input factors ({len(dc_)} weights, {max([e_['n'] for e_ in dc_.values()] or [0])} dream positions each, mix {cfg.kf_dream})")
        if tl_snap_:
            with torch.no_grad():
                for p_,o_ in tl_snap_: p_.add_(p_-o_,alpha=cfg.trunk_lr_mult-1.0)
        if stv_ is not None: stiff_post(stv_)
        if pjv_: proj_post(pjv_,model,step)
        ngp_step(model,cfg,opt.param_groups[0]['lr'])
        if cds_:
            with torch.no_grad():
                for p_,o_ in cds_: p_.copy_(o_+_cdf_*(p_-o_))
            cds_=None
        if _dsc_:                                                                       # 1-D quadratic of J along the update u just taken (s = 1)
            _t_ds=time.perf_counter()
            _dsg1,_dsJ1=_dstep_grad(model,cfg,dom_id,olds,_dsseed,_dsP)
            with torch.no_grad():
                u_=[p_.detach()-q_ for p_,q_ in zip(_dsP,_dsth)]
                d0_=float(sum((a_.float()*b_.float()).sum() for a_,b_ in zip(_dsg0,u_))); d1_=float(sum((a_.float()*b_.float()).sum() for a_,b_ in zip(_dsg1,u_))); c_=d1_-d0_
            if d0_>=0: sst_=0.5                                                         #   not downhill on the held-out score: shrink
            elif c_<=1e-12: sst_=2.0                                                    #   no measurable curvature: still descending at s = 1 -> grow
            else: sst_=min(2.0,max(0.5,-d0_/c_))
            model.__dict__['_dsm']=float(min(4.0,max(0.25,model.__dict__['_dsm']*sst_**float(cfg.dstep_k))))
            model._dsS.append(sst_); model._dsM.append(model._dsm); model._dsJ.append(float(_dsJ1-_dsJ0))
            del _dsg0,_dsg1,_dsth,u_; _ds_t+=time.perf_counter()-_t_ds
        if cfg.cdet_log and step==_L-1: _cdet_report(model,cfg)
        if _ds_on and step==_L-1 and model.__dict__.get('_dsS'):
            S_=np.array(model._dsS); M_=np.array(model._dsM); J_=np.array(model._dsJ)
            print(f"    [direct step] {len(S_)} checks | best step length median {np.median(S_):.2f}x the step taken (at the 2x cap {np.mean(S_>=2.0):.0%}, at the 0.5x cap {np.mean(S_<=0.5):.0%}) | "
                  f"lr multiplier at the end {M_[-1]:.2f} (mean over the visit {M_.mean():.2f}) | held-out objective change per checked step {J_.mean()/LOG2:+.5f} bits | {_ds_t:.0f}s")
        _ek_=max(1,int(cfg.ema_every))
        if cfg.ema_tail>0 and model.__dict__.get('_ema') is not None and ((step+1)%_ek_==0 or step==_L-1):
            with torch.no_grad():
                P_,E_,d_=model._ema
                pw_=((1.0-d_)/cfg.ema_pull_rel) if cfg.ema_pull_rel>0 else 0.0     # v9i: <= 0 -> tail averaging without the pull
                if _ek_>1: d_=d_**_ek_; pw_=1.0-(1.0-pw_)**_ek_                 # k steps at once: same decay horizon
                try: torch._foreach_lerp_(E_,[p_.detach() for p_ in P_],1.0-d_)      # one fused update (GPU: few kernels)
                except Exception:
                    for p_,e_ in zip(P_,E_): e_.mul_(d_).add_(p_.detach(),alpha=1.0-d_)
                if pw_>0:                                                          # BIDIRECTIONAL: weights pulled towards their running average
                    try: torch._foreach_lerp_([p_.data for p_ in P_],E_,pw_)      # one fused update (GPU)
                    except Exception:
                        for p_,e_ in zip(P_,E_):
                            if p_.shape==e_.shape: p_.data.lerp_(e_,pw_)
            if step==_L-1 and model.__dict__.get('_ngp') is not None: print("    [private n-grams] table norms per domain: "+' '.join(f"{_lab(cfg,d_)} {sum(float(E.weight.detach().norm()) for E in model._ngp[d_]):.2f}" for d_ in sorted(set(olds)|{dom_id})))
            if step==_L-1 and model.__dict__.get('_ngpi') is not None: print("    [private input rows] table norms per domain: "+' '.join(f"{_lab(cfg,d_)} {sum(float(E.weight.detach().norm()) for E in model._ngpi[d_]):.2f}" for d_ in sorted(set(olds)|{dom_id})))
        if ngp_ is not None:
            with torch.no_grad():
                for W_,W0_,sc_ in ngp_:
                    if sc_ is not None: W_.copy_(W0_+(W_-W0_)*sc_[:,None])
        if getattr(model,'ngram',None) is not None and cfg.ngram_sparse: ngram_lazy_step(model,cfg,opt.param_groups[0]['lr'])
        if model.__dict__.get('_slmul') is not None:
            mu_=model.__dict__.pop('_slmul')
            for g_ in opt.param_groups: g_['lr']/=mu_
        if _snrm_!=1.0:
            for g_ in opt.param_groups: g_['lr']/=_snrm_
        if model.__dict__.get('_sldb') is not None: cfg.dream_batch=model.__dict__.pop('_sldb')
        apply_update_gate(snaps, None, cfg.og_in, cfg.og_orth,
                          None, cfg.og_orth_boost,
                          getattr(model,'_og_common',None), cfg.og_common,
                          getattr(model,'_og_out',None),
                          (None))   # threshold x meta write strength, applied to the UPDATE
        _tmark(model,cfg,step,'optimiser+gates')
        if cfg.shunt and cfg.shunt_mode=='all' and (n_old>0):     # FAST STORE: all new learning tracked in the shunt
            Sd=model.__dict__.setdefault('_shunt',{})
            if model.__dict__.get('_vstart') is None: model.__dict__['_vstart']={id(p_):(p_,old_.clone()) for p_,old_,_,_ in snaps}
        if xg is not None:                                 # Adam step on the cross gains; the change enters the live weights
            X=model._xcur; X['t']+=1; X['m'].mul_(0.9).add_(0.1*xg); X['v'].mul_(0.999).add_(0.001*xg*xg)
            mh=X['m']/(1-0.9**X['t']); vh=X['v']/(1-0.999**X['t'])
            An=(X['A']-cfg.cross_lr*mh/(vh.sqrt()+1e-8)).clamp(cfg.cross_min,cfg.cross_max); dA=An-X['A']; X['A']=An
            P_=_pmap(model)
            with torch.no_grad():
                for jj,j in enumerate(X['olds']):
                    for kk,k in enumerate(X['keys']):
                        if dA[jj,kk]!=0 and k in model._spec[j]: P_[k].add_(float(dA[jj,kk])*model._spec[j][k])
        if cfg.shunt_lazy and cfg.shunt and n_old>0 and model.__dict__.get('_vstart') and ((step+1)%cfg.shunt_every==0 or step+1==_L):
            with torch.no_grad(): model._shunt={k:(p_.data-o_) for k,(p_,o_) in model._vstart.items()}   # A5: shunt = weights - segment start
            model.__dict__['_vstart']=None
        if cfg.shunt and cfg.shunt_consol=='continuous' and n_old>0 and getattr(model,'_shunt',None) and not cfg.topic_mode:
            do=False
            do=(step+1)%cfg.shunt_every==0 or step+1==_L
            if do:
                model._shunt={}                                   # MERGE: shunt absorbed into the main weights
                model.__dict__.setdefault('_merge_log',[]).append((dom_id,step+1))
        if getattr(model,'_reserve',None) and not (cfg.block_extra and cfg.block_extra_mode=='fresh'):
            for b,U in zip(model.blocks,model._uniq): U.update(b._uh,b._um,cfg)
            if (step+1)%cfg.recruit_every==0:
                before=adam_capture(opt,model); ev=recruit(model,cfg)
                if ev:
                    opt=adam_rebuild(model,cfg.lr,before)
                    if getattr(model,'_teach',None) is not None: build_teachers(model,cfg)
                    L_=model._recruit_log.setdefault(dom_id,[]); L_.extend(ev)
        orth_edge=orth is not None and (step+1)%cfg.log_every==0
        if orth_edge: orth.end(model, (x,y), dom_id)     # before any widening, so the window's width is fixed
        if (step+1)%cfg.log_every==0:
            if getattr(model,'_curve_val',None) is not None:
                model.__dict__.setdefault('_curve',{}).setdefault(dom_id,[]).append(bpc(model,model._curve_val,dom_id,n=5))
            mgm=float(np.mean([g[0].mean().item() for g in mg])) if mg is not None else float('nan')
            ee=getattr(model,'_emb_elig',None); rr=getattr(model,'_route_mean',None); ce=getattr(model,'_ch_elig',None)
            msg=f"    step {step+1}: "+("")+f"(eligible={elig:.2f}"+(f", ch_eligible={ce:.2f}" if ce is not None else "")+(f", emb_eligible={ee:.2f}" if ee is not None else "")+(f", route={rr:.2f}" if rr is not None else "")+(f", distinct={float(model.blocks[0].cur_distinct):.2f}" if model.blocks[0].cur_distinct is not None else "")+")"
            lim=(cfg.min_eligible)
            if getattr(model,'_dmax',None):
                msg+=" [dream KL max: "+" ".join(f"{_lab(cfg,d)} {float(v):.3f}" for d,v in sorted(model._dmax.items()))+"]"; model._dmax={}
            msg+=(f" [uniqueness ratio "+" ".join(f"{U.ratio():.2f}" for U in model._uniq)
                  +f" | active state {model.blocks[0].d_state} MLP {model.blocks[0].mlp_hidden}]")
            print(msg)
        if orth_edge and step+1<_L: orth.begin(model, prior_buffer, (x,y), dom_id)
        _tmark(model,cfg,step,'rest')
    if cfg.ema_tail>0 and model.__dict__.get('_ema') is not None:     # TAIL AVERAGING: the visit ends on the averaged weights
        with torch.no_grad():
            P_,E_,_=model.__dict__.pop('_ema')
            for p_,e_ in zip(P_,E_):
                if p_.shape==e_.shape: p_.copy_(e_)
    if getattr(model,'_ng_cur',None) is not None:                     # ROW PROTECTION: this domain's usage shares join the old usage
        for c_,o_ in zip(model._ng_cur,model._ng_old):
            if float(c_.sum())>0: o_+=c_/c_.sum()
            c_.zero_()
        model.__dict__.pop('_ng_sc',None)                             # protection scales change only here
    sa_=model.__dict__.pop('_surpacc',None)
    if sa_:
        f_=lambda i,ph,j: (float(sa_[(i,ph)][j])/sa_[(i,ph)][2]) if (i,ph) in sa_ else float('nan')
        print("  [surprise] per block, first 100 / last 100 steps: write scale "+" ".join(f"b{i} {f_(i,0,0):.2f}/{f_(i,1,0):.2f}" for i in range(len(model.blocks)))
              +" | memory prediction-error ratio "+" ".join(f"b{i} {f_(i,0,1):.3f}/{f_(i,1,1):.3f}" for i in range(len(model.blocks))))
    rl_=model.__dict__.pop('_relaxlog',None)
    if rl_ is not None: print(f"  [residual gates] dream coverage of the top block's units (activity on dreams / long-run activity, capped at 1): state {rl_[0]:.2f}, MLP {rl_[1]:.2f}")
    C_=model.__dict__.pop('_conf_acc',None)
    if C_: print("  [protection conflict] mean cos(old domain's protection gradient, current-loss gradient): "+" ".join(f"{_lab(cfg,d)} {float(v[0])/v[1]:+.2f}" for d,v in C_.items()))
    S_=model.__dict__.get('_share_last')
    if cfg.share_mode and S_: print(f"  [dream shares] ({cfg.share_mode}) at the visit end: "+" ".join(f"{_lab(cfg,d)} {v:.2f}" for d,v in S_.items()))
    A_=model.__dict__.pop('_borrow_acc',None)
    if A_: print("  [borrow alignment] mean cos(borrowing gradient, own-loss gradient) per old domain: "+" ".join(f"{_lab(cfg,d)} {float(v[0])/v[1]:+.2f}" for d,v in A_.items()))
    R_=model.__dict__.get('_borrow_r')
    if R_ is not None and (cfg.borrow_w>0 or cfg.em_w>0) and olds:
        print("  [relevance] old domains' relevance to this domain at the visit end (EMA of (n_old+1) x posterior, cap 1): "
              +" ".join(f"{_lab(cfg,d)} {float(R_[d]):.2f}" for d in olds))
    wa_=model.__dict__.pop('_lwfwacc',None)
    if wa_ is not None:
        print(f"  [posterior KL] mean weight per old head on this domain's text (1 = unweighted): "
              +" ".join(f"{_lab(cfg,d)} {float(wa_[0][i_])/wa_[2]:.2f}" for i_,d in enumerate(olds))+f" | posterior of the current domain {float(wa_[1])/wa_[2]:.2f}")
    ca_=model.__dict__.pop('_ctxacc',None)
    if ca_ is not None:
        print("  [context] responsibility of each old memory for this domain's text (mean causal posterior): "
              +" ".join(f"{_lab(cfg,d)} {float(ca_[0][i_])/ca_[1]:.2f}" for i_,d in enumerate(olds)))
    fa_s=model.__dict__.pop('_fastat',None)
    if fa_s:
        print("  [feature anchor] block output on dreams vs frozen teacher, mean relative error: "+" ".join(f"b{li_} {float(a_[0])/max(a_[1],1):.5f}" for li_,a_ in sorted(fa_s.items()))+f" (weight {cfg.feat_anchor} per block)")
    os_=model.__dict__.pop('_opstat',None)
    if os_ is not None and os_[2]:
        print(f"  [on-policy dreams] mean per-token KL to the teacher on LIVE-model samples {float(os_[0])/os_[2]:.4f} vs teacher samples {float(os_[1])/os_[2]:.4f}")
    pe_=model.__dict__.pop('_prio_ess',None)
    if pe_ is not None: print(f"  [dream priority] effective share of each pool drawn from (1 = uniform): {float(pe_[0])/pe_[1]:.2f}")
    ka_=model.__dict__.pop('_klacc',None)
    if ka_ is not None:                                  # fused KL statistics -> per old domain (as the loop records them)
        for i_,d in enumerate(olds):
            kst=model.__dict__.setdefault('_klstat',{}).setdefault(d,[0.0,0.0,0]); kst[0]+=ka_[0][i_]; kst[1]+=cfg.shunt_old_weight*ka_[1]; kst[2]+=ka_[1]
    if cfg.timing and model.__dict__.get('_tm',{}).get('n'):
        T_=model._tm; n_=T_['n']; ph=['prep','forward+backward','optimiser+gates','rest']; tot=sum(T_['acc'].get(k,0.0) for k in ph)/n_
        wall=time.perf_counter()-_t_dom
        model.__dict__.setdefault('_timing_log',{})[dom_id]={'ms':1000*tot,'wall':wall,'dream':model.__dict__.get('_tm_dream',0.0),
                                                            **{k:1000*T_['acc'].get(k,0.0)/n_ for k in ph}}
        print("  [timing] ms per step (sampled, synchronised): "+" | ".join(f"{k} {1000*T_['acc'].get(k,0.0)/n_:.0f}" for k in ph)
              +f" | total {1000*tot:.0f} || domain wall {wall:.0f}s ({_L/max(wall,1e-9):.1f} steps/s): regular steps {_L*tot:.0f}s,"
              +f" periodic work {max(0.0,wall-_L*tot):.0f}s (dream sampling {model.__dict__.get('_tm_dream',0.0):.0f}s, evaluation, gate refreshes)"
              +(" [bf16]" if cfg.amp=='bf16' else ""))
        sd_,sn_=T_.get('sd',[0.0,0]),T_.get('sn',[0.0,0]); nd_=model.__dict__.get('_dcount',0)-T_.get('dc0',0)
        if sd_[1] and sn_[1]:
            md_,mn_=1000*sd_[0]/sd_[1],1000*sn_[0]/sn_[1]
            print(f"  [timing] whole step, sampled: WITHOUT dreams {mn_:.0f} ms ({sn_[1]} steps) | WITH dreams {md_:.0f} ms ({sd_[1]} steps, +{md_-mn_:.0f} ms) || this visit: {nd_} dream steps -> training steps ~{(nd_*md_+(_L-nd_)*mn_)/1000:.0f}s, other work ~{max(0.0,wall-(nd_*md_+(_L-nd_)*mn_)/1000):.0f}s")
        model.__dict__['_tm']={'last':None,'acc':{},'n':0,'dc0':model.__dict__.get('_dcount',0)}
    if dom_id in tuple(cfg.state_spec_diag or ()) and n_old>0 and not cfg.topic_mode:
        was_=model.training; model.eval()
        try: state_spectrum_diag(model, cfg, olds)
        except Exception as e_: print(f"  [state spectrum] diagnostic skipped ({type(e_).__name__}: {e_})")   # never stops a run
        model.train(was_)
    sst_=model.__dict__.pop('_strandstat',None); scn_=model.__dict__.pop('_strandcnt',None)
    if scn_:
        tot_=sum(scn_.values()); top_=sorted(scn_.items(),key=lambda z:-z[1])[:6]
        print(f"  [strands] {cfg.strand_kind}, {tot_} generated; most drawn pairs ({cfg.strand_pair_mode}): "+" ".join(f"{_lab(cfg,a)}>{_lab(cfg,b)} {c/tot_:.0%}" for (a,b),c in top_)
              +(f" | KL before / after the switch {float(sst_['a'])/sst_['n']:.4f} / {float(sst_['b'])/sst_['n']:.4f}" if sst_ and sst_['n'] else "")
              +(f" | agreement-dream KL {float(sst_['ag'])/sst_['agn']:.4f}" if sst_ and sst_.get('agn') else "")
              +(" | old-subspace anchor rel. error "+" ".join(f"b{li} {float(v[0])/max(v[1],1):.4f} ({v[2]} dims)" for li,v in sorted(sst_['anc'].items())) if sst_ and sst_['anc'] else ""))
    tb_=None
    if tb_: print("  [byte-type weights] real share / dream share per byte type (clipped), per old domain: "
                  +" | ".join(f"{_lab(cfg,d)} "+" ".join(f"{_BYTE_NAMES[c][:3]} {float(v[c]):.2f}" for c in range(len(_BYTE_NAMES)-1)) for d,v in sorted(tb_.items())))
    dsa_=model.__dict__.pop('_dsteer_acc',None)
    if dsa_: print("  [drift-steered dreams] mean dream weight per old domain (fresh-chunk KL / mean, ^%.1f, clipped): "%cfg.dream_steer
                   +" ".join(f"{_lab(cfg,d)} {float(v[0])/max(v[1],1):.2f}" for d,v in sorted(dsa_.items())))
    shs_=model.__dict__.pop('_shstat',None)
    if shs_: print("  [similarities only] shared subspace (used by >= %.0f%% of old domains) per protected block, mean dims: "%(100*cfg.share_thr)
                   +" ".join(f"b{li} {a_[0]/max(a_[1],1):.1f}/{model.blocks[li].ln.weight.shape[0]}" for li,a_ in sorted(shs_.items()))
                   +f" | specific component x{cfg.share_spec}")
    bs=[getattr(b,'_bstat',None) for b in model.blocks]
    if any(bs):
        row=[(float(x[0])/x[2], float(x[1])/x[2]) if x else (float('nan'),float('nan')) for x in bs]
        model.__dict__.setdefault('_beta_log',{})[getattr(model,'_visit',dom_id)]=row
        print("  [reflections] share of writes with beta > 1 / mean beta, per block: "+" | ".join(f"b{i} {a:.2f}/{m:.2f}" for i,(a,m) in enumerate(row)))
        for b in model.blocks: b._bstat=None
    if imp_base is not None:     # restore pre-domain importance; run() then adds the full increment
        with torch.no_grad():
            for (o_,nm),b0 in zip(_imp_bufs(model),imp_base):
                cur=getattr(o_,nm); setattr(o_,nm,torch.cat([b0,torch.zeros(cur.numel()-b0.numel(),device=cur.device)]))
    return opt

@torch.no_grad()
def calibrate_affinity(model, cfg, n_domains, n_batches=6):
    """Right after the lock: run held-out-free calibration batches from each trained domain
    (fixed generator, no global RNG use) and set every unit's affinity to the drive-weighted
    distinctness of that content ('the position that activated it most')."""
    was=model.training; model.eval()
    for b in model.blocks:
        dev=b.unit_aff.device
        b.calib={'s_w':torch.zeros(b.d_state,device=dev),'s_wd':torch.zeros(b.d_state,device=dev),
                 'm_w':torch.zeros(b.mlp_hidden,device=dev),'m_wd':torch.zeros(b.mlp_hidden,device=dev)}
    for q in range(n_domains):
        get=make_loader(stream_bytes(PILE_DOMAINS[q],1000+q,4_000_000),cfg,fixed_seed=900+q)
        for _ in range(n_batches):
            x,_=get(); model(x,q)
    for b in model.blocks:
        c=b.calib
        for w,wd,aff in ((c['s_w'],c['s_wd'],b.unit_aff),(c['m_w'],c['m_wd'],b.mlp_aff)):
            ok=w>0.05*w.max()
            aff[ok]=(wd[ok]/w[ok]).clamp(0,1)
        b.calib=None
    a=model.blocks[0].mlp_aff; asg=a[a>=0]
    print(f"  [affinity calibrated] assigned {float((a>=0).float().mean()):.2f} of units | "
          f"affinity mean {float(asg.mean()) if len(asg) else float('nan'):.2f} range "
          f"[{float(asg.min()) if len(asg) else 0:.2f},{float(asg.max()) if len(asg) else 0:.2f}]")
    if was: model.train()

# ---------------- ORTHOGONALITY DIAGNOSTICS ----------------
# Question: would making writes ORTHOGONAL to what earlier domains use help, and is that
# orthogonal room already used up (so writes must be shunted elsewhere)?
# Every orth_every steps (a "window") of domains 2..5:
#  begin: from buffered earlier-domain batches, each projectable Linear's INPUT covariance ->
#         the old-domain input subspace U (top eigenvectors holding 90% / 99% of the energy).
#  end:   the window's actual weight change dW is split into its part inside U (dW U U^T, which
#         moves old-domain outputs) and the orthogonal part (which cannot, to first order).
#         Counterfactual losses, all from the same start weights:
#           actual        : what training did
#           weights_only  : only the projectable matrices changed (others kept at start)
#           orth@90/@99   : projectable matrices get only their orthogonal part
#         -> % of the window's forgetting orthogonal writes would remove, and % of the new
#            domain's learning they would keep. Eval-only: training is unchanged.
ORTH_KEYS=('Wk','gate','mlp_in','Wout','mlp_out')            # Wv shares Wk's input
ORTH_PARAM_KEY={'Wk':'Wk','Wv':'Wk','gate':'gate','mlp_in':'mlp_in','Wout':'Wout','mlp_out':'mlp_out'}

@torch.no_grad()
def _dream_cov(model, cfg):
    """Layer-input covariances of the old domains' dreams through the current network (equal weight per old domain)."""
    pools=getattr(model,'_dreams',None) or {}
    bt=[(torch.stack(pl[:cfg.og_dream_n]),None,d) for d,pl in sorted(pools.items()) if len(pl)>0]
    if not bt: return {}
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    dc=_input_cov(model,[(x,x,d) for x,_,d in bt])
    model.feat_scale.copy_(fs); model.train(was); return dc

@torch.no_grad()
def _input_cov(model, batches):
    acc={}; hooks=[]
    for li,b in enumerate(model.blocks):
        for key in ORTH_KEYS:
            def hk(m,inp,out,k=(li,key)):
                x=inp[0].reshape(-1,inp[0].shape[-1]).float()
                c,n=acc.get(k,(0.0,0)); acc[k]=(c+x.T@x, n+x.shape[0])
            hooks.append(getattr(b,key).register_forward_hook(_f32(hk)))
    for x,_,d in batches: model(x,d)
    for h in hooks: h.remove()
    return {k:c/max(n,1) for k,(c,n) in acc.items()}

def _subspaces(C, energies):
    ev,U=torch.linalg.eigh(C.float()); ev=ev.flip(0).clamp(min=0); U=U.flip(1)
    cum=torch.cumsum(ev,0)/ev.sum().clamp(min=1e-30); out={}
    for e in energies:
        k=min(int((cum<e).sum().item())+1, C.shape[0]); out[e]=(U[:,:k].float(),k)
    return out

def orth_summary(log, dom, energies):
    """Aggregate a domain's windows: sums of loss changes (BPC) over windows."""
    W=[w for w in log if w['dom']==dom]
    if not W: return None
    S=lambda mode,e,i,ref=('start',None): sum(w['L'][(mode,e)][i]-w['L'][ref][i] for w in W)
    base=('others_only',None)
    # forgetting / learning caused by the projectable matrices, given all other changes (heads, norms, biases)
    f_mat=S('actual',None,0,base); l_mat=-S('actual',None,1,base)
    r={'windows':len(W),'forget_actual':S('actual',None,0),'forget_weights_only':S('weights_only',None,0),
       'forget_matrices':f_mat,'learn_matrices':l_mat}
    for e in energies:
        r[f'removed@{e}']=(S('actual',None,0,base)-S('orth',e,0,base))/f_mat if f_mat>1e-6 else float('nan')
        r[f'kept@{e}']=(-S('orth',e,1,base))/l_mat if l_mat>1e-6 else float('nan')
    avg=lambda key,e=None: {k:float(np.mean([np.mean([v for (li,kk),v in (w[key][e] if e is not None else w[key]).items() if kk==k]) for w in W])) for k in ORTH_KEYS}
    for e in energies: r[f'rank@{e}']=avg('rank',e)
    WU=[w for w in W if ('unit_own',None) in w['L']]
    if WU:
        g=lambda mode,i: sum(w['L'][(mode,None)][i]-w['L'][('actual',None)][i] for w in WU)
        r['unit_own_old']=g('unit_own',0); r['unit_own_new']=g('unit_own',1)       # BPC change vs actual (old<0 = less forgetting)
        r['unit_layer_old']=g('unit_layer',0); r['unit_layer_new']=g('unit_layer',1)
        r['unit_rank']=float(np.mean([np.mean([v['rank'] for v in w['unit_stats'].values()]) for w in WU if w['unit_stats']]))
        r['unit_new_in']=float(np.mean([np.mean([v['new_in'] for v in w['unit_stats'].values()]) for w in WU if w['unit_stats']]))
    r['new_in_old']=avg('ovl')
    r['upd_in_old']={k:float(np.mean([w['upd'][k] for w in W if k in w['upd']])) for k in ORTH_KEYS if any(k in w['upd'] for w in W)}
    r['imp_split']={t:float(np.nanmean([w['imp_split'].get(t,float('nan')) for w in W])) for t in ('top','rest')}
    return r

def print_orth(r, name, energies):
    e_lo,e_hi=min(energies),max(energies); f=lambda d:" ".join(f"{k} {v:.2f}" for k,v in d.items())
    print(f"  [orthogonality] {name}: old-domain loss rise over {r['windows']} windows {r['forget_actual']:+.3f} BPC, "
          f"of which the weight matrices cause {r['forget_matrices']:+.3f}; matrices' new-domain gain {r['learn_matrices']:+.3f}")
    print("     LAYER-level orthogonal writes (subspace energy -> % of matrix forgetting removed / % of matrix learning kept): "
          +", ".join(f"{e:.0%}: {100*r[f'removed@{e}']:.0f}/{100*r[f'kept@{e}']:.0f}" for e in energies))
    if 'unit_own_old' in r:
        print(f"     PER-UNIT (protected MLP units, own subspace @{Config.orth_unit_energy:.0%}): each fills {r['unit_rank']:.2f} of its input dims; "
              f"new-domain input inside it {r['unit_new_in']:.2f}")
        print(f"        making only those units' writes orthogonal: own subspace old {r['unit_own_old']:+.4f} / new {r['unit_own_new']:+.4f} BPC | "
              f"layer subspace old {r['unit_layer_old']:+.4f} / new {r['unit_layer_new']:+.4f} BPC  (old<0 = less forgetting, new>0 = lost learning)")
    print(f"     old subspace fills (fraction of input dims @{e_hi:.0%}): {f(r[f'rank@{e_hi}'])}")
    print(f"     new-domain input energy already inside old subspace:  {f(r['new_in_old'])}")
    print(f"     actual update energy inside old subspace:             {f(r['upd_in_old'])}")
    print(f"     inside-old share of updates, top-20%-importance units {r['imp_split']['top']:.2f} vs others {r['imp_split']['rest']:.2f}")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _carry_save(): return {k_:_CARRY[k_] for k_ in ('on','nr','st','new','rows')}
def _carry_load(sv): _CARRY.update(sv)
@torch.no_grad()
def mk_refresh(model,cfg,x,dom):
    """MEMORY-KEYED DREAMS: episodic snapshots of mk_rows real windows -> dreams generated from them -> teacher targets (with memory)."""
    was=model.training; model.eval(); fs=model.feat_scale.clone(); sv=_carry_save(); xr=x[:int(cfg.mk_rows)]; n=xr.shape[0]
    try:
        _CARRY.update(on=True,nr=n,st={},new={},rows=None); model.features(xr); snap=dict(_CARRY['new'])        # memory at the end of each window
        if not hasattr(model,'_mk_gen'):
            g_=torch.Generator(device=x.device); g_.manual_seed(cfg.seed+23); object.__setattr__(model,'_mk_gen',g_)
        xs=torch.full((n,1),10,dtype=torch.long,device=x.device); T=int(cfg.seq_len)
        for _ in range(T-1):                                                   # generation STARTS FROM the snapshot memory
            if cfg.mk_gen=='mem': _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
            else: _CARRY.update(on=False,st={},new={})
            lg=model(xs,dom)[:,-1].float()/cfg.dream_temp; xs=torch.cat([xs,torch.multinomial(F.softmax(lg,-1),1,generator=model._mk_gen)],1)
        if cfg.mk_teacher=='mem': _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
        else: _CARRY.update(on=False,st={},new={})
        tg=F.log_softmax(model(xs,dom).float(),-1)
    finally:
        _carry_load(sv); model.feat_scale.copy_(fs); model.train(was)
    pool=model.__dict__.setdefault('_mk_pool',[]); pool.extend(zip(list(xs),list(tg))); del pool[:-int(cfg.mk_pool)]
def mk_loss(model,cfg,dom):
    pool=model.__dict__.get('_mk_pool') or []
    if not pool: return 0.0
    if not hasattr(model,'_mk_rng'): object.__setattr__(model,'_mk_rng',np.random.RandomState(cfg.seed+29))
    ix=model._mk_rng.randint(len(pool),size=int(cfg.mk_dream)); xs=torch.stack([pool[i][0] for i in ix]); tg=torch.stack([pool[i][1] for i in ix])
    lp=F.log_softmax(model(xs,dom).float(),-1)                                 # student: NO memory
    kl=(tg.exp()*(tg-lp)).sum(-1).mean(); st=model.__dict__.setdefault('_mk_stat',[0.0,0]); st[0]+=float(kl.detach()); st[1]+=1
    return kl
def _uh_save(model): return [(b.__dict__.get('_uh'),b.__dict__.get('_um')) for b in model.blocks]
def _uh_load(model,sv):
    for b,(u_,m_) in zip(model.blocks,sv):
        if u_ is not None: b.__dict__['_uh']=u_; b.__dict__['_um']=m_
@contextlib.contextmanager
def _rp_tags(model):
    """replay passes: every row is the CURRENT domain (the training batch's per-row tags _s0_dom have another row count; a mismatch
    silently switched the private trunk units / FiLM codes off for those rows, so encoder and decoder ran different models)"""
    sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=None
    try: yield
    finally: model.__dict__['_s0_dom']=sv_
def recon_loss(model,cfg,x,y,dom):
    with _rp_tags(model): return _recon_loss(model,cfg,x,y,dom)
def _recon_loss(model,cfg,x,y,dom):
    """REPLAY TRAINING: each window regenerated starting from its own (differentiable) end-of-window memory."""
    n=min(int(cfg.recon_rows),x.shape[0]); xr,yr=x[:n],y[:n]; sv=_carry_save(); suh=_uh_save(model)
    try:
        _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(xr); snap=dict(_CARRY['new'])
        _CARRY.update(on=True,nr=n,st=snap,new={},rows=None,grad=False)
        l_=F.cross_entropy(model(xr,dom).reshape(-1,256),yr.reshape(-1))
    finally:
        _CARRY.pop('grad',None); _carry_load(sv); _uh_load(model,suh)
    st=model.__dict__.setdefault('_rc_stat',[0.0,0]); st[0]+=float(l_.detach()); st[1]+=1
    return l_
def _cd_g(model,dev,cfg):
    if not hasattr(model,'_cd_gen'):
        g_=torch.Generator(device=dev); g_.manual_seed(int(cfg.seed)+41); object.__setattr__(model,'_cd_gen',g_)
    return model._cd_gen
def _cd_slices(model,cfg,x,n,L=None):
    B,T=x.shape; L=min(int(cfg.cd_seg) if L is None else int(L),T); g_=_cd_g(model,x.device,cfg)
    r_=torch.randint(0,B,(n,),generator=g_,device=x.device); s_=torch.randint(0,T-L+1,(n,),generator=g_,device=x.device)
    if L==T and n<=B: r_=torch.randperm(B,generator=g_,device=x.device)[:n]          # FULL WINDOWS: distinct rows of the batch
    return x[r_[:,None],s_[:,None]+torch.arange(L,device=x.device)[None]].long()        # v9d: one gather (was 2 syncs per window)
_RSS={'pe':[],'pool':[]}                                                   # v8: p_exact of the chosen pieces / of the pool
def _rs_rowloss(model,cfg,X,dom):
    """per-piece teacher-forced replay loss (no gradient): [mark] + piece[:-1] -> piece from the piece's own memory, live model"""
    n=X.shape[0]; sv=_carry_save(); suh=_uh_save(model)
    try:
        with torch.no_grad():
            _CARRY.update(on=True,nr=n,st={},new={},rows=None); model.features(X); snap=dict(_CARRY['new'])
            _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
            Xin=torch.cat([torch.full((n,1),int(cfg.cd_mark),dtype=torch.long,device=X.device),X[:,:-1]],1)
            l_=F.cross_entropy(model(Xin,dom).reshape(-1,256).float(),X.reshape(-1),reduction='none').view(n,-1).mean(1)
    finally: _carry_load(sv); _uh_load(model,suh)
    return torch.nan_to_num(l_,nan=1e3,posinf=1e3)
def _rs_pick(model,cfg,Cd,k,dom,g_):
    """v8 SELECTIVE REPLAY TRAINING: k of the candidate pieces Cd: round(rs_frac x k) sampled in proportion to p(1 - p) (borderline), the rest
    uniformly from the remaining candidates"""
    m=Cd.shape[0]; k=min(int(k),m)
    if k<=0 or m<=k: return Cd[:k]
    ks=int(round(float(cfg.rs_frac)*k)); pe=torch.exp(-Cd.shape[1]*_rs_rowloss(model,cfg,Cd,dom)); w=pe*(1-pe)+1e-6
    sel=torch.multinomial(w,ks,replacement=False,generator=g_) if ks>0 else torch.zeros(0,dtype=torch.long,device=Cd.device)
    if cfg.cd_nosync:                                                       # v9d: uniform among the rest via random keys (no nonzero)
        rk_=torch.rand(m,generator=g_,device=Cd.device); rk_[sel]=-1.0; ri=rk_.topk(k-ks).indices
        if ks>0: _RSS['pe'].append(pe[sel].mean())
        _RSS['pool'].append(pe.mean())
    else:
        rest=torch.ones(m,dtype=torch.bool,device=Cd.device); rest[sel]=False; ri=rest.nonzero().flatten()
        ri=ri[torch.randperm(len(ri),generator=g_,device=Cd.device)[:k-ks]]
        if ks>0: _RSS['pe'].append(float(pe[sel].mean()))
        _RSS['pool'].append(float(pe.mean()))
    return Cd[torch.cat([sel,ri])].contiguous()
def _rs_msg():
    if not _RSS['pool']: return ""
    m_=f" | selective: borderline pieces p_exact {_fm(_RSS['pe']) if _RSS['pe'] else float('nan'):.2f} (pool {_fm(_RSS['pool']):.2f})"
    _RSS['pe']=[]; _RSS['pool']=[]; return m_
def recon_loss_mark(model,cfg,x,dom):
    with _rp_tags(model): return _recon_loss_mark(model,cfg,x,dom)
def _recon_loss_mark(model,cfg,x,dom,X=None):
    """REPLAY TRAINING v2: segments decoded as [mark] + segment[:-1] -> segment from their own (differentiable) end-of-segment memory.
    X: the segments themselves (v6: exact replays in conduit passes); otherwise recon_rows slices of the batch."""
    if X is None:
        n=min(int(cfg.recon_rows),x.shape[0])
        if cfg.rs_sel: X=_rs_pick(model,cfg,_cd_slices(model,cfg,x,max(n,int(cfg.rs_pool)*n)),n,dom,_cd_g(model,x.device,cfg))   # v8
        else: X=_cd_slices(model,cfg,x,n)
    else: n=X.shape[0]
    sv=_carry_save(); suh=_uh_save(model)
    try:
        _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(X); snap=dict(_CARRY['new'])
        _CARRY.update(on=True,nr=n,st=snap,new={},rows=None,grad=False)
        Xin=torch.cat([torch.full((n,1),int(cfg.cd_mark),dtype=torch.long,device=X.device),X[:,:-1]],1)
        l_=F.cross_entropy(model(Xin,dom).reshape(-1,256).float(),X.reshape(-1))
        if not bool(torch.isfinite(l_.detach())):                        # NON-FINITE replay loss: skip this step's replay term (counted)
            model.__dict__['_rc_bad']=model.__dict__.get('_rc_bad',0)+1; l_=torch.zeros((),device=X.device)
    finally:
        _CARRY.pop('grad',None); _carry_load(sv); _uh_load(model,suh)
    st=model.__dict__.setdefault('_rc_stat',[0.0,0]); st[0]=st[0]+(l_.detach() if cfg.cd_nosync else float(l_.detach())); st[1]+=1
    return l_
@contextlib.contextmanager
def _cd_weights(model):
    """the frozen conduit teacher: visit-start weights swapped in (data pointers only, restored on exit)"""
    P=model.__dict__.get('_cd_P')
    if P is None:
        yield; return
    ps=dict(model.named_parameters())
    for k_ in P: ps[k_].data,P[k_]=P[k_],ps[k_].data
    try: yield
    finally:
        for k_ in P: ps[k_].data,P[k_]=P[k_],ps[k_].data
_ENC_STOP={'k':None}
_V97={'vp':[],'pn':[],'pd':[],'ag':[],'agd':[],'rep':0,'imp':[],'chk':[],'dis':[],'fw':[],'skip':0,'calls':0}
_HINT={'calls':0,'bytes':0,'tot':0,'wrong':0,'wcov':0,'pieces':0}               # cg12b hint statistics (device tensors, read at the visit end)
@torch.no_grad()
def _cd_hints(model,cfg,st_,m_,L,dom,M,Xc):
    """cg12b HINTS for m_ pieces: one teacher-forced pass with each piece's trace over its TRUE bytes (written with the trace, while the
    text is at hand); margin = log p(true byte) - best other; the lowest cd_hint_frac share of margins (<= cd_hint_piece per piece) are hinted.
    Returns (mask (m_,L) bool, bytes (m_,L))."""
    dev=Xc.device; sv_=_carry_save()
    try:
        Yin=torch.cat([torch.full((m_,1),M,dtype=torch.long,device=dev),Xc[:,:-1]],1)
        _CARRY.update(on=True,nr=m_,st=st_,new={},rows=None)
        lp=F.log_softmax(model(Yin,dom).float(),-1)
    finally: _carry_load(sv_)
    tl=lp.gather(-1,Xc[...,None]).squeeze(-1); ot=lp.scatter(-1,Xc[...,None],float('-inf')).max(-1).values; mg=tl-ot
    rk=torch.argsort(torch.argsort(mg,1),1); mgc=torch.where(rk<int(cfg.cd_hint_piece),mg,torch.full_like(mg,float('inf')))
    wr_=mg<0
    if float(cfg.cd_hint_adapt)>0:                                            # cg13 ADAPTIVE BUDGET: cd_hint_adapt x the teacher-forced wrong bytes,
        k=int(math.ceil(float(cfg.cd_hint_adapt)*float(wr_.sum())))          #   between cd_hint_min and cd_hint_max of the bytes (cg12b at 3 percent:
        k=min(max(k,int(round(float(cfg.cd_hint_min)*m_*L))),int(round(float(cfg.cd_hint_max)*m_*L)))   #   Wikipedia 1.95 percent wrong, 91 percent covered;
    else: k=int(round(float(cfg.cd_hint_frac)*m_*L))                          #   DM Mathematics 4.04 percent wrong, 74 percent covered)
    k=max(1,k); idx=torch.topk(-mgc.flatten(),min(k,m_*L)).indices
    hm=torch.zeros(m_*L,dtype=torch.bool,device=dev); hm[idx]=True; hm=hm.view(m_,L)&torch.isfinite(mgc)
    _HINT['calls']+=1; _HINT['bytes']=_HINT['bytes']+hm.sum(); _HINT['tot']+=m_*L
    _HINT['wrong']=_HINT['wrong']+wr_.sum(); _HINT['wcov']=_HINT['wcov']+(wr_&hm).sum(); _HINT['pieces']=_HINT['pieces']+hm.any(1).sum()
    return hm,Xc
def _cd_fb(model,snap):
    fb_=model.__dict__.get('_cd_fb')
    if fb_ is None:
        ix_=[i for i,b in enumerate(model.blocks) if id(b) in snap]; fb_=(ix_[0],id(model.blocks[ix_[0]])); model.__dict__['_cd_fb']=fb_
    return fb_
def _cd_encode_chk(model,cfg,X):
    if cfg.cd_first_block:
        _ENC_STOP['k']=model.__dict__['_cd_fb'][0]
        try: return _cd_encode(model,X)
        finally: _ENC_STOP['k']=None
    return _cd_encode(model,X)
def _cd_kdist(model,enc,ref,m,L):
    """(m, L): relative distance of the replay's first-delta-block write-key rows from the stored ones (max over heads)"""
    bid_=model.__dict__['_cd_fb'][1]; Ke=enc[bid_][1].float(); Kr=ref[bid_][1].float()
    d_=(Ke-Kr).norm(dim=-1)/Kr.norm(dim=-1).clamp_min(1e-12); return d_.reshape(m,-1,d_.shape[-1])[:,:,-L:].amax(1)
def _cd_vcheck(model,cfg,Y,ref,m,L,eps):
    """v9e: (mismatch e - exact iff e <= eps, verified prefix length j) of m replays Y against their stored traces ref"""
    enc=_cd_encode_chk(model,cfg,Y); bad_=_cd_kdist(model,enc,ref,m,L)>float(model.__dict__.get('_cd_ktol',cfg.cd_keytol))
    j_=torch.where(bad_.any(1),bad_.float().argmax(1),torch.full((m,),L,device=Y.device,dtype=torch.float32)).long()
    if cfg.cd_first_block: return (j_<L).float(),j_
    e_=_cd_mis(enc,ref,m); ok_=e_<=eps; _V97['dis'].append((~ok_&(j_>=L)).sum())
    j_=torch.where(ok_,torch.full_like(j_,L),torch.where(j_>=L,torch.zeros_like(j_),j_)); return e_,j_
_CBUF={'v':0,'ldom':None,'d':{},'mix':0,'calls':0,'src':{}}
def _cbuf_take(model,cfg,dom,n0,dev):
    """v9f: visit bookkeeping + the buffered windows for this step: list of (source domain, index)"""
    if _CBUF['ldom']!=dom:
        _CBUF['v']+=1; _CBUF['ldom']=dom; _CBUF['d'][dom]={'x':[],'t':[],'m':[],'visit':_CBUF['v']}
        for d_ in [d_ for d_,e_ in _CBUF['d'].items() if e_['visit']<_CBUF['v']-int(cfg.cd_buf_visits)]: del _CBUF['d'][d_]
    if float(cfg.cd_buf)<=0: return []
    pool_=[(d_,i_) for d_,e_ in _CBUF['d'].items() if d_!=dom for i_ in range(len(e_['x']))]
    k_=min(int(round(float(cfg.cd_buf)*n0)),len(pool_),n0-1)
    if k_<=0: return []
    pk_=torch.randperm(len(pool_),generator=_cd_g(model,dev,cfg),device=dev)[:k_].tolist()
    return [pool_[i_] for i_ in pk_]
def _cd_encode(model,X):
    _CARRY.update(on=True,nr=X.shape[0],st={},new={},rows=None); model.features(X)
    return {k_:(tuple(t_.clone() for t_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}
def _cd_rows(st,idx,n):
    f_=lambda v_: v_.reshape(n,-1,*v_.shape[1:])[idx].reshape(-1,*v_.shape[1:])
    return {k_:(tuple(f_(t_) for t_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cd_cat(sts):
    return {k_:(tuple(torch.cat([s_[k_][i] for s_ in sts]) for i in range(len(sts[0][k_]))) if isinstance(sts[0][k_],tuple) else torch.cat([s_[k_] for s_ in sts])) for k_ in sts[0]}
def _fgram(Ua,Ka,Ub,Kb): return (torch.bmm(Ua,Ub.transpose(1,2))*torch.bmm(Ka,Kb.transpose(1,2))).sum((1,2))   # <Ua^T Ka, Ub^T Kb>_F
_MIS32={'on':False}
def _fgram32(Ua,Ka,Ub,Kb):
    """<Ua^T Ka, Ub^T Kb>_F with the (L x L) Grams in full fp32 (no TF32) and their elementwise product + sum in fp64"""
    pv_=torch.backends.cuda.matmul.allow_tf32 if Ua.is_cuda else None
    if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=False
    try: G_=torch.bmm(Ua,Ub.transpose(1,2)).double(); H_=torch.bmm(Ka,Kb.transpose(1,2)).double()
    finally:
        if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=pv_
    return (G_*H_).sum((1,2))
def _cd_mis(a,b,n):
    num=0.0; den=0.0
    for k_ in b:
        if isinstance(b[k_],tuple):                                     # FACTORED: |Sa - Sb|^2 = <Sa,Sa> + <Sb,Sb> - 2 <Sa,Sb> (float64 Grams)
            if _MIS32['on']:                                            # v9c: Gram matrices in fp32 (TF32 off), their products + sums in fp64
                Ua,Ka=(t_.float() for t_ in a[k_]); Ub,Kb=(t_.float() for t_ in b[k_])
                sbb=_fgram32(Ub,Kb,Ub,Kb); d_=(_fgram32(Ua,Ka,Ua,Ka)+sbb-2.0*_fgram32(Ua,Ka,Ub,Kb)).clamp_min(0)
                num=num+d_.reshape(n,-1).sum(1).float(); den=den+sbb.reshape(n,-1).sum(1).float()
                continue
            Ua,Ka=(t_.double() for t_ in a[k_]); Ub,Kb=(t_.double() for t_ in b[k_])
            sbb=_fgram(Ub,Kb,Ub,Kb); d_=(_fgram(Ua,Ka,Ua,Ka)+sbb-2.0*_fgram(Ua,Ka,Ub,Kb)).clamp_min(0)
            num=num+d_.reshape(n,-1).sum(1).float(); den=den+sbb.reshape(n,-1).sum(1).float()
        else:
            num=num+((a[k_].float()-b[k_].float()).reshape(n,-1)**2).sum(1); den=den+(b[k_].float().reshape(n,-1)**2).sum(1)
    return num/den.clamp_min(1e-12)
_INC={'on':False,'k':1}
_RESET={'m':None}
FOKS={'skip':[],'est':[],'true':[],'k':[],'runs':[]}                                                          # v7: (B,T) bool, memory reset positions for the student pass
_CDCONF={'c':None}                                                         # v7: per-row confidence of the last _cd_gen call
_FOKD={}                                                                   # v7: per-domain gate calibration
def _fok_dom(d):
    return _FOKD.setdefault(int(d),{'c':[],'e':[],'thr':float('inf'),'exr':0.0,'n':0,'nc':0,'exrT':None,'ncall':0})
def _fm(v):
    """mean of a list of floats and/or 0-d tensors (one read-back)"""
    if not v: return 0.0
    return float(torch.stack([t_.detach().float().reshape(()).cpu() if torch.is_tensor(t_) else torch.tensor(float(t_)) for t_ in v]).mean())
def _fsum(v):
    return float(sum(float(t_) for t_ in v)) if v else 0.0
def _fok_thr_gpu(fd_,cfg):
    """v9d: the gate threshold from the GPU calibration buffer (lowest confidence with <= cd_fok_far inexact tries above it, >= 200 above)"""
    c_=torch.cat(fd_['c']); e_=torch.cat(fd_['e'])
    if c_.numel()>40000: c_=c_[-40000:]; e_=e_[-40000:]
    fd_['c']=[c_]; fd_['e']=[e_]
    o_=torch.argsort(-c_); cs_=c_[o_]; rk_=torch.arange(1,c_.numel()+1,device=c_.device,dtype=torch.float32)
    bad_=torch.cumsum(1.0-e_[o_].float(),0)/rk_; ok_=(bad_<=float(cfg.cd_fok_far))&(rk_>=200)
    li_=(ok_.float()*rk_).argmax()
    fd_['thr']=torch.where(ok_.any(),cs_[li_],torch.full((),float('inf'),device=c_.device))
_CDT={'on':False,'t':0.0,'acc':{},'n':0,'calls':0}                 # conduit timing (every 20th call, synchronised)
def _cdt(tag):
    if not _CDT['on']: return
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_=time.time(); _CDT['acc'][tag]=_CDT['acc'].get(tag,0.0)+t_-_CDT['t']; _CDT['t']=t_
# ---------------- CUDA-GRAPH CONDUIT GENERATION (cd_graph) ----------------
_CDG={'g':{},'seen':{},'dom':None,'off':None,'calls':{},'n_rep':0,'n_eag':0,'n_cap':0,'t_cap':0.0,'n_chk':0,'n_bit':0,'maxd':0.0,'minag':1.0,'pad_rows':0,'rows':0}
class _CdgFake:
    """CPU test stand-in: 'replay' re-runs the recorded function and writes into the same output tensors (static-buffer semantics)."""
    def __init__(s,fn,lg,new): s.fn=fn; s.lg=lg; s.new=new
    def replay(s):
        lg_,new_=s.fn(); s.lg.copy_(lg_)
        for k_ in s.new:
            if isinstance(s.new[k_],tuple):
                for a_,b_ in zip(s.new[k_],new_[k_]): a_.copy_(b_)
            else: s.new[k_].copy_(new_[k_])
def _cdg_leaves(st): return [t_ for k_ in sorted(st) for t_ in (st[k_] if isinstance(st[k_],tuple) else (st[k_],))]
def _cdg_sig(st,m): return tuple((k_,tuple((tuple(t_.shape[1:]),t_.shape[0]//m,str(t_.dtype)) for t_ in (st[k_] if isinstance(st[k_],tuple) else (st[k_],)))) for k_ in sorted(st))
def _cdg_fill(dst,src,m):
    """copy src (m rows, or already padded) into the static buffer dst (mb rows); extra rows repeat row 0"""
    if src.shape[0]==dst.shape[0]: dst.copy_(src); return
    r_=src.shape[0]//m; mb=dst.shape[0]//r_; dv=dst.view(mb,r_,*dst.shape[1:]); sv=src.view(m,r_,*src.shape[1:])
    dv[:m].copy_(sv)
    if mb>m: dv[m:].copy_(sv[:1].expand(mb-m,*sv.shape[1:]))
def _cdg_fill_st(dst,src,m):
    for k_ in dst:
        if isinstance(dst[k_],tuple):
            for a_,b_ in zip(dst[k_],src[k_]): _cdg_fill(a_,b_,m)
        else: _cdg_fill(dst[k_],src[k_],m)
def _cdg_slice(st,m,mb):
    """the first m rows of a padded state (for the eager fallback / self-check)"""
    if mb==m: return st
    f_=lambda t_: t_.view(mb,t_.shape[0]//mb,*t_.shape[1:])[:m].reshape(m*(t_.shape[0]//mb),*t_.shape[1:])
    return {k_:(tuple(f_(t_) for t_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cdg_once(model,dom,Yw,st,nr):
    """ONE generation forward, exactly as the eager loop in _cd_gen does it (incremental mode)"""
    _CARRY.update(on=True,nr=nr,st=st,new={},rows=None); _INC['on']=True
    try: lg=model(Yw,dom)[:,-1].float()
    finally: _INC['on']=False
    return lg,{k_:v_ for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}
def _cdg_fp(model):
    """parameters by storage (the conduit teacher is swapped in by data pointer) + the forward's global switches. (Buffers are left out: the
    training statistics among them are re-created every step; a buffer the forward reads that is replaced would be caught by the self-check.)"""
    return hash((tuple((p_.data_ptr(),tuple(p_.shape)) for p_ in model.parameters()),
                 bool(_BUFDET['on']),bool(_STF['on']),bool(_XDG['on']),_XDG.get('cur'),bool(_PJ['on']),_ENC_STOP['k'],bool(_CARRY.get('fac')),
                 bool(_NGPD['on']),int(_INC['k']),bool(model.training),torch.is_grad_enabled()))
def _cdg_plan(model,cfg,m,dom,W_,st0,dev):
    """which graph set this call uses (None: eager)"""
    if _CDG['off'] is not None or not (dev.type=='cuda' or cfg.cd_graph_fake): return None
    if _CDG['dom']!=dom:                                                       # one domain per visit: free the previous visit's graphs
        _CDG['g'].clear(); _CDG['seen'].clear(); _CDG['dom']=dom; _CDG['pool']=None
        if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
    if cfg.cd_graph_pad:
        mb=64
        while mb<m: mb*=2
    else: mb=m
    key=(dom,mb,int(W_),_cdg_sig(st0,m),_cdg_fp(model))
    if key not in _CDG['g']:
        if not cfg.cd_graph_pad:
            _CDG['seen'][key]=_CDG['seen'].get(key,0)+1
            if _CDG['seen'][key]<int(cfg.cd_graph_min): return None
        if len(_CDG['g'])>=48: _CDG['g'].pop(next(iter(_CDG['g'])))                        # bound the number of graph sets (memory)
        if dev.type=='cuda':
            fr_=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved()-torch.cuda.memory_allocated())/2**30   # free + PyTorch's unused cache
            if fr_<float(cfg.cd_graph_free_gb):                                   # MEMORY GUARD: this batch size stays eager
                _CDG['n_skip']=_CDG.get('n_skip',0)+1
                if _CDG['n_skip']==1: print(f"  [cd graph] only {fr_:.1f} GB free (< {float(cfg.cd_graph_free_gb):.0f}): new graph sets skipped while memory is short",flush=True)
                return None
            if _CDG.get('pool') is None: _CDG['pool']=torch.cuda.graph_pool_handle()   # ONE memory pool shared by every graph (each graph's outputs
        _CDG['g'][key]={'t':{},'pool':_CDG.get('pool'),'calls':0,'st':None}        #   are read before any other graph is replayed)
    G_=_CDG['g'][key]; G_['calls']+=1
    return {'G':G_,'mb':mb,'chk':(G_['calls']==1 or G_['calls']%max(1,int(cfg.cd_graph_check))==0)}
def _cdg_capture(model,cfg,G_,t,Yw,cur,m,mb,dom,padded):
    if _CDG['n_cap']>=int(getattr(cfg,'cd_graph_max_rec',800)):              # SAFETY: recording should be rare; runaway re-recording -> eager
        raise RuntimeError(f"{_CDG['n_cap']} graphs recorded so far: re-recording too often")
    t0_=time.time(); dev=Yw.device
    Ys=torch.empty((mb,Yw.shape[1]),dtype=Yw.dtype,device=dev); _cdg_fill(Ys,Yw,m)
    n_=mb if padded else m                                                     # rows of the incoming state (padded after the first byte)
    mk1=lambda t_: torch.empty(((t_.shape[0]//n_)*mb,*t_.shape[1:]),dtype=t_.dtype,device=dev)
    sk_=tuple((k_,tuple(tuple(mk1(x_).shape) for x_ in (v_ if isinstance(v_,tuple) else (v_,)))) for k_,v_ in sorted(cur.items()))
    if G_.get('st') is None: G_['st']={}
    if sk_ not in G_['st']: G_['st'][sk_]={k_:(tuple(mk1(x_) for x_ in v_) if isinstance(v_,tuple) else mk1(v_)) for k_,v_ in cur.items()}
    Ss=G_['st'][sk_]; _cdg_fill_st(Ss,cur,m)                                   # static state buffers shared by the byte positions with the same shapes
    fn=lambda: _cdg_once(model,dom,Ys,Ss,mb)
    if dev.type!='cuda':                                                       # CPU test: same plumbing, forward re-run at every 'replay'
        lg_,new_=fn(); gr=_CdgFake(fn,lg_.clone(),{k_:(tuple(x_.clone() for x_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in new_.items()})
        G_['t'][t]={'g':gr,'Y':Ys,'st':Ss,'lg':gr.lg,'new':gr.new}; _CDG['n_cap']+=1; _CDG['t_cap']+=time.time()-t0_; return
    if _CDG['n_cap']==0:                                                       # first recording of the run: a forward that synchronises with the CPU
        prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(2)   #   cannot be recorded -> find out BEFORE recording (raises here)
        try: fn()
        finally: torch.cuda.set_sync_debug_mode(prev_)
    s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): fn()                                                #   warm-up (lazy initialisation, workspaces) off the recording
    torch.cuda.current_stream().wait_stream(s_)
    gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr,pool=G_['pool']): lg_,new_=fn()
    G_['t'][t]={'g':gr,'Y':Ys,'st':Ss,'lg':lg_,'new':new_}; _CDG['n_cap']+=1
    torch.cuda.synchronize(); torch.cuda.empty_cache()                         # hand fragmented cache back (recording is rare: once per set and position)
    _CDG['t_cap']+=time.time()-t0_
def _cdg_step(model,cfg,P_,t,Yw,cur,m,dom,padded):
    """generation forward at byte position t through the recorded graph -> (logits (m, V), state, state padded?)"""
    G_,mb=P_['G'],P_['mb']
    try:
        if t not in G_['t']: _cdg_capture(model,cfg,G_,t,Yw,cur,m,mb,dom,padded)
        e_=G_['t'][t]
        if e_['Y'].shape[1]!=Yw.shape[1]: raise RuntimeError(f'window length changed ({e_["Y"].shape[1]} -> {Yw.shape[1]})')
        _cdg_fill(e_['Y'],Yw,m); _cdg_fill_st(e_['st'],cur,m); e_['g'].replay(); _CDG['n_rep']+=1; _CDG['rows']+=m; _CDG['pad_rows']+=mb-m
        lg=e_['lg'][:m]
        if P_['chk']:                                                         # SELF-CHECK: the eager forward on the same inputs
            sv_=_carry_save(); lge,_=_cdg_once(model,dom,Yw,_cdg_slice(cur,m,mb) if padded else cur,m); _carry_load(sv_)
            d_=float((lge-lg).abs().max()); sc_=float(lge.abs().max())+1e-6; ag_=float((lge.argmax(-1)==lg.argmax(-1)).float().mean())
            _CDG['n_chk']+=1; _CDG['n_bit']+=int(torch.equal(lge,lg)); _CDG['maxd']=max(_CDG['maxd'],d_/sc_); _CDG['minag']=min(_CDG['minag'],ag_)
            if d_/sc_>float(cfg.cd_graph_tol) or ag_<0.95:                    #   (a stale or broken graph gives unrelated logits; padding only rounding)
                raise RuntimeError(f'self-check failed at byte {t}: max logit difference {d_/sc_:.2e} of the largest logit, same top byte {ag_:.4f}')
        return lg,e_['new'],True
    except Exception as ex_:
        _CDG['off']=repr(ex_)[:300]; _CDG['g'].clear(); _CDG['pool']=None
        if Yw.is_cuda: torch.cuda.synchronize(); torch.cuda.empty_cache()          # free the graphs' memory before carrying on eagerly
        print(f"  [cd graph] OFF from here on (eager generation, unchanged results): {_CDG['off']}",flush=True)
        lg_,new_=_cdg_once(model,dom,Yw,_cdg_slice(cur,m,mb) if padded else cur,m); return lg_,new_,False
def _cdg_rows(st,a,b,m):
    """rows a..b of a state dict whose tensors have m row groups"""
    f_=lambda t_: t_.view(m,t_.shape[0]//m,*t_.shape[1:])[a:b].reshape((b-a)*(t_.shape[0]//m),*t_.shape[1:])
    return {k_:(tuple(f_(x_) for x_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cdg_plans(model,cfg,m,dom,W_,st0,dev):
    """ONE plan per chunk of at most cd_graph_rows rows, made once per generation call from the starting state (the state's shapes grow
    with the byte position, so planning per position would give every position its own graph set)"""
    R_=max(64,int(getattr(cfg,'cd_graph_rows',512))); out=[]
    for a_ in range(0,m,R_):
        b_=min(m,a_+R_); out.append((a_,b_,_cdg_plan(model,cfg,b_-a_,dom,W_,_cdg_rows(st0,a_,b_,m) if (a_>0 or b_<m) else st0,dev)))
    return out
def _cdg_gen_step(model,cfg,t,Yw,cur,m,dom,plans):
    """one byte position for m rows through the graphs, chunk by chunk (state passed between positions unpadded)"""
    if len(plans)==1:
        a_,b_,P_=plans[0]
        if P_ is None: return None
        lg,new,pd_=_cdg_step(model,cfg,P_,t,Yw,cur,m,dom,False)
        return lg,(_cdg_slice(new,m,P_['mb']) if pd_ else new)
    lgs=[]; news=[]
    for a_,b_,P_ in plans:
        ca_=_cdg_rows(cur,a_,b_,m)
        if P_ is None or _CDG['off'] is not None:                             # (eager for this chunk: memory guard or graphs off)
            lg_,new_=_cdg_once(model,dom,Yw[a_:b_],ca_,b_-a_)
        else:
            lg_,new_,pd_=_cdg_step(model,cfg,P_,t,Yw[a_:b_],ca_,b_-a_,dom,False)
            if pd_: new_=_cdg_slice(new_,b_-a_,P_['mb'])
        lgs.append(lg_.clone()); news.append({k_:(tuple(x_.clone() for x_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in new_.items()})
    def catst(ds):
        out={}
        for k_ in ds[0]:
            if isinstance(ds[0][k_],tuple): out[k_]=tuple(torch.cat([d_[k_][i] for d_ in ds],0) for i in range(len(ds[0][k_])))
            else: out[k_]=torch.cat([d_[k_] for d_ in ds],0)
        return out
    return torch.cat(lgs,0),catst(news)
def _cdg_msg():
    if not (_CDG['n_rep'] or _CDG['off']): return ''
    return (f" || cuda graphs: {_CDG['n_rep']} graph forwards ({_CDG['n_cap']} recorded, {_CDG['t_cap']:.1f}s), padding {_CDG['pad_rows']/max(1,_CDG['rows']+_CDG['pad_rows']):.0%} of rows"
            f" | self-checks {_CDG['n_chk']}: bit-identical {_CDG['n_bit']}, max logit diff {_CDG['maxd']:.1e}, same top byte >= {_CDG['minag']:.4f}"+(f" | {torch.cuda.mem_get_info()[0]/2**30:.1f} GB free, peak allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if torch.cuda.is_available() else '')+(f" | sets skipped for memory {_CDG.get('n_skip',0)}" if _CDG.get('n_skip') else '')+(f" | OFF: {_CDG['off'][:120]}" if _CDG['off'] else ''))
# ---------------- v9.8 cg5: SLOT MEMORY GENERATION (cd_slot), with CUDA graphs bound to one persistent buffer ----------------
_CG5={'key':None,'cap':0,'B':None,'Y':None,'LG':None,'g':{},'pool':None,'nchk':0,'chk_calls':0}
def _slot_ok(st0,m):
    """the starting state is a set of factored memories (U, K) for m rows (no other carried statistics)"""
    why=None
    if not st0: why='empty state'
    for k_,v_ in (st0 or {}).items():
        if why: break
        if isinstance(k_,tuple): why=f'extra carried statistic {k_[1:]}'
        elif not (isinstance(v_,tuple) and len(v_)==2): why=f'memory not factored ({type(v_).__name__})'
        elif not (v_[0].dim()==3 and v_[0].shape[0]%m==0): why=f'factor shape {tuple(v_[0].shape)} for {m} rows'
        elif v_[0].dtype!=torch.float32 or v_[1].dtype!=torch.float32: why=f'factor type {v_[0].dtype}'
    if why and not _CG5.get('why'): _CG5['why']=why; print(f"  [slot memory] not used for this state: {why} (old generation path)",flush=True)
    return why is None
def _slot_alloc(st0,m,cap,R,dev):
    return {k_:(torch.zeros((cap*(v_[0].shape[0]//m),R,v_[0].shape[2]),device=dev),torch.zeros((cap*(v_[0].shape[0]//m),R,v_[1].shape[2]),device=dev),
                torch.zeros((cap*(v_[0].shape[0]//m),R),device=dev),torch.zeros((cap*(v_[0].shape[0]//m),),device=dev)) for k_,v_ in st0.items()}
def _slot_fill(B_,st0,m):
    for k_,(U,K,Gs,Gt) in B_.items():
        U0,K0=st0[k_]; n_=U0.shape[0]; r0=U0.shape[1]
        U[:n_,:r0].copy_(U0); U[:n_,r0:].zero_(); K[:n_,:r0].copy_(K0); K[:n_,r0:].zero_(); Gs[:n_].zero_(); Gt[:n_].zero_()
def _slot_view(B_,a,b,m_tot,r0,i,mh_of):
    return {k_:_SlotF(U[a*mh_of[k_]:b*mh_of[k_]],K[a*mh_of[k_]:b*mh_of[k_]],Gs[a*mh_of[k_]:b*mh_of[k_]],Gt[a*mh_of[k_]:b*mh_of[k_]],r0,i) for k_,(U,K,Gs,Gt) in B_.items()}
def _slot_fwd(model,dom,Yv,sl,nr):
    _CARRY.update(on=True,nr=nr,st=sl,new={},rows=None); _INC['on']=True
    try: return model(Yv,dom)[:,-1].float()
    finally: _INC['on']=False
def _slot_clone(sl): return {k_:_SlotF(v_.U.clone(),v_.K.clone(),v_.Gs.clone(),v_.Gt.clone(),v_.r0,v_.i) for k_,v_ in sl.items()}
def _slot_copy(dst,src):
    for k_ in dst: dst[k_].U.copy_(src[k_].U); dst[k_].K.copy_(src[k_].K); dst[k_].Gs.copy_(src[k_].Gs); dst[k_].Gt.copy_(src[k_].Gt)
def _cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    """as _cd_gen (same bytes, same log-probabilities up to rounding), with the memories in fixed-size slot buffers written in place"""
    dev=next(model.parameters()).device; W_=int(cfg.cd_inc_win); _INC['k']=next((i for i,b in enumerate(model.blocks) if not getattr(b,'loc',False)),1)
    k0_=next(iter(st0)); r0=st0[k0_][0].shape[1]; R=r0+L; mh_of={k_:v_[0].shape[0]//m for k_,v_ in st0.items()}
    gr_=bool(getattr(cfg,'cd_graph',False) and (dev.type=='cuda' or getattr(cfg,'cd_graph_fake',False)) and _CDG['off'] is None)
    RB=max(64,int(getattr(cfg,'cd_graph_rows',512)))
    if gr_:
        key=(dom,R,r0,L,W_,tuple(sorted((k_,mh_of[k_],tuple(v_[0].shape[2:]),tuple(v_[1].shape[2:])) for k_,v_ in st0.items())),_cdg_fp(model))
        need=-(-m//RB)*RB
        if _CG5['key']!=key or _CG5['cap']<need:
            _CG5['g'].clear(); _CG5['B']=None; _CG5['Y']=None; _CG5['LG']=None; _CG5['pool']=None   # (a pool whose graphs are all freed must not be reused)
            if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
            cap=max(need,_CG5['cap'] if _CG5['key'] is not None and _CG5['key'][:6]==key[:6] else 0)
            ok_=True
            if dev.type=='cuda':
                byt=sum(cap*mh_of[k_]*R*(v_[0].shape[2]+v_[1].shape[2]+1)*4 for k_,v_ in st0.items())
                fr_=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
                ok_=fr_-byt>float(cfg.cd_graph_free_gb)*2**30
                if not ok_:
                    _CDG['n_skip']=_CDG.get('n_skip',0)+1
                    if _CDG['n_skip']==1: print(f"  [cd graph] slot buffers need {byt/2**30:.1f} GB, {fr_/2**30:.1f} GB free: eager slot generation for this call",flush=True)
            if ok_:
                _CG5['B']=_slot_alloc(st0,m,cap,R,dev); _CG5['Y']=torch.full((cap,L+1),M,dtype=torch.long,device=dev); _CG5['LG']=torch.zeros((cap,256),device=dev)
                _CG5['key']=key; _CG5['cap']=cap
                if dev.type=='cuda' and _CG5.get('pool') is None: _CG5['pool']=torch.cuda.graph_pool_handle()
            else: gr_=False; _CG5['key']=None; _CG5['cap']=0
    if gr_: B_=_CG5['B']; Ybuf=_CG5['Y']; LG=_CG5['LG']; cap=_CG5['cap']
    else: cap=m; B_=_slot_alloc(st0,m,m,R,dev); Ybuf=torch.full((m,L+1),M,dtype=torch.long,device=dev); LG=None
    _slot_fill(B_,st0,m); Ybuf[:m,0]=M
    nch=-(-m//RB); lps=[]
    for t in range(L):
        y0=max(0,t+1-W_) if W_>0 else 0
        if gr_ and _CDG['off'] is None:
            for j in range(nch):
                a=j*RB; b=a+RB; Yv=Ybuf[a:b,y0:t+1]; sl=_slot_view(B_,a,b,cap,r0,r0+t,mh_of)
                def fn(Yv=Yv,sl=sl,a=a,b=b): LG[a:b].copy_(_slot_fwd(model,dom,Yv.contiguous(),sl,RB))
                gk=(j,t)
                try:
                    if gk not in _CG5['g']:
                        if _CDG['n_cap']>=int(getattr(cfg,'cd_graph_max_rec',800)): raise RuntimeError(f"{_CDG['n_cap']} graphs recorded so far: re-recording too often")
                        t0_=time.time(); i_=r0+t
                        sv_={k_:(v_.U[:,i_].clone(),v_.K[:,i_].clone(),v_.Gs[:,i_].clone(),v_.Gt.clone()) for k_,v_ in sl.items()}
                        def rst():
                            for k_,v_ in sl.items(): v_.U[:,i_].copy_(sv_[k_][0]); v_.K[:,i_].copy_(sv_[k_][1]); v_.Gs[:,i_].copy_(sv_[k_][2]); v_.Gt.copy_(sv_[k_][3])
                        if dev.type!='cuda': _CG5['g'][gk]=fn
                        else:
                            if _CDG['n_cap']==0:                                       # a forward that synchronises cannot be recorded: find out first
                                prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(2)
                                try: fn()
                                finally: torch.cuda.set_sync_debug_mode(prev_); rst()
                            s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s_):
                                for _ in range(2): fn(); rst()                         # warm-up runs write slot i: undone
                            torch.cuda.current_stream().wait_stream(s_)
                            g5=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g5,pool=_CG5['pool']): fn()           # (recording executes nothing)
                            _CG5['g'][gk]=g5; torch.cuda.synchronize(); torch.cuda.empty_cache()
                        _CDG['n_cap']+=1; _CDG['t_cap']+=time.time()-t0_; del sv_
                    _CG5['chk_calls']+=1; ce_=int(cfg.cd_graph_check) if _CDG['n_chk']<40 else int(getattr(cfg,'cd_graph_check2',250))
                    chk=(_CG5['chk_calls']%max(1,ce_)==0)
                    if chk: slc=_slot_clone(sl); Yc=Yv.contiguous().clone()
                    g5=_CG5['g'][gk]
                    (g5() if dev.type!='cuda' else g5.replay()); _CDG['n_rep']+=1; nv_=max(0,min(b,m)-a); _CDG['rows']+=nv_; _CDG['pad_rows']+=RB-nv_
                    if chk:                                                        # SELF-CHECK: the eager forward on copies of the same inputs
                        lge=_slot_fwd(model,dom,Yc,slc,RB); lgg=LG[a:a+nv_]; lge_=lge[:nv_]
                        d_=float((lge_-lgg).abs().max()) if nv_ else 0.0; sc_=float(lge_.abs().max())+1e-6 if nv_ else 1.0
                        ag_=float((lge_.argmax(-1)==lgg.argmax(-1)).float().mean()) if nv_ else 1.0
                        ds_=max(float((slc[k_].Gt-sl[k_].Gt).abs().max()) for k_ in sl)
                        _CDG['n_chk']+=1; _CDG['n_bit']+=int(torch.equal(lge_,lgg)); _CDG['maxd']=max(_CDG['maxd'],d_/sc_); _CDG['minag']=min(_CDG['minag'],ag_)
                        if d_/sc_>float(cfg.cd_graph_tol) or ag_<0.95 or ds_>1e-4:
                            _slot_copy(sl,slc); LG[a:b].copy_(lge)
                            raise RuntimeError(f'self-check failed at byte {t}: logit difference {d_/sc_:.2e}, same top byte {ag_:.4f}, memory difference {ds_:.1e}')
                        del slc,Yc
                except Exception as ex_:
                    _CDG['off']=repr(ex_)[:300]; _CG5['g'].clear(); _CG5['key']=None; _CG5['pool']=None
                    if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
                    print(f"  [cd graph] OFF from here on (eager slot generation, unchanged results): {_CDG['off']}",flush=True)
                    if 'self-check failed' not in _CDG['off']:                    # (a failed self-check already restored the correct result)
                        for jj in range(j,nch):
                            aa=jj*RB; bb=min(aa+RB,m)
                            if bb>aa: LG[aa:bb].copy_(_slot_fwd(model,dom,Ybuf[aa:bb,y0:t+1].contiguous(),_slot_view(B_,aa,bb,cap,r0,r0+t,mh_of),bb-aa))
                        break
            lg=LG[:m].clone()
        else:
            lg=_slot_fwd(model,dom,Ybuf[:m,y0:t+1].contiguous(),_slot_view(B_,0,m,cap,r0,r0+t,mh_of),m)
        lps.append(F.log_softmax(lg,-1))
        if ban is not None: lg=lg.masked_fill((flen==t)[:,None]&(torch.arange(lg.shape[-1],device=dev)[None]==ban[:,None]),float('-inf'))
        nx_=lg.argmax(-1,keepdim=True) if temp<=0 else torch.multinomial(F.softmax(lg/float(temp),-1),1,generator=g_)
        if forced is not None: nx_=torch.where((t<flen)[:,None],forced[:,t:t+1],nx_)
        if hint is not None: nx_=torch.where(hint[0][:,t:t+1],hint[1][:,t:t+1],nx_)   # cg12b: hinted bytes forced
        cf_=lps[-1].gather(1,nx_).squeeze(1); _CDCONF['c']=cf_ if t==0 else _CDCONF['c']+cf_
        Ybuf[:m,t+1]=nx_[:,0]
    return Ybuf[:m,1:L+1].clone(), torch.stack(lps,1)
_SLOTCHK={'n':0}
def _cd_gen(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    if getattr(cfg,'cd_slot',False) and bool(cfg.cd_inc) and _slot_ok(st0,m):
        if int(getattr(cfg,'cd_slot_check',0))>_SLOTCHK['n']:                  # TEST: old path vs slot path on the same inputs (same random state)
            _SLOTCHK['n']+=1; gs_=g_.get_state(); c0_=_CDCONF.get('c')
            Yo,lo=_cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint); g_.set_state(gs_)
            Yn,ln=_cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
            same_=float((Yo==Yn).float().mean()); fin_=torch.isfinite(lo)&torch.isfinite(ln)
            print(f"  [slot check {_SLOTCHK['n']}] {m} rows x {L} bytes: same bytes {same_:.4f} | largest log-probability difference {float((lo-ln).abs()[fin_].max()):.2e}",flush=True)
            return Yn,ln
        return _cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
    return _cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
def _cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    """m segments generated from their traces st0 (temp 0: greedy). Returns the bytes (m, L) and the teacher's log-probabilities at every step
    (m, L, 256): lp[:,j] predicts byte j from [mark] + bytes[:j] with the trace (the same as the parallel teacher pass over the result)."""
    dev=next(model.parameters()).device; Y=torch.full((m,1),M,dtype=torch.long,device=dev); lps=[]; cur=st0
    inc_=bool(cfg.cd_inc); W_=int(cfg.cd_inc_win)
    if inc_: _INC['k']=next((i for i,b in enumerate(model.blocks) if not getattr(b,'loc',False)),1)
    gr_=bool(inc_ and getattr(cfg,'cd_graph',False) and (dev.type=='cuda' or getattr(cfg,'cd_graph_fake',False)))   # CUDA GRAPHS (cd_graph)
    pl_=_cdg_plans(model,cfg,m,dom,W_,st0,dev) if (gr_ and _CDG['off'] is None) else None
    for t in range(L):
        o_=_cdg_gen_step(model,cfg,t,Y[:,-W_:] if W_>0 else Y,cur,m,dom,pl_) if (pl_ is not None and _CDG['off'] is None) else None
        if o_ is not None: lg,cur=o_
        else:
          _CARRY.update(on=True,nr=m,st=cur,new={},rows=None)
          if inc_:
            _INC['on']=True
            try: lg=model(Y[:,-W_:] if W_>0 else Y,dom)[:,-1].float()
            finally: _INC['on']=False
            cur={k_:v_ for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}    # memory after this byte (factored)
          else: lg=model(Y,dom)[:,-1].float()
        lps.append(F.log_softmax(lg,-1))
        if ban is not None: lg=lg.masked_fill((flen==t)[:,None]&(torch.arange(lg.shape[-1],device=dev)[None]==ban[:,None]),float('-inf'))   # v9e repair
        nx_=lg.argmax(-1,keepdim=True) if temp<=0 else torch.multinomial(F.softmax(lg/float(temp),-1),1,generator=g_)
        if forced is not None: nx_=torch.where((t<flen)[:,None],forced[:,t:t+1],nx_)                # v9e: the verified prefix kept
        if hint is not None: nx_=torch.where(hint[0][:,t:t+1],hint[1][:,t:t+1],nx_)                 # cg12b: hinted bytes forced
        cf_=lps[-1].gather(1,nx_).squeeze(1); _CDCONF['c']=cf_ if t==0 else _CDCONF['c']+cf_   # confidence: sum log p(chosen byte)
        Y=torch.cat([Y,nx_],1)
    return Y[:,1:], torch.stack(lps,1)
@torch.no_grad()
def _prot_minv(model,cfg,dev):
    """(A_old + lam I)^-1 for the head input: uncentred covariance of the old domains' head inputs on their own dream-bank rows (K-FAC A factor)."""
    vi_=model.__dict__.get('_visit'); C_=model.__dict__.get('_protM')
    if C_ is not None and C_[0]==vi_: return C_[1]
    BK=model.__dict__.get('_bank',{}); ol=[d for d in (getattr(model,'_olds',None) or []) if d in BK and d!=model.__dict__.get('_self_slot')]
    if not ol: model.__dict__['_protM']=(vi_,None); return None
    sv_=_carry_save(); suh_=_uh_save(model); was_=model.training; model.eval(); fs_=model.feat_scale.clone(); A_=None; N_=0
    _CARRY.update(on=False,st={},new={})
    try:
        for d in ol:
            X_=BK[d][:int(cfg.cd_prot_rows),:int(cfg.seq_len)].long().to(dev)
            _HCAP['on']=True; _HCAP['h']=None
            try: model(X_,d)
            finally: _HCAP['on']=False
            z_=_HCAP['h'].float().reshape(-1,_HCAP['h'].shape[-1]); _HCAP['h']=None
            A_=z_.t()@z_ if A_ is None else A_+z_.t()@z_; N_+=z_.shape[0]
    finally: _carry_load(sv_); _uh_load(model,suh_); model.feat_scale.copy_(fs_); model.train(was_)
    A_=A_/max(1,N_); lam_=float(cfg.cd_prot_lam)*float(torch.diagonal(A_).mean())
    M_=torch.linalg.inv(A_+lam_*torch.eye(A_.shape[0],device=A_.device))
    model.__dict__['_protM']=(vi_,M_); return M_
@torch.no_grad()
def _head_dz(model,d,z,lq):
    """dL/dz of KL(q||p) through domain d's (soft-capped) head: ((p - q) * cap') @ W.  z (B,T,D), lq target log-probs (B,T,256)."""
    hd_=model.heads[d]; u_=F.linear(z.float(),hd_.weight.float(),hd_.bias.float())
    if hd_.cap>0: t_=torch.tanh(u_/hd_.cap); lg_=hd_.cap*t_; dc_=1-t_*t_
    else: lg_=u_; dc_=torch.ones_like(u_)
    g_=(F.softmax(lg_,-1)-lq.float().exp())*dc_
    return g_@hd_.weight.float()
@torch.no_grad()
def _sig_update(model,d,z,lp,lq):
    dz_=_head_dz(model,d,z,lq)
    u_=z.float().mean((0,1)); v_=dz_.mean((0,1)); S_=model.__dict__.get('_cur_sig')
    model.__dict__['_cur_sig']=(u_,v_) if S_ is None else (0.8*S_[0]+0.2*u_,0.8*S_[1]+0.2*v_)
@torch.no_grad()
def _sim_rows(model,cfg,dm,ri,pick,BK,TV,r_):
    """SWIL row choice: for each dream slot (domain d), dream_sim_k candidate rows of d's bank, scored by the rank-1 signature
    interaction with the current learning; the best candidate replaces the random row (dream_sim_floor of slots stay random)."""
    S_=model.__dict__.get('_cur_sig'); k_=int(cfg.dream_sim_k)
    if S_ is None or k_<2 or not TV: return ri
    uc_,vc_=S_; dev=model.embed.weight.device; T_=min(int(cfg.seq_len),BK[dm[0]].shape[1]); kb_=int(cfg.bank_targets)
    slots=[j for j in range(len(dm)) if r_.rand()>=float(cfg.dream_sim_floor) and dm[j] in TV]
    if not slots: return ri
    sv_=_carry_save(); suh_=_uh_save(model); was_=model.training; model.eval(); fs_=model.feat_scale.clone(); _CARRY.update(on=False,st={},new={})
    out=list(ri); SC_=model.__dict__.setdefault('_simstat',[0,0.0,0.0,0.0])
    try:
        for d in sorted({dm[j] for j in slots}):
            js_=[j for j in slots if dm[j]==d]; cand_=[[ri[j]]+[pick(d) for _ in range(k_-1)] for j in js_]; flat_=[c for cc in cand_ for c in cc]
            X_=torch.stack([BK[d][i][:T_] for i in flat_]).long().to(dev)
            V_=torch.stack([TV[d][0][i][:T_] for i in flat_]).to(dev).float(); I_=torch.stack([TV[d][1][i][:T_] for i in flat_]).to(dev).long()
            rest_=(1-V_.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-kb_); TG_=rest_.log().expand(X_.shape[0],X_.shape[1],256).clone(); TG_.scatter_(-1,I_,V_)
            _HCAP['on']=True; _HCAP['h']=None
            try: model(X_,d)                                                     # (bank targets are stored per position, aligned with the logits)
            finally: _HCAP['on']=False
            z_=_HCAP['h']; _HCAP['h']=None
            dz_=_head_dz(model,d,z_,TG_)
            u_=z_.float().mean(1); v_=dz_.mean(1)
            a_=(v_@vc_)*(u_@uc_)/(vc_.norm()*uc_.norm()).clamp_min(1e-12)        # ~ <G_row, G_cur>: < 0 = the current step HARMS this row
            sc_=(-a_) if cfg.dream_sim=='harm' else a_.abs()
            sc_=sc_.view(len(js_),k_)
            best_=sc_.argmax(1).tolist()
            for jj,(j,b_) in enumerate(zip(js_,best_)): out[j]=cand_[jj][b_]
            SC_[0]+=len(js_); SC_[1]+=float((a_<0).float().mean())*len(js_); SC_[2]+=float(sc_.max(1).values.mean()-sc_.mean())*len(js_)
            SC_[3]+=sum(1 for b_ in best_ if b_!=0)
    finally: _carry_load(sv_); _uh_load(model,suh_); model.feat_scale.copy_(fs_); model.train(was_)
    return out
@torch.no_grad()
def _store_score(tq,lpy,vm,C,L):
    """per window: the mean over its pieces of the POSITIVE part of each piece's reducible signal (log q(y) - log p(y) at the verified bytes)"""
    T1=tq.shape[1]; pg=((torch.arange(T1,device=tq.device)+1)//max(1,L)).clamp(max=C-1)
    r=(tq-lpy).float()*vm; sm=torch.zeros(tq.shape[0],C,device=tq.device).index_add_(1,pg,r); ct=torch.zeros_like(sm).index_add_(1,pg,vm.float())
    return (sm/ct.clamp_min(1)).clamp_min(0).mean(1)
def _nrem_loss(model,cfg,dom):
    """NREM: re-replay the long-term store's windows (any domain), highest remaining signal first; re-teach, re-measure, retire."""
    SB_=model.__dict__.get('_sbuf'); z0_=next(model.parameters()).sum()*0.0
    if not SB_ or not SB_['e']: return z0_
    ST_=SB_['st']; L=int(cfg.cd_seg); M=int(cfg.cd_mark); dev=model.embed.weight.device
    sel_=sorted(SB_['e'],key=lambda e_:-e_['v'])[:int(cfg.cd_store_k)]; drop_=set(); KS_=0.0; NS_=0
    for d_ in sorted({e_['dom'] for e_ in sel_}):
        sd_=[e_ for e_ in sel_ if e_['dom']==d_]; k_l=len(sd_); C=sd_[0]['x'].numel()//L
        with torch.no_grad():
            trs_=_cd_cat([e_['tr'] for e_ in sd_]) if k_l>1 else sd_[0]['tr']
            xb_=torch.stack([e_['x'] for e_ in sd_]); pcs_=xb_.reshape(k_l*C,L)
            Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
            svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
            try:
                with _cd_weights(model):
                    _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,d_).float(),-1)
            finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
            tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sd_]).float(); vml_=(wl_>0).float()
            tqyl_=tgl_.gather(-1,ybl_[...,None]).squeeze(-1)
            if float(cfg.cd_evid_w)>0:
                dgp_=(tgl_.argmax(-1)!=ybl_).float()*vml_; ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1)
                tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
        try: lpl_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
        kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KS_=KS_+(kll_*wl_).sum(); NS_+=wl_.numel()
        with torch.no_grad(): rn_=_store_score(tqyl_,lpl_.gather(-1,ybl_[...,None]).squeeze(-1),vml_,C,L).tolist()
        for e_,r_ in zip(sd_,rn_):
            e_['n']+=1; ST_['nrem']=ST_.get('nrem',0)+1
            if r_<=max(1e-4,float(cfg.cd_store_retire)*e_['v0']): drop_.add(id(e_)); ST_['ret']=ST_.get('ret',0)+1; ST_['ret_n']=ST_.get('ret_n',0)+e_['n']; ST_['nrem_ret']=ST_.get('nrem_ret',0)+1; continue
            e_['v']=r_
    SB_['e']=[e_ for e_ in SB_['e'] if id(e_) not in drop_]
    return KS_/max(1,NS_) if NS_>0 else z0_
def lora_merge(model,cfg,olds,cur):
    """LOW-RANK DREAM MERGE: per old domain, the trunk update that restores its dreams (stored targets) -> rank-r -> merged -> no-harm check -> applied."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[d for d in olds if d in BK and d in TV and d!=cur]
    if not ol: return
    t0=time.time(); dev=model.embed.weight.device; kb=int(cfg.bank_targets); T=min(int(cfg.seq_len),BK[ol[0]].shape[1]); V_=int(cfg.lm_val)
    PR=[(n_,p_) for n_,p_ in model.named_parameters() if p_.ndim==2 and n_.startswith('blocks.') and '.pu_' not in n_ and '.film_' not in n_]
    rs=np.random.RandomState(int(cfg.seed)+211+int(model.__dict__.get('_visit',0)))
    def tg_(d,idx):
        X=torch.stack([BK[d][i][:T] for i in idx]).long().to(dev); Vv=torch.stack([TV[d][0][i][:T] for i in idx]).to(dev).float(); Ii=torch.stack([TV[d][1][i][:T] for i in idx]).to(dev).long()
        rest=(1-Vv.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-kb); G=rest.log().expand(X.shape[0],X.shape[1],256).clone(); G.scatter_(-1,Ii,Vv); return X,G
    def kl_(d,X,G): lp=F.log_softmax(model(X,d).float(),-1); return (G.exp()*(G-lp)).sum(-1).mean()
    SBk_=model.__dict__.get('_sbuf') if cfg.lm_store_check else None
    STW=[e_ for e_ in (SBk_['e'] if SBk_ else []) if e_['dom'] in ol][:32]
    def store_ce():                                                        # real-text check: the student's loss on the stored VERIFIED bytes of old domains
        if not STW: return None
        tot=0.0; cnt=0.0
        for d_ in sorted({e_['dom'] for e_ in STW}):
            xs=torch.stack([e_['x'] for e_ in STW if e_['dom']==d_]); ws=torch.stack([e_['w'] for e_ in STW if e_['dom']==d_]).float()
            lp=F.log_softmax(model(xs[:,:-1],d_).float(),-1); ce=-lp.gather(-1,xs[:,1:,None]).squeeze(-1)
            tot+=float((ce*(ws>0).float()).sum()); cnt+=float((ws>0).float().sum())
        return tot/max(1.0,cnt)
    sv_=_carry_save(); suh_=_uh_save(model); was_=model.training; fs_=model.feat_scale.clone(); _CARRY.update(on=False,st={},new={}); model.eval()
    LR_=model.__dict__.get('_last_rep')
    try:
        VAL={}; base={}
        with torch.no_grad():
            for d in ol: n_=BK[d].shape[0]; VAL[d]=tg_(d,list(range(min(V_,n_//4)))); base[d]=float(kl_(d,*VAL[d]))
            cur_p=F.log_softmax(model(LR_[0][:8,:-1],cur).float(),-1) if (LR_ is not None and LR_[1]==cur) else None
            sce0_=store_ce()
        P0={n_:p_.detach().clone() for n_,p_ in PR}; DL={}; fit={}; lowr={}
        for d in ol:
            op_=torch.optim.Adam([p_ for _,p_ in PR],lr=float(cfg.lm_lr)); n_=BK[d].shape[0]; lo_=min(V_,n_//4)
            for _ in range(int(cfg.lm_steps)):
                X,G=tg_(d,rs.randint(lo_,n_,size=int(cfg.lm_rows)).tolist()); op_.zero_grad(set_to_none=True); kl_(d,X,G).backward(); op_.step()
            with torch.no_grad():
                fit[d]=float(kl_(d,*VAL[d])); D={}
                for n_,p_ in PR:
                    dd=p_.detach()-P0[n_]; p_.data.copy_(P0[n_])
                    if int(cfg.lm_rank)>0 and min(dd.shape)>int(cfg.lm_rank):
                        U,S,Vh=torch.linalg.svd(dd.float(),full_matrices=False); r=int(cfg.lm_rank); dd=(U[:,:r]*S[:r])@Vh[:r]
                    D[n_]=dd.to(p_.dtype)
                for n_,p_ in PR: p_.data.add_(D[n_])
                lowr[d]=float(kl_(d,*VAL[d]))
                for n_,p_ in PR: p_.data.copy_(P0[n_])
            DL[d]=D
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            MG={}; conf=[]
            for n_,_ in PR:
                S_=torch.stack([DL[d][n_].float() for d in ol])
                if cfg.lm_merge=='ties':
                    k_=max(1,int(0.2*S_[0].numel())); th_=S_.abs().flatten(1).kthvalue(S_[0].numel()-k_+1,dim=1).values.view(-1,*([1]*(S_.dim()-1)))
                    St_=torch.where(S_.abs()>=th_,S_,torch.zeros_like(S_)); sg_=torch.sign(St_.sum(0)); ag_=(torch.sign(St_)==sg_)&(St_!=0)
                    MG[n_]=(St_*ag_).sum(0)/ag_.sum(0).clamp_min(1); nz_=(St_!=0).sum(0); conf.append(float(((nz_>ag_.sum(0))&(nz_>0)).float().mean()))
                elif cfg.lm_merge=='sum': MG[n_]=S_.sum(0)
                else: MG[n_]=S_.mean(0)
            res=[]; acc=None
            for sc in cfg.lm_scales:
                for n_,p_ in PR: p_.data.copy_(P0[n_]+float(sc)*MG[n_].to(p_.dtype))
                ks={d:float(kl_(d,*VAL[d])) for d in ol}
                cd_=float((cur_p.exp()*(cur_p-F.log_softmax(model(LR_[0][:8,:-1],cur).float(),-1))).sum(-1).mean()) if cur_p is not None else 0.0
                ok_=all(ks[d]<=base[d]*(1+float(cfg.lm_eps)) for d in ol) and np.mean(list(ks.values()))<np.mean(list(base.values())) and cd_<=float(cfg.lm_cur_eps)
                if sce0_ is not None:
                    sce_=store_ce(); ok_=ok_ and sce_<=sce0_; cd_=(cd_,sce_-sce0_)
                res.append((sc,ks,cd_,ok_))
                if ok_ and acc is None: acc=sc; break
            if acc is None:
                for n_,p_ in PR: p_.data.copy_(P0[n_])
    finally: _carry_load(sv_); _uh_load(model,suh_); model.feat_scale.copy_(fs_); model.train(was_)
    mb_=np.mean(list(base.values())); mf_=np.mean(list(fit.values())); ml_=np.mean(list(lowr.values())); last_=res[-1]
    cap_=(mb_-ml_)/max(1e-9,mb_-mf_)
    print(f"  [dream merge] {len(ol)} old domains, rank {cfg.lm_rank if int(cfg.lm_rank)>0 else 'full'}, {cfg.lm_merge}: held-out dream KL {mb_:.4f} -> own full fit {mf_:.4f}, own rank-r {ml_:.4f}"
          f" (rank-r keeps {cap_:.0%} of the restoring effect) | sign conflicts {np.mean(conf) if conf else 0:.2f} | "
          +(f"ACCEPTED scale {acc:g}: merged KL {np.mean(list(last_[1].values())):.4f}, visited-domain drift {last_[2] if not isinstance(last_[2],tuple) else last_[2][0]:.4f}"+(f", stored real windows {last_[2][1]:+.4f} nats" if isinstance(last_[2],tuple) else "") if acc is not None else
            "REJECTED at every scale ("+"; ".join(f"x{r_[0]:g}: KL {np.mean(list(r_[1].values())):.4f}, worst x{max(r_[1][d]/max(1e-9,base[d]) for d in ol):.2f}, drift {r_[2] if not isinstance(r_[2],tuple) else r_[2][0]:.4f}"+(f", stored real {r_[2][1]:+.4f}" if isinstance(r_[2],tuple) else "") for r_ in res)+")")
          +f" | {time.time()-t0:.0f}s")
@torch.no_grad()
def sem_rescore(model,cfg,d):
    """fresh targets for domain d's semantic rows: the current model's top-k log-probs on the stored real bytes (no trace), as a dream bank's"""
    SM_=model.__dict__.get('_sem',{}).get(int(d))
    if not SM_ or not SM_['x']: return
    k_=int(cfg.bank_targets); was=model.training; model.eval(); fs=model.feat_scale.clone(); sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=int(d)
    svc_=_carry_save(); _CARRY.update(on=False,st={},new={}); dev=model.embed.weight.device; n_=len(SM_['x'])
    try:
        for c0 in range(0,n_,32):
            Xc=torch.stack(SM_['x'][c0:c0+32]).long().to(dev)
            lp=F.log_softmax(model.heads[d](priv(model,Xc,model.features(Xc),d)).float()/cfg.shunt_T,-1); v_,i_=lp.topk(k_,-1)
            for j in range(Xc.shape[0]): SM_['v'][c0+j]=v_[j].to(torch.float16).cpu(); SM_['i'][c0+j]=i_[j].to(torch.uint8).cpu()
    finally: _carry_load(svc_); model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs); model.train(was)
    print(f"  [semantic rows] {_lab(cfg,d)}: {n_} rows re-scored by the current model (fresh targets)")

# =====================================================================================================================
# cg20 OFFLINE INFERENCE IN SLEEP (retrieve -> infer -> verify -> consolidate) on the hippocampal sentence store
#   PROBES   inf_probes chains A -r1-> B (planted in domain dX's text) and B -r2-> C (domain dY != dX), a third of them with C -r3-> D (dZ):
#            every pass, each fact sentence is inserted inf_plant_k times into its domain's (fresh) training text at sentence boundaries,
#            so in passes 2+ the facts reach the weights ONLY through the closed loop (trace -> regeneration -> verification).
#            inf_probe_file: optional JSONL from the API probe script (otherwise templated chains with invented names).
#   STORE    pass with real text: the training windows; passes with real_w 0: ONLY the VERIFIED bytes of the checked replays (unverified
#            positions blanked). Kept: sentences with >= 2 candidate names (clause segments); name frequencies over ALL windows seen.
#   INFER    after each visit, bridge-centred chaining: for each rare name B in this visit's new sentences, its partners (other rare names in
#            the same sentence) across ALL stored sentences; partners from different sentences give conclusions A ~ C. Abstain when B
#            has more than inf_maxfan partners (hub). Typed conclusion "A <r1> someone who <r2> C" when both clauses are subject-first.
#            Conclusions re-enter the store as INFERRED items (they may chain once more with real/verified items -> three-hop, never
#            inferred + inferred). Grounding is built in: each conclusion carries its two source sentences.
#   CONSOLIDATE  inf_steps steps at lr x inf_lr: conclusion sentences planted alone into kept verified windows (CE), interleaved with the
#            visit's kept verified windows (share 1 - inf_mix) under their own slots.
#   DIAGNOSTICS  per sleep: proposals, abstentions, probe chains found (true / wrong), store size; probe recall FROM THE WEIGHTS
#            (one-hop, two-hop, typed two-hop, three-hop) vs 19 distractor names, false-memory preference, BPC before / after.
# =====================================================================================================================
import re as _re_inf
def _ng_zero(model):
    """clear the SPARSE gradients of the hashed n-gram tables (kept in model.__dict__, so not model.parameters()): any extra backward
    (consolidation, sleep phases) otherwise ACCUMULATES them until the next n-gram optimiser step - a huge stale update, and on the GPU an
    int32 overflow in coalesce (the illegal-address crash after the first closed-loop consolidation)"""
    for k_ in ('_ngpi_opt','_ngp_opt','_ngo_opt','_fcp_opt'):
        O_=model.__dict__.get(k_)
        if O_ is not None: O_.zero_grad(set_to_none=True)
def _inf_sync(cfg,tag_):
    """DEBUG (inf_debug_sync): force the GPU to finish everything queued so far; an illegal memory access then surfaces HERE, naming the stage"""
    if not int(getattr(cfg,'inf_debug_sync',0)) or not torch.cuda.is_available(): return
    torch.cuda.synchronize(); n_=_INF.setdefault('dbgn',{}); n_[tag_]=n_.get(tag_,0)+1
    if n_[tag_]<=3 or n_[tag_]%50==0: print(f"  [inference debug] GPU state OK after {tag_} (#{n_[tag_]})",flush=True)
# ---- cg20h: CONSOLIDATION INTO A FAST STORE (inf_cons='mem') instead of the weights -----------------------------------------------
#   a sparse error-memory corrector keyed on the model's PRE-HEAD features z (centred, normalised, random expansion to inf_mem_dx units,
#   top inf_mem_act share kept): logits += gain * phi(z) M. Written in sleep by the delta rule with the residual (onehot - q) on the
#   conclusion bytes, and 'do-no-harm' anchor writes (p_model - q) on recent verified windows. Active only in no-grad evaluation
#   (bpc, probe recall): the weights and the closed loop never see it.
_INFM={'M':None,'on':False,'X':None}
_INFM_TAUS=(8.0,32.0,128.0)
def _infm_init(model,cfg):
    if _INFM['M'] is not None: return
    dev_=model.embed.weight.device; d_=int(model.embed.weight.shape[1]); key_=str(getattr(cfg,'inf_mem_key','z')); g_=torch.Generator().manual_seed(int(cfg.seed)+2222)
    Dk_={'name':d_,'nameb':d_,'namemix':d_,'z':d_,'slow':4*d_,'slowgate':4*d_,'latch':2*d_}[key_]
    DX_=int(getattr(cfg,'inf_mem_dx',0) or (4096 if key_ in ('name','nameb','namemix','z') else 16384)); DXB_=int(getattr(cfg,'inf_mem_dxb',65536)) if key_.startswith('name') else DX_
    _INFM['bpc_on']=str(getattr(cfg,'inf_mem_bpc','on'))!='off'
    _INFM.update(M=torch.zeros(DXB_,256,device=dev_),WX=(torch.randn(DX_,Dk_,generator=g_)/math.sqrt(Dk_)).to(dev_),mu=None,key=key_,DXB=DXB_,names={},d=d_,
                 gates=torch.ones(len(_INFM_TAUS),device=dev_),lthr=None,model=model,dream=str(getattr(cfg,'inf_xfer','none'))=='dream',
                 ka=max(1,int(DX_*float(getattr(cfg,'inf_mem_act',0.02)))),g=float(getattr(cfg,'inf_mem_gain',1.0)),wz=float(getattr(cfg,'inf_mem_wz',0.5)),gthr=float(getattr(cfg,'inf_mem_gate',0.0)))
def _infm_ids(X_):
    """key 'name': per position, the hashed id of the last complete name before it if that name is in the store (else -1: no correction)"""
    Xc_=X_.detach().to('cpu',torch.uint8).numpy(); out_=np.full(Xc_.shape,-1,np.int64); nm_=_INFM['names']
    for r_ in range(Xc_.shape[0]):
        if not _UNITS.get('on'):
            for m_ in _INF_NAME.finditer(bytes(Xc_[r_])): out_[r_,m_.end():]=nm_.get(m_.group(0),-1)
            continue
        L_=_UNITS['lex']
        for m_ in _UEVT.finditer(bytes(Xc_[r_])):                             # rule-free: a stored unit opens its bank; another unit or a sentence end closes it
            g_=m_.group(0)
            if g_ in nm_: out_[r_,m_.end():]=nm_[g_]
            elif len(g_)==1 and g_ in b'.!?\n' or (g_ in L_ and not _UNITS.get('all')): out_[r_,m_.end():]=-1
    return torch.from_numpy(out_).to(X_.device)
def _infm_nrm(v_): return v_/(v_.norm(dim=-1,keepdim=True)+1e-6)
def _infm_bytes(X_):
    """DRIFT-FREE key features: a FIXED random code of the last 8 bytes (order-aware, decaying); never trained, so it cannot drift with the trunk"""
    E_=_INFM.get('EB')
    if E_ is None or E_.device!=X_.device:
        g_=torch.Generator().manual_seed(4321); E_=(torch.randn(8,256,_INFM['d'],generator=g_)/math.sqrt(_INFM['d'])).to(X_.device); _INFM['EB']=E_
    B_,T_=X_.shape; Xl_=X_.long(); out_=torch.zeros(B_,T_,E_.shape[2],device=X_.device)
    for j_ in range(min(8,T_)): out_[:,j_:]+=(0.8**j_)*E_[j_][Xl_[:,:T_-j_]]
    return out_
def _infm_latch(z_,X_,lg_):
    """SURPRISE LATCH: a word whose first two bytes were surprising (above the running 85th percentile of ordinary text) is latched (mean
    feature over the word) and held until the next latched word; positions before any latch get id -1 (no correction)"""
    B_,T_,d_=z_.shape; out_=torch.zeros(B_,T_,d_,device=z_.device); id_=torch.full((B_,T_),-1,dtype=torch.long,device=z_.device)
    lp_=F.log_softmax(lg_.float(),-1); su_=torch.zeros(B_,T_,device=z_.device); su_[:,1:]=-lp_[:,:-1].gather(-1,X_[:,1:,None]).squeeze(-1)/math.log(2)
    Xc_=X_.detach().to('cpu',torch.uint8).numpy(); suc_=su_.detach().cpu().numpy(); thr_=_INFM['lthr'] if _INFM['lthr'] is not None else 1e9; ws_=[]
    isl_=lambda c: (65<=c<=90) or (97<=c<=122)
    for r_ in range(B_):
        st_=-1; cur_=None
        for t_ in range(T_):
            c_=int(Xc_[r_,t_]); 
            if isl_(c_):
                if st_<0: st_=t_
            elif st_>=0:
                w_=float(suc_[r_,st_]+(suc_[r_,st_+1] if st_+1<t_ else 0.0)); ws_.append(w_)
                if w_>thr_ and st_>0: cur_=z_[r_,st_:t_].mean(0)
                st_=-1
            if cur_ is not None: out_[r_,t_]=cur_; id_[r_,t_]=0
    return out_,id_,ws_
def _infm_keys(z_,X_,lg_=None):
    """per-position store keys K (centred, per-piece normalised) and ids (-1: inactive). z (B,T,d), X (B,T) bytes, lg (B,T,256) logits"""
    k_=_INFM['key']; zc_=z_.float()-_INFM['mu']
    if k_=='name': return zc_,_infm_ids(X_)
    if k_=='nameb': return _infm_bytes(X_),_infm_ids(X_)                                    # name bank + byte-context units: no trunk features at all
    if k_=='namemix': return _infm_nrm(_infm_bytes(X_))+_INFM['wz']*_infm_nrm(zc_),_infm_ids(X_)   # mostly stable, partly trunk (rewording)
    if k_=='z': return zc_,torch.zeros(X_.shape,dtype=torch.long,device=X_.device)
    if k_ in ('slow','slowgate'):
        B_,T_,d_=zc_.shape; st_=torch.zeros(B_,len(_INFM_TAUS),d_,device=zc_.device); al_=torch.tensor([1.0/t for t in _INFM_TAUS],device=zc_.device)[None,:,None]; S_=[]
        for t_ in range(T_): st_=(1-al_)*st_+al_*zc_[:,t_,None,:]; S_.append(st_)
        S_=torch.stack(S_,1)                                                        # (B,T,ntau,d)
        K_=torch.cat([_infm_nrm(zc_)]+[_INFM['gates'][j_]*_infm_nrm(S_[:,:,j_]) for j_ in range(len(_INFM_TAUS))],-1)
        return K_,torch.zeros(X_.shape,dtype=torch.long,device=X_.device)
    L_,id_,_=_infm_latch(zc_,X_,lg_)
    return torch.cat([_infm_nrm(zc_),_infm_nrm(L_)],-1),id_
def _infm_sparse(k_,id_):
    y_=_infm_nrm(k_.float())@_INFM['WX'].T
    v_,i_=y_.topk(_INFM['ka'],-1); v_=v_.clamp(min=0)+1e-6; v_=v_/v_.norm(dim=-1,keepdim=True)
    if _INFM['key'].startswith('name'): i_=(i_+((id_[:,None]%1000003)*2654435761)%_INFM['DXB'])%_INFM['DXB']      # conjunction: the name picks the bank, z the units (no int64 overflow)
    i_=i_.clamp_(0,_INFM['M'].shape[0]-1)
    return i_,v_
def _infm_corr(z_,id_):
    sh_=z_.shape; zf_=z_.reshape(-1,sh_[-1]); idf_=id_.reshape(-1); out_=torch.zeros(zf_.shape[0],256,device=z_.device)
    if idf_.numel()!=zf_.shape[0]: return (0.0*out_).view(*sh_[:-1],256)            # misaligned ids: no correction (never index out of range)
    sel_=(idf_>=0).nonzero().squeeze(-1); Mw_=_INFM.get('Mw') if _INFM.get('wm_on') else None; lt_=not _INFM.get('lt_off')
    for c_ in range(0,sel_.numel(),512):
        s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_])
        if lt_: out_[s_]=(v_[...,None]*_INFM['M'][i_]).sum(1)
        if Mw_ is not None: out_[s_]+=(v_[...,None]*Mw_[i_]).sum(1)                 # WORKING MEMORY (question-time scratch store)
    return (_INFM['g']*out_).view(*sh_[:-1],256)
def _infm_write(zf_,idf_,R_,beta_,gate_lr=0.0):
    if idf_.numel()!=zf_.shape[0] or R_.shape[0]!=zf_.shape[0]: return
    sel_=((idf_>=0)&(R_.abs().sum(-1)>0)).nonzero().squeeze(-1)
    for c_ in range(0,sel_.numel(),512):
        s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_])
        if gate_lr>0:                                                               # SLOWGATE: timescale gates follow the retrieval error
            dk_=(((_INFM['M'][i_]@R_[s_][:,:,None]).squeeze(-1)*v_)[...,None]*_INFM['WX'][i_]).sum(1); d_=_INFM['d']
            for j_ in range(len(_INFM_TAUS)):
                pc_=zf_[s_][:,d_*(j_+1):d_*(j_+2)]; _INFM['gates'][j_]+=gate_lr*float((dk_[:,d_*(j_+1):d_*(j_+2)]*pc_).sum(-1).mean())
            _INFM['gates'].clamp_(0.0,3.0)
        _INFM['M'].index_add_(0,i_.reshape(-1),(beta_*v_[...,None]*R_[s_][:,None,:]).reshape(-1,256))
_Model_fwd0=Model.forward
def _model_fwd_m(s,x,domain=0):
    if _INFM['M'] is not None: _INFM['X']=x
    return _Model_fwd0(s,x,domain)
Model.forward=_model_fwd_m
def _infm_head_corr(x,lg_=None):
    X_=_INFM['X']
    if X_ is None or x.dim()!=3 or X_.dim()!=2 or X_.shape[0]!=x.shape[0] or X_.shape[1]<x.shape[1] or X_.device!=x.device: return 0.0   # only full (rows, positions, features) reads aligned with the stashed bytes
    K_,id_=_infm_keys(x,X_[:,-x.shape[1]:],lg_); x=K_
    if _INF.get('corr') and not _INFM.get('dbg_pr'):
        _INFM['dbg_pr']=1; print(f"  [inference eval | fast store debug] stashed row 0: {bytes(X_[0].to('cpu',torch.uint8).tolist())!r} | x {tuple(x.shape)} X {tuple(X_.shape)} | ids row 0 {id_[0].tolist()[:40]} | first store names {list(_INFM['names'])[:5]}",flush=True)
    c_=_infm_corr(x,id_)
    gt_=float(_INFM.get('gthr',0.0))
    if gt_>0: c_=c_*(c_.amax(-1,keepdim=True)>=gt_).float()                       # CONFIDENCE GATE: a correction is applied only if it backs some byte by >= gthr logits
    d_=_INFM.setdefault('dbg',[0,0,0.0]); d_[0]+=int(id_.numel()); d_[1]+=int((id_>=0).sum()); d_[2]+=float(c_.abs().amax(-1).sum())
    return c_
_Head_fwd0=Head.forward
def _head_fwd_m(s,x):
    z_=_Head_fwd0(s,x)
    if _INFM['M'] is not None and _INFM['mu'] is not None and not torch.is_grad_enabled():
        act_=_INFM['force'] if 'force' in _INFM else (_INFM['on'] or (_INFM['dream'] and float(_INFM['model'].__dict__.get('_real_w',1.0))==0))
        if act_: z_=z_+_infm_head_corr(x,z_)
    return z_
Head.forward=_head_fwd_m
_bpc_w0=bpc
def bpc(model,get,domain,n=15):
    """evaluation with the fast store active (cg20h); identical to the original when no store exists"""
    if _INFM['M'] is None: return _bpc_w0(model,get,domain,n)
    sv_=_INFM['on']; _INFM['on']=bool(_INFM['bpc_on'])
    try: return _bpc_w0(model,get,domain,n)
    finally: _INFM['on']=sv_
_INF_REL=["{a} works with {b}.","{a} lives near {b}.","{a} reports to {b}.","{a} is a partner of {b}.","{a} trained under {b}.","{a} is linked to {b}."]
_INF_SYL=['ka','to','mi','ra','ve','lo','du','sen','tar','bel','nor','qui','fa','zo','pel','dra','mon','ris','tev','gal','hun','cor','vin','sal','bri','ed','an','ul','or','is']
_INF_NAME=_re_inf.compile(rb'\b[A-Z][a-z]{3,11}\b')
_INF_SENT=_re_inf.compile(rb'(?<=[.!?])\s+')
_INF_TRIP=_re_inf.compile(rb'(?<![A-Za-z])([A-Z][a-z]{3,11}) ((?:[a-z]+ ){0,3}[a-z]+) ([A-Z][a-z]{3,11})(?![A-Za-z])')
_INF_NAME_RX=_INF_NAME; _INF_TRIP_RX=_INF_TRIP                                  # the hand-written patterns (kept for question-time readouts / prompts)
# ---- cg20s RULE-FREE UNITS: a LEARNED lexicon replaces the capitalised-name pattern. A word type becomes a unit the first time the model,
#      reading it in context, cannot COMPLETE it (mean surprisal of its bytes after the first, in the top inf_units_q of recent words).
_UNITS={'lex':set()}
_UWORD=_re_inf.compile(rb"(?<![\x80-\xff])[A-Za-z0-9][A-Za-z0-9'-]*(?![\x80-\xff])")
_UEVT=_re_inf.compile(rb"(?<![\x80-\xff])[A-Za-z0-9][A-Za-z0-9'-]*(?![\x80-\xff])|[.!?\n]")
class _UnitMatch:
    def __init__(s,g,sp): s._g=g; s._sp=sp
    def group(s,i=0): return s._g[i]
    def start(s,i=0): return s._sp[i][0]
    def end(s,i=0): return s._sp[i][1]
def _u_rare(w_):
    """'all' mode (the word-level inference layer): ANY word of 3+ letters whose window frequency so far is at most the rarity limit"""
    return len(w_)>=3 and _UNITS['df'].get(w_,0)<=_UNITS['mx']
class _UnitName:
    """drop-in for the name regex: matches only word types in the learned lexicon ('all' mode: any statistically rare word)"""
    def finditer(s,b_):
        if _UNITS.get('all'):
            for m_ in _UWORD.finditer(b_):
                if _u_rare(m_.group(0)): yield m_
            return
        L_=_UNITS['lex']
        for m_ in _UWORD.finditer(b_):
            if m_.group(0) in L_: yield m_
    def findall(s,b_): return [m_.group(0) for m_ in s.finditer(b_)]
    def search(s,b_): return next(s.finditer(b_),None)
    def fullmatch(s,b_): return _UWORD.fullmatch(b_) if (_u_rare(b_) if _UNITS.get('all') else b_ in _UNITS['lex']) else None
class _UnitTrip:
    """drop-in for the fact pattern: two consecutive units with 1-4 non-unit words between them (single spaces, no punctuation).
    'all' mode: the clause's two RAREST words (lowest window frequency) are its arguments, the 1-6 words between them its relation"""
    def finditer(s,b_):
        if _UNITS.get('all'):
            for cl_ in _re_inf.finditer(rb'[^.,;:!?()\"]+',b_):
                ws_=[m_ for m_ in _UWORD.finditer(cl_.group(0)) if _u_rare(m_.group(0))]
                if len(ws_)<2: continue
                two_=sorted(sorted(ws_,key=lambda m_: (_UNITS['df'].get(m_.group(0),0),m_.start()))[:2],key=lambda m_: m_.start())
                a_,c_=two_; o_=cl_.start(); mid_=cl_.group(0)[a_.end():c_.start()]
                if a_.group(0)==c_.group(0) or not _re_inf.fullmatch(rb" \S+(?: \S+){0,5} ",mid_): continue
                yield _UnitMatch((b_[o_+a_.start():o_+c_.end()],a_.group(0),mid_[1:-1],c_.group(0)),((o_+a_.start(),o_+c_.end()),(o_+a_.start(),o_+a_.end()),(o_+a_.end()+1,o_+c_.start()-1),(o_+c_.start(),o_+c_.end())))
            return
        L_=_UNITS['lex']; ws_=list(_UWORD.finditer(b_)); us_=[k_ for k_,m_ in enumerate(ws_) if m_.group(0) in L_]
        for x_,y_ in zip(us_,us_[1:]):
            if not 1<=y_-x_-1<=4: continue
            a_,c_=ws_[x_],ws_[y_]; mid_=b_[a_.end():c_.start()]
            if not _re_inf.fullmatch(rb" [^\s.,;:!?()\"]+(?: [^\s.,;:!?()\"]+){0,3} ",mid_): continue
            yield _UnitMatch((b_[a_.start():c_.end()],a_.group(0),mid_[1:-1],c_.group(0)),((a_.start(),c_.end()),(a_.start(),a_.end()),(a_.end()+1,c_.start()-1),(c_.start(),c_.end())))
    def search(s,b_): return next(s.finditer(b_),None)
def _units_on(cfg): return int(getattr(cfg,'inf_units',0))>0
@torch.no_grad()
def _units_update(model,cfg,X,S_):
    """read a few rows of the batch with the CURRENT model (first encounter, before it trains on them) and flag word types it cannot complete"""
    S_['un']=S_.get('un',0)+1
    if S_['un']%max(1,int(getattr(cfg,'inf_units_every',4))): return
    R_=min(int(getattr(cfg,'inf_units_rows',8)),X.shape[0]); Xs_=X[:R_].long()
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sv_=_INFM['on']; _INFM['on']=False; fz_=_INFM.pop('force',None); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(Xs_[:,:-1],_XDG['cur']).float(),-1)
    finally:
        _carry_load(sv4_); _INFM['on']=sv_; model.train(was_)
        if fz_ is not None: _INFM['force']=fz_
    bits_=(-lp_.gather(-1,Xs_[:,1:,None]).squeeze(-1)/math.log(2)).cpu().numpy(); Xc_=Xs_.to('cpu',torch.uint8).numpy()
    rv_=S_.setdefault('ures',[]); L_=S_.setdefault('lex',set()); _UNITS['lex']=L_; c0_=int(getattr(cfg,'inf_units_ctx',32)); q_=float(getattr(cfg,'inf_units_q',0.2))
    thr_=float(np.quantile(rv_,1-q_)) if len(rv_)>=int(getattr(cfg,'inf_units_warm',2000)) else None; nf_=0
    for r_ in range(R_):
        w_=bytes(Xc_[r_])
        for m_ in _UWORD.finditer(w_):
            a_,e_=m_.start(),m_.end()
            if e_-a_<3 or a_<c0_ or e_>=len(w_)-1: continue
            c_=float(bits_[r_,a_:e_-1].mean())            # bytes a+1..e-1 are predicted at positions a..e-2
            rv_.append(c_)                                # RECENT words only (FIFO): the threshold tracks the model as it is NOW (a long reservoir lagged and missed names)
            w0_=m_.group(0)
            if str(getattr(cfg,'inf_units_mode','rank'))=='mdl':   # MDL CHUNKING: a word type becomes a unit when treating it as ONE symbol would have saved bits:
                g0_=S_.setdefault('ugain',{}).setdefault(w0_,[0,0.0])   #   every repeat after the first saves its byte surprisal minus the cost of a pointer into the lexicon
                if g0_[0]>0: g0_[1]+=c_*(e_-a_-1)-math.log2(len(L_)+1024)
                g0_[0]+=1
                if g0_[1]>=float(getattr(cfg,'inf_units_gain',16.0)) and w0_ not in L_ and w0_ not in S_.get('lexdem',()): L_.add(w0_); nf_+=1
            elif thr_ is not None and c_>=thr_ and w0_ not in L_ and w0_ not in S_.get('lexdem',()): L_.add(w0_); nf_+=1
    del rv_[:-int(getattr(cfg,'inf_units_win',3000))]
    S_['unf']=S_.get('unf',0)+nf_
_INF={'probes':None}
def _inf_phrase(t): return t.split('{a} ')[1].split(' {b}')[0]
def inf_probes(cfg):
    """the probe chains (cached): list of dicts {names:[A,B,C(,D)], rels:[phrase...], doms:[dX,dY(,dZ)], facts:[(dom, sentence)]}"""
    if _INF['probes'] is not None: return _INF['probes']
    global _INF_NAME
    if _units_on(cfg):                                                          # cg20s: RULE-FREE: names and facts come from the learned lexicon
        global _INF_TRIP; _INF_NAME=_UnitName(); _INF_TRIP=_UnitTrip(); _UNITS['on']=True
    elif int(getattr(cfg,'inf_name_strict',0)):                                 # cg20i: a name never touches a letter or a non-ASCII byte (no 'Katar' out of 'Katariny')
        _INF_NAME=_re_inf.compile(rb'(?<![A-Za-z\x80-\xff])[A-Z][a-z]{3,11}(?![A-Za-z\x80-\xff])')
    nd_=len(PILE_DOMAINS); rs_=np.random.RandomState(int(cfg.seed)+2020); used=set(); ch=[]
    def nm():
        while True:
            n_=''.join(rs_.choice(_INF_SYL,rs_.randint(2,4))); n_=n_[0].upper()+n_[1:]
            if int(getattr(cfg,'inf_probe_rare',0)) and not any(ch_ in n_.lower() for ch_ in 'qxz'): continue   # cg20j: invented names carry q, x or z (no collision with real words)
            if 5<=len(n_)<=10 and n_ not in used: used.add(n_); return n_
    if cfg.inf_probe_file:
        import json as _js
        for ln_ in open(cfg.inf_probe_file):
            o_=_js.loads(ln_); c_=o_['chain']; dn_=[i_ for i_,d_ in enumerate(PILE_DOMAINS) if o_['domain'].split()[0].lower() in d_.lower()]
            d0_=dn_[0] if dn_ else rs_.randint(nd_); rl_=c_['relations']; ds_=[d0_]+[int(rs_.randint(nd_)) for _ in rl_[1:]]
            ch.append({'names':c_['entities'][:len(rl_)+1],'rels':[r_['phrase'] for r_ in rl_],'doms':ds_,'facts':[(ds_[i_],r_['snippet_explicit']) for i_,r_ in enumerate(rl_)],'api':True})
    else:
        for k_ in range(int(cfg.inf_probes)):
            h_=3 if rs_.rand()<float(cfg.inf_hops3) else 2; n_=[nm() for _ in range(h_+1)]; tp_=[_INF_REL[rs_.randint(len(_INF_REL))] for _ in range(h_)]
            ds_=[int(rs_.randint(nd_))]
            for _ in range(h_-1): ds_.append(int((ds_[-1]+1+rs_.randint(nd_-1))%nd_))   # consecutive facts in DIFFERENT domains
            ch.append({'names':n_,'rels':[_inf_phrase(t_) for t_ in tp_],'doms':ds_,'facts':[(ds_[i_],tp_[i_].format(a=n_[i_],b=n_[i_+1])) for i_ in range(h_)],'api':False,'tp':tp_})
        uf_=float(getattr(cfg,'inf_upd_frac',0.0)); rsu_=np.random.RandomState(int(cfg.seed)+3030)
        for c_ in ch:                                                           # BELIEF REVISION probes (two-hop chains): from pass 2 the first fact changes
            if uf_<=0 or len(c_['names'])!=3 or rsu_.rand()>=uf_: continue
            A_=c_['names'][0]; t0_,t1_=c_['tp'][0],c_['tp'][1]
            if rsu_.rand()<2/3:                                                 #   TRUE UPDATE: 'A r0 B' becomes 'A r0 B2' (planted as often as any fact), plus 'B2 r1 C2'
                B2_,C2_=nm(),nm(); c_['upd']={'type':'true','B2':B2_,'C2':C2_,'new':[(c_['doms'][0],t0_.format(a=A_,b=B2_)),(c_['doms'][1],t1_.format(a=B2_,b=C2_))]}
            else:                                                               #   NOISE: one single contradicting sentence 'A r0 B3'; the old fact stays
                B3_=nm(); c_['upd']={'type':'noise','B3':B3_,'new':[(c_['doms'][0],t0_.format(a=A_,b=B3_))]}
    pf_=float(getattr(cfg,'inf_para_frac',0.0))
    if pf_>0:                                                                   # PARAPHRASE PROBES: facts stated in the usual wording, the paraphrase, or BOTH
        PP_={_inf_phrase(t_):_INF_PARA[_inf_phrase(t_)] for t_ in _INF_REL}; rsq_=np.random.RandomState(int(cfg.seed)+4040); ck_=int(getattr(cfg,'inf_comp_k',2))
        for c_ in ch:
            if c_.get('upd') or c_.get('api') or rsq_.rand()>=pf_: continue
            c_['wd']=[]
            for j_,(dd_,s_) in enumerate(list(c_['facts'])):
                r0_=c_['rels'][j_]; ps_=s_.replace(f" {r0_} ",f" {PP_[r0_]} ",1); u0_=rsq_.rand()
                w0_='both' if u0_<0.4 else ('para' if u0_<0.6 else 'can'); c_['wd'].append(w0_)
                if w0_=='para': c_['facts'][j_]=(dd_,ps_)
                elif w0_=='both': c_.setdefault('extra',[]).append((dd_,ps_))
            oth_=[r_ for r_ in PP_ if r_!=c_['rels'][0]]; rsq_.shuffle(oth_)   # COMPETING RELATIONS: the subject also has facts under other relations
            for r1_ in oth_[:ck_]:
                X_=nm(); u1_=rsq_.rand(); ph_=r1_ if u1_<0.6 else PP_[r1_]
                c_.setdefault('extra',[]).append((c_['doms'][0],f"{c_['names'][0]} {ph_} {X_}."))
                if u1_<0.3: c_['extra'].append((c_['doms'][0],f"{c_['names'][0]} {PP_[r1_]} {X_}."))
    _INF['probes']=ch
    print(f"  [inference] {len(ch)} probe chains ({sum(len(c_['rels'])==3 for c_ in ch)} three-hop){' from '+cfg.inf_probe_file if cfg.inf_probe_file else ' (templated, invented names)'}"
          f", {sum(len(c_['facts']) for c_ in ch)} fact sentences, each planted x{cfg.inf_plant_k} per pass into its domain's text")
    return ch
def inf_plant(raw,cfg,d,p):
    """insert domain d's probe fact sentences (each inf_plant_k times) at sentence boundaries of its pass-p text"""
    fs_=[]; once_=[]
    for c_ in inf_probes(cfg):
        u_=c_.get('upd') if int(p)>=1 else None
        for j_,(dd_,s_) in enumerate(c_['facts']):
            if u_ and u_['type']=='true' and j_==0: continue                   # the old fact is no longer true
            if dd_==int(d): fs_.append(s_)
        for dd_,s_ in c_.get('extra',[]):
            if dd_==int(d): fs_.append(s_)
        if u_:
            for dd_,s_ in u_['new']:
                if dd_==int(d): (fs_ if u_['type']=='true' else once_).append(s_)
    if not fs_ and not once_: return raw
    b_=bytes(raw); rs_=np.random.RandomState(9000+97*int(d)+int(p)); cuts=[m_.end() for m_ in _re_inf.finditer(rb'[.!?] (?=[A-Z])',b_)]
    if len(cuts)<2: return raw
    ins=sorted([(int(cuts[rs_.randint(len(cuts))]),s_) for s_ in fs_ for _ in range(int(cfg.inf_plant_k))]+[(int(cuts[rs_.randint(len(cuts))]),s_) for s_ in once_]); out=bytearray(); j_=0
    for c_,s_ in ins: out+=b_[j_:c_]+s_.encode('latin-1')+b' '; j_=c_
    out+=b_[j_:]
    print(f"  [inference] domain {d} pass {p}: {len(ins)} probe fact sentences planted ({len(fs_)} facts)")
    return bytes(out)
def _inf_store(model):
    return model.__dict__.setdefault('_infS',{'it':[],'df':{},'idx':{},'new':[],'nwin':0,'pairs':set(),'conc':[],'stat':{},'ring':[],'lw':set(),'strict':bool(int(getattr(model.cfg,'inf_name_strict',0)))})
@torch.no_grad()
def inf_store_add(model,cfg,X,dom,mask=None):
    """add windows X (rows of bytes, CPU-converted here) to the sentence store; mask (same shape as X[:,1:]): verified positions only"""
    _inf_sync(cfg,'store add, before ('+('closed loop' if mask is not None else 'real text')+')')
    S_=_inf_store(model)
    if _units_on(cfg):
        global _INF_NAME,_INF_TRIP
        if not isinstance(_INF_NAME,_UnitName): _INF_NAME=_UnitName(); _INF_TRIP=_UnitTrip(); _UNITS['on']=True   # (also after a checkpoint restore)
        _UNITS['lex']=S_.setdefault('lex',set())
        if str(getattr(cfg,'inf_units_mode','rank'))=='all': _UNITS.update(all=True,df=S_['df'],mx=int(cfg.inf_df))
        elif mask is None: _units_update(model,cfg,X,S_)                       # real text only (closed-loop bytes are the model's own)
    Xc_=X.detach().to('cpu',torch.uint8).numpy()
    if mask is not None:
        Mc_=(mask.detach()>0).to('cpu').numpy(); Xc_=Xc_.copy(); Xc_[:,1:][~Mc_]=0                # unverified bytes blanked
    cap_=int(cfg.inf_store_cap)
    for r_ in range(Xc_.shape[0]):
        w_=bytes(Xc_[r_]); S_['nwin']+=1
        S_['ring'].append((w_,int(dom))); del S_['ring'][:-512]                    # recent windows (real / verified-only bytes): filler + anchors
        if int(getattr(cfg,'inf_sofl',0)):                                       # SLEEP-ONLY FEATURE LEARNING: the visit's windows, buffered (reservoir) for the sleep
            sb_=S_.setdefault('sbuf',[]); S_['sbn']=S_.get('sbn',0)+1; cb_=int(getattr(cfg,'inf_sofl_buf',256))
            if len(sb_)<cb_: sb_.append((w_,int(dom)))
            else:
                j1_=np.random.randint(S_['sbn'])
                if j1_<cb_: sb_[j1_]=(w_,int(dom))
        rv_=S_.setdefault('res',{}).setdefault(int(dom),[]); rn0_=S_.setdefault('resn',{}); rn0_[int(dom)]=rn0_.get(int(dom),0)+1   # PER-DOMAIN RESERVOIR (64 windows, uniform over the domain's history)
        if len(rv_)<64: rv_.append(w_)
        else:
            j0_=np.random.randint(S_['resn'][int(dom)])
            if j0_<64: rv_[j0_]=w_
        for n_ in set((_UWORD.findall(w_) if _UNITS.get('all') else _INF_NAME.findall(w_))): S_['df'][n_]=S_['df'].get(n_,0)+1
        if S_.get('strict'): S_['lw'].update(_re_inf.findall(rb'(?<![A-Za-z])[a-z]{4,12}(?![A-Za-z])',w_))   # lower-case words seen (a capitalised common word is not a name)
        w2_=_re_inf.sub(rb'^[A-Za-z]*',b'',w_[1:] if mask is not None else w_)   # a word cut by the window start is not a name
        w2_=_re_inf.sub(rb'[A-Za-z]*$',b'',w2_)                                       #   nor one cut by the window end
        w2_=_re_inf.sub(rb'[A-Za-z]*\x00+[A-Za-z]*',b'. ',w2_)                       # blanked (unverified) runs: hard sentence breaks, adjacent word fragments dropped
        for s_ in _INF_SENT.split(w2_):
            ns_=_INF_NAME.findall(s_)
            if len(set(ns_))<2 or len(s_)>300: continue
            if len(S_['it'])>=cap_: continue
            i_=len(S_['it']); S_['it'].append({'s':s_,'d':int(dom),'k':'real' if mask is None else 'ver','dep':0,'v':int(model.__dict__.get('_visit',0))}); S_['new'].append(i_)
            if int(getattr(cfg,'inf_audit',0)):                                 # EVIDENCE RECORD: subject-first triples (subject, relation, object)
                m3_=_INF_TRIP.search(s_)
                if m3_: S_.setdefault('rel',{}).setdefault((m3_.group(1),m3_.group(2)),{}).setdefault(m3_.group(3),[]).append(i_)
            for n_ in set(ns_): S_['idx'].setdefault(n_,[]).append(i_)
def _inf_rel(s_,a_,b_):
    """relation phrase if the sentence reads '<a> <1-6 lowercase words> <b>' (subject-first), else None"""
    w0_=rb'[^\s.,;:!?()"]+' if _UNITS.get('on') else rb'[a-z]+'
    m_=_re_inf.search(rb'(?<![A-Za-z0-9])'+_re_inf.escape(a_)+rb'((?: '+w0_+rb'){1,6}) '+_re_inf.escape(b_)+rb'(?![A-Za-z0-9])',s_)
    return m_.group(1).strip().decode('latin-1') if m_ else None
def inf_infer(model,cfg):
    """bridge-centred chaining over the store; returns this sleep's NEW conclusions"""
    S_=_inf_store(model); df_=S_['df']; mx_=int(cfg.inf_df); new_=[]; st_={'bridges':0,'hub':0,'prop':0}
    lw_=S_.get('lw') if S_.get('strict') else None; ok_=lambda n_: df_.get(n_,0)<=mx_ and (lw_ is None or n_.lower() not in lw_)
    br_=set(n_ for i_ in S_['new'] for n_ in _INF_NAME.findall(S_['it'][i_]['s']) if ok_(n_))
    if str(getattr(cfg,'inf_bridge_order','any'))=='rare': br_=sorted(br_,key=lambda n_: (df_.get(n_,0),n_))   # cg20i: rarest bridges first (the budget goes to rare links)
    for B in br_:
        part={}
        for i_ in S_['idx'].get(B,[]):
            it_=S_['it'][i_]
            if it_.get('sup'): continue                                         # superseded by the grounding audit
            for A in set(_INF_NAME.findall(it_['s'])):
                if A!=B and ok_(A): part.setdefault(A,[]).append(i_)
        st_['bridges']+=1
        if len(part)>int(cfg.inf_maxfan): st_['hub']+=1; continue
        ks_=sorted(part)
        for x_ in range(len(ks_)):
            for y_ in range(x_+1,len(ks_)):
                A,C=ks_[x_],ks_[y_]; ia_=part[A][0]; ic_=part[C][0]
                if ia_==ic_ or (S_['it'][ia_]['dep']+S_['it'][ic_]['dep'])>1: continue   # different sentences; never inferred + inferred
                key_=(min(A,C),max(A,C))
                if key_ in S_['pairs']: continue
                if set(S_['idx'].get(A,[]))&set(S_['idx'].get(C,[])): S_['pairs'].add(key_); continue   # already directly linked: nothing to infer
                S_['pairs'].add(key_)
                sa_,sc_=S_['it'][ia_]['s'],S_['it'][ic_]['s']
                r1_=_inf_rel(sa_,A,B); r2_=_inf_rel(sc_,B,C); fw_=True
                if r1_ is None or r2_ is None:                                   # try the other direction (C ... B, B ... A)
                    r1b_=_inf_rel(sc_,C,B); r2b_=_inf_rel(sa_,B,A)
                    if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
                typed_=(r1_ is not None and r2_ is not None)
                sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
                f0_=len(part); nw0_=max(2,S_['nwin'])
                gain_=math.log2(nw0_/max(1,df_.get(B,1)))-math.log2(max(1,f0_*(f0_-1)/2))   # bits: how surprising a bridge this rare is, minus the links it proposes
                new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                             'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':gain_})
                st_['prop']+=1
                if len(new_)>=int(cfg.inf_budget): break
            if len(new_)>=int(cfg.inf_budget): break
        if len(new_)>=int(cfg.inf_budget): break
    cap_=int(getattr(cfg,'inf_pool',0))
    if cap_>0:                                                                  # THE POOL (rule-free inference layer): links compete on bits saved
        P_=[c_ for c_ in S_['conc'] if not c_.get('freed')]+new_; rel_=float(getattr(cfg,'inf_pool_rel',0.0)); ev_=0; dr_=0
        if rel_>0 and len(P_)>=8:
            md_=float(np.median([c_.get('gain',0.0) for c_ in P_]))
            for c_ in P_:
                if c_.get('gain',0.0)<rel_*md_ and md_>0: c_['freed']=True; c_['evicted']='below going rate'; dr_+=1
        P_=sorted([c_ for c_ in P_ if not c_.get('freed')],key=lambda c_: -c_.get('gain',0.0))
        for c_ in P_[cap_:]: c_['freed']=True; c_['evicted']='pool full'; ev_+=1
        new_=[c_ for c_ in new_ if not c_.get('freed')]
        st_['pool']=f"pool {min(len(P_),cap_)}/{cap_}, {ev_} evicted (worst), {dr_} dropped (below {rel_:.2f} x going rate)"
    for c_ in new_:                                                             # conclusions re-enter the store (one more hop allowed)
        i_=len(S_['it']); S_['it'].append({'s':f"So {c_['A']} is connected to {c_['C']}.".encode(),'d':c_['d'],'k':'inf','dep':c_['dep']-1})
        for n_ in (c_['A'].encode(),c_['C'].encode()): S_['idx'].setdefault(n_,[]).append(i_)
    S_['new']=[]; S_['conc']+=new_; return new_,st_
def _inf_truth(cfg):
    """probe ground truth: set of true (A,C) pairs at 2 and 3 hops"""
    t2=set(); t3=set()                                                          # 'connected to' is symmetric: both orders count
    for c_ in inf_probes(cfg):
        n_=c_['names']; t2|={(n_[0],n_[2]),(n_[2],n_[0])}
        if len(n_)>3: t2|={(n_[1],n_[3]),(n_[3],n_[1])}; t3|={(n_[0],n_[3]),(n_[3],n_[0])}
    return t2,t3
@torch.no_grad()
def inf_idmem(model,cfg,S_,tag,D_=None):
    """IDENTIFIER MEMORY (cg20s): every live recorded fact (A, r, B) is stored as code(A)*code(r) -> code(B) (fixed random +-1 codes from hashes,
    summed outer products; superseded / freed facts are simply left out). Questions are answered by lookups with clean-up to the nearest known
    unit; the returned identifier is the next key (no text round trip). Unknown subject -> low clean-up similarity -> ABSTAIN."""
    import zlib as _zl
    F_=[f_ for f_ in (S_.get('facts') or []) if not f_.get('freed')]
    if not F_: return
    D_=int(D_ or getattr(cfg,'inf_id_dim',1024))                                   # capacity: interference grows as sqrt(facts / D); 4096 for thousands of facts
    def code(s_):
        g_=np.random.RandomState(_zl.crc32(s_.encode('latin-1','replace'))%(2**31)); return torch.tensor(g_.choice([-1.0,1.0],D_),dtype=torch.float32)
    M_=torch.zeros(D_,D_); un_=sorted(set(x_ for f_ in F_ for x_ in (f_['A'],f_['B']))); cc_={}
    gc_=lambda s_: cc_[s_] if s_ in cc_ else cc_.setdefault(s_,code(s_))
    for i0_ in range(0,len(F_),256):
        ch_=F_[i0_:i0_+256]; K_=torch.stack([gc_(f_['A'])*gc_(f_['r']) for f_ in ch_]); V_=torch.stack([gc_(f_['B']) for f_ in ch_]); M_+=K_.T@V_
    C_=torch.stack([gc_(u_) for u_ in un_]); sim_=float(getattr(cfg,'inf_qt_sim',0.5))
    def look(cur_,rels_):
        for r_ in rels_:
            v_=(gc_(cur_)*gc_(r_))@M_; s_=F.cosine_similarity(v_[None,:],C_); j_=int(s_.argmax())
            if float(s_[j_])<sim_: return None
            cur_=un_[j_]
        return cur_
    st_={}; P_=inf_probes(cfg)
    for c_ in P_:
        n_,R_=c_['names'],c_['rels']; u_=c_.get('upd')
        if u_ and u_['type']=='true':                                           # (as in the probe eval: always scored against the NEW answer)
            for k_,(qs_,rr_,tr_,old_) in {'UPDATED one-hop':(n_[0],R_[:1],u_['B2'],n_[1]),'UPDATED two-hop':(n_[0],R_[:2],u_['C2'],n_[2])}.items():
                a_=look(qs_,rr_); s0_=st_.setdefault(k_,[0,0,0,0]); s0_[0]+=1; s0_[1]+=int(a_ is not None); s0_[2]+=int(a_==tr_); s0_[3]+=int(a_==old_)
            continue
        for k_,h_ in (('one-hop',1),('two-hop',2),('three-hop',3)):
            if len(R_)<h_: continue
            a_=look(n_[0],R_[:h_]); s0_=st_.setdefault(k_,[0,0,0,0]); s0_[0]+=1; s0_[1]+=int(a_ is not None); s0_[2]+=int(a_==n_[h_])
    rs_=np.random.RandomState(77); ab_=0; nu_=40
    for _ in range(nu_):
        q_=''.join(rs_.choice(list('abcdefghiklmnoprstuvy'),7)).capitalize(); ab_+=int(look(q_,[P_[0]['rels'][0]]) is None)
    out_=[f"{k_} {v_[2]/max(1,v_[0]):.2f} correct, answers {v_[1]/max(1,v_[0]):.2f}"+(f", stale {v_[3]/max(1,v_[0]):.2f}" if k_.startswith('UPDATED') else '') for k_,v_ in st_.items()]
    print(f"  [inference IDENTIFIER MEMORY | {tag}] {len(F_)} live recorded facts, {len(un_)} units | "+" | ".join(out_)+f" | unknown subjects abstained {ab_/nu_:.2f}",flush=True)
def inf_eval(model,cfg,S_,tag):
    """probe recall from the weights; with a fast store (cg20h) also WITH the store active"""
    _inf_sync(cfg,'eval start'); _INF['corr']=False; r_=_inf_eval1(model,cfg,S_,tag); _inf_sync(cfg,'eval from the weights')
    if _INFM['M'] is not None and _INFM['mu'] is not None:
        _INF['corr']=True; _INFM['dbg']=[0,0,0.0]
        try:
            r_=_inf_eval1(model,cfg,S_,tag+" | WITH THE FAST STORE"); d_=_INFM['dbg']; _inf_sync(cfg,'eval with the store')
            print(f"  [inference eval | fast store use] {d_[1]}/{d_[0]} probe positions store-keyed, mean max |logit correction| there {d_[2]/max(1,d_[1]):.3f} | names in the store {len(_INFM.get('names',{}))}",flush=True)
        finally: _INF['corr']=False
    if int(getattr(cfg,'inf_splice',0)) and ('END' in tag or (int(model.__dict__.get('_visit',0))+1)%max(1,int(getattr(cfg,'inf_splice_every',1)))==0):
        try: inf_splice(model,cfg,tag)
        except Exception as e_: print(f"  [inference splice-and-appraise] skipped: {e_!r}"[:300],flush=True)
    if int(getattr(cfg,'inf_eval_qtime',0)) and not _INF.get('corr') and ('END' in tag or (int(model.__dict__.get('_visit',0))+1)%max(1,int(getattr(cfg,'inf_qt_every',getattr(cfg,'inf_splice_every',1))))==0):
        try: inf_qtime2(model,cfg,tag)
        except Exception as e_: print(f"  [inference question-time chaining] skipped: {e_!r}"[:300],flush=True)
        if int(getattr(cfg,'inf_idmem',0)):
            try: inf_idmem(model,cfg,S_,tag)
            except Exception as e_: print(f"  [inference IDENTIFIER MEMORY] skipped: {e_!r}"[:300],flush=True)
        if int(getattr(cfg,'inf_eval_qtime3',0)):
            try:
                for nc0_ in sorted(set([0,int(getattr(cfg,'inf_qt_comp',2))])):
                    cfg.inf_qt_comp=nc0_; inf_qtime3(model,cfg,tag)
            except Exception as e_: print(f"  [inference multi-hop question-time] skipped: {e_!r}"[:300],flush=True)
    return r_
@torch.no_grad()
def _inf_eval1(model,cfg,S_,tag):
    """probe recall FROM THE WEIGHTS (no context): one-hop, two-hop ('So A is connected to'), typed two-hop, three-hop; 19 distractor names"""
    ch_=inf_probes(cfg); dev_=model.embed.weight.device; was_=model.training; model.eval(); rs_=np.random.RandomState(77)
    alln_=[n_ for c_ in ch_ for n_ in c_['names']]; res={}
    def score(prompts,cands,slot):
        out=[]
        for p_,cs_ in zip(prompts,cands):
            seqs=[(p_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cs_]; L_=max(len(s_) for s_ in seqs)
            X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
            for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
            try: lp_=F.log_softmax(model(X_[:,:-1],slot).float(),-1)
            finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
            n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cs_)]
            sc2_=[float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))]
            out.append(int(np.argmax(sc2_)))
            if _INF.get('cal_on'):
                pp_=np.exp(np.array(sc2_)-max(sc2_)); pp_=pp_/pp_.sum(); A0_=_INF['cal_on'][len(_INF['calrec'])] if len(_INF['calrec'])<len(_INF['cal_on']) else ('','','')
                _INF['calrec'].append((int(np.argmax(sc2_))==0,float(pp_.max()),float(pp_[0]),int(A0_[0].encode('latin-1') in (_INFM.get('names') or {})),A0_[0],cs_[int(np.argmax(sc2_))],A0_[1],A0_[2]))
        return out
    try:
        UPD_=float(getattr(cfg,'inf_upd_frac',0.0))>0
        KINDS_=('one-hop','two-hop','typed two-hop','three-hop')+(('UPDATED one-hop','UPDATED two-hop','NOISE-CONTRADICTED one-hop') if UPD_ else ())+(('reworded two-hop','context two-hop','pronoun two-hop') if (int(getattr(cfg,'inf_eval_hard',0)) and ('END' in tag or _INF.get('corr'))) else ())+(('IN-CONTEXT two-hop','IN-CONTEXT decoy bridge','IN-CONTEXT three-hop') if (int(getattr(cfg,'inf_eval_ctx',0)) and not _INF.get('corr')) else ())+(('CTX2 order AB','CTX2 order BA','CTX2 decoy AB','CTX2 decoy BA') if (int(getattr(cfg,'inf_eval_ctx2',0)) and not _INF.get('corr')) else ())
        if float(getattr(cfg,'inf_para_frac',0.0))>0: KINDS_=KINDS_+('PARAPHRASED one-hop (unseen wording)','SEEN-WORDING one-hop (para probes)')
        act_={}
        for kind in KINDS_:
            pr_=[];cd_=[];sl_=[];bn_=[]; _INFM['dbg']=[0,0,0.0]
            for c_ in ch_:
                n_=c_['names']; r_=c_['rels']
                if kind in ('three-hop','IN-CONTEXT three-hop') and len(n_)<4: continue
                if kind.startswith(('PARAPHRASED','SEEN-WORDING')) and (not c_.get('wd') or (kind.startswith('PARAPHRASED') and c_['wd'][0]=='both')): continue
                u0_=c_.get('upd')
                if kind.startswith('UPDATED') and not (u0_ and u0_['type']=='true'): continue
                if kind.startswith('NOISE') and not (u0_ and u0_['type']=='noise'): continue
                if UPD_ and u0_ and u0_['type']=='true' and not kind.startswith('UPDATED'): continue   # their old answers are no longer true
                if kind.startswith(('IN-CONTEXT','CTX2')) and str(getattr(cfg,'inf_xfer','none'))=='skill' and n_[0] not in _inf_heldout(cfg): continue
                if kind.startswith('CTX2') and len(n_)!=3: continue   # skill runs: in-context probes on HELD-OUT chains only
                fs_=[f_ for _,f_ in c_['facts']]; oth0_=[x_ for x_ in alln_ if x_ not in n_]
                tgt={'one-hop':n_[1],'three-hop':n_[-1],'IN-CONTEXT three-hop':n_[-1],'UPDATED one-hop':(u0_ or {}).get('B2'),'UPDATED two-hop':(u0_ or {}).get('C2'),'NOISE-CONTRADICTED one-hop':n_[1]}.get(kind,n_[2])
                pp={'one-hop':f"{n_[0]} {r_[0]} ",'two-hop':f"So {n_[0]} is connected to ",'typed two-hop':f"{n_[0]} {r_[0]} someone who {r_[1]} ",'three-hop':f"So {n_[0]} is connected to ",
                    'reworded two-hop':f"{n_[0]}, it turns out, is connected to ",'context two-hop':f"The weather was mild that year. So {n_[0]} is connected to ",
                    'pronoun two-hop':f"Consider {n_[0]}. They are connected to ",
                    'IN-CONTEXT two-hop':f"{fs_[0]} {fs_[1]} So {n_[0]} is connected to ",                       # S1: the two source facts in the context
                    'IN-CONTEXT decoy bridge':f"{fs_[0].replace(n_[1],oth0_[len(pr_)%len(oth0_)])} {fs_[1]} So {n_[0]} is connected to ",   # chain broken at the bridge: C still in context
                    'IN-CONTEXT three-hop':(' '.join(fs_[:3])+f" So {n_[0]} is connected to "),
                    'UPDATED one-hop':f"{n_[0]} {r_[0]} ",'UPDATED two-hop':f"So {n_[0]} is connected to ",'NOISE-CONTRADICTED one-hop':f"{n_[0]} {r_[0]} "}.get(kind,'')
                if kind.startswith(('PARAPHRASED','SEEN-WORDING')):
                    pz_=_INF_PARA[r_[0]]; seen_para_=c_['wd'][0]=='para'
                    pp=f"{n_[0]} {(r_[0] if seen_para_ else pz_) if kind.startswith('PARAPHRASED') else (pz_ if seen_para_ else r_[0])} "; tgt=n_[1]
                if kind.startswith('CTX2'):                                     # SHARPER IN-CONTEXT TEST: a distractor fact LAST (the answer is not the last name),
                    X0_,Y0_=oth0_[(3*len(pr_))%len(oth0_)],oth0_[(3*len(pr_)+1)%len(oth0_)]   #   both sentence orders, in-context names among the candidates
                    fx_=f"{X0_} {r_[1]} {Y0_}."; f0_=fs_[0] if 'decoy' not in kind else fs_[0].replace(n_[1],oth0_[(3*len(pr_)+2)%len(oth0_)])
                    pp=(f"{f0_} {fs_[1]} {fx_} " if kind.endswith('AB') else f"{fs_[1]} {f0_} {fx_} ")+f"So {n_[0]} is connected to "
                oth=[x_ for x_ in alln_ if x_ not in n_]; cs_=[tgt]+list(rs_.choice(oth,19,replace=False))
                if kind=='UPDATED one-hop': cs_=[tgt,n_[1]]+cs_[1:19]                 # the OLD answer is a candidate: picking it = a stale belief
                elif kind=='UPDATED two-hop': cs_=[tgt,n_[2]]+cs_[1:19]
                elif kind=='NOISE-CONTRADICTED one-hop': cs_=[tgt,u0_['B3']]+cs_[1:19]  # the single false contradiction is a candidate: picking it = a wrong flip
                elif kind.startswith('CTX2'): ic_=[n_[1],X0_,Y0_]; cs_=[tgt]+ic_+[x_ for x_ in cs_[1:] if x_ not in ic_][:16]   # in-context names compete with the answer
                pr_.append(pp); cd_.append(cs_); bn_.append(n_[1]); sl_.append(int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])))
            hits=[]
            for sl in sorted(set(sl_)):
                ix_=[i_ for i_,s_ in enumerate(sl_) if s_==sl]
                if kind=='two-hop' and _INF.get('corr'): _INF['cal_on']=[(pr_[i_].split(' ')[1],cd_[i_][0],bn_[i_]) for i_ in ix_]; _INF['calrec']=[]
                hh_=score([pr_[i_] for i_ in ix_],[cd_[i_] for i_ in ix_],sl); hits+=[int(h_==0) for h_ in hh_]
                if kind.startswith(('UPDATED','NOISE')): _INF.setdefault('stale',{}).setdefault(kind,[]).extend([int(h_==1) for h_ in hh_])
                if kind.startswith('CTX2'): _INF.setdefault('copy',{}).setdefault(kind,[]).extend([int(1<=h_<=3) for h_ in hh_])
                if _INF.get('cal_on') is not None and kind=='two-hop' and _INF.get('corr'): _INF.setdefault('calrec_all',[]).extend(_INF['calrec']); _INF['cal_on']=None
            res[kind]=(float(np.mean(hits)) if hits else float('nan'),len(hits)); d_=_INFM.get('dbg',[0,0,0.0]); act_[kind]=d_[2]/max(1,d_[0])
            if kind.startswith('CTX2'): cp0_=_INF['copy'].pop(kind,[]); res[kind+f" (picked another in-context name {np.mean(cp0_) if cp0_ else float('nan'):.2f})"]=res.pop(kind)
            if kind.startswith(('UPDATED','NOISE')): st0_=_INF['stale'].pop(kind,[]); res[kind+(' (STALE: old answer preferred' if kind.startswith('UPDATED') else ' (WRONG FLIP: false contradiction preferred')+f" {np.mean(st0_) if st0_ else float('nan'):.2f})"]=res.pop(kind)
        if _INF.get('corr') and int(getattr(cfg,'inf_eval_hard',0)) and 'END' in tag:   # FALSE HITS: invented names never seen anywhere
            rsf_=np.random.RandomState(555); fn_=[]
            while len(fn_)<40:
                n0_=''.join(rsf_.choice(['xa','qe','wu','yo','zi','vex','kru','plo','gne','fyr'],rsf_.randint(2,4))); n0_=n0_[0].upper()+n0_[1:]
                if 5<=len(n0_)<=10 and n0_ not in fn_: fn_.append(n0_)
            _INFM['dbg']=[0,0,0.0]; score([f"So {x_} is connected to " for x_ in fn_],[list(rs_.choice(alln_,20,replace=False)) for _ in fn_],int(model.__dict__.get('_t2s',{}).get(0,0)))
            d_=_INFM['dbg']; act_['FALSE-HIT unknown names']=d_[2]/max(1,d_[0])
        if 'END' in tag and not _INF.get('corr') and S_['conc']:              # TEMPORAL GRADIENT: weights-only recall of found chains, by when their conclusion was made
            vm_={}
            for c0_ in S_['conc']: vm_.setdefault(frozenset((c0_['A'],c0_['C'])),c0_.get('v',0))
            rows_=[]
            for c_ in ch_:
                n_=c_['names']
                if len(n_)<3 or (c_.get('upd') or {}).get('type')=='true': continue
                k0_=frozenset((n_[0],n_[2]))
                if k0_ not in vm_: continue
                oth=[x_ for x_ in alln_ if x_ not in n_]; rows_.append((vm_[k0_],f"So {n_[0]} is connected to ",[n_[2]]+list(rs_.choice(oth,19,replace=False)),int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))))
            if rows_:
                vs_=sorted(r_[0] for r_ in rows_); cut_=vs_[len(vs_)//2]; res_tg_={}
                for nm0_,sel_ in (('OLDER half',lambda v_: v_<cut_),('RECENT half',lambda v_: v_>=cut_)):
                    rr_=[r_ for r_ in rows_ if sel_(r_[0])]; hh_=[]
                    for sl in sorted(set(r_[3] for r_ in rr_)):
                        q_=[r_ for r_ in rr_ if r_[3]==sl]; hh_+=[int(h_==0) for h_ in score([r_[1] for r_ in q_],[r_[2] for r_ in q_],sl)]
                    res_tg_[nm0_]=(float(np.mean(hh_)) if hh_ else float('nan'),len(hh_))
                print(f"  [inference temporal gradient | {tag}] two-hop recall FROM THE WEIGHTS ALONE (store off) by conclusion age (split at visit {cut_}): "+" | ".join(f"{k_} {v_[0]:.2f} (n {v_[1]})" for k_,v_ in res_tg_.items())+" | CLS predicts OLDER > RECENT once transfer works",flush=True)
        tt_=_inf_truth(cfg); wr_=[c_ for c_ in S_['conc'] if (c_['A'],c_['C']) not in tt_[0]|tt_[1] and any(c_['A']==q_['names'][0] for q_ in ch_)]
        fm_=float('nan')
        if wr_:
            pm_={q_['names'][0]:q_ for q_ in ch_}; pp_=[];cc_=[];ss_=[]
            for c_ in wr_[:60]:
                q_=pm_[c_['A']]; pp_.append(f"So {c_['A']} is connected to "); cc_.append([c_['C'],q_['names'][2]]); ss_.append(int(model.__dict__.get('_t2s',{}).get(q_['doms'][0],q_['doms'][0])))
            fh_=[]
            for sl in sorted(set(ss_)):
                ix_=[i_ for i_,s_ in enumerate(ss_) if s_==sl]; fh_+=[int(h_==0) for h_ in score([pp_[i_] for i_ in ix_],[cc_[i_] for i_ in ix_],sl)]
            fm_=float(np.mean(fh_))
    finally: model.train(was_)
    print(f"  [inference eval | {tag}] probe recall {'WEIGHTS + FAST STORE' if _INF.get('corr') else 'FROM THE WEIGHTS'} (chance 0.05): "+" | ".join(f"{k_} {v_[0]:.2f} (n {v_[1]})" for k_,v_ in res.items())
          +(f" | false memory: wrong probe conclusions preferred over the truth in {fm_:.2f} of {min(60,len(wr_))}" if wr_ else " | no wrong probe conclusions"),flush=True)
    cr_ok_=list(_INF.get('calrec_all') or []) if _INF.get('corr') else []
    if _INF.get('corr') and _INF.get('calrec_all'):                               # CALIBRATION: do the 'knows what it knows' signals predict a correct answer?
        cr_=_INF.pop('calrec_all'); ok_=np.array([r_[0] for r_ in cr_]); cf_=np.array([r_[1] for r_ in cr_]); hit_=np.array([r_[3] for r_ in cr_])
        def auc_(sg_,y_):
            P_=sg_[y_==1]; N_=sg_[y_==0]
            return float(np.mean([(p0_>n0_)+0.5*(p0_==n0_) for p0_ in P_ for n0_ in N_])) if len(P_) and len(N_) else float('nan')
        print(f"  [inference eval | calibration, two-hop with store] accuracy {ok_.mean():.2f} | confidence (top probability) AUC {auc_(cf_,ok_):.2f}, mean when right {cf_[ok_==1].mean() if ok_.any() else float('nan'):.2f} / wrong {cf_[ok_==0].mean() if (~ok_.astype(bool)).any() else float('nan'):.2f} | "
              f"store hit (subject in the store) AUC {auc_(hit_.astype(float),ok_):.2f}, accuracy with hit {ok_[hit_==1].mean() if (hit_==1).any() else float('nan'):.2f} (n {int(hit_.sum())}) / without {ok_[hit_==0].mean() if (hit_==0).any() else float('nan'):.2f} (n {int((hit_==0).sum())})",flush=True)
    if _INF.get('corr') and cr_ok_:
        _inf_abstain_sources(S_,cr_ok_,tag)
    if _INF.get('corr') and int(getattr(cfg,'inf_eval_hard',0)): print("  [inference eval | store activity] mean max |logit correction| per probe position: "+" | ".join(f"{k_} {v_:.3f}" for k_,v_ in act_.items()),flush=True)
    return res
def inf_sleep(model,cfg,t,ts,vals,olds_,S_slot):
    """after a visit: infer on the store, verify, consolidate the conclusions (planted alone into kept verified windows), report"""
    _inf_sync(cfg,'sleep start')
    if int(getattr(cfg,'inf_sofl',0)): _inf_sofl(model,cfg,t,ts,vals,olds_,S_slot)
    S_=_inf_store(model); nid_=list(S_['new']); vis0_=int(model.__dict__.get('_visit',0)); ord_=str(getattr(cfg,'inf_sleep_order','nrem'))
    if int(getattr(cfg,'inf_disc',1)): new_,st_=inf_infer(model,cfg)
    else: new_,st_=[],{'bridges':0,'hub':0,'prop':0}; S_['new']=[]                 # sleep-order test: no co-occurrence discovery (REM links instead)
    t2_,t3_=_inf_truth(cfg)
    if ord_!='nrem':
        nf_=_inf_facts(cfg,S_,nid_,vis0_)
        if int(getattr(cfg,'inf_rules',0)): _inf_rules(cfg,S_,vis0_)
        if ord_=='rem_nrem': _inf_rem(model,cfg,S_,S_slot,int(getattr(cfg,'inf_rem_n',48)),'REM BEFORE NREM')
    elif int(getattr(cfg,'inf_rem_shadow',0)): _inf_facts(cfg,S_,nid_,vis0_)
    if int(getattr(cfg,'inf_hyp',0)): new_=new_+_inf_hyp(model,cfg,S_,nid_,st_)
    for c_ in new_: c_.setdefault('v',vis0_)
    if int(getattr(cfg,'inf_prune',0)) or int(getattr(cfg,'inf_xfer_salience',0)): _inf_use(S_,nid_,vis0_)
    if str(getattr(cfg,'inf_self','off'))!='off' and new_: new_=inf_self(model,cfg,S_,new_,S_slot,t2_)
    pt_=[c_ for c_ in new_ if (c_['A'],c_['C']) in t2_|t3_]; pn_=set(n_ for q_ in inf_probes(cfg) for n_ in q_['names']); pw_=[c_ for c_ in new_ if (c_['A'] in pn_ or c_['C'] in pn_) and c_ not in pt_]
    allc_=S_['conc']; found2_=len(set(frozenset((c_['A'],c_['C'])) for c_ in allc_ if (c_['A'],c_['C']) in t2_)); found3_=len(set(frozenset((c_['A'],c_['C'])) for c_ in allc_ if (c_['A'],c_['C']) in t3_))
    print(f"  [inference] {_lab(cfg,t)}: store {len(S_['it'])} sentences from {S_['nwin']} windows | bridges {st_['bridges']} ({st_['hub']} hubs abstained) | "
          f"{len(new_)} new conclusions ({sum(c_['typed'] for c_ in new_)} typed, {sum(c_['dep']>1 for c_ in new_)} three-hop) | probe-related: {len(pt_)} TRUE, {len(pw_)} wrong | "
          f"probe chains found so far: two-hop {found2_}/{len(t2_)//2}, three-hop {found3_}/{len(t3_)//2}"+(f" | {st_['pool']}" if st_.get('pool') else ''),flush=True)
    if new_: print("    e.g. "+" || ".join(c_['s'] for c_ in new_[:3]),flush=True)
    if _units_on(cfg) and _UNITS.get('all'):
        pnb_=set(n_.encode() for n_ in pn_); df0_=S_['df']; mx0_=int(cfg.inf_df)
        rare_=sum(1 for w_,c0_ in df0_.items() if c0_<=mx0_ and len(w_)>=3); pool_=[c_ for c_ in S_['conc'] if not c_.get('freed')]
        gs_=[c_.get('gain',0.0) for c_ in pool_]
        print(f"  [inference WORD-LEVEL LAYER] {len(df0_)} word types counted, {rare_} currently rare (window frequency <= {mx0_}) | probe names rare {sum(1 for n_ in pnb_ if 0<df0_.get(n_,0)<=mx0_)}/{len(pnb_)} (seen {sum(1 for n_ in pnb_ if df0_.get(n_,0)>0)}) | "
              f"links in the pool {len(pool_)}, gain median {float(np.median(gs_)) if gs_ else float('nan'):.1f} bits (probe-true links {sum(1 for c_ in pool_ if (c_['A'],c_['C']) in t2_|t3_)}, gain median {float(np.median([c_.get('gain',0) for c_ in pool_ if (c_['A'],c_['C']) in t2_|t3_] or [float('nan')])):.1f})",flush=True)
    elif _units_on(cfg):
        lx_=S_.get('lex',set()); md_=int(getattr(cfg,'inf_units_maxdf',0)) or int(cfg.inf_df)
        dem_=[w_ for w_ in lx_ if S_['df'].get(w_,0)>md_]                       # FREQUENCY DEMOTION: a unit seen in too many windows is too common to be an entity
        for w_ in dem_: lx_.discard(w_); S_.setdefault('lexdem',set()).add(w_)
        pnb_=set(n_.encode() for n_ in pn_); rv0_=S_.get('ures',[])
        print(f"  [inference RULE-FREE UNITS] learned lexicon {len(lx_)} word types (+{S_.get('unf',0)} since last sleep, {len(dem_)} demoted as too common, e.g. {[w_.decode('latin-1') for w_ in dem_[:6]]}) | probe names in it {len(pnb_&lx_)}/{len(pnb_)} | "
              f"completion threshold {float(np.quantile(rv0_,1-float(getattr(cfg,'inf_units_q',0.2)))) if len(rv0_)>=int(getattr(cfg,'inf_units_warm',2000)) else float('nan'):.2f} bits (from {len(rv0_)} recent words) | e.g. {[w_.decode('latin-1','replace') for w_ in list(lx_)[-12:]]}",flush=True)
        S_['unf']=0
    for c_ in pw_[:3]: print(f"  [inference wrong] probe-related: {c_['A']} ~ {c_['C']} via {c_['B']} | sources: {S_['it'][c_['src'][0]]['s'][:120]!r} || {S_['it'][c_['src'][1]]['s'][:120]!r}",flush=True)
    if (not allc_ and not (ord_!='nrem' and S_.get('facts'))) or int(cfg.inf_steps)<=0: return
    nb_=int(cfg.pu_cons_eval) or 4; b0_=bpc(model,vals[t],ts,n=nb_); o0_=float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_])) if olds_ else float('nan')
    RV_=model.__dict__.get('_nres') or {}; BK_=model.__dict__.get('_nres_bank',{}); kept_=list(RV_.get('e',[]))+[e_ for l_ in BK_.values() for e_ in l_]
    dev_=model.embed.weight.device; rs_=model.__dict__.setdefault('_inf_rs',np.random.RandomState(int(cfg.seed)+2121)); Tq=int(cfg.seq_len)
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; op_=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(cfg.inf_lr))
    TP_=[p_.detach().clone() for p_ in PA_]                                       # ANCHORS keep the PRE-SLEEP model's output (KL), as cg19 consolidation
    def teach(X_,d_):
        with torch.no_grad():
            sv_=[p_.data for p_ in PA_]
            for p_,q_ in zip(PA_,TP_): p_.data=q_
            try: return F.log_softmax(fwd(X_,d_),-1)
            finally:
                for p_,v_ in zip(PA_,sv_): p_.data=v_
    ring_=S_['ring']; Lr_=min(int(cfg.inf_rowlen)+1,len(ring_[0][0]) if ring_ else Tq+1)                    # every row has the store's window length
    R_=int(cfg.inf_rows); kc_=max(1,int(round(R_*float(cfg.inf_mix)))); ka_=R_-kc_ if (kept_ or ring_) else 0; ls_=[]
    pool_=[c_ for c_ in allc_ if c_.get('n',0)<int(cfg.inf_target)] or allc_                 # conclusions not yet consolidated (exposure target)
    def filler(dom_):
        sd_=[w0_ for w0_,dd_ in ring_ if dd_==dom_]                              # same-domain text around a conclusion (no foreign text under a head)
        if sd_: return sd_[rs_.randint(len(sd_))]
        if ring_: return ring_[rs_.randint(len(ring_))][0]
        return b' '*Lr_
    def fwd(X_,d_):
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_
        try: return model(X_[:,:-1],d_).float()
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_
    if str(getattr(cfg,'inf_cons','w'))=='mem': return _inf_mem_write(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,kept_,Lr_,R_,kc_,ka_,filler,fwd,rs_,dev_,b0_,o0_,nb_)
    model.train()
    for st in range(int(cfg.inf_steps)):
        loss_=0.0; grp_={}
        for c_ in [pool_[i_] for i_ in rs_.choice(len(pool_),kc_)]:
            c_['n']=c_.get('n',0)+1
            w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); u_=rs_.rand(); sn_=c_['s'] if (c_['typed'] and u_<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
            it_=(' '+sn_+' ').encode('latin-1')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_; w_=w_[:Lr_]
            mk_=torch.zeros(Lr_-1); mk_[max(0,o_):o_+len(it_)-1]=1.0                   # targets = the conclusion's bytes only
            grp_.setdefault(int(S_slot(c_['d'])),[]).append((torch.tensor(list(bytes(w_)),dtype=torch.long),mk_))
        for _ in range(ka_):                                                    # ANCHORS: kept verified windows (closed loop) or recent windows
            if ring_:
                w0_,dd_=ring_[rs_.randint(len(ring_))]; o0_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0_:o0_+Lr_]),dtype=torch.long); da_=int(S_slot(dd_))
            else:
                e_=kept_[rs_.randint(len(kept_))]; xa_=e_['x'].to('cpu',torch.long)[:Lr_]; da_=int(e_['d'])
            if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
            grp_.setdefault(da_,[]).append((xa_,None))
        for d_,rows_ in grp_.items():
            cr_=[r_ for r_ in rows_ if r_[1] is not None]; ar_=[r_ for r_ in rows_ if r_[1] is None]
            if cr_:
                X_=torch.stack([r_[0] for r_ in cr_]).to(dev_); M_=torch.stack([r_[1] for r_ in cr_]).to(dev_); lg_=fwd(X_,d_)
                ce_=F.cross_entropy(lg_.reshape(-1,256),X_[:,1:].reshape(-1),reduction='none').view(M_.shape)
                loss_=loss_+(ce_*M_).sum()/M_.sum().clamp(min=1)*len(cr_)/R_
            if ar_:
                X_=torch.stack([r_[0] for r_ in ar_]).to(dev_); T_=teach(X_,d_); L_=F.log_softmax(fwd(X_,d_),-1)
                kl_=(T_.exp()*(T_-L_)).sum(-1); nz_=(X_[:,1:]>0).float()                 # blanked positions excluded
                loss_=loss_+float(cfg.inf_anchor_w)*(kl_*nz_).sum()/nz_.sum().clamp(min=1)*len(ar_)/R_
        op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(loss_)); _ng_zero(model)
        for p_ in PA_: p_.grad=None
    b1_=bpc(model,vals[t],ts,n=nb_); o1_=float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_])) if olds_ else float('nan')
    print(f"  [inference sleep] {int(cfg.inf_steps)} steps at lr x{cfg.inf_lr} on {len(pool_)} conclusions ({kc_} rows/step) + {ka_} kept verified rows/step | "
          f"loss {ls_[0]:.3f} -> {ls_[-1]:.3f} | held-out BPC visited {b0_:.3f} -> {b1_:.3f}, old domains {o0_:.3f} -> {o1_:.3f}",flush=True)
    if int(cfg.inf_eval_every)>0 and (int(model.__dict__.get('_visit',0))+1)%int(cfg.inf_eval_every)==0: inf_eval(model,cfg,S_,f"visit {int(model.__dict__.get('_visit',0))+1}")

@torch.no_grad()
def _inf_mem_write(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,kept_,Lr_,R_,kc_,ka_,filler,fwd,rs_,dev_,b0_,o0_,nb_):
    """cg20h: write the verified conclusions into the fast store (delta rule on the residual), with do-no-harm anchor writes"""
    _infm_init(model,cfg); was_=model.training; model.eval(); beta_=float(getattr(cfg,'inf_mem_beta',0.5)); fa_=float(getattr(cfg,'inf_mem_nofill',0.5))
    def feats(X_,d_):
        sh_=_HCAP['on']; _HCAP['on']=True; sv_=_INFM['on']; _INFM['on']=False
        try: lg_=fwd(X_,d_); return lg_,_HCAP['h'].float()
        finally: _HCAP['on']=sh_; _INFM['on']=sv_
    if _INFM['mu'] is None:                                                     # centre of the feature space: recent verified / real windows
        zs_=[]
        for _ in range(4):
            if not ring_: break
            rows_=[ring_[rs_.randint(len(ring_))] for _ in range(8)]; X_=torch.stack([torch.tensor(list((w_+b' '*Lr_)[:Lr_]),dtype=torch.long) for w_,_ in rows_]).to(dev_)
            _,z_=feats(X_,int(S_slot(rows_[0][1]))); zs_.append(z_.reshape(-1,z_.shape[-1]))
        _INFM['mu']=torch.cat(zs_).mean(0) if zs_ else torch.zeros(_INFM['WX'].shape[1],device=dev_)
    import zlib as _zl
    ord_=str(getattr(cfg,'inf_sleep_order','nrem')); FX_=S_.get('facts',[]) if ord_!='nrem' else []; REMk_=int(getattr(cfg,'inf_sleep_cycles',1)) if ord_=='nrem_rem' else 0; remd_=0
    PN_=S_.setdefault('pruned_names',set())
    for c_ in [c0_ for c0_ in list(allc_)+FX_ if not c0_.get('freed')]:
        for n_ in ((c_['A'],c_['B']) if c_.get('fact') else ((c_['A'],c_['B'],c_['C']) if float(getattr(cfg,'inf_mem_src',0.0))>0 else (c_['A'],c_['C']))):
            if n_.encode('latin-1') not in PN_: _INFM['names'].setdefault(n_.encode('latin-1'),int(_zl.crc32(n_.encode('latin-1'))))
    ainfo_=_inf_audit(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_)
    nmt_=int(getattr(cfg,'inf_mem_maint',0))
    if nmt_>0 and _INFM.get('mu') is not None:                                  # STORE MAINTENANCE: the trunk keeps learning, so the store's keys drift;
        old_=[c_ for c_ in allc_ if not c_.get('freed') and c_.get('n',0)>0]       #   stored conclusions it no longer recalls are rewritten first
        nm_=[n_.decode('latin-1') for n_ in _INFM['names']]; fail_=0; ck_=[]
        if old_ and len(nm_)>=25:
            for c_ in [old_[i_] for i_ in rs_.choice(len(old_),min(len(old_),nmt_),replace=False)]:
                oth_=[n_ for n_ in nm_ if n_ not in (c_['A'],c_['C'])]; cs_=[c_['C']]+list(rs_.choice(oth_,19,replace=False))
                ok_=_inf_score1(model,cfg,f"So {c_['A']} is connected to ",cs_,int(S_slot(c_['d'])),True)==0; ck_.append(ok_)
                if not ok_: c_['n']=0; fail_+=1
        ainfo_+=f" | MAINTENANCE: {fail_} of {len(ck_)} stored conclusions no longer recalled (store recall {np.mean(ck_) if ck_ else float('nan'):.2f}) -> rewritten first"
    rc_=[];ra_=[]; wn_=0
    rn_=[i_ for i_,(w0_,_) in enumerate(ring_) if any(n_ in _INFM['names'] for n_ in _INF_NAME.findall(w0_))] if float(getattr(cfg,'inf_mem_anchor_named',0.0))>0 else []
    for st in range(int(cfg.inf_steps)):
        grp_={}
        live_=([c0_ for c0_ in allc_ if not c0_.get('freed')] or list(allc_))+[f0_ for f0_ in FX_ if not f0_.get('freed')]       # sleep-order test: NREM also replays this visit's recorded facts
        if not live_: break
        if int(getattr(cfg,'inf_mem_prio',0)):                                  # cg20i: conclusions below the exposure target are written first
            lo_=[c0_ for c0_ in live_ if c0_.get('n',0)<int(cfg.inf_target)]; live_=lo_ or live_
            if int(getattr(cfg,'inf_mem_prio',0))>=2:                              # cg20j: RARE LINKS FIRST - the rarest-named conclusions take the write budget
                dfm_=S_['df']; live_=sorted(live_,key=lambda c0_: dfm_.get(c0_['A'].encode('latin-1'),0)+dfm_.get(c0_.get('C',c0_['B']).encode('latin-1'),0))[:max(4*kc_,256)]
        sf_=float(getattr(cfg,'inf_mem_src',0.0)); add_=str(getattr(cfg,'inf_mem_src_mode','split'))=='add'
        sel_=[(live_[i_],'c' if add_ else None) for i_ in rs_.choice(len(live_),kc_)]
        if add_ and sf_>0: sel_+=[(live_[i_],'s') for i_ in rs_.choice(len(live_),max(1,int(round(kc_*sf_))))]   # one-hop facts IN ADDITION to the conclusion rows
        for c_,kd_ in sel_:
            if kd_!='s': c_['n']=c_.get('n',0)+1
            if c_.get('fact'): sn_=c_['s']                                       # a recorded one-hop fact (NREM replay)
            elif kd_=='s' or (kd_ is None and sf_>0 and rs_.rand()<sf_): sn_=S_['it'][c_['src'][rs_.randint(2)]]['s'].decode('latin-1').strip()   # a verified SOURCE fact (one-hop) of this conclusion
            else: sn_=c_['s'] if (c_['typed'] and rs_.rand()<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
            if rs_.rand()<fa_:                                                  # no context before the conclusion (as the probes are asked)
                it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); o_=0
            else:
                w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); it_=(' '+sn_+' ').encode('latin-1')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_
            w_=w_[:Lr_]; mk_=torch.zeros(Lr_-1); mk_[max(0,o_-1) if o_>0 else 0:o_+len(it_)-1]=1.0
            grp_.setdefault(int(S_slot(c_['d'])),[]).append((torch.tensor(list(bytes(w_)),dtype=torch.long),mk_))
        for _ in range(ka_):
            if ring_:
                af_=float(getattr(cfg,'inf_mem_anchor_named',0.0))                 # cg20i: do-no-harm anchors aimed at text where stored names occur
                if af_>0 and rn_ and rs_.rand()<af_: w0_,dd_=ring_[rn_[rs_.randint(len(rn_))]]
                else: w0_,dd_=ring_[rs_.randint(len(ring_))]
                o0b_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0b_:o0b_+Lr_]),dtype=torch.long); da_=int(S_slot(dd_))
            else:
                e_=kept_[rs_.randint(len(kept_))]; xa_=e_['x'].to('cpu',torch.long)[:Lr_]; da_=int(e_['d'])
            if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
            grp_.setdefault(da_,[]).append((xa_,None))
        for d_,rows_ in grp_.items():
            X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); lg_,z_=feats(X_,d_); Y_=X_[:,1:]
            if _INFM['key']=='latch':                                                    # threshold: 85th percentile of word-start surprise on ordinary (anchor) rows
                ar_=[i_ for i_,r_ in enumerate(rows_) if r_[1] is None]
                if ar_:
                    _,_,ws_=_infm_latch(z_[ar_].float()-_INFM['mu'],X_[ar_,:-1],lg_[ar_])
                    if ws_: q85_=float(np.percentile(ws_,85)); _INFM['lthr']=q85_ if _INFM['lthr'] is None else 0.9*_INFM['lthr']+0.1*q85_
            K_,id2_=_infm_keys(z_,X_[:,:-1],lg_); zf_=K_.reshape(-1,K_.shape[-1]); idf_=id2_.reshape(-1); c0_=_infm_corr(zf_,idf_); lgf_=lg_.reshape(-1,256)
            q_=F.softmax(lgf_+c0_,-1); p0_=F.softmax(lgf_,-1); oh_=F.one_hot(Y_.reshape(-1),256).float()
            isc_=torch.tensor([r_[1] is not None for r_ in rows_],device=dev_)[:,None].expand(-1,Lr_-1).reshape(-1)
            mk_=torch.stack([r_[1] if r_[1] is not None else torch.zeros(Lr_-1) for r_ in rows_]).to(dev_).reshape(-1)
            nz_=(Y_.reshape(-1)>0).float()*(~isc_).float()
            for it_ in range(int(getattr(cfg,'inf_mem_iters',1))):                  # repeated delta writes on the same rows (features fixed): fit to the fixed point
                if it_>0: q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1)
                R_c_=(oh_-q_)*mk_[:,None]; R_a_=((oh_ if int(getattr(cfg,'inf_mem_replay',0)) and float(model.__dict__.get('_real_w',1.0))==0 else p0_)-q_)*nz_[:,None]   # conclusions: fit the bytes | anchors: undo any effect (or, inf_mem_replay in closed-loop passes: LEARN the verified replay bytes)
                _infm_write(zf_,idf_,R_c_+R_a_,beta_,gate_lr=(float(getattr(cfg,'inf_mem_gate_lr',0.002)) if (_INFM['key']=='slowgate' and it_==0) else 0.0))
            q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1); mk_=mk_*(idf_>=0).float(); _INFM['wpos']=_INFM.get('wpos',0)+int(mk_.sum())     # fit after the writes, on the store-keyed conclusion bytes
            if mk_.sum()>0: rc_.append(float(-(q_.gather(-1,Y_.reshape(-1,1)).squeeze(-1).clamp(min=1e-9).log()*mk_).sum()/mk_.sum()/math.log(2)))
            if nz_.sum()>0: ra_.append(float(((q_*(q_.clamp(min=1e-9).log()-p0_.clamp(min=1e-9).log())).sum(-1)*nz_).sum()/nz_.sum()))
            wn_+=1
        if REMk_ and remd_<REMk_ and (st+1)%max(1,int(cfg.inf_steps)//REMk_)==0:   # NREM chunk done -> a REM episode (cycles: its links are written by the next chunk)
            remd_+=1; _inf_rem(model,cfg,S_,S_slot,max(1,int(getattr(cfg,'inf_rem_n',48))//REMk_),f"NREM -> REM, cycle {remd_}/{REMk_}")
    nh_=int(getattr(cfg,'inf_prune_harm',0))
    if nh_>0 and rn_:                                                           # HARM-BASED PRUNING: each stored name's measured effect on recent REAL text
        H_=S_.setdefault('harm',{}); inv_={v_:k_ for k_,v_ in _INFM['names'].items()}; acc_={}
        pick_=[ring_[rn_[i_]] for i_ in rs_.choice(len(rn_),min(len(rn_),nh_),replace=False)]
        for dd0_ in sorted(set(int(d_) for _,d_ in pick_)):
            rows_=[w_ for w_,d_ in pick_ if int(d_)==dd0_]
            for b0_ in range(0,len(rows_),16):
                X_=torch.stack([torch.tensor(list((w_+b' '*Lr_)[:Lr_]),dtype=torch.long) for w_ in rows_[b0_:b0_+16]]).to(dev_); sl0_=int(S_slot(dd0_)); tg_=X_[:,1:,None]
                _INFM['force']=False; lo_=F.log_softmax(fwd(X_,sl0_),-1).gather(-1,tg_).squeeze(-1)
                _INFM['force']=True; ln_=F.log_softmax(fwd(X_,sl0_),-1).gather(-1,tg_).squeeze(-1); _INFM.pop('force',None)
                dh_=(lo_-ln_).detach().cpu().numpy(); ids_=_infm_ids(X_[:,:-1]).cpu().numpy()     # > 0: the store makes the true byte LESS likely
                for v_ in np.unique(ids_[ids_>=0]):
                    m_=ids_==v_; a0_=acc_.setdefault(int(v_),[0.0,0,[]]); a0_[0]+=float(dh_[m_].sum()); a0_[1]+=int(m_.sum())
                    for j0_ in range(m_.shape[0]):
                        if m_[j0_].any(): a0_[2].append(float(dh_[j0_][m_[j0_]].sum())/math.log(2))   # one SAMPLE per window: bits the store costs (+) or saves (-) there
        thr_=float(getattr(cfg,'inf_prune_harm_thr',0.02)); pn0_=set(x_.encode('latin-1') for q_ in inf_probes(cfg) for x_ in q_['names']); cut_=[]
        if int(getattr(cfg,'inf_prune_mdl',0)):                                # MDL / SEQUENTIAL BIT ACCOUNTING: every stored name is a 'rule' that must pay for itself.
            G_=S_.setdefault('mdl',{}); z_=float(getattr(cfg,'inf_mdl_z',2.0)); nmin_=int(getattr(cfg,'inf_mdl_min',8)); cst_=float(getattr(cfg,'inf_mdl_cost',0.0))
            for v_,(s_,n_,ws_) in acc_.items():                                #   bits it costs or saves accumulate over ALL sleeps (one sample per window it fires in);
                nm_=inv_.get(v_)                                               #   drop only when the whole evidence says it costs bits with confidence (lower bound > 0)
                if nm_ is None: continue
                g_=G_.setdefault(nm_,[0,0.0,0.0])
                for w0_ in ws_: g_[0]+=1; g_[1]+=w0_+cst_; g_[2]+=(w0_+cst_)**2
            nu_=0; nh0_=0; nd_=0; rel_=float(getattr(cfg,'inf_mdl_rel',0.0)); n0_=float(getattr(cfg,'inf_mdl_n0',8.0)); rinfo_=''
            live0_=[(nm_,v_) for nm_,v_ in G_.items() if nm_ in _INFM['names'] and v_[0]>0]
            med_=float(np.median([-v_[1]/v_[0] for _,v_ in live0_])) if live0_ else 0.0      # GOING RATE: median bits SAVED per window across stored names
            nrel_=0
            for nm_,(n_,s_,ss_) in G_.items():
                if nm_ not in _INFM['names'] or n_<nmin_: nd_+=int(nm_ in _INFM['names']); continue
                mu_=s_/n_; se_=math.sqrt(max(ss_/n_-mu_*mu_,1e-12)/n_)
                if mu_-z_*se_>0: cut_.append(nm_)
                elif rel_>0 and med_>0 and (n_*(-mu_)+n0_*med_)/(n_+n0_)<rel_*med_: cut_.append(nm_); nrel_+=1   # EMPIRICAL BAYES: the name's gain shrunk towards the going rate
                elif mu_+z_*se_<0: nu_+=1
                else: nd_+=1
            if rel_>0: rinfo_=f" | going rate {med_:+.3f} bits saved per window; {nrel_} removed as below {rel_:.2f} x the going rate after shrinkage (prior weight {n0_:.0f} windows)"
            ainfo_+=f" | MDL ACCOUNTING: {len(G_)} names with evidence; proven to SAVE bits {nu_}, undecided {nd_}, removed {len(cut_)} (probe names among them {sum(1 for x_ in cut_ if x_ in pn0_)})"+rinfo_
            acc_={}
        for v_,(s_,n_,*_) in acc_.items():
            nm_=inv_.get(v_)
            if nm_ is None or n_<int(getattr(cfg,'inf_prune_harm_min',8)): continue
            h0_=H_.get(nm_); H_[nm_]=s_/n_ if h0_ is None else 0.5*h0_+0.5*s_/n_
            if H_[nm_]>thr_ and h0_ is not None and h0_>thr_: cut_.append(nm_)      # harmful in two sleeps running
        for nm_ in cut_:
            _INFM['names'].pop(nm_,None); PN_.add(nm_); s0_=nm_.decode('latin-1')
            for c_ in list(allc_)+FX_:
                if s0_ in (c_.get('A'),c_.get('B'),c_.get('C')): c_['freed']=True; c_['pruned']=True
        ainfo_+=f" | HARM PRUNING: {len(acc_)} stored names measured on {len(pick_)} recent real windows, {len(cut_)} keys removed as net-harmful (probe names among them {sum(1 for x_ in cut_ if x_ in pn0_)}), total pruned names {len(PN_)}"
    if int(getattr(cfg,'inf_rem_shadow',0)) and ord_=='nrem':                  # SHADOW REM: lookup recombination after the writes, counted only
        try: _inf_rem(model,cfg,S_,S_slot,int(getattr(cfg,'inf_rem_n',48)),'SHADOW after NREM writes (counted, not written; "already known" = found by discovery)',shadow_=True)
        except Exception as e_: print(f"  [inference REM recombination] skipped: {e_!r}"[:300],flush=True)
    _inf_sync(cfg,'store writes'); xinfo_=_inf_xfer(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,Lr_,kc_,ka_,R_,filler,fwd,rs_,dev_,nb_,fa_)
    xinfo_+=_inf_free(model,cfg,S_,allc_,S_slot,Lr_,filler,fwd,feats,rs_,dev_,beta_)
    xinfo_+=_inf_prune(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_)
    model.train(was_); _inf_sync(cfg,'transfer / freeing / pruning')
    b1_=bpc(model,vals[t],ts,n=nb_); _inf_sync(cfg,'held-out BPC with the store'); o1_=float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_])) if olds_ else float('nan')
    print(f"  [inference sleep | FAST STORE] keyed positions written {_INFM.get('wpos',0)} | {int(cfg.inf_steps)} write steps ({kc_} conclusion rows + {ka_} anchor rows) on {len(allc_)} conclusions | "
          f"keyed conclusion bytes after writing {rc_[0] if rc_ else float('nan'):.3f} (first step), {np.mean(rc_[-5:]) if rc_ else float('nan'):.3f} (last steps) bits | anchor KL(store||model) {np.mean(ra_[-5:]) if ra_ else float('nan'):.4f} | "
          f"|M| {float(_INFM['M'].norm()):.1f}{ainfo_} | trunk weight sum {float(sum(p0_.detach().double().abs().sum() for p0_ in model.parameters())):.6e}{' | gates '+str([round(float(g0_),2) for g0_ in _INFM['gates']]) if _INFM['key']=='slowgate' else ''}{xinfo_} | held-out BPC (store on) visited {b0_:.3f} -> {b1_:.3f}, old domains {o0_:.3f} -> {o1_:.3f}",flush=True)
    if int(cfg.inf_eval_every)>0 and (int(model.__dict__.get('_visit',0))+1)%int(cfg.inf_eval_every)==0: inf_eval(model,cfg,S_,f"visit {int(model.__dict__.get('_visit',0))+1}")

def inf_final_bpc(model,cfg,vals,S_):
    inf_need_report(model)
    """end of run: every domain's held-out BPC with the fast store OFF (weights only) and ON (same model): its direct cost"""
    if _INFM['M'] is None or _INFM.get('mu') is None: return
    r_={}
    try:
        for on_ in (False,True): _INFM['force']=on_; r_[on_]=[bpc(model,vals[d_],S_(d_)) for d_ in range(len(vals))]
    finally: _INFM.pop('force',None)
    print(f"  [inference final BPC] store OFF (weights only) {np.mean(r_[False]):.3f} | store ON {np.mean(r_[True]):.3f} | direct cost of the store {np.mean(r_[True])-np.mean(r_[False]):+.3f} | per domain off/on "
          +" ".join(f"{a_:.3f}/{b_:.3f}" for a_,b_ in zip(r_[False],r_[True]))+f" | store used in BPC during training: {_INFM.get('bpc_on',True)}",flush=True)

def _inf_conc_row(c_,Lr_,filler,rs_,fa_):
    """a conclusion row (as the store writes): the sentence alone at the start (share fa_) or planted in same-domain filler; mask = its bytes"""
    sn_=c_['s'] if (c_['typed'] and rs_.rand()<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
    if rs_.rand()<fa_:
        it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); o_=0
    else:
        w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); it_=(' '+sn_+' ').encode('latin-1')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_
    w_=w_[:Lr_]; mk_=torch.zeros(Lr_-1); mk_[max(0,o_-1) if o_>0 else 0:o_+len(it_)-1]=1.0
    return torch.tensor(list(bytes(w_)),dtype=torch.long),mk_
def _inf_xfer(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,Lr_,kc_,ka_,R_,filler,fwd,rs_,dev_,nb_,fa_):
    """STORE -> WEIGHTS TRANSFER (cg20 test 2): 'teach' = small steps of KL(pre-sleep weights + store || weights) on conclusion rows, anchors
    KL(pre-sleep weights || weights) on recent windows; 'brake' = the same, rolled back if store-off held-out BPC rises more than inf_xfer_eps"""
    mode_=str(getattr(cfg,'inf_xfer','none')); xs_=int(getattr(cfg,'inf_xfer_steps',0)); live_=[c_ for c_ in allc_ if not c_.get('freed')]
    if mode_ not in ('teach','brake','skill') or xs_<=0 or not live_: return ''
    if mode_=='skill':                                                          # SKILL TRANSFER: worked examples (sources + conclusion) from the store, never a held-out chain
        ho_=_inf_heldout(cfg); live_=[c_ for c_ in live_ if not ({c_['A'],c_['B'],c_['C']}&ho_)]
        if not live_: return ''
    V0_=int(model.__dict__.get('_visit',0)); sinfo_=''
    if int(getattr(cfg,'inf_xfer_salience',0)):                                 # SALIENCE: used (re-encountered) and older conclusions are consolidated first
        live_=sorted(live_,key=lambda c_: -(c_.get('use',0)+(V0_-c_.get('v',V0_))))[:int(getattr(cfg,'inf_xfer_n',64))]
        sinfo_=f", salience-selected {len(live_)} (mean use {np.mean([c_.get('use',0) for c_ in live_]):.1f}, mean age {np.mean([V0_-c_.get('v',V0_) for c_ in live_]):.1f} visits)"
    sp_=int(getattr(cfg,'inf_xfer_space',0)); sched_=None
    if sp_>0:                                                                   # SPACING: every selected conclusion gets exactly inf_xfer_space rows this sleep
        sched_=[c_ for c_ in live_ for _ in range(sp_)]; rs_.shuffle(sched_); xs_=min(xs_,max(1,int(math.ceil(len(sched_)/max(1,kc_)))))
    old_=[d_ for d_ in (S_.get('res') or {}) if d_!=int(t) and S_['res'][d_]]; fo_=float(getattr(cfg,'inf_xfer_old',0.0))
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; TP_=[p_.detach().clone() for p_ in PA_]
    def bpc_off():
        _INFM['force']=False
        try: return float(np.mean([bpc(model,vals[t],ts,n=nb_)]+[bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_]))
        finally: _INFM.pop('force',None)
    b0_=bpc_off() if mode_ in ('brake','skill') else float('nan')
    def teacher(X_,d_):
        sv_=[p_.data for p_ in PA_]
        for p_,q_ in zip(PA_,TP_): p_.data=q_
        _INFM['force']=True
        try:
            with torch.no_grad(): return F.log_softmax(fwd(X_,d_),-1)
        finally:
            _INFM.pop('force',None)
            for p_,v_ in zip(PA_,sv_): p_.data=v_
    op_=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(getattr(cfg,'inf_xfer_lr',0.05))); was_=model.training; model.train(); ls_=[]
    with torch.enable_grad():
        for st_ in range(xs_):
            grp_={}
            nk_=0; tries_=0; edge_=(mode_=='skill' and str(getattr(cfg,'inf_xfer_sel','random'))=='edge'); cand_=[]
            while (len(cand_) if edge_ else nk_)<(4*kc_ if edge_ else kc_) and tries_<16*kc_:
                if sched_ is not None:
                    if not sched_: break
                    c_=sched_.pop(); tries_+=1
                else: c_=live_[rs_.randint(len(live_))]; tries_+=1
                r0_=_inf_skill_row(c_,S_,Lr_,rs_) if mode_=='skill' else _inf_conc_row(c_,Lr_,filler,rs_,fa_)
                if r0_ is None: continue
                if edge_: cand_.append((c_,r0_)); continue
                grp_.setdefault(int(S_slot(c_['d'])),[]).append((r0_[0],r0_[1],mode_=='skill')); nk_+=1
            if edge_ and cand_:                                                 # EDGE OF COMPETENCE: the model's own in-context answer probability picks the examples
                with torch.no_grad():
                    lpa_=[]
                    for c_,r0_ in cand_:
                        X1_=r0_[0][None].to(dev_); lg1_=F.log_softmax(fwd(X1_,int(S_slot(c_['d']))),-1)[0]; m1_=r0_[1].to(dev_)
                        lpa_.append(float((lg1_.gather(-1,X1_[0,1:,None]).squeeze(-1)*m1_).sum()))
                tg_=math.log(float(getattr(cfg,'inf_xfer_edge_p',0.3))); sc_=[(abs(l_-tg_) if l_<math.log(0.8) else 1e9) for l_ in lpa_]
                for j_ in np.argsort(sc_)[:kc_]:
                    c_,r0_=cand_[int(j_)]; grp_.setdefault(int(S_slot(c_['d'])),[]).append((r0_[0],r0_[1],True)); nk_+=1
                model.__dict__.setdefault('_edge_log',[]).append(float(np.mean([math.exp(lpa_[int(j_)]) for j_ in np.argsort(sc_)[:kc_]])))
            for _ in range(ka_ if ring_ else 0):
                if old_ and rs_.rand()<fo_: dd_=old_[rs_.randint(len(old_))]; w0_=S_['res'][dd_][rs_.randint(len(S_['res'][dd_]))]   # INTERLEAVING: old domains' windows
                else: w0_,dd_=ring_[rs_.randint(len(ring_))]
                o0_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0_:o0_+Lr_]),dtype=torch.long)
                if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
                grp_.setdefault(int(S_slot(dd_)),[]).append((xa_,None,False))
            loss_=0.0
            for d_,rows_ in grp_.items():
                X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); T_=teacher(X_,d_); L_=F.log_softmax(fwd(X_,d_),-1); kl_=(T_.exp()*(T_-L_)).sum(-1)
                ce_=-L_.gather(-1,X_[:,1:,None]).squeeze(-1); hard_=torch.tensor([bool(r_[2]) for r_ in rows_],device=dev_)[:,None]
                W_=torch.stack([r_[1] if r_[1] is not None else (X_[j_,1:]>0).float().cpu() for j_,r_ in enumerate(rows_)]).to(dev_)
                loss_=loss_+(torch.where(hard_,ce_,kl_)*W_).sum()/W_.sum().clamp(min=1)*len(rows_)/R_   # skill rows: CE on the answer bytes; others: KL
            op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(loss_.detach())); _ng_zero(model)
            for p_ in PA_: p_.grad=None
    model.train(was_); info_=f" | TRANSFER {mode_}: {xs_} steps at lr x{getattr(cfg,'inf_xfer_lr',0.05)}{sinfo_}{', spacing '+str(sp_)+' rows per conclusion' if sp_ else ''}{', '+str(int(100*fo_))+'% of anchors from '+str(len(old_))+' old domains' if fo_>0 else ''}, KL {ls_[0] if ls_ else float('nan'):.4f} -> {np.mean(ls_[-5:]) if ls_ else float('nan'):.4f}"+(f", edge selection: mean in-context answer probability of the chosen examples {np.mean(model.__dict__['_edge_log'][-xs_:]):.2f}" if model.__dict__.get('_edge_log') and str(getattr(cfg,'inf_xfer_sel','random'))=='edge' else '')
    if mode_ in ('brake','skill'):
        b1_=bpc_off(); eps_=float(getattr(cfg,'inf_xfer_eps',0.01))
        if b1_-b0_>eps_:
            with torch.no_grad():
                for p_,q_ in zip(PA_,TP_): p_.data.copy_(q_)
            info_+=f", BRAKE: store-off BPC {b0_:.3f} -> {b1_:.3f} > +{eps_} ROLLED BACK"
        else: info_+=f", brake ok: store-off BPC {b0_:.3f} -> {b1_:.3f}"
        model.__dict__.setdefault('_xfer_rb',[]).append(int(b1_-b0_>eps_))
    return info_
@torch.no_grad()
def _inf_score1(model,cfg,prompt_,cands_,slot_,store_):
    seqs=[(prompt_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cands_]; L_=max(len(s_) for s_ in seqs); dev_=model.embed.weight.device
    X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
    for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cands_)]
    return int(np.argmax([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))]))
@torch.no_grad()
def _inf_free(model,cfg,S_,allc_,S_slot,Lr_,filler,fwd,feats,rs_,dev_,beta_):
    """FREEING (cg20 test 2d): a conclusion the WEIGHTS ALONE now recall (C beats 19 store names after 'So A is connected to') has its store
    entries faded (delta writes towards the weights' own prediction) and is no longer rehearsed"""
    if not int(getattr(cfg,'inf_xfer_free',0)): return ''
    live_=[c_ for c_ in allc_ if not c_.get('freed')]; nm_=[n_.decode('latin-1') for n_ in _INFM['names']]
    if not live_ or len(nm_)<25: return ''
    chk_=[live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),int(getattr(cfg,'inf_xfer_check',64))),replace=False)]; fr_=0
    for c_ in chk_:
        oth_=[n_ for n_ in nm_ if n_ not in (c_['A'],c_['C'])]; cs_=[c_['C']]+list(rs_.choice(oth_,19,replace=False))
        if _inf_score1(model,cfg,f"So {c_['A']} is connected to ",cs_,int(S_slot(c_['d'])),False)!=0: continue
        c_['freed']=True; fr_+=1
        for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []):
            it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
            X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(c_['d']))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
            Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
            for _ in range(10): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
    return f" | FREEING: {fr_} of {len(chk_)} checked recalled by the weights alone, entries faded (total freed {sum(int(c_.get('freed',False)) for c_ in allc_)} of {len(allc_)})"

# ---- cg20 SELF-TAUGHT INFERENCE (S2-S4): the model completes and checks the conclusions; the engine only pairs the source sentences --------
def _inf_words(t_):
    """generic word split (no name rule): letter runs of length >= 3"""
    return list(dict.fromkeys(w_ for w_ in _re_inf.findall(r'[A-Za-z]{3,}',t_)))
@torch.no_grad()
def _inf_lp(model,cfg,ctx_,cands_,slot_,store_):
    """log-prob of each candidate (+'.') after ctx_ (no carried context; store on/off)"""
    seqs=[(ctx_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cands_]; L_=max(len(s_) for s_ in seqs); dev_=model.embed.weight.device
    X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
    for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cands_)]
    return [float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))]
@torch.no_grad()
def _inf_gen(model,cfg,ctx_,slot_,store_,n_=14):
    """greedy continuation of ctx_ up to '.' (re-reads the whole prefix each step; no carried context)"""
    out_=b''; dev_=model.embed.weight.device; b_=ctx_.encode('latin-1')
    sv4_=_carry_save(); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try:
        for _ in range(n_):
            x_=(b_+out_)[-int(cfg.seq_len):]; _CARRY.update(on=False,st={},new={})
            lg_=model(torch.tensor([list(x_)],dtype=torch.long,device=dev_),slot_)[0,-1]; c_=int(lg_.argmax())
            if c_ in (46,10,0): break
            out_+=bytes([c_])
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    return out_.decode('latin-1').strip()
def inf_self(model,cfg,S_,new_,S_slot,t2_):
    """per sleep, for up to inf_self_n engine conclusions A ~ C (via bridge B, sources sa / sc):
      S2 proposal: the model completes 'sa sc So A is connected to ' (greedy text; and its pick among the context words, no name rule)
      S3 evidence check: gap(w) = lp(w | sa sc Q) - lp(w | sa[B->decoy] sc Q); AUC of gap(C) vs gap(other context words)
      S4 handover (inf_self='hand', from sleep inf_self_hand on): the engine only pairs the sentences; the conclusion is the MODEL'S pick,
         kept only if its gap > inf_self_m nats (else abstain)"""
    mode_=str(cfg.inf_self); vis_=int(model.__dict__.get('_visit',0))+1; hand_=(mode_=='hand' and vis_>=int(getattr(cfg,'inf_self_hand',3)))
    rs_=np.random.RandomState(int(cfg.seed)+31*vis_); m_=float(getattr(cfg,'inf_self_m',0.5)); st_={'n':0,'gen':0,'pick':0,'kept':0,'abst':0,'ktrue':0,'kwrong':0}
    nm_=list(S_['df'].keys()); aucs_=[]; gt_=[];go_=[]; out_=[]
    idx_=list(range(len(new_))); rs_.shuffle(idx_); chk_=set(idx_[:int(getattr(cfg,'inf_self_n',48))])
    for i_,c_ in enumerate(new_):
        if i_ not in chk_:
            if not hand_: out_.append(c_)
            continue
        A,B,C=c_['A'],c_['B'],c_['C']; sa_=S_['it'][c_['src'][0]]['s'].decode('latin-1').strip(); sc_=S_['it'][c_['src'][1]]['s'].decode('latin-1').strip()
        if A not in sa_: sa_,sc_=sc_,sa_
        ctx_=f"{sa_} {sc_} So {A} is connected to "; sl_=int(S_slot(c_['d']))
        dec_=nm_[rs_.randint(len(nm_))].decode('latin-1') if nm_ else 'Someone'; ctxd_=f"{sa_.replace(B,dec_)} {sc_} So {A} is connected to "
        cw_=[w_ for w_ in _inf_words(sa_+' '+sc_) if w_!=A]
        if C not in cw_ or len(cw_)<2:
            if not hand_: out_.append(c_)
            continue
        st_['n']+=1; g_=_inf_gen(model,cfg,ctx_,sl_,True); st_['gen']+=int(g_.startswith(C))
        l1_=_inf_lp(model,cfg,ctx_,cw_,sl_,True); l0_=_inf_lp(model,cfg,ctxd_,cw_,sl_,True); gap_=[a_-b_ for a_,b_ in zip(l1_,l0_)]
        pk_=cw_[int(np.argmax(l1_))]; st_['pick']+=int(pk_==C); ic_=cw_.index(C)
        oth_=[gap_[j_] for j_ in range(len(cw_)) if j_!=ic_]; gt_.append(gap_[ic_]); go_+=oth_
        aucs_.append(float(np.mean([gap_[ic_]>o_ for o_ in oth_]+[0.5*(gap_[ic_]==o_) for o_ in oth_]) if oth_ else 0.5))
        if not hand_: out_.append(c_); continue
        gp_=gap_[cw_.index(pk_)]
        if gp_<=m_ or pk_==A: st_['abst']+=1; continue
        c2_=dict(c_); c2_['C']=pk_; c2_['typed']=False; c2_['s']=f"So {A} is connected to {pk_}."; c2_['self']=True; out_.append(c2_); st_['kept']+=1
        tr_=(A,pk_) in t2_; st_['ktrue']+=int(tr_); st_['kwrong']+=int((not tr_) and ((A,C) in t2_))
    if hand_:                                                                   # the conclusions list in the store follows the model's choices
        S_['conc']=[c_ for c_ in S_['conc'] if c_ not in new_]+out_
    n0_=max(1,st_['n'])
    print(f"  [self-taught inference | {'HANDOVER: model completes + checks, engine only pairs' if hand_ else 'shadow'}] {st_['n']} conclusions tried | S2 proposal: greedy text = C {st_['gen']/n0_:.2f}, "
          f"pick among {np.mean([0]) if False else ''}context words = C {st_['pick']/n0_:.2f} | S3 evidence gap (real vs decoy bridge): true C {np.mean(gt_) if gt_ else float('nan'):+.2f} nats, other words "
          f"{np.mean(go_) if go_ else float('nan'):+.2f}, AUC {np.mean(aucs_) if aucs_ else float('nan'):.2f}"+(f" | S4: kept {st_['kept']} (probe-true {st_['ktrue']}, probe-wrong {st_['kwrong']}), abstained {st_['abst']} (margin {m_} nats)" if hand_ else ''),flush=True)
    return out_

_INF_FUNC={w_.encode() for w_ in 'and or of the a an von van der den de du des del della di da la le les el los las und y e et en in on at to for by from with as is was'.split()}
def _inf_facts(cfg,S_,nid_,vis0_):
    """EPISODIC FACT RECORDS (sleep-order test): this visit's salient one-hop facts 'A r B' (seen >= inf_rem_minrep times, rare names)"""
    F_=S_.setdefault('facts',[]); K_=S_.setdefault('fkey',{}); cnt_={}; first_={}; df_=S_['df']; mx_=int(cfg.inf_df); lw_=S_.get('lw') if S_.get('strict') else None
    ok_=lambda n_: df_.get(n_,0)<=mx_ and (lw_ is None or n_.lower() not in lw_)
    for i_ in nid_:
        for m_ in _INF_TRIP.finditer(S_['it'][i_]['s']):
            a_,r_,b_=m_.group(1),m_.group(2),m_.group(3)
            if a_==b_ or not ok_(a_) or not ok_(b_): continue
            if int(getattr(cfg,'inf_rem_stop',0)) and set(r_.split())<=_INF_FUNC: continue   # 'Konrad von Wordship', 'Chaloner and Carley': no relation word
            k_=(a_.decode('latin-1'),r_.decode('latin-1'),b_.decode('latin-1')); cnt_[k_]=cnt_.get(k_,0)+1; first_.setdefault(k_,i_)
    mr_=int(getattr(cfg,'inf_rem_minrep',3)); nw_=0; cap_=int(getattr(cfg,'inf_rem_fact_cap',0))
    ks_=[k_ for k_,c_ in cnt_.items() if c_>=mr_ and k_ not in K_]
    if cap_>0 and len(ks_)>cap_:                                                # PRIORITY = LENGTH / SPREAD: long, rarely shared names first
        ks_=sorted(ks_,key=lambda k_: -(len(k_[0])+len(k_[2]))/(df_.get(k_[0].encode('latin-1'),1)+df_.get(k_[2].encode('latin-1'),1)))[:cap_]
    ks_=set(ks_)
    for k_,c_ in cnt_.items():
        if c_<mr_: continue
        if k_ not in K_ and k_ not in ks_: continue
        if k_ in K_: K_[k_]['rep']+=c_; continue
        f_={'A':k_[0],'r':k_[1],'B':k_[2],'s':f"{k_[0]} {k_[1]} {k_[2]}.",'d':S_['it'][first_[k_]]['d'],'i':first_[k_],'v':vis0_,'rep':c_,'n':0,'fact':True}
        F_.append(f_); K_[k_]=f_; S_.setdefault('fpair',{}).setdefault((k_[0],k_[2]),f_); nw_+=1
    return nw_
def _inf_rules(cfg,S_,vis0_):
    """RULE LAYER (rules that pay for themselves): from the recorded fact records, propose 'phrase p => phrase q' for every pair of relation
    phrases sharing subjects; score each by the bits it saves on held-out evidence (leave-one-out two-part code); keep a fixed-size pool of
    the best; APPLY only confident rules that reach a fraction of the going rate (median gain of rules that save bits): derived facts
    'A q B' are added to the fact records, so NREM replay writes them into the fast store. Derived facts die with their source."""
    F_=S_.get('facts') or []; K_=S_.setdefault('fkey',{})
    for f_ in F_:                                                               # a derived fact whose source was freed (superseded) is freed too
        if f_.get('derived') and not f_.get('freed') and K_.get(f_['srcf'],{}).get('freed'): f_['freed']=True
    src_=[f_ for f_ in F_ if not f_.get('freed') and not f_.get('derived')]
    by_={}
    for f_ in src_: by_.setdefault(f_['r'],{})[f_['A']]=f_['B']
    objs_=set(f_['B'] for f_ in src_); N_=max(2,len(objs_)); R_=[r_ for r_ in by_ if len(by_[r_])>=2]; cst_=math.log2(max(2,len(R_)*(len(R_)-1)))
    mn_=int(getattr(cfg,'inf_rule_min',3)); cand_=[]
    for p_ in R_:
        for q_ in R_:
            if p_==q_: continue
            sh_=[a_ for a_ in by_[p_] if a_ in by_[q_]]
            if len(sh_)<mn_: continue
            b0_=b1_=0.0; h_=0
            for a_ in sh_:
                oth_=[x_ for x_ in sh_ if x_!=a_]; hh_=sum(by_[p_][x_]==by_[q_][x_] for x_ in oth_); pi_=(hh_+1)/(len(oth_)+2)
                hit_=by_[p_][a_]==by_[q_][a_]; h_+=hit_; b0_+=math.log2(N_); b1_+=-math.log2(pi_) if hit_ else (-math.log2(1-pi_)+math.log2(N_))
            cand_.append({'p':p_,'q':q_,'n':len(sh_),'h':h_,'gain':b0_-b1_-cst_,'conf':(h_+1)/(len(sh_)+2)})
    pos_=[c_['gain'] for c_ in cand_ if c_['gain']>0]; md_=float(np.median(pos_)) if pos_ else 0.0
    cap_=int(getattr(cfg,'inf_rules_cap',24)); pool_=sorted([c_ for c_ in cand_ if c_['gain']>0],key=lambda c_: -c_['gain'])[:cap_]
    use_=[c_ for c_ in pool_ if c_['conf']>=float(getattr(cfg,'inf_rule_conf',0.85)) and c_['gain']>=float(getattr(cfg,'inf_rule_rel',0.5))*md_]
    PP_={_inf_phrase(t_):_INF_PARA[_inf_phrase(t_)] for t_ in _INF_REL}; true_={(k_,v_) for k_,v_ in PP_.items()}|{(v_,k_) for k_,v_ in PP_.items()}
    nd_=0
    for c_ in use_:
        for a_,b_ in by_[c_['p']].items():
            if a_ in by_.get(c_['q'],{}): continue                              # the subject already has a q-fact: never overwrite evidence
            k_=(a_,c_['q'],b_)
            if k_ in K_: continue
            sf_=next((f_ for f_ in src_ if f_['A']==a_ and f_['r']==c_['p'] and f_['B']==b_),None)
            if sf_ is None: continue
            f_={'A':a_,'r':c_['q'],'B':b_,'s':f"{a_} {c_['q']} {b_}.",'d':sf_['d'],'i':sf_['i'],'v':vis0_,'rep':1,'n':0,'fact':True,'derived':True,'srcf':(a_,c_['p'],b_)}
            F_.append(f_); K_[k_]=f_; nd_+=1
    S_['rules']=use_
    print(f"  [inference RULE LAYER] {len(cand_)} candidate rules between {len(R_)} relation phrases; {len(pos_)} save bits (going rate {md_:.1f} bits); pool {len(pool_)}/{cap_}; "
          f"APPLIED {len(use_)} (true paraphrase pairs {sum(1 for c_ in use_ if (c_['p'],c_['q']) in true_)}, other {sum(1 for c_ in use_ if (c_['p'],c_['q']) not in true_)}); {nd_} derived facts added"
          +(" | e.g. "+"; ".join(f"'{c_['p']}' => '{c_['q']}' {c_['gain']:+.0f} bits conf {c_['conf']:.2f}" for c_ in use_[:4]) if use_ else ''),flush=True)
def _inf_nm1(g_):
    w_=(g_.split() or [''])[0].strip('.,;:!?')
    return w_ if _INF_NAME.fullmatch(w_.encode('latin-1','ignore')) else ''
@torch.no_grad()
def _inf_rem(model,cfg,S_,S_slot,n_,tag_,shadow_=False):
    """REM-LIKE RECOMBINATION (sleep-order test): links proposed from MEMORY (weights + fast store), not from co-occurrence.
      subject: a recorded fact 'A r0 B' (half from this visit, half older); lookup 1: complete 'A r0 ' -> bridge B^;
      lookup 2: for each common relation r1, complete 'B^ r1 ' -> C^; APPRAISAL (knowing it knows): log p(C^ | 'B^ r1 ') - log p(C^ | 'Someone r1 ')
      must exceed inf_rem_m nats (the completion belongs to the bridge, not just to the relation); GROUNDING: kept as a conclusion only if both links are recorded facts, else a hypothesis (not written)"""
    F_=S_.get('facts') or []
    if not F_ or n_<=0: return []
    t2_,_=_inf_truth(cfg); pn_=set(x_ for q_ in inf_probes(cfg) for x_ in q_['names']); vis0_=int(model.__dict__.get('_visit',0))
    rs_=np.random.RandomState(int(cfg.seed)+97*vis0_+len(S_.setdefault('rem_log',[]))); m_=float(getattr(cfg,'inf_rem_m',2.0))
    rc_={}
    for f_ in F_: rc_[f_['r']]=rc_.get(f_['r'],0)+f_['rep']
    rv_=[r_ for r_,_ in sorted(rc_.items(),key=lambda x_:-x_[1])[:int(getattr(cfg,'inf_rem_rels',8))]]
    pool_=sorted(set(x_ for f_ in F_ for x_ in (f_['A'],f_['B'])))
    rec_=[f_ for f_ in F_ if f_['v']==vis0_]; old_=[f_ for f_ in F_ if f_['v']!=vis0_]
    pf_=float(getattr(cfg,'inf_rem_probe_frac',0.0)); sel_=[]
    if pf_>0:                                                                   # MEASUREMENT SAMPLE: a share of subjects from probe facts (scorable), labelled in the report
        pr_=[f_ for f_ in F_ if f_['A'] in pn_]; sel_=[pr_[i_] for i_ in rs_.permutation(len(pr_))[:int(round(n_*pf_))]]
    m0_=n_-len(sel_); h_=m0_//2 if old_ else m0_; sel_+=[rec_[i_] for i_ in rs_.permutation(len(rec_))[:h_] if rec_[i_] not in sel_]; sel_+=[old_[i_] for i_ in rs_.permutation(len(old_))[:n_-len(sel_)] if old_[i_] not in sel_]
    st_={k_:0 for k_ in ('n','bok','nob','prop','abst','known','acc','at','aw','hyp','ht','hw')}; out_=[]
    for f_ in sel_:
        A_,r0_=f_['A'],f_['r']; d_=int(S_slot(f_['d'])); st_['n']+=1
        Bh_=_inf_nm1(_inf_gen(model,cfg,f"{A_} {r0_} ",d_,True)); st_['bok']+=int(Bh_==f_['B'])
        if not Bh_ or Bh_==A_: st_['nob']+=1; continue
        best_=None; ex_=[]; dB_=sorted(set(int(S_slot(g_['d'])) for g_ in F_ if g_['A']==Bh_)) or [d_]   # lookup 2 in the domain(s) where B^ was met (fix: was A's domain)
        for r1_,dd_ in [(r_,x_) for r_ in rv_ for x_ in dB_]:
            Ch_=_inf_nm1(_inf_gen(model,cfg,f"{Bh_} {r1_} ",dd_,True)); ex_.append(f"{r1_}->{Ch_ or '-'}")
            if not Ch_ or Ch_ in (A_,Bh_): continue
            mg_=_inf_lp(model,cfg,f"{Bh_} {r1_} ",[Ch_],dd_,True)[0]-_inf_lp(model,cfg,f"Someone {r1_} ",[Ch_],dd_,True)[0]   # specific to the bridge: not just the relation's usual object
            if int(getattr(cfg,'inf_rem_all',0)):                              # ALL CANDIDATES: the best GROUNDED proposal above the margin wins
                fp0_=S_.get('fpair',{}); gr_=(fp0_.get((Bh_,Ch_)) or fp0_.get((Ch_,Bh_))) is not None and (fp0_.get((A_,Bh_)) or fp0_.get((Bh_,A_))) is not None
                if gr_ and mg_>m_ and (best_ is None or not best_[3] or mg_>best_[0]): best_=(mg_,r1_,Ch_,True); continue
                if best_ is not None and best_[3]: continue
            if best_ is None or mg_>best_[0]: best_=(mg_,r1_,Ch_,False)
        if best_ is None: continue
        best_=best_[:3]
        st_['prop']+=1; mg_,r1_,C_=best_
        if len(st_.setdefault('eg',[]))<4: tc_=[q_['names'][2] for q_ in inf_probes(cfg) if q_['names'][0]==A_]; st_['eg'].append(f"{A_} {r0_} {Bh_} (true {f_['B']}) | {', '.join(ex_)} | chose {r1_}->{C_} margin {mg_:.1f} (true C {tc_[0] if tc_ else '?'})")
        if mg_<=m_: st_['abst']+=1; continue
        key_=(min(A_,C_),max(A_,C_)); tr_=(A_,C_) in t2_; pw_=(A_ in pn_ or C_ in pn_) and not tr_
        if key_ in S_['pairs'] or key_ in S_.setdefault('rem_seen',set()): st_['known']+=1; continue
        fp_=S_.get('fpair',{}); fa_=fp_.get((A_,Bh_)) or fp_.get((Bh_,A_)); fb_=fp_.get((Bh_,C_)) or fp_.get((C_,Bh_))   # grounded: recorded episodes link A-B^ and B^-C^
        if fa_ is not None and fb_ is not None and shadow_:                    # SHADOW: proposals are only counted (nothing written; discovery is unchanged)
            S_['rem_seen'].add(key_); st_['acc']+=1; st_['at']+=int(tr_); st_['aw']+=int(pw_)
        elif fa_ is not None and fb_ is not None:
            S_['pairs'].add(key_); r1_=fb_['r'] if fb_['A']==Bh_ else r1_; c_={'A':A_,'B':Bh_,'C':C_,'r':(r0_,r1_),'typed':True,'s':f"{A_} {r0_} someone who {r1_} {C_}.",'src':(fa_['i'],fb_['i']),
                                    'd':f_['d'],'dep':1,'v':vis0_,'rem':True}
            S_['conc'].append(c_); out_.append(c_); st_['acc']+=1; st_['at']+=int(tr_); st_['aw']+=int(pw_)
        else: st_['hyp']+=1; st_['ht']+=int(tr_); st_['hw']+=int(pw_)
    eg_=st_.pop('eg',[]); S_['rem_log'].append(st_); n0_=max(1,st_['n'])
    for e_ in eg_[:3]: print(f"    [inference REM e.g.] {e_}",flush=True)
    f2_=len(set(frozenset((c_['A'],c_['C'])) for c_ in S_['conc'] if (c_['A'],c_['C']) in t2_))
    print(f"  [inference REM recombination | {tag_}] {st_['n']} subjects{f' ({pf_:.0%} sampled from probe facts for scoring)' if pf_>0 else ''} ({len(rec_)} facts from this visit, {len(old_)} older; {len(rv_)} relations) | bridge recalled {st_['bok']/n0_:.2f} (no name {st_['nob']}) | "
          f"{st_['prop']} proposals, {st_['abst']} abstained (margin {m_} nats), {st_['known']} already known | ACCEPTED grounded {st_['acc']} (probe-true {st_['at']}, probe-wrong {st_['aw']}) | "
          f"hypotheses (ungrounded, not written) {st_['hyp']} (probe-true {st_['ht']}, probe-wrong {st_['hw']}) | facts recorded {len(F_)} | probe chains found so far: two-hop {f2_}/{len(t2_)//2}",flush=True)
    return out_
@torch.no_grad()
def _inf_wm_write(model,cfg,ctx_,sl_,iters_=20,beta_=0.5):
    """WORKING MEMORY: the facts in a prompt ('Name relation Name.') are written into a scratch store with the same name-keyed delta rule as
    sleep writes (to convergence, on top of whatever the long-term store already predicts); returns the names added for the question"""
    import zlib as _zl
    fs_=[m_.group(0).decode('latin-1') for m_ in _INF_TRIP_RX.finditer(ctx_.encode('latin-1'))]   # question-time readout: prompt facts parsed by the template pattern
    if not fs_: return []
    dev_=model.embed.weight.device
    if _INFM.get('Mw') is None or _INFM['Mw'].shape!=_INFM['M'].shape: _INFM['Mw']=torch.zeros_like(_INFM['M'])
    add_=[]
    for f_ in fs_:
        for n_ in _INF_NAME_RX.findall(f_.encode('latin-1')):
            if n_ not in _INFM['names']: _INFM['names'][n_]=int(_zl.crc32(n_)); add_.append(n_)
    rows_=[(' '+f_+'.').encode('latin-1') for f_ in fs_]; Lr_=max(len(r_) for r_ in rows_)+2
    X_=torch.full((len(rows_),Lr_),32,dtype=torch.long,device=dev_); mk_=torch.zeros(len(rows_),Lr_-1,device=dev_)
    for j_,(r_,f_) in enumerate(zip(rows_,fs_)):
        X_[j_,:len(r_)]=torch.tensor(list(r_),device=dev_); a0_=1+len(f_.split(' ')[0]); mk_[j_,a0_:len(r_)-1]=1.0   # targets: everything after the subject
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl_; sh_=_HCAP['on']; _HCAP['on']=True; sv_=_INFM['on']; _INFM['on']=False; fz_=_INFM.pop('force',None); was_=model.training; model.eval()
    try: lg_=model(X_[:,:-1],sl_).float(); z_=_HCAP['h'].float()
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; _HCAP['on']=sh_; _INFM['on']=sv_; model.train(was_)
        if fz_ is not None: _INFM['force']=fz_
    K_,id_=_infm_keys(z_,X_[:,:-1],lg_); zf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256)
    oh_=F.one_hot(X_[:,1:].reshape(-1),256).float(); mf_=(mk_.reshape(-1)*(idf_>=0).float())[:,None]
    Ms_=_INFM['M']
    for _ in range(int(iters_)):
        q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1); R_=(oh_-q_)*mf_; sel_=((idf_>=0)&(R_.abs().sum(-1)>0)).nonzero().squeeze(-1)
        for c_ in range(0,sel_.numel(),512):
            s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_]); _INFM['Mw'].index_add_(0,i_.reshape(-1),(beta_*v_[...,None]*R_[s_][:,None,:]).reshape(-1,256))
    return add_
def _inf_wm_clear(add_):
    if _INFM.get('Mw') is not None: _INFM['Mw'].zero_()
    for n_ in add_: _INFM['names'].pop(n_,None)
@torch.no_grad()
def inf_qtime2(model,cfg,tag=''):
    """QUESTION-TIME CHAINING WITH A WORKING MEMORY. Prompts 'A r0 B. B r1 C. X r1 Y.' (both orders) and the broken-bridge decoy (A r0 D);
    PROBE chains (also in long-term memory) and NEW chains (invented names that appear only in the prompt). Readers:
      model       the weights alone (in-context copying)          long-term   + the sleep-time fast store
      WM          + a working memory written from the prompt only (long-term store off)      WM+LT  both
    Each: DIRECT 'So A is connected to ' and TWO LOOKUPS ('A r0 ' -> B^, then 'B^ r ' for each relation in the prompt). Decoys: answering
    C is a FALSE chain. 16 candidates (chance 0.06)."""
    if _INFM.get('M') is None or _INFM.get('mu') is None: return
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')][:int(getattr(cfg,'inf_qt_n',60))]
    alln_=[n_ for c_ in inf_probes(cfg) for n_ in c_['names']]; t2s_=model.__dict__.get('_t2s',{}); rs_=np.random.RandomState(99)
    used_=set(alln_)|set(n_.decode('latin-1') for n_ in _INFM['names'])
    def fresh():
        while True:
            n_=''.join(rs_.choice(['qa','xo','zu','vi','ke','lom','dar','fen','sul','gor','pim','tas'],rs_.randint(2,4))); n_=n_[0].upper()+n_[1:]
            if 5<=len(n_)<=10 and n_ not in used_: used_.add(n_); return n_
    items_=[]
    for src_ in ('probe','NEW'):
        for i_,c_ in enumerate(ch_):
            r0_,r1_=c_['rels']; sl_=int(t2s_.get(c_['doms'][0],c_['doms'][0]))
            if src_=='probe': A_,B_,C_=c_['names']; oth_=[x_ for x_ in alln_ if x_ not in c_['names']]; X_,Y_,D_=oth_[(3*i_)%len(oth_)],oth_[(3*i_+1)%len(oth_)],oth_[(3*i_+2)%len(oth_)]; ex_=list(rs_.choice([x_ for x_ in oth_ if x_ not in (X_,Y_,D_)],12,replace=False))
            else: A_,B_,C_,X_,Y_,D_=[fresh() for _ in range(6)]; ex_=[fresh() for _ in range(12)]
            f0_,f1_=c_['tp'][0].format(a=A_,b=B_)[:-1],c_['tp'][1].format(a=B_,b=C_)[:-1]; fx_=f"{X_} {r1_} {Y_}"
            for kind_ in ('AB','BA','decoy AB','decoy BA'):
                dec_='decoy' in kind_; g0_=f0_.replace(B_,D_) if dec_ else f0_
                ctx_=(f"{g0_}. {f1_}. {fx_}. " if kind_.endswith('AB') else f"{f1_}. {g0_}. {fx_}. ")
                items_.append(dict(src=src_,kind=kind_,dec=dec_,ctx=ctx_,A=A_,B=(D_ if dec_ else B_),C=C_,r=sorted({r0_,r1_}),r0=r0_,cand=[B_,X_,Y_,C_]+([D_] if dec_ else [])+ex_[:12-int(dec_)],sl=sl_,oth=(B_,X_,Y_,D_)))
    res_={}
    for mode_ in ('model','long-term','WM','WM+LT'):
        for it_ in items_:
            add_=[]; st0_=mode_!='model'
            if mode_=='WM': _INFM['lt_off']=True
            if mode_.startswith('WM'): _INFM['wm_on']=True; add_=_inf_wm_write(model,cfg,it_['ctx'],it_['sl'])
            try:
                cand_=it_['cand']; lp_=_inf_lp(model,cfg,it_['ctx']+f"So {it_['A']} is connected to ",cand_,it_['sl'],st0_); pk_=cand_[int(np.argmax(lp_))]
                lb_=_inf_lp(model,cfg,it_['ctx']+f"{it_['A']} {it_['r0']} ",cand_,it_['sl'],st0_); Bh_=cand_[int(np.argmax(lb_))]; best_=None
                for r_ in it_['r']:
                    cc_=[x_ for x_ in cand_ if x_!=Bh_]; lc_=_inf_lp(model,cfg,it_['ctx']+f"{Bh_} {r_} ",cc_,it_['sl'],st0_); j_=int(np.argmax(lc_))
                    if best_ is None or lc_[j_]>best_[0]: best_=(lc_[j_],cc_[j_])
            finally: _INFM.pop('lt_off',None); _INFM.pop('wm_on',None); _inf_wm_clear(add_)
            k_=(it_['src'],'decoy' if it_['dec'] else 'real',mode_); s_=res_.setdefault(k_,[0,0,0,0,0])
            s_[0]+=1; s_[1]+=int(pk_==it_['C']); s_[2]+=int(Bh_==it_['B']); s_[3]+=int(best_[1]==it_['C']); s_[4]+=int(pk_ in it_['oth'])
    for src_ in ('probe','NEW'):
        out_=[]
        for mode_ in ('model','long-term','WM','WM+LT'):
            r_=res_.get((src_,'real',mode_)); d_=res_.get((src_,'decoy',mode_))
            if not r_: continue
            n_,nd_=max(1,r_[0]),max(1,d_[0])
            out_.append(f"{mode_}: direct {r_[1]/n_:.2f} (copy {r_[4]/n_:.2f}), two lookups bridge {r_[2]/n_:.2f} answer {r_[3]/n_:.2f} | decoy FALSE chain direct {d_[1]/nd_:.2f}, two lookups {d_[3]/nd_:.2f}")
        print(f"  [inference question-time chaining WITH WORKING MEMORY | {tag} | {src_} chains{' (in long-term memory too)' if src_=='probe' else ' (names only in the prompt)'}, {len(ch_)} x 2 orders] "+" || ".join(out_),flush=True)
_INF_PARA={'works with':'is a colleague of','lives near':'resides close to','reports to':'answers to','is a partner of':'partners with','trained under':'was taught by','is linked to':'is connected with'}
@torch.no_grad()
def inf_qtime3(model,cfg,tag='',D_=None):
    """MULTI-HOP QUESTION-TIME CHAINING, depth 2-5, on NEW chains (names only in the prompt): facts N0 r0 N1. N1 r1 N2. ... plus one distractor
    fact per hop (other names, same relation), shuffled; the question supplies the relation path ('N0's r0's r1 ...').
      TEXT  : working memory written from the prompt; each hop asks the store 'cur r ' and picks a name (bytes), then re-reads it as the next key
      ID    : an IDENTIFIER memory: key = code(subject) * code(relation), value = code(object) (fixed random codes from hashes, never trained),
              summed outer products, read + clean-up to the nearest known name code; the returned IDENTIFIER is the next key (no text round trip).
              Background load: every recorded long-term fact is stored in the same memory (interference).
    Decoys: one middle link broken (its object replaced by an outsider) -> a correct reader ABSTAINS. Abstention: TEXT if the best hop margin over the
    runner-up < inf_qt_m nats; ID if the clean-up similarity < inf_qt_sim. Paraphrase: the question uses synonyms of the relations."""
    if _INFM.get('M') is None or _INFM.get('mu') is None: return
    import zlib as _zl
    D_=int(D_ or getattr(cfg,'inf_id_dim',1024))
    rs_=np.random.RandomState(1234); used_=set(n_.decode('latin-1') for n_ in _INFM['names'])|set(x_ for c_ in inf_probes(cfg) for x_ in c_['names'])
    def fresh():
        while True:
            n_=''.join(rs_.choice(['qa','xo','zu','vi','ke','lom','dar','fen','sul','gor','pim','tas','bru','nix'],rs_.randint(2,5))); n_=n_[0].upper()+n_[1:]
            if 5<=len(n_)<=10 and n_ not in used_: used_.add(n_); return n_
    def code(s_):
        g_=np.random.RandomState(_zl.crc32(s_.encode('latin-1'))%(2**31)); return torch.tensor(g_.choice([-1.0,1.0],D_),dtype=torch.float32)
    rels_=[_inf_phrase(t_) for t_ in _INF_REL]; tpl_={_inf_phrase(t_):t_ for t_ in _INF_REL}
    F0_=(_inf_store(model).get('facts') or [])[:3000]; m_=float(getattr(cfg,'inf_qt_m',1.0)); sim_=float(getattr(cfg,'inf_qt_sim',0.5))
    Mbg_=torch.zeros(D_,D_); bg_=[(f_['A'],f_['r'],f_['B']) for f_ in F0_]; nb_=int(getattr(cfg,'inf_qt_bg',2000)); nl_=[n_.decode('latin-1') for n_ in _INFM['names']]
    while len(bg_)<nb_ and len(nl_)>=2: a0_,b0_=rs_.choice(nl_,2,replace=False); bg_.append((a0_,rels_[rs_.randint(len(rels_))],b0_))   # background load: stored-name facts
    for a0_,r0_,b0_ in bg_: Mbg_+=torch.outer(code(a0_)*code(r0_),code(b0_))
    F0_=bg_
    sl_=int(model.__dict__.get('_t2s',{}).get(0,0)); out_=[]; nc_=int(getattr(cfg,'inf_qt_comp',0))
    for depth_ in (2,3,4,5):
        st_={k_:0 for k_ in ('n','t_ok','t_ans','i_ok','i_ans','p_ok','p_ans','ip_ok','ip_ans','dn','t_dab','i_dab','t_dfalse','i_dfalse')}
        for trial_ in range(int(getattr(cfg,'inf_qt_n',40))//2):
            for dec_ in (False,True):
                N_=[fresh() for _ in range(depth_+1)]; R_=[rels_[rs_.randint(len(rels_))] for _ in range(depth_)]
                facts_=[(N_[i_],R_[i_],N_[i_+1]) for i_ in range(depth_)]; brk_=None
                if dec_: brk_=1+rs_.randint(depth_-1) if depth_>1 else 0; facts_[brk_]=(fresh(),R_[brk_],N_[brk_+1])   # the chain breaks: N_brk has no R_brk fact
                facts_+=[(fresh(),R_[i_],fresh()) for i_ in range(depth_)]
                for i_ in range(depth_):   # COMPETING RELATIONS: the chain node also has facts under OTHER relations -> the reader must use the relation
                    oth_=[r_ for r_ in rels_ if r_!=R_[i_]]; rs_.shuffle(oth_)
                    facts_+=[(N_[i_],r_,fresh()) for r_ in oth_[:nc_]]
                rs_.shuffle(facts_)
                ctx_=' '.join(tpl_[r_].format(a=a_,b=b_) for a_,r_,b_ in facts_)+' '
                names_=sorted(set(x_ for a_,_,b_ in facts_ for x_ in (a_,b_))); cand_=names_+[fresh() for _ in range(4)]
                # TEXT: working memory, hop by hop
                _INFM['lt_off']=True; _INFM['wm_on']=True; add_=_inf_wm_write(model,cfg,ctx_,sl_)
                try:
                    res_={}
                    rmap_={c0_['p']:c0_['q'] for c0_ in (_inf_store(model).get('rules') or []) if c0_['q'] in rels_}   # LEARNED RULES rewrite a paraphrased relation to one the memory holds
                    for mode_,RR_ in (('t',R_),('p',[rmap_.get(_INF_PARA[r_],_INF_PARA[r_]) for r_ in R_])):
                        cur_=N_[0]; ok_=True; ab_=False
                        for r_ in RR_:
                            cc_=[x_ for x_ in cand_ if x_!=cur_]; lp_=np.array(_inf_lp(model,cfg,f"{cur_} {r_} ",cc_,sl_,True)); o_=np.argsort(-lp_)
                            if lp_[o_[0]]-lp_[o_[1]]<m_: ab_=True; break
                            cur_=cc_[o_[0]]
                        res_[mode_]=(ab_,cur_)
                finally: _INFM.pop('lt_off',None); _INFM.pop('wm_on',None); _inf_wm_clear(add_)
                # ID: identifier memory (prompt facts + long-term background), identifiers returned and reused
                M_=Mbg_.clone()
                for a_,r_,b_ in facts_: M_+=torch.outer(code(a_)*code(r_),code(b_))
                C_=torch.stack([code(x_) for x_ in cand_]); ires_={}
                for mode_,RR_ in (('i',R_),('ip',[_INF_PARA[r_] for r_ in R_])):   # ip: paraphrased relation identifiers (no canonicalisation)
                    cur_=N_[0]; iab_=False
                    for r_ in RR_:
                        v_=(code(cur_)*code(r_))@M_; s_=F.cosine_similarity(v_[None,:],C_); j_=int(s_.argmax())
                        if float(s_[j_])<sim_: iab_=True; break
                        cur_=cand_[j_]
                    ires_[mode_]=(iab_,cur_)
                iab_,cur_=ires_['i']
                st_['ip_ans']+=int(not dec_ and not ires_['ip'][0]); st_['ip_ok']+=int(not dec_ and (not ires_['ip'][0]) and ires_['ip'][1]==N_[-1])
                if not dec_:
                    st_['n']+=1
                    st_['t_ans']+=int(not res_['t'][0]); st_['t_ok']+=int((not res_['t'][0]) and res_['t'][1]==N_[-1])
                    st_['p_ans']+=int(not res_['p'][0]); st_['p_ok']+=int((not res_['p'][0]) and res_['p'][1]==N_[-1])
                    st_['i_ans']+=int(not iab_); st_['i_ok']+=int((not iab_) and cur_==N_[-1])
                else:
                    st_['dn']+=1; st_['t_dab']+=int(res_['t'][0]); st_['i_dab']+=int(iab_)
                    st_['t_dfalse']+=int((not res_['t'][0]) and res_['t'][1]==N_[-1]); st_['i_dfalse']+=int((not iab_) and cur_==N_[-1])
        n_,d_=max(1,st_['n']),max(1,st_['dn']); acc_=lambda ok,an: f"{ok/n_:.2f} correct, answers {an/n_:.2f}, accuracy when answering {ok/max(1,an):.2f}"
        out_.append(f"DEPTH {depth_}: TEXT {acc_(st_['t_ok'],st_['t_ans'])}; PARAPHRASED question {acc_(st_['p_ok'],st_['p_ans'])}; ID {acc_(st_['i_ok'],st_['i_ans'])}; ID PARAPHRASED {acc_(st_['ip_ok'],st_['ip_ans'])} | "
                    f"decoys: abstained TEXT {st_['t_dab']/d_:.2f} / ID {st_['i_dab']/d_:.2f}, FALSE chain TEXT {st_['t_dfalse']/d_:.2f} / ID {st_['i_dfalse']/d_:.2f}")
    print(f"  [inference multi-hop question-time | {tag} | new chains only in the prompt, one distractor per hop, {nc_} competing-relation facts per chain node, {len(F0_)} long-term facts as background load, abstain margin {m_} nats / similarity {sim_}] "+" || ".join(out_),flush=True)
@torch.no_grad()
def inf_qtime(model,cfg,tag='',store_=False):
    """QUESTION-TIME CHAINING BY TWO LOOKUPS on the CTX2 prompts (facts 'A r0 B. B r1 C. X r1 Y.' in both orders, plus the broken-bridge decoy
    'A r0 D. B r1 C. X r1 Y.'). DIRECT: context + 'So A is connected to ' -> pick. TWO LOOKUPS (copying, twice): (1) context + 'A r0 ' -> the
    bridge B^ among the candidate names; (2) context + 'B^ r ' for each relation in the context -> the best answer. Decoy: a correct reader
    has NO chain to C, so answering C there is a FALSE chain. Candidates: every in-context name + 12 other probe names."""
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')][:int(getattr(cfg,'inf_splice_n',150))]
    alln_=[n_ for c_ in inf_probes(cfg) for n_ in c_['names']]; t2s_=model.__dict__.get('_t2s',{}); rs_=np.random.RandomState(99); out_=[]
    for kind_ in ('AB','BA','decoy AB','decoy BA'):
        st_={'n':0,'dir':0,'cp':0,'b':0,'c':0}; dec_='decoy' in kind_
        for i_,c_ in enumerate(ch_):
            A_,B_,C_=c_['names']; r0_,r1_=c_['rels']; fs_=[f_ for _,f_ in c_['facts']]; oth_=[x_ for x_ in alln_ if x_ not in c_['names']]
            X_,Y_,D_=oth_[(3*i_)%len(oth_)],oth_[(3*i_+1)%len(oth_)],oth_[(3*i_+2)%len(oth_)]
            f0_=fs_[0].replace(B_,D_) if dec_ else fs_[0]; fx_=f"{X_} {r1_} {Y_}."
            ctx_=(f"{f0_} {fs_[1]} {fx_} " if kind_.endswith('AB') else f"{fs_[1]} {f0_} {fx_} ")
            ic_=[B_,X_,Y_,C_]+([D_] if dec_ else []); cand_=ic_+list(rs_.choice([x_ for x_ in oth_ if x_ not in ic_],12,replace=False)); sl_=int(t2s_.get(c_['doms'][0],c_['doms'][0]))
            lp_=_inf_lp(model,cfg,ctx_+f"So {A_} is connected to ",cand_,sl_,store_); pk_=cand_[int(np.argmax(lp_))]
            st_['n']+=1; st_['dir']+=int(pk_==C_); st_['cp']+=int(pk_ in (B_,X_,Y_,D_))
            lb_=_inf_lp(model,cfg,ctx_+f"{A_} {r0_} ",cand_,sl_,store_); Bh_=cand_[int(np.argmax(lb_))]; st_['b']+=int(Bh_==(D_ if dec_ else B_))
            best_=None
            for r_ in sorted({r0_,r1_}):
                cc_=[x_ for x_ in cand_ if x_!=Bh_]; lc_=_inf_lp(model,cfg,ctx_+f"{Bh_} {r_} ",cc_,sl_,store_); j_=int(np.argmax(lc_))
                if best_ is None or lc_[j_]>best_[0]: best_=(lc_[j_],cc_[j_])
            st_['c']+=int(best_[1]==C_)
        n0_=max(1,st_['n'])
        if dec_: out_.append(f"DECOY {kind_[-2:]}: FALSE chain to C - direct {st_['dir']/n0_:.2f}, two lookups {st_['c']/n0_:.2f} (bridge read as the decoy {st_['b']/n0_:.2f})")
        else: out_.append(f"{kind_}: direct {st_['dir']/n0_:.2f} (copied another in-context name {st_['cp']/n0_:.2f}) | two lookups: bridge {st_['b']/n0_:.2f}, answer {st_['c']/n0_:.2f}")
    print(f"  [inference question-time chaining | {tag}{' | store on' if store_ else ''}] {len(ch_)} two-hop chains, facts in the context with a distractor fact last, 16 candidates (chance 0.06) || "+" || ".join(out_),flush=True)
def _inf_auc(p_,n_):
    p_=np.asarray(p_,float)[:,None]; n_=np.asarray(n_,float)[None,:]
    return float(((p_>n_).sum()+0.5*(p_==n_).sum())/max(1,p_.size*n_.size))
@torch.no_grad()
def inf_splice(model,cfg,tag='',nf_=3):
    """SPLICE-AND-APPRAISE (cg20o test): can the model tell a true link from a false recombination by 'knowing what it knows'?
    For each two-hop chain A-r0-B, B-r1-C the TRUE splice is (A, C); the FALSE splices put the C of other chains (rare names from other
    episodes) into the same common frame. Each candidate is appraised with the store OFF by:
      feels right      log p(X | 'So A is connected to ')                       (raw confidence: fluency of the splice)
      specific to A    the same minus log p(X | 'So someone is connected to ')  (does the link belong to A, or is X just familiar?)
      linked (own)     the weights recall A's bridge (greedy 'A r0 ' -> B^), then log p(X | 'B^ r1 ')   (self-retrieval: two one-hop lookups)
      linked, specific the same minus log p(X | 'Someone r1 ')
      linked (oracle)  log p(X | 'B r1 ') with the true bridge (upper bound if bridge recall were perfect)
      store-backed     'feels right' with the fast store on (memory, not self-appraisal)
    Reported: pooled AUC true vs false (usable as an accept threshold), paired accuracy (true beats each false splice of the same A),
    and the share of true splices accepted when at most 5% of false splices are accepted."""
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')][:int(getattr(cfg,'inf_splice_n',150))]
    if len(ch_)<nf_+2: return None
    rs_=np.random.RandomState(4242); t2s_=model.__dict__.get('_t2s',{}); sl_=lambda d_: int(t2s_.get(d_,d_))
    st_=_INFM.get('M') is not None and _INFM.get('mu') is not None
    K_=['feels right','specific to A','linked (own bridge)','linked, specific','linked (oracle bridge)']+(['store-backed','linked (store lookups)'] if st_ else [])
    V_={k_:[] for k_ in K_}; br_=[]; brs_=[]; dg_=[0,0]
    for i_,c_ in enumerate(ch_):
        A_,B_,C_=c_['names']; r0_,r1_=c_['rels']; d0_,d1_=sl_(c_['doms'][0]),sl_(c_['doms'][1])
        cs_=[C_]+[ch_[j_]['names'][2] for j_ in rs_.choice([j_ for j_ in range(len(ch_)) if j_!=i_],nf_,replace=False)]
        f_=np.array(_inf_lp(model,cfg,f"So {A_} is connected to ",cs_,d0_,False)); b_=np.array(_inf_lp(model,cfg,"So someone is connected to ",cs_,d0_,False))
        g_=_inf_gen(model,cfg,f"{A_} {r0_} ",d0_,False); Bh_=(g_.split() or [''])[0].strip('.,;:'); br_.append(int(Bh_==B_))
        l_=np.array(_inf_lp(model,cfg,f"{Bh_ or 'Someone'} {r1_} ",cs_,d1_,False)); lb_=np.array(_inf_lp(model,cfg,f"Someone {r1_} ",cs_,d1_,False))
        o_=np.array(_inf_lp(model,cfg,f"{B_} {r1_} ",cs_,d1_,False))
        for k_,v_ in zip(K_,[f_,f_-b_,l_,l_-lb_,o_]): V_[k_].append(v_)
        if st_:
            _INFM['dbg']=[0,0,0.0]; V_['store-backed'].append(np.array(_inf_lp(model,cfg,f"So {A_} is connected to ",cs_,d0_,True))); dg_[0]+=_INFM['dbg'][1]; dg_[1]+=_INFM['dbg'][0]
            gs_=_inf_gen(model,cfg,f"{A_} {r0_} ",d0_,True); Bs_=(gs_.split() or [''])[0].strip('.,;:'); brs_.append(int(Bs_==B_))
            V_['linked (store lookups)'].append(np.array(_inf_lp(model,cfg,f"{Bs_ or 'Someone'} {r1_} ",cs_,d1_,True)))
    out_=[]; res_={}
    for k_ in K_:
        M_=np.stack(V_[k_]); P_=M_[:,0]; N_=M_[:,1:].ravel(); thr_=np.quantile(N_,0.95)
        res_[k_]=(_inf_auc(P_,N_),float((M_[:,:1]>M_[:,1:]).mean()),float((P_>thr_).mean()))
        out_.append(f"{k_}: AUC {res_[k_][0]:.2f}, paired {res_[k_][1]:.2f}, accepted at 5% false {res_[k_][2]:.2f}")
    print(f"  [inference splice-and-appraise | {tag}] {len(ch_)} two-hop chains, {nf_} false splices each (the C of another chain in the same frame), store OFF unless noted"
          f" | bridge recalled by the weights (A r0 -> B) {np.mean(br_):.2f}"+(f", with the store {np.mean(brs_):.2f} (store-keyed positions {dg_[0]}/{dg_[1]}, names in the store {len(_INFM.get('names') or {})})" if st_ else ", no fast store")+" | "+" | ".join(out_),flush=True)
    return res_
def _inf_heldout(cfg):
    """skill-transfer runs: every second probe chain is held out (its names never appear in a worked example)"""
    if _INF.get('ho') is None: _INF['ho']={n_ for i_,c_ in enumerate(inf_probes(cfg)) if i_%2==1 for n_ in c_['names']}
    return _INF['ho']
def _inf_skill_row(c_,S_,Lr_,rs_):
    """a WORKED EXAMPLE: the conclusion's two verified source sentences, then 'So A is connected to C.'; targets = C and the full stop"""
    A,C=c_['A'],c_['C']; sa_=S_['it'][c_['src'][0]]['s'].decode('latin-1').strip(); sc_=S_['it'][c_['src'][1]]['s'].decode('latin-1').strip()
    if A not in sa_: sa_,sc_=sc_,sa_
    if rs_.rand()<0.5: sa_,sc_=sc_,sa_
    t_=f"{sa_} {sc_} So {A} is connected to {C}."; b_=t_.encode('latin-1','replace')
    if len(b_)>Lr_-1: return None
    w_=b_+b' '*(Lr_-len(b_)); a0_=len(b_)-len((C+'.').encode('latin-1','replace')); mk_=torch.zeros(Lr_-1); mk_[a0_-1:len(b_)-1]=1.0
    return torch.tensor(list(w_),dtype=torch.long),mk_

def _inf_hyp(model,cfg,S_,nid_,st_):
    """TWO-TIER STORE: LOOSE hypothesis generation (rarity threshold x inf_hyp_df_mult, fan-out x inf_hyp_fan_mult) into an unverified tier
    that never touches predictions; a hypothesis is PROMOTED to a conclusion only with independent support: two different bridges"""
    H_=S_.setdefault('hyp',{}); df_=S_['df']; mx_=int(cfg.inf_df)*float(getattr(cfg,'inf_hyp_df_mult',2.0)); fan_=int(cfg.inf_maxfan)*float(getattr(cfg,'inf_hyp_fan_mult',3.0))
    lw_=S_.get('lw') if S_.get('strict') else None; ok_=lambda n_: df_.get(n_,0)<=mx_ and (lw_ is None or n_.lower() not in lw_)
    done_={frozenset((c_['A'],c_['C'])) for c_ in S_['conc']}; br_=set(n_ for i_ in nid_ if i_<len(S_['it']) for n_ in _INF_NAME.findall(S_['it'][i_]['s']) if ok_(n_)); nh0_=len(H_)
    for B in br_:
        part={}
        for i_ in S_['idx'].get(B,[]):
            it_=S_['it'][i_]
            if it_['dep']>0 or it_.get('k')=='inf' or it_.get('sup'): continue   # hypotheses come from real / verified, current sentences only
            for A in set(_INF_NAME.findall(it_['s'])):
                if A!=B and ok_(A): part.setdefault(A,[]).append(i_)
        if len(part)>fan_: continue
        ks_=sorted(part)
        for x_ in range(len(ks_)):
            for y_ in range(x_+1,len(ks_)):
                A,C=ks_[x_],ks_[y_]; ia_=part[A][0]; ic_=part[C][0]
                if ia_==ic_: continue
                H_.setdefault((A,C),{})[B]=(ia_,ic_)
    cap_=int(getattr(cfg,'inf_hyp_cap',300000))
    while len(H_)>cap_: H_.pop(next(iter(H_)))                                    # oldest hypotheses forgotten first
    out_=[]
    for (A,C),bs_ in list(H_.items()):
        if len(bs_)<int(getattr(cfg,'inf_hyp_support',2)): continue
        key_=frozenset((A.decode('latin-1'),C.decode('latin-1')))
        if key_ in done_ or (set(S_['idx'].get(A,[]))&set(S_['idx'].get(C,[]))): H_.pop((A,C)); continue
        B,(ia_,ic_)=sorted(bs_.items())[0]
        c_={'A':A.decode('latin-1'),'B':B.decode('latin-1'),'C':C.decode('latin-1'),'r':(None,None),'typed':False,'s':f"So {A.decode('latin-1')} is connected to {C.decode('latin-1')}.",
            'src':(ia_,ic_),'d':S_['it'][ia_]['d'],'dep':1,'promoted':len(bs_)}
        out_.append(c_); done_.add(key_); H_.pop((A,C))
        i2_=len(S_['it']); S_['it'].append({'s':c_['s'].encode('latin-1'),'d':c_['d'],'k':'inf','dep':0})
        for n_ in (A,C): S_['idx'].setdefault(n_,[]).append(i2_)
    S_['conc']+=out_
    print(f"  [inference two-tier store] hypotheses (unverified, never used for prediction): {len(H_)} (+{len(H_)-nh0_+len(out_)} this sleep) | PROMOTED with two independent bridges: {len(out_)}",flush=True)
    return out_
def _inf_use(S_,nid_,vis_):
    """usage: a conclusion is USED when one of its names appears again in newly stored (real or verified) text after it was made"""
    nm_=set(n_.decode('latin-1') for i_ in nid_ if i_<len(S_['it']) for n_ in _INF_NAME.findall(S_['it'][i_]['s']))
    for c_ in S_['conc']:
        if c_.get('v',vis_)<vis_ and (c_['A'] in nm_ or c_['C'] in nm_): c_['use']=c_.get('use',0)+1
@torch.no_grad()
def _inf_prune(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_):
    """PRUNING (selective sleep): conclusions never used (no name re-encountered) inf_prune_age visits after they were made have their store
    entries faded and leave the rehearsal pool"""
    if not int(getattr(cfg,'inf_prune',0)) or _INFM['M'] is None: return ''
    vis_=int(model.__dict__.get('_visit',0)); age_=int(getattr(cfg,'inf_prune_age',2))
    cand_=[c_ for c_ in allc_ if not c_.get('freed') and c_.get('use',0)==0 and vis_-c_.get('v',vis_)>=age_][:int(getattr(cfg,'inf_prune_n',200))]
    for c_ in cand_:
        c_['freed']=True; c_['pruned']=True
        for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []):
            it_=(sn_+' ').encode('latin-1','replace')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
            X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(c_['d']))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
            Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
            for _ in range(10): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
    t2_,t3_=_inf_truth(cfg); pr_=sum(1 for c_ in cand_ if (c_['A'],c_['C']) in t2_|t3_)
    return f" | PRUNED {len(cand_)} unused conclusions ({pr_} of them true probe chains; total pruned {sum(int(c_.get('pruned',False)) for c_ in allc_)} of {len(allc_)})"

def _inf_abstain_sources(S_,cr_,tag):
    """(1) ANSWER OR ABSTAIN: accuracy on the questions answered vs the share answered, for three policies; bluffs = answers to chains the
    system never found. (2) SHOW SOURCES: answers backed by a stored conclusion cite its two verified source sentences; checked against the truth"""
    cm_={frozenset((c_['A'],c_['C'])):c_ for c_ in S_['conc'] if not c_.get('freed')}
    ok_=np.array([r_[0] for r_ in cr_],bool); cf_=np.array([r_[1] for r_ in cr_]); hit_=np.array([r_[3] for r_ in cr_],bool)
    fnd_=np.array([frozenset((r_[4],r_[6])) in cm_ for r_ in cr_],bool); cite_=np.array([frozenset((r_[4],r_[5])) in cm_ for r_ in cr_],bool)
    pol_={'answer always':np.ones(len(cr_),bool),'answer if confidence >= 0.5':cf_>=0.5,'answer if a stored conclusion backs the answer':cite_,'backed and confidence >= 0.5':cite_&(cf_>=0.5)}
    seg_=[]
    for k_,a_ in pol_.items():
        seg_.append(f"{k_}: answers {a_.mean():.2f}, accuracy when answering {ok_[a_].mean() if a_.any() else float('nan'):.2f}, bluffs on unfound chains {(a_&~fnd_&~ok_).sum()}/{int((~fnd_).sum())}")
    print(f"  [inference answer-or-abstain | {tag}] two-hop with the store, n {len(cr_)} ({int(fnd_.sum())} chains found) || "+" || ".join(seg_),flush=True)
    ex_=[];ver_=0;nc_=0
    for r_ in cr_:
        c_=cm_.get(frozenset((r_[4],r_[5])))
        if c_ is None: continue
        nc_+=1; s1_=S_['it'][c_['src'][0]]['s'].decode('latin-1','replace').strip(); s2_=S_['it'][c_['src'][1]]['s'].decode('latin-1','replace').strip()
        good_=(r_[5]==r_[6]) and (r_[7] in s1_+' '+s2_) and (r_[4] in s1_+' '+s2_) and (r_[6] in s1_+' '+s2_); ver_+=int(good_)
        if len(ex_)<2: ex_.append(f"{r_[4]} -> {r_[5]} because \"{s1_[:80]}\" + \"{s2_[:80]}\" ({'checks out' if good_ else 'WRONG'})")
    print(f"  [inference show-sources | {tag}] {nc_} answers backed by a stored conclusion; cited sources contain the true chain (A, bridge, C) and the answer is right in {ver_}/{nc_}"+(" | e.g. "+" || ".join(ex_) if ex_ else ''),flush=True)

@torch.no_grad()
def _inf_fade_sent(sn_,d_,S_slot,Lr_,filler,feats,dev_,beta_,it_n=10):
    """fade the store's entries for one sentence (writes towards the model's own prediction)"""
    it_=(sn_+' ').encode('latin-1','replace')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(d_)[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
    X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(d_))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
    Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
    for _ in range(it_n): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
def _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_):
    for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []): _inf_fade_sent(sn_,c_['d'],S_slot,Lr_,filler,feats,dev_,beta_)
@torch.no_grad()
def _inf_audit(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_):
    """GROUNDING AUDIT (consolidation phase): every (subject, relation) with competing objects is re-grounded on its evidence record.
    Support of an object = sum over its verified / real sentences of exp(-(visits since stored)/tau). For relations LEARNED to be single-valued
    (>= inf_audit_func of their subjects have one object, >= 20 subjects): top >= k x second -> CONFIRMED (the others superseded, their
    conclusions RETRACTED, their store entries faded); otherwise UNCERTAIN (dependent conclusions demoted to the hypothesis tier).
    Then STORE DRIFT: sampled subjects whose store answer is backed by no belief and no fact have that association erased."""
    if not int(getattr(cfg,'inf_audit',0)) or _INFM['M'] is None or _INFM.get('mu') is None: return ''
    R_=S_.get('rel',{}); V_=int(model.__dict__.get('_visit',0)); tau_=float(getattr(cfg,'inf_audit_tau',5.0)); k_=float(getattr(cfg,'inf_audit_k',2.0))
    subj_={}
    for (A,r),ob_ in R_.items(): subj_.setdefault(r,[]).append(sum(1 for o_,ids_ in ob_.items() if any(not S_['it'][i_].get('sup') for i_ in ids_)))
    func_={r:(np.mean([n_==1 for n_ in v_])>=float(getattr(cfg,'inf_audit_func',0.6)) and len(v_)>=int(getattr(cfg,'inf_audit_minsubj',20))) for r,v_ in subj_.items()}
    w_=lambda i_: math.exp(-(V_-S_['it'][i_].get('v',V_))/tau_)
    sup_new_=set(); unc_=set(); conf_=0
    for (A,r),ob_ in R_.items():
        if not func_.get(r) or len(ob_)<2: continue
        sc_={o_:sum(w_(i_) for i_ in ids_ if not S_['it'][i_].get('sup')) for o_,ids_ in ob_.items()}; rk_=sorted(sc_,key=lambda o_: -sc_[o_])
        if sc_[rk_[1]]<=0: continue
        if str(getattr(cfg,'inf_audit_rule','ratio'))=='recent':          # RECENCY RULE: decayed support keeps its RATIO forever (both decay alike), so a
            cn_={o_:sum(1 for i_ in ids_ if not S_['it'][i_].get('sup')) for o_,ids_ in ob_.items()}   #   near-tie never resolves. Instead: the object stated
            lv_={o_:max([S_['it'][i_].get('v',0) for i_ in ids_ if not S_['it'][i_].get('sup')] or [-1]) for o_,ids_ in ob_.items()}   # most recently wins,
            mc_=max(float(getattr(cfg,'inf_audit_minrep',3)),float(getattr(cfg,'inf_audit_minfrac',0.25))*max(cn_.values()))   # minrep: 3 locally; 2 at GPU exposure (facts seen 1-2 times)   #   if it is REPEATED (>= minfrac of the best count: one noisy line never is)
            el_=[o_ for o_ in rk_ if cn_[o_]>=mc_ and sc_[o_]>0]
            if el_:
                top_=max(el_,key=lambda o_: lv_[o_])
                if all(lv_[top_]>lv_[o_] for o_ in rk_ if o_!=top_ and sc_[o_]>0): rk_=[top_]+[o_ for o_ in rk_ if o_!=top_]; sc_[top_]=float('inf')
        if sc_[rk_[0]]>=k_*sc_[rk_[1]]:
            conf_+=1
            for o_ in rk_[1:]:
                for i_ in ob_[o_]:
                    if not S_['it'][i_].get('sup'): S_['it'][i_]['sup']=True; sup_new_.add(i_)
            for i_ in ob_[rk_[0]]: S_['it'][i_].pop('unc',None)
        else:
            for o_ in rk_:
                for i_ in ob_[o_]: S_['it'][i_]['unc']=True; unc_.add(i_)
    nret_=0; ndem_=0; nrep_=0; t2_,t3_=_inf_truth(cfg); upd_={(c_['names'][0],c_['names'][2]):c_['upd']['type'] for c_ in inf_probes(cfg) if c_.get('upd')}
    good_=0; falser_=0
    for c_ in allc_:
        sr_=c_.get('src',())
        if any(i_ in sup_new_ for i_ in sr_) and not c_.get('retracted'):
            c_['retracted']=True; c_['freed']=True; nret_+=1; _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
            ty_=upd_.get((c_['A'],c_['C'])) or upd_.get((c_['C'],c_['A']))
            if ty_=='true': good_+=1
            elif (c_['A'],c_['C']) in t2_|t3_: falser_+=1
        elif any(S_['it'][i_].get('unc') for i_ in sr_ if i_<len(S_['it'])) and not c_.get('retracted'):
            if not c_.get('uncertain'): c_['uncertain']=True; c_['freed']=True; ndem_+=1; _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
        elif c_.get('uncertain') and not any(S_['it'][i_].get('unc') for i_ in sr_ if i_<len(S_['it'])):
            c_['uncertain']=False; c_['freed']=False; nrep_+=1                  # re-confirmed: back into the store (rewritten by the normal writes)
    nff_=0                                                                      # REPLAYED FACT RECORDS follow the audit: a fact whose every sentence is superseded
    for f_ in S_.get('facts',[]):                                               #   is freed (else NREM replay keeps writing the OLD fact back into the store)
        if f_.get('freed'): continue
        ids_=R_.get((f_['A'].encode('latin-1'),f_['r'].encode('latin-1')),{}).get(f_['B'].encode('latin-1'))
        if ids_ and all(S_['it'][i_].get('sup') for i_ in ids_):
            f_['freed']=True; nff_+=1; fp0_=S_.get('fpair',{})
            for k0_ in ((f_['A'],f_['B']),(f_['B'],f_['A'])):
                if fp0_.get(k0_) is f_: fp0_.pop(k0_,None)
            _inf_fade_sent(f_['s'],f_['d'],S_slot,Lr_,filler,feats,dev_,beta_)
    ncas_=0; ncas_t_=0
    if int(getattr(cfg,'inf_audit_cascade',1)):                                 # CASCADING RETRACTION: a retracted conclusion's re-entered sentence is superseded too,
        ik_={}                                                                  #   so every conclusion built ON it (three-hop, REM) is retracted in turn, to a fixed point
        for i_,it_ in enumerate(S_['it']):
            if it_.get('k')=='inf':
                m_=_re_inf.match(rb'So (\S+) is connected to (\S+)\.',it_['s'])
                if m_: ik_.setdefault(frozenset((m_.group(1).decode('latin-1'),m_.group(2).decode('latin-1'))),[]).append(i_)
        fr_=[c_ for c_ in allc_ if c_.get('retracted') and not c_.get('casc_done')]
        while fr_:
            dead_=set()
            for c_ in fr_:
                c_['casc_done']=True
                for i_ in ik_.get(frozenset((c_['A'],c_['C'])),[]):
                    if not S_['it'][i_].get('sup'): S_['it'][i_]['sup']=True; dead_.add(i_)
            fr_=[]
            for c_ in allc_:
                if not c_.get('retracted') and any(i_ in dead_ for i_ in c_.get('src',())):
                    c_['retracted']=True; c_['freed']=True; ncas_+=1; ncas_t_+=int((c_['A'],c_['C']) in t2_|t3_); fr_.append(c_); _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
    dg_=''
    if int(getattr(cfg,'inf_audit_debug',0)):                                   # BELIEF-REVISION DIAGNOSTIC: the evidence for each true-update probe subject
        ps_=[]
        for c0_ in inf_probes(cfg):
            u0_=c0_.get('upd')
            if not u0_ or u0_['type']!='true': continue
            ob_=R_.get((c0_['names'][0].encode(),c0_['rels'][0].encode()),{})
            sc0_={o_.decode('latin-1'):(len(ids_),round(sum(w_(i_) for i_ in ids_ if not S_['it'][i_].get('sup')),2),sum(1 for i_ in ids_ if S_['it'][i_].get('sup'))) for o_,ids_ in ob_.items()}
            ps_.append(f"{c0_['names'][0]} {c0_['rels'][0]}: old {c0_['names'][1]} {sc0_.get(c0_['names'][1],'-')} new {u0_['B2']} {sc0_.get(u0_['B2'],'-')} (func {func_.get(c0_['rels'][0].encode())})")
        dg_=" | REVISION EVIDENCE (items, support, superseded): "+"; ".join(ps_[:8])
    for i_ in sup_new_: _inf_fade_sent(S_['it'][i_]['s'].decode('latin-1','replace').strip(),S_['it'][i_]['d'],S_slot,Lr_,filler,feats,dev_,beta_)   # the superseded one-hop fact
    flips_=sum(1 for c_ in inf_probes(cfg) if c_.get('upd',{}).get('type')=='noise' and any(S_['it'][i_].get('sup') for i_ in R_.get((c_['names'][0].encode(),c_['rels'][0].encode()),{}).get(c_['names'][1].encode(),[])))
    nd_=int(getattr(cfg,'inf_audit_drift',20)); er_=0; ck_=0
    live_=[c_ for c_ in allc_ if not c_.get('freed')]; nm_=[n_.decode('latin-1') for n_ in _INFM['names']]
    if live_ and len(nm_)>=25 and nd_>0:                                        # STORE DRIFT: does the store push associations nothing grounds?
        bel_={}
        for c_ in live_: bel_.setdefault(c_['A'],set()).add(c_['C']); bel_.setdefault(c_['C'],set()).add(c_['A'])
        for X in list(dict.fromkeys(c_['A'] for c_ in [live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),3*nd_),replace=False)]))[:nd_]:
            cs_=list(bel_.get(X,set()))+[n_ for n_ in rs_.choice(nm_,24,replace=False) if n_!=X]; sl_=int(S_slot(live_[0]['d']))
            l1_=_inf_lp(model,cfg,f"So {X} is connected to ",cs_,sl_,True); l0_=_inf_lp(model,cfg,f"So {X} is connected to ",cs_,sl_,False); ck_+=1
            Y=cs_[int(np.argmax(np.array(l1_)-np.array(l0_)))]
            gr_=Y in bel_.get(X,set()) or any(Y.encode() in S_['it'][i_]['s'] for i_ in S_['idx'].get(X.encode(),[]) if not S_['it'][i_].get('sup'))
            if not gr_ and max(np.array(l1_)-np.array(l0_))>1.0:
                er_+=1; _inf_fade_sent(f"So {X} is connected to {Y}.",live_[0]['d'],S_slot,Lr_,filler,feats,dev_,beta_)
    nf_=sum(func_.values())
    return (f" | GROUNDING AUDIT: {nf_} relations learned single-valued; {conf_} (subject, relation) conflicts resolved, {len(sup_new_)} sentences superseded, "
            f"{nret_} conclusions RETRACTED ({good_} stale probe beliefs, {falser_} true probe chains wrongly), {ndem_} demoted as uncertain, {nrep_} re-confirmed; "
            f"noise contradictions that flipped a belief {flips_}; CASCADE: {ncas_} dependent conclusions retracted ({ncas_t_} of them true probe chains), {nff_} replayed fact records freed; store drift: {er_} ungrounded associations erased of {ck_} subjects checked"+dg_)

# ---- DEBUG (inf_debug_sync): validate the sparse n-gram gradients before every n-gram optimiser step ------------------------------------
_ngp_step_0=ngp_step
def ngp_step(model,cfg,lr):
    if int(getattr(cfg,'inf_debug_sync',0)) and torch.cuda.is_available():
        for on_ in ('_ngpi_opt','_ngp_opt','_ngo_opt'):
            O_=model.__dict__.get(on_)
            if O_ is None: continue
            for g0_ in O_.param_groups:
                for p_ in g0_['params']:
                    g_=p_.grad
                    if g_ is None or not g_.is_sparse: continue
                    torch.cuda.synchronize(); ix_=g_._indices(); bad_=int(((ix_<0)|(ix_>=p_.shape[0])).sum()) if ix_.numel() else 0
                    if bad_ or not torch.isfinite(g_._values()).all():
                        print(f"  [inference debug] BAD SPARSE GRAD in {on_}: param {tuple(p_.shape)}, {bad_} of {ix_.numel()} indices out of range (min {int(ix_.min()) if ix_.numel() else 0}, max {int(ix_.max()) if ix_.numel() else 0}), values finite {bool(torch.isfinite(g_._values()).all())}",flush=True)
        _inf_sync(cfg,'n-gram gradients validated')
    return _ngp_step_0(model,cfg,lr)

# ---- cg20 SLEEP-ONLY FEATURE LEARNING (inf_sofl): no gradient learning while awake (the visit's learning rate is 0; fast learners only:
#      in-context memory, closed-form head refits, the fast store); at the visit's end the SLEEP runs inf_sofl_frac x steps_per_task
#      backprop steps on the buffered (real or verified) windows, interleaved with old domains' windows under KL to the pre-sleep model.
#      FEATURE-NEED SIGNALS, computed forward-only before the sleep, are logged with the sleep's measured BENEFIT (held-out BPC gain).
def _inf_fwd(model,X_,d_):
    sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_
    try: return model(X_[:,:-1],d_).float()
    finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_
def _inf_bpc_off(model,vals,d_,s_,n_=4):
    sv_=_INFM.get('force','__none__'); _INFM['force']=False
    try: return float(bpc(model,vals[d_],s_,n=n_))
    finally:
        if sv_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=sv_
@torch.no_grad()
def _inf_need(model,cfg,buf_,S_slot,vals,t,ts):
    """forward-only FEATURE-NEED signals on the buffered windows: (A) a linear softmax readout of the pre-head features fitted by the delta rule
    (its held-out bits) vs the same readout on a random nonlinear expansion (gap = structure the features miss) and vs the model's own head;
    (B) delta-rule KEY COLLISIONS in the last block (near-identical keys, different next bytes); (C) FAST-SYSTEM LOAD (store-off minus store-on
    held-out BPC); (D) REPEAT vs NOVEL (bits on the buffered windows minus held-out bits)"""
    dev_=model.embed.weight.device; rs_=np.random.RandomState(len(buf_)+7); was_=model.training; model.eval()
    sel_=[buf_[i_] for i_ in rs_.choice(len(buf_),min(16,len(buf_)),replace=False)]; Zs_=[];Ys_=[];Ls_=[];Ks_=[]
    blk_=model.blocks[-1]
    for w_,d0_ in sel_:
        X_=torch.tensor([list(w_)],dtype=torch.long,device=dev_); sh_=_HCAP['on']; _HCAP['on']=True; _DQ['on']=True; _DQ['kv']={}
        try: lg_=_inf_fwd(model,X_,int(S_slot(d0_))); z_=_HCAP['h'].float()
        finally: _HCAP['on']=sh_; _DQ['on']=False
        y_=X_[0,1:]; ok_=y_>0; Zs_.append(z_[0][ok_]); Ys_.append(y_[ok_]); Ls_.append(F.log_softmax(lg_[0][ok_],-1).gather(-1,y_[ok_][:,None]).squeeze(-1))
        kv_=_DQ['kv'].get(id(blk_))
        if kv_ is not None: Ks_.append((kv_[0].reshape(-1,kv_[0].shape[-1])[:y_.numel()][ok_],y_[ok_]))
    model.train(was_); Z_=torch.cat(Zs_); Y_=torch.cat(Ys_); N_=Z_.shape[0]; out_={}
    if N_>200:
        pm_=torch.randperm(N_,generator=torch.Generator().manual_seed(3)).to(dev_)[:4096]; Z_=Z_[pm_]; Y_=Y_[pm_]; Lh_=torch.cat(Ls_)[pm_]; n1_=int(0.75*Z_.shape[0])
        mu_=Z_[:n1_].mean(0); sd_=Z_[:n1_].std(0)+1e-6; Zn_=(Z_-mu_)/sd_
        def fit(F_):                                                            # softmax readout by the (batch) delta rule: W += lr F^T (onehot - p) / n
            W_=torch.zeros(F_.shape[1],256,device=dev_); b_=torch.zeros(256,device=dev_); oh_=F.one_hot(Y_[:n1_],256).float()
            for _ in range(150):
                P_=F.softmax(F_[:n1_]@W_+b_,-1); W_+=0.5*F_[:n1_].T@(oh_-P_)/n1_; b_+=0.5*(oh_-P_).mean(0)
            return float(F.cross_entropy(F_[n1_:]@W_+b_,Y_[n1_:]))/math.log(2)
        g_=torch.Generator().manual_seed(11); Rr_=(torch.randn(Zn_.shape[1],4*Zn_.shape[1],generator=g_)/math.sqrt(Zn_.shape[1])).to(dev_)
        out_['lin']=fit(Zn_); out_['rf']=fit(torch.relu(Zn_@Rr_)/math.sqrt(4.0)); out_['head']=float(-Lh_[n1_:].mean())/math.log(2)
        out_['A_gap']=out_['lin']-out_['rf']
    if Ks_:
        K_=F.normalize(torch.cat([k_ for k_,_ in Ks_]).float(),dim=-1); Yk_=torch.cat([y_ for _,y_ in Ks_])
        if K_.shape[0]>2000: ix_=torch.randperm(K_.shape[0],generator=torch.Generator().manual_seed(5))[:2000].to(dev_); K_=K_[ix_]; Yk_=Yk_[ix_]
        C_=K_@K_.T; C_.fill_diagonal_(0); hi_=C_>0.95; npair_=int(hi_.sum())
        out_['B_coll']=float((hi_&(Yk_[:,None]!=Yk_[None,:])).sum())/max(1,npair_) if npair_ else 0.0; out_['B_pairs']=npair_/max(1,K_.shape[0])
    sv_=_INFM.get('force','__none__'); _INFM['force']=True
    try: on_=float(bpc(model,vals[t],ts,n=4))
    finally:
        if sv_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=sv_
    off_=_inf_bpc_off(model,vals,t,ts); out_['C_load']=off_-on_; out_['D_rep']=(float(-torch.cat(Ls_).mean())/math.log(2))-off_
    return out_,off_
def _inf_sofl(model,cfg,t,ts,vals,olds_,S_slot):
    S_=_inf_store(model); ev_=int(getattr(cfg,'inf_sofl_every',1))
    if ev_>1 and (int(model.__dict__.get('_visit',0))+1)%ev_!=0: return     # sleep only every ev_ visits: the buffer keeps filling meanwhile
    buf_=S_.get('sbuf') or []; S_['sbuf']=[]; S_['sbn']=0
    if len(buf_)<4: return
    dev_=model.embed.weight.device; rs_=np.random.RandomState(int(cfg.seed)+5151+int(model.__dict__.get('_visit',0)))
    need_,b0_=_inf_need(model,cfg,buf_,S_slot,vals,t,ts); _inf_sync(cfg,'feature-need signals')
    o0_=float(np.mean([_inf_bpc_off(model,vals,d_,s_) for d_,s_ in olds_])) if olds_ else float('nan')
    st_=max(1,int(round(int(cfg.steps_per_task)*float(getattr(cfg,'inf_sofl_frac',1.0))))); R_=int(getattr(cfg,'inf_sofl_rows',16)); fa_=float(getattr(cfg,'inf_sofl_old',0.25))
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; TP_=[p_.detach().clone() for p_ in PA_]
    op_=model.__dict__.get('_sofl_opt')
    if op_ is None: op_=model.__dict__['_sofl_opt']=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(getattr(cfg,'inf_sofl_lr',1.0)))
    old_=[d_ for d_ in (S_.get('res') or {}) if d_!=int(t) and S_['res'][d_]]
    def teacher(X_,d_):
        sv_=[p_.data for p_ in PA_]
        for p_,q_ in zip(PA_,TP_): p_.data=q_
        try:
            with torch.no_grad(): return F.log_softmax(_inf_fwd(model,X_,d_),-1)
        finally:
            for p_,v_ in zip(PA_,sv_): p_.data=v_
    was_=model.training; model.train(); ls_=[]; L0_=len(buf_[0][0])
    with torch.enable_grad():
        for k_ in range(st_):
            grp_={}
            for _ in range(R_):
                if old_ and rs_.rand()<fa_: d0_=old_[rs_.randint(len(old_))]; w0_=S_['res'][d0_][rs_.randint(len(S_['res'][d0_]))]; an_=True
                else: w0_,d0_=buf_[rs_.randint(len(buf_))]; an_=False
                w0_=(bytes(w0_)+b' '*L0_)[:L0_]; grp_.setdefault(int(S_slot(d0_)),[]).append((torch.tensor(list(w0_),dtype=torch.long),an_))
            loss_=0.0
            for d_,rows_ in grp_.items():
                X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); L_=F.log_softmax(_inf_fwd(model,X_,d_),-1); nz_=(X_[:,1:]>0).float()
                an_=torch.tensor([r_[1] for r_ in rows_],device=dev_)[:,None]
                ce_=-L_.gather(-1,X_[:,1:,None]).squeeze(-1)
                if bool(an_.any()): T_=teacher(X_,d_); kl_=(T_.exp()*(T_-L_)).sum(-1)
                else: kl_=torch.zeros_like(ce_)
                loss_=loss_+(torch.where(an_,kl_,ce_)*nz_).sum()/nz_.sum().clamp(min=1)*len(rows_)/R_
            op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(loss_.detach())); _ng_zero(model)
    for p_ in PA_: p_.grad=None
    model.train(was_); _inf_sync(cfg,'sleep feature learning')
    b1_=_inf_bpc_off(model,vals,t,ts); o1_=float(np.mean([_inf_bpc_off(model,vals,d_,s_) for d_,s_ in olds_])) if olds_ else float('nan')
    ben_=b0_-b1_; model.__dict__.setdefault('_need_log',[]).append((need_,ben_,o1_-o0_ if olds_ else 0.0))
    print(f"  [inference sleep feature learning] {st_} backprop steps in SLEEP on {len(buf_)} buffered windows ({int(100*fa_)}% rows from {len(old_)} old domains, KL to the pre-sleep model) | loss {ls_[0]:.3f} -> {np.mean(ls_[-5:]):.3f} | "
          f"held-out BPC (store off) AWAKE {b0_:.3f} -> AFTER SLEEP {b1_:.3f} (benefit {ben_:+.3f}), old domains {o0_:.3f} -> {o1_:.3f} | FEATURE NEED: "
          +" ".join(f"{k0_} {v0_:+.3f}" if isinstance(v0_,float) else f"{k0_} {v0_}" for k0_,v0_ in need_.items()),flush=True)
def inf_need_report(model):
    """end of run: which forward-only signal predicts how much a sleep's backprop helped (Pearson r over the sleeps)"""
    lg_=model.__dict__.get('_need_log') or []
    if len(lg_)<4: return
    ben_=np.array([b_ for _,b_,_ in lg_]); ks_=sorted(set(k_ for n_,_,_ in lg_ for k_ in n_)); seg_=[]
    for k_ in ks_:
        x_=np.array([n_.get(k_,np.nan) for n_,_,_ in lg_],float); ok_=np.isfinite(x_)
        if ok_.sum()>=4 and np.std(x_[ok_])>0 and np.std(ben_[ok_])>0: seg_.append(f"{k_} r {np.corrcoef(x_[ok_],ben_[ok_])[0,1]:+.2f}")
    print(f"  [inference sleep feature learning | FEATURE-NEED CALIBRATION] {len(lg_)} sleeps, benefit (held-out BPC gain) mean {ben_.mean():+.3f} | correlation of each forward-only signal with the benefit: "+" | ".join(seg_),flush=True)

def pu_consolidate(model,cfg,t,ts,vals,olds_,S_):
    """cg19 (1)+(3): PRIVATE-INTO-SHARED CONSOLIDATION on the visit's VERIFIED windows, with trace-level retirement and a capacity cap.
    Teacher = the model frozen at the visit end (private units ON). Student = the live model with slot ts's private units OFF on the visit's
    windows; old domains' kept windows (private units on) are taught to keep the frozen output. Only the blocks' SHARED weights learn."""
    RV_=model.__dict__.get('_nres'); BK_=model.__dict__.get('_nres_bank',{})
    if not RV_ or RV_.get('td')!=int(t) or not RV_.get('e') or not hasattr(model.blocks[0],'pu_out'):
        print(f"  [consolidation] {_lab(cfg,t)}: skipped (no verified windows or no private units)"); return
    own_=[e_ for e_ in RV_['e']]; anc_=[e_ for d_,l_ in BK_.items() if d_!=int(t) for e_ in l_]
    nb_=int(cfg.pu_cons_eval); dev_=model.embed.weight.device
    def _ev(off):
        _PUOFF['d']=(ts if off else None)
        try: b_=float(bpc(model,vals[t],ts,n=nb_))
        finally: _PUOFF['d']=None
        return b_
    def _evo():
        if not olds_: return float('nan')
        return float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_]))
    was_=model.training
    on0_=_ev(False); off0_=_ev(True); old0_=_evo(); model.train()
    PA_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad]
    TP_=[p_.detach().clone() for p_ in PA_]                                       # the frozen teacher
    SH_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad and n_.startswith('blocks.') and '.pu_' not in n_]
    PU_=[getattr(b_,k_) for b_ in model.blocks for k_ in ('pu_in','pu_b','pu_out')]   # only slot ts's slice is updated (others' grads zeroed)
    SH_=SH_+PU_
    op_=torch.optim.Adam(SH_,lr=float(cfg.lr)*float(cfg.pu_cons_lr))
    rs_=model.__dict__.setdefault('_pucons_rs',np.random.RandomState(int(cfg.seed)+6161))
    R_=int(cfg.pu_cons_rows); ka_=min(len(anc_),int(round(R_*float(cfg.pu_cons_anchor)))) if anc_ else 0; ko_=R_-ka_
    kl_o=[]; kl_a=[]
    def _fwd(X_,d_,off):
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_; _PUOFF['d']=(d_ if off else None)
        try: return F.log_softmax(model(X_[:,:-1],d_).float()/cfg.shunt_T,-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_; _PUOFF['d']=None
    def _teach(X_,d_):
        with torch.no_grad():
            sv_=[p_.data for p_ in PA_]
            for p_,q_ in zip(PA_,TP_): p_.data=q_
            try: return _fwd(X_,d_,False)
            finally:
                for p_,v_ in zip(PA_,sv_): p_.data=v_
    _cs_snap_=True if float(getattr(cfg,'pu_cons_brake',0.0))>0 else None; _cs_rv_=list(RV_['e'])   # brake: TP_ (the frozen copy taken above) is the snapshot
    for e_ in own_: e_.pop('ck0',None); e_.pop('ck',None)
    for st_ in range(int(cfg.pu_cons_steps)):
        live_=[e_ for e_ in own_ if not e_.get('cret')]
        if not live_: break
        so_=[live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),ko_),replace=False)]
        sa_=[anc_[i_] for i_ in rs_.choice(len(anc_),ka_,replace=False)] if ka_>0 else []
        loss_=0.0
        Xo_=torch.stack([e_['x'] for e_ in so_]).to(dev_); To_=_teach(Xo_,ts); Lo_=_fwd(Xo_,ts,True)
        ko2_=(To_.exp()*(To_-Lo_)).sum(-1).mean(-1)                               # per-row KL, private units OFF
        Ln_=_fwd(Xo_,ts,False); kn2_=(To_.exp()*(To_-Ln_)).sum(-1).mean(-1)   # and WITH them (no double counting)
        loss_=loss_+(ko2_.mean()+kn2_.mean())*len(so_)/R_
        with torch.no_grad():
            for e_,k_ in zip(so_,ko2_.tolist()):
                e_.setdefault('ck0',k_); e_['ck']=k_
                if float(cfg.pu_cons_retire)>=0 and k_<=float(cfg.pu_cons_retire)*e_['ck0'] and st_>0: e_['cret']=True
        kl_o.append(float(ko2_.mean()))
        for d_ in sorted(set(e_['d'] for e_ in sa_)):                               # ANCHORS: old domains keep the frozen output
            g_=[e_ for e_ in sa_ if e_['d']==d_]; Xa_=torch.stack([e_['x'] for e_ in g_]).to(dev_)
            Ta_=_teach(Xa_,d_); La_=_fwd(Xa_,d_,False); ka2_=(Ta_.exp()*(Ta_-La_)).sum(-1).mean(-1)
            loss_=loss_+ka2_.sum()/R_; kl_a.append(float(ka2_.mean()))
        op_.zero_grad(set_to_none=True); loss_.backward()
        with torch.no_grad():
            for q_ in PU_:                                                        # private units: only this slot's slice learns
                if q_.grad is not None: k_=q_.grad[ts].clone(); q_.grad.zero_(); q_.grad[ts]=k_
        op_.step()
        for p_ in PA_:
            if p_.grad is not None: p_.grad=None
        _ng_zero(model)                                                         # FIX: the n-gram tables' sparse grads accumulated here (GPU crash)
    del op_
    nret_=sum(1 for e_ in own_ if e_.get('cret'))
    RV_['e']=[e_ for e_ in RV_['e'] if not e_.get('cret')]                       # (3) retired windows are not kept
    cap_=int(cfg.pu_cons_cap); nev_=0
    if cap_>0 and len(RV_['e'])>cap_:                                             # (3) capacity cap: weakest-first eviction
        RV_['e']=sorted(RV_['e'],key=lambda e_:-float(e_.get('ck',1e9)))
        nev_=len(RV_['e'])-cap_; RV_['e']=RV_['e'][:cap_]
    if float(cfg.pu_cons_shrink)>0:
        with torch.no_grad():
            for b_ in model.blocks: b_.pu_out.data[ts]*=(1.0-float(cfg.pu_cons_shrink))
    on1_=_ev(False); off1_=_ev(True); old1_=_evo(); _cs_rb_=''
    _cs_b_=float(getattr(cfg,'pu_cons_brake',0.0))
    if _cs_snap_ is not None and (old1_-old0_>_cs_b_ or on1_-on0_>_cs_b_):     # BRAKE: the phase hurt old domains or the domain itself -> undo it
        with torch.no_grad():
            for q_,s0_ in zip(PA_,TP_): q_.data.copy_(s0_)
        RV_['e']=_cs_rv_
        for e_ in RV_['e']: e_.pop('cret',None)
        _cs_rb_=f" | ROLLED BACK (old domains {old1_-old0_:+.4f}, own {on1_-on0_:+.4f}; brake {_cs_b_})"; on1_,off1_,old1_=on0_,off0_,old0_; nret_=0; nev_=0
    _cs_snap_=None
    model.train(was_)
    rec_=(off0_-off1_)/max(1e-6,off0_-on0_)
    CL_=model.__dict__.setdefault('_pucons_log',[]); CL_.append((int(t),on0_,on1_,off0_,off1_,rec_,old0_,old1_,nret_,nev_,len(RV_['e'])))
    print(f"  [consolidation] {_lab(cfg,t)}: {len(kl_o)} steps on {len(own_)} verified windows ({ko_} rows/step) + {ka_} anchor rows/step from {len(set(e_['d'] for e_ in anc_))} old domains"
          f" | held-out WITH its private units {on0_:.3f} -> {on1_:.3f} | WITHOUT {off0_:.3f} -> {off1_:.3f} (gap recovered {100*rec_:.0f}%)"
          f" | old domains mean {old0_:.3f} -> {old1_:.3f} ({old1_-old0_:+.4f})"
          f" | KL own (private off) {kl_o[0] if kl_o else float('nan'):.4f} -> {kl_o[-1] if kl_o else float('nan'):.4f}, anchors {np.mean(kl_a) if kl_a else float('nan'):.4f}"
          f" | retired {nret_} (absorbed), evicted {nev_} (cap {cap_ or 'none'}), kept {len(RV_['e'])}"
          +(f" | private outputs x{1-float(cfg.pu_cons_shrink):.2f}" if float(cfg.pu_cons_shrink)>0 else "")+_cs_rb_)
def _nrem_loss_v1(model,cfg,dom):
    """NREM v1: re-teach the visit's reservoir windows (stored bytes + stored teacher targets), highest remaining signal first; retire consolidated ones.
    nrem_mix>0: part of each batch comes from other domains' kept samples (cross-domain NREM), each taught under its own domain."""
    RV_=model.__dict__.get('_nres'); z0_=next(model.parameters()).sum()*0.0
    td_=int(model.__dict__.get('_cur_dom',dom))
    if not RV_ or RV_.get('td')!=td_: return z0_
    k_=int(cfg.nrem_k); BK_=model.__dict__.get('_nres_bank',{}) if float(cfg.nrem_mix)>0 else {}
    oth_=[e_ for d_,l_ in BK_.items() if d_!=td_ for e_ in l_]
    ko_=min(len(oth_),int(round(k_*float(cfg.nrem_mix)))) if oth_ else 0
    if cfg.nrem_mix_add: k_=k_+ko_
    if cfg.nrem_own_sel=='rand' and RV_['e']:
        rs3_=model.__dict__.setdefault('_nres_rs3',np.random.RandomState(int(cfg.seed)+919)); own_=[RV_['e'][i_] for i_ in rs3_.choice(len(RV_['e']),min(len(RV_['e']),k_-ko_),replace=False)]
    else: own_=sorted(RV_['e'],key=lambda e_:-e_['v'])[:k_-ko_]
    if len(own_)<k_-ko_: ko_=min(len(oth_),k_-len(own_))
    if ko_>0:
        if cfg.nrem_mix_sel=='rand':
            rs_=model.__dict__.setdefault('_nres_rs2',np.random.RandomState(int(cfg.seed)+717)); osel_=[oth_[i_] for i_ in rs_.choice(len(oth_),ko_,replace=False)]
        else: osel_=sorted(oth_,key=lambda e_:-e_['v'])[:ko_]
    else: osel_=[]
    sel_=own_+osel_; OWN_=set(id(e_) for e_ in own_)
    if not sel_: return z0_
    L=int(cfg.cd_seg); kb_=int(cfg.bank_targets); C=sel_[0]['x'].numel()//L; tot_=sum(e_['w'].numel() for e_ in (own_ if (cfg.nrem_mix_add and own_) else sel_)); loss_=z0_; drop_=set()
    for d_ in sorted(set(e_['d'] for e_ in sel_)):
        g_=[e_ for e_ in sel_ if e_['d']==d_]
        with torch.no_grad():
            xb_=torch.stack([e_['x'] for e_ in g_]); wl_=torch.stack([e_['w'] for e_ in g_]).float(); vml_=(wl_>0).float()
            V_=torch.stack([e_['tv'] for e_ in g_]).float(); I_=torch.stack([e_['ti'] for e_ in g_]).long()
            rest_=(1-V_.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-kb_); TG_=rest_.log().expand(*V_.shape[:-1],256).clone(); TG_.scatter_(-1,I_,V_)
            qy_=torch.stack([e_['q'] for e_ in g_]).float()
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
        try: lp_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
        if cfg.nrem_tgt=='bytes': loss_=loss_-(lp_.gather(-1,xb_[:,1:,None]).squeeze(-1)*wl_).sum()/max(1,tot_)
        else: loss_=loss_+((TG_.exp()*(TG_-lp_)).sum(-1)*wl_).sum()/max(1,tot_)
        with torch.no_grad():
            rn_=_store_score(qy_,lp_.gather(-1,xb_[:,1:,None]).squeeze(-1),vml_,C,L).tolist()
            for e_,r_ in zip(g_,rn_):
                o_=id(e_) not in OWN_; e_['n']+=1; RV_['rep_o' if o_ else 'rep']=RV_.get('rep_o' if o_ else 'rep',0)+1
                if float(cfg.nrem_retire)>=0 and r_<=max(1e-4,float(cfg.nrem_retire)*e_['v0']): drop_.add(id(e_)); RV_['ret_o' if o_ else 'ret']=RV_.get('ret_o' if o_ else 'ret',0)+1
                else: e_['v']=r_
    if drop_:
        RV_['e']=[e_ for e_ in RV_['e'] if id(e_) not in drop_]
        for d_ in list(BK_): BK_[d_]=[e_ for e_ in BK_[d_] if id(e_) not in drop_]
    return loss_
def _cd_lrf(cfg,x):
    if float(cfg.cd_lr)>=0: return float(cfg.cd_lr)
    C=int(cfg.cd_chain) if int(cfg.cd_chain)>0 else max(1,x.shape[1]//int(cfg.cd_seg))
    return min(1.0,int(cfg.cd_rows)*min(int(cfg.cd_seg)*C,x.shape[1])/float(x.numel()))
_CDR=[0,0,0.0,0]                                                           # v9: reused conduit steps, conduit steps, teaching-share sum / count
_CDP={'n':0,'todo':None}
def _cd_will_reuse(model,cfg,dom):
    rc_=model.__dict__.get('_cd_cache')
    return bool(int(cfg.cd_reuse)>1 and rc_ is not None and rc_['dom']==dom and rc_['k']<int(cfg.cd_reuse)-1)
def cd_loss(model,cfg,x,dom):
    _CDP['n']+=1
    if int(getattr(cfg,'cd_sync_find',-1))==_CDP['n'] and torch.cuda.is_available():   # v9.8 SYNC FINDER: every CPU wait in one conduit call, by code line
        import warnings as _wn
        prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(1)
        try:
            with _wn.catch_warnings(record=True) as wl_:
                _wn.simplefilter('always')
                with _rp_tags(model): r_=_cd_loss(model,cfg,x,dom)
        finally: torch.cuda.set_sync_debug_mode(prev_)
        cnt_={}
        for w_ in wl_:
            if 'synchroniz' in str(w_.message).lower(): kk_=(os.path.basename(w_.filename),w_.lineno); cnt_[kk_]=cnt_.get(kk_,0)+1
        print(f"  [sync finder] conduit call {_CDP['n']}: {sum(cnt_.values())} CPU waits for the GPU, by code line:"+''.join(f"\n    {f_}:{l_}  x{c_}" for (f_,l_),c_ in sorted(cnt_.items(),key=lambda z:-z[1])[:25]),flush=True)
        return r_
    if cfg.cd_prof:
        if _CDP['todo'] is None: _CDP['todo']=sorted(int(c_) for c_ in cfg.cd_prof_calls)
        if _CDP['todo'] and _CDP['n']>=_CDP['todo'][0] and not _cd_will_reuse(model,cfg,dom):   # v9b: a GENERATION step
            _CDP['todo'].pop(0); cu_=torch.cuda.is_available()
            try:
                acts_=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cu_ else [])
                if cu_: torch.cuda.synchronize()
                t0_=time.time(); _PRF['on']=True
                try:
                    with torch.profiler.profile(activities=acts_) as pr_:
                        with _rp_tags(model): r_=_cd_loss(model,cfg,x,dom)
                        if cu_: torch.cuda.synchronize()
                finally: _PRF['on']=False
                wt_=time.time()-t0_; ka_=pr_.key_averages()
                gt_=lambda e: (e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total)
                nl_=sum(e_.count for e_ in ka_ if e_.key in ('cudaLaunchKernel','cuLaunchKernel'))
                gpu_=sum((e_.self_device_time_total if hasattr(e_,'self_device_time_total') else e_.self_cuda_time_total) for e_ in ka_)/1e3 if cu_ else 0.0
                print(f"  [conduit profile] GENERATION step (conduit call {_CDP['n']}, domain {dom}): wall {1000*wt_:.0f} ms (profiled) | GPU busy {gpu_:.0f} ms | {nl_} kernel launches")
                rg_=[e for e in ka_ if e.key.startswith('R:')]
                if rg_:
                    rg_.sort(key=lambda e:-(gt_(e) if cu_ else e.cpu_time_total))
                    print("  [conduit profile] NAMED REGIONS (GPU time | CPU time, incl. everything inside):\n"+"\n".join(f"    {e.key[2:]:20} GPU {gt_(e)/1e3:8.1f} ms | CPU {e.cpu_time_total/1e3:8.1f} ms  ({e.count} calls)" for e in rg_[:16]))
                print("  [conduit profile] top operations by "+("GPU" if cu_ else "CPU")+" time:\n"+ka_.table(sort_by=('self_cuda_time_total' if cu_ else 'self_cpu_time_total'),row_limit=14))
                print("  [conduit profile] top operations by CPU time (launch / Python overhead):\n"+ka_.table(sort_by='self_cpu_time_total',row_limit=10))
                return r_
            except Exception as e_: print(f"  [conduit profile] failed ({repr(e_)[:160]})")
    with _rp_tags(model): return _cd_loss(model,cfg,x,dom)
def _cd_loss(model,cfg,x,dom):
    _MIS32['on']=bool(cfg.cd_mis32)
    return _cd_loss_(model,cfg,x,dom)
def _cd_loss_(model,cfg,x,dom):
    """CONDUIT REPLAY (see cd_w): trace -> exactness-checked free replay (frozen teacher) -> KL into the student reading ordinary text."""
    ph_=model.__dict__.get('_sleep_phase')
    if ph_=='rem': return next(model.parameters()).sum()*0.0                  # SLEEP REM: no hippocampal replay (dreams only)
    if ph_=='nrem': return _nrem_loss_v1(model,cfg,dom) if int(cfg.nrem_res)>0 else _nrem_loss(model,cfg,dom)   # SLEEP NREM (v1: reservoir, no traces)
    n0=int(cfg.cd_rows); Ls_=int(cfg.cd_seg); C=int(cfg.cd_chain) if int(cfg.cd_chain)>0 else max(1,x.shape[1]//Ls_)   # CHAINED SUB-SEGMENTS: n0 windows of
    SELB_=_cbuf_take(model,cfg,dom,n0,x.device); n0=n0-len(SELB_); _CBUF['calls']+=1; _CBUF['mix']+=len(SELB_)   # v9f: buffered windows replace fresh ones
    if 0<float(cfg.cd_win_keep)<1.0 and n0>1:                                 # WINDOW-LEVEL SURPRISE GATING: only the most surprising share of the rows is replayed
        k_w=max(1,int(round(n0*float(cfg.cd_win_keep))))
        with torch.no_grad():
            if cfg.cd_win_rand: iw_=torch.randperm(x.shape[0],generator=_cd_g(model,x.device,cfg),device=x.device)[:k_w]
            else:
                was_w=model.training; model.eval(); fs_w=model.feat_scale.clone(); sv_w=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=dom
                try: lw_=F.cross_entropy(model(x[:,:-1].long(),dom).float().reshape(-1,256),x[:,1:].long().reshape(-1),reduction='none').view(x.shape[0],-1).mean(1)
                finally: model.__dict__['_s0_dom']=sv_w; model.feat_scale.copy_(fs_w); model.train(was_w)
                iw_=torch.topk(lw_,min(k_w,x.shape[0])).indices
        x=x[iw_]; n0=min(k_w,x.shape[0])
    Xw=_cd_slices(model,cfg,x,n0,Ls_*C)                                       #   sub-segment its own trace; all replayed together; the student reads
    if Xw.shape[1]!=Ls_*C: C=1                                                #   the JOINED windows (full context), positions weighted by exactness
    n=n0*C; X=Xw.reshape(n,-1); L=X.shape[1]; dev=X.device; M=int(cfg.cd_mark); g_=_cd_g(model,dev,cfg)
    fdf_=_fok_dom(dom); fw_=1.0; _V97['calls']+=1
    if float(cfg.cd_floor_hi)>float(cfg.cd_floor_lo) and fdf_.get('vps') is not None:      # v9e LOW-EXACT FLOOR
        fw_=min(1.0,max(0.0,(float(fdf_['vps'])-float(cfg.cd_floor_lo))/(float(cfg.cd_floor_hi)-float(cfg.cd_floor_lo))))
        fdf_['fcall']=fdf_.get('fcall',0)+1
        if fw_<=0.0 and fdf_['fcall']%max(1,int(cfg.cd_floor_every))!=0:
            _V97['skip']+=1; _V97['fw'].append(0.0); return next(model.parameters()).sum()*0.0
    rc_=model.__dict__.get('_cd_cache')                                       # v9 ROUTE 3: learn again from the last verified batch
    ru_=bool(int(cfg.cd_reuse)>1 and rc_ is not None and rc_['dom']==dom and rc_['k']<int(cfg.cd_reuse)-1 and rc_['best'].shape[0]==n)
    if ru_: rc_['k']+=1; _CDR[0]+=1
    _CDR[1]+=1
    sv=_carry_save(); was=model.training; model.eval(); fs=model.feat_scale.clone(); kh=getattr(model,'_keep_h',False); model._keep_h=False
    _CARRY['fac']=bool(cfg.cd_fac)                                             # FACTORED TRACES (cd_fac)
    _CDT['on']=(_CDT['calls']%20==0); _CDT['calls']+=1
    if _CDT['on']: _CDT['n']+=1; _cdt('_start')
    try:
        with torch.no_grad(), _cd_weights(model):
            if ru_:
                best,tg,ex_,chkd_,X,tries,eps=rc_['best'],rc_['tg'],rc_['ex'],rc_['chk'],rc_['X'],rc_['tries'],rc_['eps']
                pm_=rc_.get('pm',ex_[:,None].expand(n,L))
            else:
                fdq_=_fok_dom(dom); one_=bool(float(cfg.cd_fok_1try)>0 and fdq_['exr']>=float(cfg.cd_fok_1try))
                Gc_=max(1,int(cfg.cd_gen_chunk1) if ((one_ or cfg.cd_repair) and int(cfg.cd_gen_chunk1)>0) else int(cfg.cd_gen_chunk)); chs_=[(a_,min(n,a_+Gc_)) for a_ in range(0,n,Gc_)]   # v9c
                snap=_cd_cat([_cd_encode(model,X[a_:b_]) for a_,b_ in chs_]) if len(chs_)>1 else _cd_encode(model,X)
                _cdt('encode')
                ec_=model.__dict__.get('_cd_epsc')
                if cfg.cd_repair: _cd_fb(model,snap)
                if float(cfg.cd_eps)>0: eps=float(cfg.cd_eps)
                elif ec_ is not None and ec_[0]%max(1,int(cfg.cd_eps_every))!=0: eps=ec_[1]; model.__dict__['_cd_epsc']=(ec_[0]+1,eps)
                elif cfg.cd_repair and cfg.cd_first_block:                         # v9e: exact iff no key row differs; tolerance = key-row noise
                    a_,b_=chs_[0]; d0_=_cd_kdist(model,_cd_encode_chk(model,cfg,X[a_:b_]),_cd_rows(snap,torch.arange(a_,b_,device=dev),n),b_-a_,L)
                    model.__dict__['_cd_ktol']=max(float(cfg.cd_keytol),10.0*float(d0_.max())); eps=0.5; model.__dict__['_cd_epsc']=((ec_[0] if ec_ else 0)+1,eps)
                else:
                    a_,b_=chs_[0]; e1_=_cd_encode(model,X[a_:b_]); rf_=_cd_rows(snap,torch.arange(a_,b_,device=dev),n); e0_=_cd_mis(e1_,rf_,b_-a_)
                    eps=max(float(cfg.cd_eps_floor),10.0*float(e0_.max())); model.__dict__['_cd_epsc']=((ec_[0] if ec_ else 0)+1,eps)
                    if cfg.cd_repair: model.__dict__['_cd_ktol']=max(float(cfg.cd_keytol),10.0*float(_cd_kdist(model,e1_,rf_,b_-a_,L).max()))
                _cdt('tolerance')
                best=None; beste=torch.full((n,),float('inf'),device=dev); tries=0
                tgb=None; chkd_=torch.ones(n,dtype=torch.bool,device=dev)            # v7: which pieces were verified by the re-encode check
                JV_=torch.full((n,),L,dtype=torch.long,device=dev)                 # v9e: verified prefix length per piece
                if cfg.cd_inc or cfg.cd_greedy:                                    # v5: generation logits = teacher targets; greedy first try
                    best=torch.zeros(n,L,dtype=torch.long,device=dev); tgb=torch.empty(n,L,256,device=dev); K_=max(1,int(cfg.cd_tries))
                    for a_,b_ in chs_:
                        m_=b_-a_; st_=_cd_rows(snap,torch.arange(a_,b_,device=dev),n)
                        if cfg.cd_greedy:
                            Ys,lp_=_cd_gen(model,cfg,st_,m_,L,dom,M,0.0,g_); tries+=m_
                            best[a_:b_]=Ys; tgb[a_:b_]=lp_; beste[a_:b_]=_cd_mis(_cd_encode(model,Ys),st_,m_)
                            for _ in range(K_-1):                                      # sampled retries: still-inexact segments only
                                fl_=(beste[a_:b_]>eps).nonzero().flatten()
                                if len(fl_)==0: break
                                sf_=_cd_rows(st_,fl_,m_); Ys,lp_=_cd_gen(model,cfg,sf_,len(fl_),L,dom,M,float(cfg.cd_temp),g_); tries+=len(fl_)
                                e_=_cd_mis(_cd_encode(model,Ys),sf_,len(fl_)); bt_=e_<beste[a_+fl_]; ix_=a_+fl_[bt_]
                                best[ix_]=Ys[bt_]; tgb[ix_]=lp_[bt_]; beste[ix_]=e_[bt_]
                        elif cfg.cd_repair:                                            # v9e RECALL FIX: greedy try -> key-row check -> repair
                            fd_=_fok_dom(dom)
                            hn_=_cd_hints(model,cfg,st_,m_,L,dom,M,X[a_:b_]) if (float(cfg.cd_hint_frac)>0 or float(cfg.cd_hint_adapt)>0) else None   # cg12b
                            Ys,lp_=_cd_gen(model,cfg,st_,m_,L,dom,M,0.0,g_,hint=hn_); tries+=m_; cf_=_CDCONF['c'].clone()
                            if cfg.cd_fok:                                             # the GATE on the greedy try (as v7)
                                cnf_=cf_>=fd_['thr']; cal_=cnf_&(torch.rand(m_,generator=g_,device=dev)<float(cfg.cd_fok_chk)); chk_=(~cnf_)|cal_
                            else: cnf_=torch.zeros(m_,dtype=torch.bool,device=dev); cal_=cnf_; chk_=~cnf_
                            best[a_:b_]=Ys; tgb[a_:b_]=lp_; beste[a_:b_]=0.0; chkd_[a_:b_]=chk_
                            pc_=chk_.nonzero().flatten()
                            if len(pc_)>0:
                                sP_=_cd_rows(st_,pc_,m_); e_,j_=_cd_vcheck(model,cfg,Ys[pc_].contiguous(),sP_,len(pc_),L,eps)
                                beste[a_+pc_]=e_; JV_[a_+pc_]=j_; _V97['chk'].append(len(pc_))
                                if cfg.cd_fok:
                                    fd_['c'].append(cf_[pc_].flatten()); fd_['e'].append((e_<=eps).flatten()); fd_['nc']+=len(pc_)
                                    cl_=cal_[pc_]; FOKS.setdefault('estn',[]).append((cl_&(e_>eps)).sum()); FOKS.setdefault('estd',[]).append(cl_.sum())
                                act_=torch.ones(len(pc_),dtype=torch.bool,device=dev)
                                for _ in range(max(0,int(cfg.cd_repair_n))):           # REPAIR from the first wrong byte (excluded), greedy after it
                                    nd_=((j_<L)&act_).nonzero().flatten()
                                    if len(nd_)==0: break
                                    rr_=pc_[nd_]; fl_=j_[nd_]; fo_=best[a_+rr_]; bn_=fo_.gather(1,fl_[:,None]).squeeze(1)
                                    sR_=_cd_rows(st_,rr_,m_); Yr,lpr=_cd_gen(model,cfg,sR_,len(rr_),L,dom,M,0.0,g_,forced=fo_,flen=fl_,ban=bn_,hint=((hn_[0][rr_],hn_[1][rr_]) if hn_ is not None else None))
                                    tries+=len(rr_); er_,jr_=_cd_vcheck(model,cfg,Yr,sR_,len(rr_),L,eps); bt_=jr_>fl_; ix_=a_+rr_   # v9.8: selects on the GPU
                                    best[ix_]=torch.where(bt_[:,None],Yr,best[ix_]); tgb[ix_]=torch.where(bt_[:,None,None],lpr,tgb[ix_])   #   (no boolean-mask
                                    beste[ix_]=torch.where(bt_,er_,beste[ix_]); JV_[ix_]=torch.where(bt_,jr_,JV_[ix_])                   #   indexing: no CPU wait)
                                    j_=j_.clone(); j_[nd_]=torch.where(bt_,jr_,j_[nd_]); act_=torch.zeros_like(act_); act_[nd_]=bt_
                                    _V97['rep']+=len(rr_); _V97['imp'].append(bt_.sum()); del sR_
                                del sP_
                            if cfg.cd_fok:
                                nch_=~chk_; FOKS['skip'].append(nch_.float().mean()); FOKS['k'].append(1)
                                FOKS.setdefault('trn',[]).append(((best[a_:b_]!=X[a_:b_]).any(1)&nch_).sum()); FOKS.setdefault('trd',[]).append(nch_.sum())
                                fd_['n']+=1
                                if fd_['nc']>=int(cfg.cd_fok_min) and fd_['n']%5==1: _fok_thr_gpu(fd_,cfg)
                        else:                                                          # all tries sampled, in one batch
                            fd_=_fok_dom(dom); Kq_=1 if (float(cfg.cd_fok_1try)>0 and fd_['exr']>=float(cfg.cd_fok_1try)) else K_   # v7: adaptive tries
                            stK=_cd_rows(st_,torch.arange(m_,device=dev).repeat(Kq_),m_)
                            Ys,lp_=_cd_gen(model,cfg,stK,m_*Kq_,L,dom,M,float(cfg.cd_temp),g_); tries+=m_*Kq_
                            cf_=_CDCONF['c'].reshape(Kq_,m_); ar_=torch.arange(m_,device=dev); Ysr=Ys.reshape(Kq_,m_,L); lpr=lp_.reshape(Kq_,m_,L,-1)
                            cmx_,ci_=cf_.max(0)
                            if cfg.cd_fok:                                             # v7 GATE: confident -> accepted unchecked (+ random calibration checks)
                                cnf_=cmx_>=fd_['thr']; cal_=cnf_&(torch.rand(m_,generator=g_,device=dev)<float(cfg.cd_fok_chk)); chk_=(~cnf_)|cal_
                            else: cnf_=torch.zeros(m_,dtype=torch.bool,device=dev); cal_=cnf_; chk_=~cnf_
                            best[a_:b_]=Ysr[ci_,ar_]; tgb[a_:b_]=lpr[ci_,ar_]; beste[a_:b_]=0.0; chkd_[a_:b_]=chk_
                            pc_=chk_.nonzero().flatten()
                            if len(pc_)>0:
                                rows_=(torch.arange(Kq_,device=dev)[:,None]*m_+pc_[None]).flatten(); sK_=_cd_rows(stK,rows_,m_*Kq_)
                                e_=_cd_mis(_cd_encode(model,Ys[rows_]),sK_,len(rows_)).reshape(Kq_,len(pc_)); be_,bi_=e_.min(0)
                                best[a_+pc_]=Ysr[bi_,pc_]; tgb[a_+pc_]=lpr[bi_,pc_]; beste[a_+pc_]=be_; del sK_
                                if cfg.cd_fok and cfg.cd_nosync:                       # v9d: calibration data and statistics stay on the GPU
                                    fd_['c'].append(cf_[:,pc_].flatten()); fd_['e'].append((e_<=eps).flatten()); fd_['nc']+=Kq_*len(pc_)
                                    cl_=cal_[pc_]; okc_=(e_[ci_[pc_],torch.arange(len(pc_),device=dev)]<=eps)
                                    FOKS.setdefault('estn',[]).append((cl_&~okc_).sum()); FOKS.setdefault('estd',[]).append(cl_.sum())
                                elif cfg.cd_fok:
                                    fd_['c']+=cf_[:,pc_].flatten().tolist(); fd_['e']+=(e_<=eps).float().flatten().tolist()
                                    fd_['c']=fd_['c'][-40000:]; fd_['e']=fd_['e'][-40000:]
                                    cl_=cal_[pc_]
                                    if bool(cl_.any()):                                # estimated wrong-accept: calibration pieces whose most confident try failed
                                        ok_=(e_[ci_[pc_],torch.arange(len(pc_),device=dev)]<=eps)[cl_]; FOKS['est'].append(float(1.0-ok_.float().mean()))
                            if cfg.cd_fok and cfg.cd_nosync:                           # v9d: skipped share / true wrong accepts as tensors; threshold on the GPU
                                nch_=~chk_; FOKS['skip'].append(nch_.float().mean()); FOKS['k'].append(Kq_)
                                FOKS.setdefault('trn',[]).append(((best[a_:b_]!=X[a_:b_]).any(1)&nch_).sum()); FOKS.setdefault('trd',[]).append(nch_.sum())
                                fd_['n']+=1
                                if fd_['nc']>=int(cfg.cd_fok_min) and fd_['n']%5==1: _fok_thr_gpu(fd_,cfg)
                            elif cfg.cd_fok:
                                ac_=(~chk_).nonzero().flatten(); FOKS['skip'].append(float((~chk_).float().mean())); FOKS['k'].append(Kq_)
                                if len(ac_)>0: FOKS['true'].append(float((best[a_+ac_]!=X[a_+ac_]).any(1).float().mean()))   # TRUE bytes: diagnostic
                                fd_['n']+=1
                                if len(fd_['c'])>=int(cfg.cd_fok_min) and fd_['n']%5==1:
                                    c_=np.array(fd_['c']); e2_=np.array(fd_['e']); o_=np.argsort(-c_); bad_=np.cumsum(1.0-e2_[o_])/np.arange(1,len(o_)+1)
                                    okk_=np.where((bad_<=float(cfg.cd_fok_far))&(np.arange(1,len(o_)+1)>=200))[0]
                                    fd_['thr']=float(c_[o_][okk_.max()]) if len(okk_) else float('inf')
                            del stK
                elif cfg.cd_par:                                                   # PARALLEL TRIES: all rows x tries generated as ONE batch (one call per byte)
                    K_=max(1,int(cfg.cd_tries)); best=torch.zeros(n,L,dtype=torch.long,device=dev)
                    for a_,b_ in chs_:                                             #   (in chunks of cd_gen_chunk segments)
                        m_=b_-a_; stK=_cd_rows(snap,torch.arange(a_,b_,device=dev).repeat(K_),n); Y=torch.full((m_*K_,1),M,dtype=torch.long,device=dev)
                        for _ in range(L):
                            _CARRY.update(on=True,nr=m_*K_,st=stK,new={},rows=None)
                            lg=model(Y,dom)[:,-1].float()/float(cfg.cd_temp); Y=torch.cat([Y,torch.multinomial(F.softmax(lg,-1),1,generator=g_)],1)
                        Ys=Y[:,1:]; e_=_cd_mis(_cd_encode(model,Ys),stK,m_*K_).reshape(K_,m_); be_,bi_=e_.min(0)
                        best[a_:b_]=Ys.reshape(K_,m_,L)[bi_,torch.arange(m_,device=dev)]; beste[a_:b_]=be_; tries+=m_*K_; del stK
                for k_ in (range(0) if (cfg.cd_par or cfg.cd_inc or cfg.cd_greedy) else range(max(1,int(cfg.cd_tries)))):
                    rows=torch.arange(n,device=dev) if k_==0 else (beste>eps).nonzero().flatten()
                    if len(rows)==0: break
                    st_=_cd_rows(snap,rows,n); m=len(rows); Y=torch.full((m,1),M,dtype=torch.long,device=dev)
                    for _ in range(L):
                        _CARRY.update(on=True,nr=m,st=st_,new={},rows=None)
                        lg=model(Y,dom)[:,-1].float()/float(cfg.cd_temp); Y=torch.cat([Y,torch.multinomial(F.softmax(lg,-1),1,generator=g_)],1)
                    Ys=Y[:,1:]; tries+=m; e_=_cd_mis(_cd_encode(model,Ys),st_,m)
                    if best is None: best=Ys.clone()
                    bt_=e_<beste[rows]; best[rows[bt_]]=Ys[bt_]; beste[rows[bt_]]=e_[bt_]
                _cdt('generate + check')
                ex_=beste<=eps
                if cfg.cd_repair:                                                  # v9e: the positions that teach
                    pm_=(torch.arange(L,device=dev)[None]<JV_[:,None]) if cfg.cd_partial else ex_[:,None].expand(n,L)
                    tcp_=torch.cumprod((best==X).long(),1).bool()                      #   (diagnostic) the TRULY correct prefix
                    _V97['vp'].append(pm_.float().mean()); _V97['pn'].append((pm_&tcp_).sum()); _V97['pd'].append(pm_.sum())
                    _V97['ag'].append(((JV_==tcp_.sum(1))&chkd_).sum()); _V97['agd'].append(chkd_.sum())
                    vm_=pm_.float().mean(); fdv_=_fok_dom(dom); fdv_['vpsT']=vm_ if fdv_.get('vpsT') is None else 0.9*fdv_['vpsT']+0.1*vm_
                    fdv_['vcall']=fdv_.get('vcall',0)+1
                    if fdv_['vcall']%10==1: fdv_['vps']=float(fdv_['vpsT'])
                fdd_=_fok_dom(dom)
                if cfg.cd_nosync:                                                  # v9d: EMA on the device, read back every 10 conduit calls
                    exm_=ex_.float().mean(); fdd_['exrT']=exm_ if fdd_['exrT'] is None else 0.9*fdd_['exrT']+0.1*exm_; fdd_['ncall']+=1
                    if fdd_['ncall']%10==1: fdd_['exr']=float(fdd_['exrT'])
                else: fdd_['exr']=float(ex_.float().mean()) if fdd_['exr']==0.0 else 0.9*fdd_['exr']+0.1*float(ex_.float().mean())
                Yin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),best[:,:-1]],1); tg=[]
                for a_,b_ in (chs_ if tgb is None else ()):
                    _CARRY.update(on=True,nr=b_-a_,st=_cd_rows(snap,torch.arange(a_,b_,device=dev),n) if len(chs_)>1 else snap,new={},rows=None)
                    tg.append(F.log_softmax(model(Yin[a_:b_],dom).float(),-1))       # teacher WITH the trace (replay mode): tg[:,j] predicts best[:,j]
                tg=torch.cat(tg) if tgb is None else tgb
                if (cfg.cd_reuse_mode=='loop' and int(cfg.cd_reuse_k)>0) or int(cfg.cd_store_cap)>0: model.__dict__['_snap_last']=snap   # kept for the maintenance loop
                del snap
                _cdt('teacher targets')
    finally:
        _INC['on']=False; _CARRY.pop('fac',None); _carry_load(sv); model.feat_scale.copy_(fs); model.train(was); model._keep_h=kh
    if int(cfg.cd_reuse)>1 and not ru_: model.__dict__['_cd_cache']={'dom':dom,'k':0,'best':best,'tg':tg,'ex':ex_,'chk':chkd_,'X':X,'tries':tries,'eps':eps,'pm':(pm_ if cfg.cd_repair else None)}
    bj_=best.reshape(n0,-1); tgj_=tg.reshape(n0,-1,tg.shape[-1])
    if int(cfg.lm_rank)>=0: model.__dict__['_last_rep']=(bj_.detach(),int(dom))     # (dream merge: the visit's verified replays, for its drift check)               # joined windows (C = 1: unchanged)
    if float(cfg.cd_buf)>0 and dom in _CBUF['d']:                               # v9f: this step's verified windows into the buffer
        eb_=_CBUF['d'][dom]; pmb_=(pm_ if cfg.cd_repair else ex_[:,None].expand(n,L))&chkd_[:,None]
        for r_ in range(min(int(cfg.cd_buf_per_step),n0)):
            if len(eb_['x'])>=int(cfg.cd_buf_cap): break
            eb_['x'].append(bj_[r_].clone()); eb_['t'].append(tgj_[r_].half()); eb_['m'].append(pmb_.reshape(n0,-1)[r_].clone())
    rz_=None
    if cfg.cd_runs and C>1:                                                    # v7 EXACT RUNS: piece j starts a run after an inexact piece j-1
        exw_=ex_.view(n0,C); st0_=torch.zeros(n0,C,dtype=torch.bool,device=dev); st0_[:,1:]=(~exw_[:,:-1]) if cfg.cd_runs_any else (exw_[:,1:]&~exw_[:,:-1])
        rb_=st0_.repeat_interleave(L,1)&(torch.arange(C*L,device=dev)%L==0)[None]   # (n0, C*L): first byte p of such a run
        if cfg.cd_runs_mask:                                                   # v9 FIX 1: the run is its own clean segment from byte p
            _RESET['m']=rb_[:,:-1].contiguous()                                #   memory reset AT input byte p
            ps_=torch.arange(C*L-1,device=dev)[None].expand(n0,-1)
            _SEG['s']=torch.cummax(torch.where(rb_[:,:-1],ps_,torch.zeros_like(ps_)),1).values   #   local filter + n-grams stop at p
            sk_=int(cfg.cd_run_excl)
        else:
            sk_=int(cfg.cd_run_skip); rs_=torch.zeros_like(rb_); rs_[:,sk_:]=rb_[:,:C*L-sk_]
            _RESET['m']=rs_[:,:-1].contiguous()                                #   memory reset at input byte p + skip
        rz_=torch.zeros(n0,C*L,dtype=torch.bool,device=dev)
        for o_ in range(sk_+1): rz_[:,o_:]|=rb_[:,:C*L-o_]                     #   target bytes p .. p + skip do not teach (fix 1: p .. p + excl)
    sv2=_carry_save(); suh=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
    if cfg.cd_ngp_mask and not cfg.cd_ngp_frozen:                              # v8b: n-gram rows learn only at positions that teach (input t -> target t+1)
        wm_=torch.where(ex_,torch.ones(n,device=dev),torch.full((n,),float(cfg.cd_inexact_w),device=dev)).view(n0,C).repeat_interleave(L,1)[:,1:]
        if rz_ is not None: wm_=wm_*(~rz_[:,1:]).float()
        _NGPD['m']=wm_>0
    KLB_=0.0; NPB_=0
    try:
        _HCAP['on']=((float(cfg.cd_gate_keep)>0 and str(cfg.cd_gate_mode)=='prot') or bool(cfg.dream_sim)); _HCAP['h']=None
        try: lp=F.log_softmax(model(bj_[:,:-1],dom).float(),-1)
        finally: _HCAP['on']=False
        hcap_=_HCAP['h']; _HCAP['h']=None                                       # (protected gate: the student's head input)
        if cfg.dream_sim and hcap_ is not None:                                 # SWIL: the current learning's rank-1 signature (u = mean z, v = mean dL/dz)
            with torch.no_grad():
                _sig_update(model,dom,hcap_,lp.detach(),tgj_[:,1:])
        if False: lp=F.log_softmax(model(bj_[:,:-1],dom).float(),-1)            # student: the replay as ORDINARY text, no memory, no marker
        if SELB_:                                                               # v9f: buffered windows, each with ITS OWN domain
            cur_=_XDG.get('cur')
            try:
                for d_ in sorted({d_ for d_,_ in SELB_}):
                    ii_=[i_ for e_,i_ in SELB_ if e_==d_]; eb_=_CBUF['d'][d_]
                    xb_=torch.stack([eb_['x'][i_] for i_ in ii_]); tb_=torch.stack([eb_['t'][i_] for i_ in ii_]).float()[:,1:]; mb_=torch.stack([eb_['m'][i_] for i_ in ii_]).float()[:,1:]
                    _XDG['cur']=d_; _BUFDET['on']=bool(cfg.cd_buf_detach)
                    try: lpb_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
                    finally: _BUFDET['on']=False
                    KLB_=KLB_+((tb_.exp()*(tb_-lpb_)).sum(-1)*mb_).sum(); NPB_+=mb_.numel()
                    _CBUF['src'][d_]=_CBUF['src'].get(d_,0)+len(ii_)
            finally: _XDG['cur']=cur_
    finally: _carry_load(sv2); _uh_load(model,suh); _NGPD['on']=False; _NGPD['m']=None; _RESET['m']=None; _SEG['s']=None
    _cdt('student forward'); _CDT['on']=False
    iw_=float(cfg.cd_inexact_w); w_=torch.where(ex_,torch.ones(n,device=dev),torch.full((n,),iw_,device=dev)); tg2=tgj_[:,1:]
    wp_=w_.view(n0,C).repeat_interleave(L,1)[:,1:]                              # per-position weight = its sub-segment's
    if cfg.cd_repair: wp_=pm_.float().reshape(n0,C*L)[:,1:]                         # v9e: per-position (verified prefix) weights
    if rz_ is not None: wp_=wp_*(~rz_[:,1:]).float(); FOKS['runs'].append(wp_.mean().detach())   # v9.8: no CPU wait (read at the visit end)   # v7: targets whose context crosses a run start
    if float(cfg.cd_gate_keep)>0 and C>1:                                       # SURPRISE-GATED REPLAY (test): low-surprise pieces do not teach
        with torch.no_grad():
            T1g_=lp.shape[1]; pg_=((torch.arange(T1g_,device=lp.device)+1)//max(1,L)).clamp(max=C-1)
            ceg_=-lp.gather(-1,bj_[:,1:,None]).squeeze(-1)
            sg_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,ceg_.float()); cg_=torch.zeros_like(sg_).index_add_(1,pg_,torch.ones_like(ceg_,dtype=torch.float))
            sur_=(sg_/cg_.clamp_min(1)).flatten(); k_=max(1,int(sur_.numel()*float(cfg.cd_gate_keep)))
            ksur_=torch.zeros_like(sur_,dtype=torch.bool); ksur_[torch.topk(sur_,k_).indices]=True
            gm_=str(cfg.cd_gate_mode); gg_=_cd_g(model,lp.device,cfg)
            if gm_ in ('rho','rhopos'):                                         # REDUCIBLE SIGNAL ('rhopos': every piece with a POSITIVE signal, no fixed share) (data-averaged trace benefit, RHO-LOSS form):
                yr_=bj_[:,1:,None]; vm_=(wp_>0).float()                         #   log q(y) - log p(y) at the VERIFIED byte = student surprise
                rr_=(tg2.gather(-1,yr_)-lp.gather(-1,yr_)).squeeze(-1)*vm_      #   minus teacher surprise (teacher = with the trace, pre-blend)
                srh_=(torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,rr_.float())/
                      torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,vm_).clamp_min(1)).flatten()
                keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.topk(srh_,k_).indices]=True
                if gm_=='rhopos': keep_=srh_>0
                GR_=model.__dict__.setdefault('_grho',[0,0.0,0.0]); GR_[0]+=1; GR_[1]=GR_[1]+(keep_&ksur_).sum().float()/max(1,k_)   # (device tensors:
                GR_[2]=GR_[2]+(srh_<0).float().mean()                                                                           #  no CPU wait)
            elif gm_ in ('kl','prot','kl0'):                                      # THE OBJECTIVE ITSELF: per-piece KL to the target actually taught
                qg_=tg2.exp()                                                   #   ('kl0': to the teacher's OWN distribution, no evidence blend)
                if float(cfg.cd_evid_w)>0 and gm_!='kl0':                                      #   (evidence blend applied, as the loss will see it)
                    yg_=bj_[:,1:]; dg_=(tg2.argmax(-1)!=yg_).to(qg_.dtype).unsqueeze(-1)*float(cfg.cd_evid_w)
                    qg_=(1-dg_)*qg_+dg_*F.one_hot(yg_,qg_.shape[-1]).to(qg_.dtype)
                klg_=(qg_*(torch.log(qg_.clamp_min(1e-12))-lp)).sum(-1)*(wp_>0).float()
                skl_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,klg_.float()).flatten()
                kkl_=torch.zeros_like(sur_,dtype=torch.bool); kkl_[torch.topk(skl_,k_).indices]=True; keep_=kkl_
                if gm_=='prot':                                                 # PROTECTED GATE: KL x context novelty to the old domains,
                    M_=_prot_minv(model,cfg,lp.device) if hcap_ is not None else None   #   h^T (A_old + lam)^-1 h  (the K-FAC factorisation)
                    if M_ is not None:
                        hz_=hcap_.float(); nov_=torch.einsum('btd,de,bte->bt',hz_,M_,hz_)/hz_.shape[-1]
                        spr_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,(klg_*nov_).float()).flatten()
                        keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.topk(spr_,k_).indices]=True
                        GP_=model.__dict__.setdefault('_gprot',[0,0.0,0.0,0.0]); GP_[0]+=1; GP_[1]+=float((keep_&kkl_).sum())/max(1,k_)
                        GP_[2]+=float((keep_&ksur_).sum())/max(1,k_)
                        nvp_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,nov_.float()*(wp_>0).float()).flatten()
                        GP_[3]+=float(nvp_[keep_].mean()/nvp_.mean().clamp_min(1e-12))
                if gm_ in ('kl','kl0'): GK_=model.__dict__.setdefault('_gkl',[0,0.0,0.0]); GK_[0]+=1; GK_[1]+=float((keep_&ksur_).sum())/max(1,k_)
                if gm_ in ('kl','kl0'): GK_[2]+=float(skl_[ksur_].sum()/skl_.sum().clamp_min(1e-12))
            elif gm_=='rand':
                keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.randperm(sur_.numel(),generator=gg_,device=lp.device)[:k_]]=True
            elif gm_=='kpp':                                                    # ORTHOGONALITY / DIVERSITY GATE (BADGE-style k-means++)
                gp_=(lp.exp()-tg2.exp())*wp_.unsqueeze(-1)                    #   teaching gradient w.r.t. the logits, per position
                E_=torch.zeros(lp.shape[0],C,gp_.shape[-1],device=lp.device).index_add_(1,pg_,gp_.float()).reshape(-1,gp_.shape[-1])
                D2_=(E_*E_).sum(1); keep_=torch.zeros_like(sur_,dtype=torch.bool)
                for _ in range(k_):
                    tot_=float(D2_.sum())
                    if not tot_>0: break
                    j_=int(torch.multinomial(D2_/tot_,1,generator=gg_)); keep_[j_]=True
                    D2_=torch.minimum(D2_,((E_-E_[j_])**2).sum(1)); D2_[j_]=0.0
                nk_=int(keep_.sum())
                if nk_<k_:                                                      #   (all signal used up: fill at random)
                    rest_=(~keep_).nonzero().flatten(); keep_[rest_[torch.randperm(rest_.numel(),generator=gg_,device=lp.device)[:k_-nk_]]]=True
                GS_=model.__dict__.setdefault('_gsel',[0,0.0,0.0,0.0])
                sigp_=D2_.new_zeros(1); en_=(E_*E_).sum(1)
                GS_[0]+=1; GS_[1]+=float((keep_&ksur_).sum())/max(1,k_); GS_[2]+=float(en_[keep_].sum()/en_.sum().clamp_min(1e-12)); GS_[3]+=float(en_[ksur_].sum()/en_.sum().clamp_min(1e-12))
            else: keep_=ksur_
            wp_=wp_*keep_.view(lp.shape[0],C)[:,pg_].float()
    if int(cfg.cd_store_cap)>0 or int(cfg.nrem_res)>0: model.__dict__['_tqy']=tg2.gather(-1,bj_[:,1:,None]).squeeze(-1).detach()   # (store: pre-blend teacher log q(y))
    if float(cfg.cd_evid_w)>0:                                                  # EVIDENCE TARGETS: blend the verified replayed byte into the teacher target
        yb_=bj_[:,1:]; ew_=float(cfg.cd_evid_w)
        model.__dict__['_dis_last']=(tg2.argmax(-1)!=yb_)                     # (pre-blend disagreement: the reuse fragility measure)
        dis_=torch.ones_like(yb_,dtype=torch.bool) if cfg.cd_evid_all else (tg2.argmax(-1)!=yb_)
        pt_=tg2.exp(); oh_=F.one_hot(yb_,pt_.shape[-1]).to(pt_.dtype); lam_=(ew_*dis_.to(pt_.dtype)).unsqueeze(-1)
        if float(cfg.cd_evid_adapt)>0:                                          # ADAPTIVE: the piece's share of disagreeing positions raises the blend
            T1e_=yb_.shape[1]; pe_=((torch.arange(T1e_,device=yb_.device)+1)//max(1,L)).clamp(max=C-1)
            dsum_=torch.zeros(yb_.shape[0],C,device=yb_.device).index_add_(1,pe_,dis_.float()); dcnt_=torch.zeros_like(dsum_).index_add_(1,pe_,torch.ones_like(dis_,dtype=torch.float))
            dshare_=(dsum_/dcnt_.clamp_min(1))[:,pe_]
            lam_=((ew_+(float(cfg.cd_evid_adapt)-ew_)*dshare_)*dis_.float()).clamp(0,1).unsqueeze(-1).to(pt_.dtype)
        tg2=torch.log(((1-lam_)*pt_+lam_*oh_).clamp_min(1e-12))
        EV_=model.__dict__.setdefault('_evid',[0.0,0.0]); EV_[0]+=(dis_.float()*wp_).sum().detach(); EV_[1]+=wp_.sum().detach()
    klp_=(tg2.exp()*(tg2-lp)).sum(-1)
    KLR_=0.0; NPR_=0
    if int(cfg.cd_reuse_k)>0 and cfg.cd_reuse_mode!='loop':                     # SPACED REPLAY REUSE
        RB_=model.__dict__.setdefault('_rbuf',{}); st_r=model.__dict__.get('_rstep',0)+1; model.__dict__['_rstep']=st_r
        if RB_.get('dom')!=dom: RB_.clear(); RB_['dom']=dom; RB_['e']=[]
        with torch.no_grad():
            T1r_=tg2.shape[1]; prr_=((torch.arange(T1r_,device=tg2.device)+1)//max(1,L)).clamp(max=C-1)
            dis0_=model.__dict__.pop('_dis_last',None)                          # fragility from the PRE-evidence teacher (the blend always agrees)
            dz_=(dis0_ if dis0_ is not None else (tg2.argmax(-1)!=bj_[:,1:])).float()*(wp_>0).float()
            fp_=(torch.zeros(dz_.shape[0],C,device=dz_.device).index_add_(1,prr_,dz_)>0); frag_=fp_.float().mean(1)
            for r_ in range(bj_.shape[0]):
                RB_['e'].append({'s':st_r,'x':bj_[r_].clone(),'t':tg2[r_].half(),'w':wp_[r_].clone(),'fp':fp_[r_][prr_].clone(),'f':(float(frag_[r_]) if cfg.cd_reuse_mode=='mix' else frag_[r_])})
            RB_['e']=[e_ for e_ in RB_['e'] if st_r-e_['s']<=3*int(cfg.cd_reuse_lag)]
        el_=[e_ for e_ in RB_['e'] if st_r-e_['s']>=int(cfg.cd_reuse_lag)]
        if el_:
            k_r=min(int(cfg.cd_reuse_k),len(el_)); gr_=_cd_g(model,dev,cfg)
            if cfg.cd_reuse_mode=='mix':
                kp_=k_r//2; srt_=sorted(range(len(el_)),key=lambda i_:-el_[i_]['f']); pick_=srt_[:kp_]
                rest_=[i_ for i_ in range(len(el_)) if i_ not in set(pick_)]; ru2_=torch.randperm(len(rest_),generator=gr_,device=dev)[:k_r-kp_].tolist()
                pick_=pick_+[rest_[i_] for i_ in ru2_]
            else: pick_=model.__dict__.setdefault('_ru_rs',np.random.RandomState(int(cfg.seed)+97)).permutation(len(el_))[:k_r].tolist()   # (CPU draw: no GPU wait)
            sel_=[el_[i_] for i_ in pick_]; idset_={id(e_) for e_ in sel_}; RB_['e']=[e_ for e_ in RB_['e'] if id(e_) not in idset_]
            xr_=torch.stack([e_['x'] for e_ in sel_]); trr_=torch.stack([e_['t'] for e_ in sel_]).float(); wr_=torch.stack([e_['w'] for e_ in sel_]).float()
            if cfg.cd_reuse_mode=='upw': wr_=wr_*(1.0+(float(cfg.cd_reuse_upw)-1.0)*torch.stack([e_['fp'] for e_ in sel_]).float())
            sv3_=_carry_save(); suh3_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
            try: lpr_=F.log_softmax(model(xr_[:,:-1],dom).float(),-1)
            finally: _carry_load(sv3_); _uh_load(model,suh3_); _NGPD['on']=False
            KLR_=((trr_.exp()*(trr_-lpr_)).sum(-1)*wr_).sum(); NPR_=wr_.numel()
            RU_=model.__dict__.setdefault('_rustat',[0,0.0,0.0]); RU_[0]+=k_r; _fs_=lambda L_: (torch.stack(L_).sum() if torch.is_tensor(L_[0]) else float(sum(L_))); RU_[1]=RU_[1]+_fs_([e_['f'] for e_ in sel_]); RU_[2]=RU_[2]+_fs_([e_['f'] for e_ in el_])/max(1,len(el_))*k_r   # (device sums)
    if int(cfg.cd_reuse_k)>0 and cfg.cd_reuse_mode=='loop':                     # DEGRADATION-DRIVEN MAINTENANCE LOOP
        LB_=model.__dict__.setdefault('_lbuf',{}); st_l=model.__dict__.get('_lstep',0)+1; model.__dict__['_lstep']=st_l
        if LB_.get('dom')!=dom: LB_.clear(); LB_['dom']=dom; LB_['e']=[]; LB_['st']=[0,0,0,0,0,0.0]
        snp_=model.__dict__.pop('_snap_last',None)
        if snp_ is not None:
            with torch.no_grad():
                for r_ in range(bj_.shape[0]):
                    LB_['e'].append({'due':st_l+int(cfg.cd_reuse_lag),'iv':int(cfg.cd_reuse_lag),'n':0,'second':False,'x':bj_[r_].clone(),'w':wp_[r_].clone(),
                                     'tr':_cd_rows(snp_,torch.arange(r_*C,(r_+1)*C,device=dev),n)})
        due_=[e_ for e_ in LB_['e'] if e_['due']<=st_l]
        if due_:
            due_.sort(key=lambda e_:(not e_['second'],e_['due'])); sel_=due_[:int(cfg.cd_reuse_k)]; k_l=len(sel_)
            with torch.no_grad():
                trs_=_cd_cat([e_['tr'] for e_ in sel_]) if k_l>1 else sel_[0]['tr']
                xb_=torch.stack([e_['x'] for e_ in sel_]); pcs_=xb_.reshape(k_l*C,L)
                Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
                svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
                try:
                    with _cd_weights(model):
                        _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,dom).float(),-1)
                finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
                tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sel_]).float()
                dgp_=(tgl_.argmax(-1)!=ybl_).float()*(wl_>0).float(); dg_=dgp_.sum(1)/(wl_>0).float().sum(1).clamp_min(1)
                if float(cfg.cd_evid_w)>0:
                    ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1); tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
                T1l_=ybl_.shape[1]; pl_=((torch.arange(T1l_,device=dev)+1)//max(1,L)).clamp(max=C-1)
                fpl_=(torch.zeros(k_l,C,device=dev).index_add_(1,pl_,dgp_)>0).float()[:,pl_]; wl2_=wl_*(1.0+(float(cfg.cd_reuse_upw)-1.0)*fpl_)
            sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
            try: lpl_=F.log_softmax(model(xb_[:,:-1],dom).float(),-1)
            finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
            kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KLR_=(kll_*wl2_).sum(); NPR_=wl2_.numel()
            with torch.no_grad():
                klw_=((kll_*wl_).sum(1)/wl_.sum(1).clamp_min(1)).tolist(); dgl_=dg_.tolist(); S_=LB_['st']; drop_=set()
                for e_,d_,kw_ in zip(sel_,dgl_,klw_):
                    S_[0]+=1; S_[5]+=d_; e_['n']+=1
                    if kw_<float(cfg.cd_loop_kldrop): S_[4]+=1; drop_.add(id(e_)); continue
                    if d_>float(cfg.cd_loop_fail):
                        if e_['second']: S_[3]+=1; drop_.add(id(e_)); continue
                        e_['second']=True; e_['due']=st_l+1; S_[2]+=1; continue
                    e_['second']=False
                    if d_<=float(cfg.cd_loop_easy): S_[1]+=1; e_['iv']=e_['iv']*2
                    else: S_[2]+=1; e_['iv']=max(2,e_['iv']//2)
                    if e_['n']>=int(cfg.cd_loop_maxuse): drop_.add(id(e_)); continue
                    e_['due']=st_l+e_['iv']
                LB_['e']=[e_ for e_ in LB_['e'] if id(e_) not in drop_]
    if int(cfg.cd_sig_every)>0:                                                 # REPLAY-SIGNAL DIAGNOSTIC (no training effect): per piece, the learning signal
        sc_=model.__dict__.get('_rsig_n',0); model.__dict__['_rsig_n']=sc_+1    #   (KL to the trace-reading teacher over its teaching positions) and its
        if sc_%int(cfg.cd_sig_every)==0:                                        #   SURPRISE at experience (the plain model's loss on the bytes)
            with torch.no_grad():
                T1_=klp_.shape[1]; pid_=((torch.arange(T1_,device=klp_.device)+1)//max(1,L)).clamp(max=C-1)
                ce_=-lp.gather(-1,bj_[:,1:,None]).squeeze(-1)/LOG2; w_s=wp_.float()
                ks_=torch.zeros(klp_.shape[0],C,device=klp_.device).index_add_(1,pid_,(klp_*w_s).float())
                cs_=torch.zeros_like(ks_).index_add_(1,pid_,(ce_*w_s).float()); ns_=torch.zeros_like(ks_).index_add_(1,pid_,w_s)
                ok_=ns_>0; RS_=model.__dict__.setdefault('_rsig',[])
                RS_.append(torch.stack([(ks_/ns_.clamp_min(1))[ok_],(cs_/ns_.clamp_min(1))[ok_],ns_[ok_]]).half())
    if cfg.cd_kl_norm: kl=(klp_*wp_).sum()/wp_.sum().clamp_min(1.0)               # v9 FIX 2a: averaged over the positions that teach
    else: kl=(klp_*wp_).mean() if iw_==0 else (klp_*wp_).sum()/wp_.sum().clamp_min(1e-6)   # (iw_ = 0: EXACT replays only; fewer exact positions teach less)
    if fw_<1.0: kl=kl*fw_                                                       # v9e LOW-EXACT FLOOR
    if model.__dict__.get('_ctrl_bytes') or cfg.cd_bytes_all:
        ceb_=(-(lp.gather(-1,bj_[:,1:,None]).squeeze(-1))*wp_).mean(); bw_=1.0 if model.__dict__.get('_ctrl_bytes') else float(cfg.cd_bytes_w)
        kl=ceb_ if bw_>=1.0 else bw_*ceb_+(1.0-bw_)*kl   # sleep_ctrl_bytes: next-byte CE on the fresh checked replays only
    if NPB_>0: kl=(kl*wp_.numel()+KLB_)/(wp_.numel()+NPB_)                      # v9f: averaged over generated + buffered windows
    if int(cfg.cd_store_cap)>0:                                                 # BOUNDED LONG-TERM STORE (admission / spaced re-checks / retirement / cap)
        SB_=model.__dict__.setdefault('_sbuf',{'e':[],'st':{}}); st_s=model.__dict__.get('_sstep',0)+1; model.__dict__['_sstep']=st_s
        ST_=SB_['st']; snp_=model.__dict__.pop('_snap_last',None); tqy_=model.__dict__.pop('_tqy',None)
        if snp_ is not None and tqy_ is not None:
            with torch.no_grad():
                rl_=_store_score(tqy_,lp.gather(-1,bj_[:,1:,None]).squeeze(-1),(wp_>0).float(),C,L).tolist()
            rs_=model.__dict__.setdefault('_s_rs',np.random.RandomState(int(cfg.seed)+131)); ka_=int(round(len(rl_)*float(cfg.cd_store_adm)))
            if cfg.cd_store_admit=='rand': top_=set(rs_.permutation(len(rl_))[:ka_].tolist())
            else: top_=set(i_ for i_ in sorted(range(len(rl_)),key=lambda i_:-rl_[i_])[:ka_] if rl_[i_]>0)
            rnd_=set(i_ for i_ in range(len(rl_)) if rs_.rand()<float(cfg.cd_store_rand))-top_
            if float(cfg.sem_share)>0 and (top_|rnd_):                          # SEMANTIC COPIES (trace-free): bytes + evidence-blended teacher targets
                with torch.no_grad():
                    SM_=model.__dict__.setdefault('_sem',{}).setdefault(int(dom),{'x':[],'v':[],'i':[],'p':0}); kb_=int(cfg.bank_targets)
                    for r_ in sorted(top_|rnd_):
                        if float((wp_[r_]>0).float().mean())<float(cfg.sem_minver): continue
                        lt_=torch.cat([tg2[r_],tg2[r_][-1:]],0)                    #   (T positions; the last one repeats)
                        v_,i_=torch.topk(lt_.float(),kb_,dim=-1)
                        it_=(bj_[r_].to(torch.uint8).cpu(),v_.to(torch.float16).cpu(),i_.to(torch.uint8).cpu())
                        if len(SM_['x'])<int(cfg.sem_cap): SM_['x'].append(it_[0]); SM_['v'].append(it_[1]); SM_['i'].append(it_[2])
                        else: j_=SM_['p']%int(cfg.sem_cap); SM_['x'][j_],SM_['v'][j_],SM_['i'][j_]=it_; SM_['p']+=1
                        ST_['sem']=ST_.get('sem',0)+1
            for r_ in sorted(top_|rnd_):
                SB_['e'].append({'dom':int(dom),'due':st_s+int(cfg.cd_reuse_lag),'iv':int(cfg.cd_reuse_lag),'n':0,'x':bj_[r_].clone(),'w':wp_[r_].clone(),
                                 'tr':_cd_rows(snp_,torch.arange(r_*C,(r_+1)*C,device=dev),n),'v':float(rl_[r_]),'v0':float(rl_[r_])})
            ST_['seen']=ST_.get('seen',0)+len(rl_); ST_['adm']=ST_.get('adm',0)+len(top_); ST_['admr']=ST_.get('admr',0)+len(rnd_)
            while len(SB_['e'])>int(cfg.cd_store_cap):                           #   full: evict the MOST CONSOLIDATED (lowest current signal)
                j_=min(range(len(SB_['e'])),key=lambda i_:SB_['e'][i_]['v']); SB_['e'].pop(j_); ST_['evict']=ST_.get('evict',0)+1
        due_=[e_ for e_ in SB_['e'] if e_['due']<=st_s]
        if due_:
            due_.sort(key=lambda e_:e_['due']); sel_=due_[:int(cfg.cd_store_k)]; drop_=set()
            for d_ in sorted({e_['dom'] for e_ in sel_}):
                sd_=[e_ for e_ in sel_ if e_['dom']==d_]; k_l=len(sd_)
                with torch.no_grad():
                    trs_=_cd_cat([e_['tr'] for e_ in sd_]) if k_l>1 else sd_[0]['tr']
                    xb_=torch.stack([e_['x'] for e_ in sd_]); pcs_=xb_.reshape(k_l*C,L)
                    Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
                    svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
                    try:
                        with _cd_weights(model):                                    #   the frozen teacher reads the STORED traces (its own domain)
                            _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,d_).float(),-1)
                    finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
                    tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sd_]).float(); vml_=(wl_>0).float()
                    tqyl_=tgl_.gather(-1,ybl_[...,None]).squeeze(-1)
                    if float(cfg.cd_evid_w)>0:
                        dgp_=(tgl_.argmax(-1)!=ybl_).float()*vml_; ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1)
                        tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
                sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
                try: lpl_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)            #   the student, its own domain's head (old domains too)
                finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
                kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KLR_=KLR_+(kll_*wl_).sum(); NPR_+=wl_.numel()
                with torch.no_grad(): rn_=_store_score(tqyl_,lpl_.gather(-1,ybl_[...,None]).squeeze(-1),vml_,C,L).tolist()
                for e_,r_ in zip(sd_,rn_):
                    e_['n']+=1; ST_['chk']=ST_.get('chk',0)+1; ST_['chk_old']=ST_.get('chk_old',0)+(1 if e_['dom']!=int(dom) else 0)
                    if r_<=max(1e-4,float(cfg.cd_store_retire)*e_['v0']): drop_.add(id(e_)); ST_['ret']=ST_.get('ret',0)+1; ST_['ret_n']=ST_.get('ret_n',0)+e_['n']; continue
                    e_['iv']=e_['iv']*2 if r_<=e_['v'] else max(2,e_['iv']//2)      #   consolidating: wait longer; being forgotten: sooner
                    e_['v']=r_; e_['due']=st_s+e_['iv']
            SB_['e']=[e_ for e_ in SB_['e'] if id(e_) not in drop_]
    if cfg.inf_on and model.__dict__.get('_tqy') is not None: inf_store_add(model,cfg,bj_,int(model.__dict__.get('_cur_dom',dom)),wp_)   # cg20 store (verified)
    if int(cfg.nrem_res)>0 and model.__dict__.get('_tqy') is not None:            # NREM v1 RESERVOIR (uniform over the visit's verified windows)
        with torch.no_grad():
            RV_=model.__dict__.setdefault('_nres',{'dom':None,'e':[],'seen':0,'ret':0,'rep':0})
            td_=int(model.__dict__.get('_cur_dom',dom))                       # the TRUE domain (slots can be shared); windows are taught under their slot 'd'
            if RV_.get('td')!=td_:
                if float(cfg.nrem_mix)>0 and RV_.get('td') is not None and RV_['e']: model.__dict__.setdefault('_nres_bank',{})[RV_['td']]=RV_['e']   # keep it for cross-domain NREM
                model.__dict__.setdefault('_nres_bank',{}).pop(td_,None)
                RV_.update(dom=int(dom),td=td_,e=[],seen=0,ret=0,rep=0,rep_o=0,ret_o=0)
            tq_=model.__dict__['_tqy']; vm_=(wp_>0).float()
            sc_=_store_score(tq_,lp.gather(-1,bj_[:,1:,None]).squeeze(-1),vm_,C,L).tolist()
            rsr_=model.__dict__.setdefault('_nres_rs',np.random.RandomState(int(cfg.seed)+313)); kb_=int(cfg.bank_targets)
            for r_ in range(bj_.shape[0]):
                RV_['seen']+=1; j_=len(RV_['e']) if len(RV_['e'])<int(cfg.nrem_res) else rsr_.randint(RV_['seen'])
                if j_>=int(cfg.nrem_res): continue
                v_,i_=torch.topk(tg2[r_].float(),kb_,dim=-1)
                xr_=bj_[r_].clone()
                if cfg.nrem_real:
                    PR_=model.__dict__.get('_p1win',{}).get(td_)
                    if PR_ and PR_['r']:
                        rw_=PR_['r'][rsr_.randint(len(PR_['r']))]; Wn_=xr_.numel()
                        if rw_.numel()>=Wn_: o_=rsr_.randint(rw_.numel()-Wn_+1); xr_=rw_[o_:o_+Wn_].to(xr_.device,xr_.dtype).clone()
                en_={'d':int(dom),'x':xr_,'w':wp_[r_].clone(),'tv':v_.half(),'ti':i_.to(torch.uint8),'q':tq_[r_].half(),'v0':float(sc_[r_]),'v':float(sc_[r_]),'n':0}
                if j_<len(RV_['e']): RV_['e'][j_]=en_
                else: RV_['e'].append(en_)
    if NPR_>0: kl=(kl*wp_.numel()+KLR_)/(wp_.numel()+NPR_)                      # reuse / maintenance: averaged over fresh + reused windows
    _V97['fw'].append(fw_)
    if cfg.cd_lr_teach and model.__dict__.get('_cdf') is not None:             # v9 FIX 2b: step scaled by the teaching share
        tsh_=max(1e-3,float(wp_.float().mean())); model.__dict__['_cdf']=float(model.__dict__['_cdf'])*tsh_; _CDR[2]+=tsh_; _CDR[3]+=1
    st=model.__dict__.setdefault('_cd_stat',[0.0,0,0.0,0.0,0.0,0.0,0.0])
    if cfg.cd_nosync:                                                          # v9d: accumulated on the device, read at the visit end
        st[0]=st[0]+kl.detach(); st[1]+=1; st[2]=st[2]+ex_.float().mean(); st[3]+=tries/n; st[4]=st[4]+(best==X).float().mean(); st[5]+=eps
        st[6]=st[6]+(ex_==(best==X).all(1)).float().mean()
    else:
        st[0]+=float(kl.detach()); st[1]+=1; st[2]+=float(ex_.float().mean()); st[3]+=tries/n; st[4]+=float((best==X).float().mean()); st[5]+=eps
        st[6]+=float((ex_==(best==X).all(1)).float().mean())                     # exactness test agrees with the TRUE bytes (diagnostic)
    if float(cfg.cd_recon_w)>0 and not model.__dict__.get('_ctrl_bytes'):     # v6: REPLAY TRAINING ON EXACT REPLAYS (live model, no real bytes)
        ix_=(ex_&chkd_).nonzero().flatten()                                    # v7: CHECKED exact replays only (never gate-accepted)
        if len(ix_)>0:
            kr_=int(cfg.cd_recon_rows)
            if cfg.rs_sel and len(ix_)>kr_:                                    # v8: borderline + random among the checked exact replays
                pl_=ix_[torch.randperm(len(ix_),generator=g_,device=dev)[:max(kr_,int(cfg.rs_pool)*kr_)]]
                Xr_=_rs_pick(model,cfg,best[pl_].contiguous(),kr_,dom,g_)
            else: Xr_=best[ix_[torch.randperm(len(ix_),generator=g_,device=dev)[:kr_]]].contiguous()
            rl_=_recon_loss_mark(model,cfg,None,dom,X=Xr_)
            kl=kl+float(cfg.cd_recon_w)*rl_
            if len(st)<9: st.extend([0.0]*(9-len(st)))
            st[7]=st[7]+(rl_.detach() if cfg.cd_nosync else float(rl_.detach())); st[8]+=1
    return kl
_RDIAG={'rec':False,'v':[]}                                          # replay diagnostics: memory decay over a segment (recorded in the carry path)
def replay_diag(model,cfg,vals,NDOM,_S,n=16):
    with _rp_tags(model): return _replay_diag(model,cfg,vals,NDOM,_S,n)
def _replay_diag(model,cfg,vals,NDOM,_S,n=16):
    """ENCODER/DECODER DIAGNOSTICS (recon_mark runs): per segment length L, teacher-forced replay BPC of [mark]+segment[:-1] -> segment read from
    the segment's OWN end-of-segment memory vs from an empty memory (information the trace carries), greedy and sampled (T=0.7) replay byte
    accuracy, and the memory decay across the segment; plus which blocks carry a memory and whether replay training reaches the encoder."""
    print("\n=== ENCODER/DECODER DIAGNOSTICS (replay of a segment from its own end-of-segment memory) ===")
    was=model.training; model.eval(); fs=model.feat_scale.clone(); kh=getattr(model,'_keep_h',False); model._keep_h=False
    sv=_carry_save(); M=int(cfg.cd_mark); dev=DEVICE; g=torch.Generator(device=dev); g.manual_seed(99); res={}; cur0_=_XDG.get('cur')
    print(f"  replay-training gradient into the encoder: {'ON' if _RC_GRAD['on'] else 'OFF (detached snapshot)'}")
    try:
        data={}
        for d in range(NDOM):
            g_=vals[d]
            if hasattr(g_,'reset'): g_.reset()
            data[d]=torch.cat([g_()[0] for _ in range(2)],0).to(dev)
        X0=data[0]; T0=X0.shape[1]; Ls=[L_ for L_ in (8,12,16,24,32,48,64,96,128) if L_<=T0]
        with torch.no_grad():
            Xs=X0[:n,:min(32,T0)].long(); snap=_cd_encode(model,Xs)
            print("  blocks: "+" | ".join(f"b{i} "+("local filter" if getattr(b,'loc',False) else "attention" if getattr(b,'lattn',False) else "delta memory")
                                           +f" {getattr(b,'mh',1)} heads x dim {b.d_state//getattr(b,'mh',1)}" for i,b in enumerate(model.blocks)))
            print(f"  trace size: {sum(v_.numel() for v_ in snap.values())//n} numbers per segment, in {len(snap)} blocks")
        with torch.enable_grad():
            sl=_S(0); _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(Xs); sn=dict(_CARRY['new'])
            _CARRY.update(on=True,nr=n,st=sn,new={},rows=None,grad=False)
            Xin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),Xs[:,:-1]],1)
            l_=F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),Xs.reshape(-1)); _CARRY.pop('grad',None)
            tr=[v_ for k_,v_ in sn.items() if not isinstance(k_,tuple)]; rg=all(t_.requires_grad for t_ in tr)
            gn=sum(float(gr.norm()**2) for gr in torch.autograd.grad(l_,tr,allow_unused=True) if gr is not None)**0.5 if rg else 0.0
            print(f"  gradient check: snapshot differentiable {rg} | grad norm at the snapshot {gn:.3e}"); model.zero_grad(set_to_none=True)
        print(f"  {'L':>4} | {'empty BPC':>9} {'trace BPC':>9} {'carries':>8} | {'greedy acc':>10} {'sampled':>8} | {'decay over L':>12}")
        for Lx in Ls:
            e_=t_=a_=s_=0.0; dec=[]
            for d in range(NDOM):
                x=data[d]; B,T=x.shape
                r_=torch.randint(0,B,(n,),generator=g,device=dev); o_=torch.randint(0,T-Lx+1,(n,),generator=g,device=dev)
                X=torch.stack([x[r_[i],o_[i]:o_[i]+Lx] for i in range(n)]).long(); sl=_S(d)
                Xin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),X[:,:-1]],1)
                if isinstance(sl,int): _XDG['cur']=sl                     # this domain's private units / FiLM code (not the last-trained domain's)
                with torch.no_grad():
                    _RDIAG.update(rec=True,v=[]); snap=_cd_encode(model,X); _RDIAG['rec']=False
                    _CARRY.update(on=False,st={},new={})
                    e_+=float(F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),X.reshape(-1)))/LOG2
                    _RDIAG.update(rec=True,v=[]); _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
                    t_+=float(F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),X.reshape(-1)))/LOG2
                    _RDIAG['rec']=False; dec+=_RDIAG['v']
                    for mode in ('greedy','sampled'):
                        Y=torch.full((n,1),M,dtype=torch.long,device=dev)
                        for _ in range(Lx):
                            _CARRY.update(on=True,nr=n,st=snap,new={},rows=None); lg=model(Y,sl)[:,-1].float()
                            nx=lg.argmax(-1,keepdim=True) if mode=='greedy' else torch.multinomial(F.softmax(lg/0.7,-1),1,generator=g)
                            Y=torch.cat([Y,nx],1)
                        acc=float((Y[:,1:]==X).float().mean())
                        if mode=='greedy': a_+=acc
                        else: s_+=acc
            k_=NDOM; res[Lx]=(e_/k_,t_/k_,a_/k_,s_/k_)
            print(f"  {Lx:4d} | {e_/k_:9.3f} {t_/k_:9.3f} {(e_-t_)/k_:+8.3f} | {a_/k_:10.2f} {s_/k_:8.2f} | {sum(dec)/max(1,len(dec)):12.3f}")
        print("  [replay diag] "+" ".join(f"L{k}:carries {v[0]-v[1]:+.3f} greedy {v[2]:.2f}" for k,v in res.items()))
    finally:
        _carry_load(sv); model.feat_scale.copy_(fs); model.train(was); model._keep_h=kh; _XDG['cur']=cur0_
    return res
@torch.no_grad()
def recon_fidelity(model,cfg,vals,NDOM,_S,n=8):
    print("\n=== REPLAY FIDELITY: held-out BPC of a window read from its OWN end-of-window memory (snapshot) vs from an empty memory ===")
    out={}; was=model.training; model.eval()
    for d in range(NDOM):
        sl_=_S(d); g_=vals[d]
        if hasattr(g_,'reset'): g_.reset()
        a_=b_=0.0
        for _ in range(n):
            x,y=g_(); Ts_=int(cfg.seq_len); x,y=x[:,-Ts_:],y[:,-Ts_:]; sv=_carry_save()
            try:
                a_+=float(F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)))/n
                _CARRY.update(on=True,nr=x.shape[0],st={},new={},rows=None); model.features(x); snap=dict(_CARRY['new'])
                _CARRY.update(on=True,nr=x.shape[0],st=snap,new={},rows=None)
                b_+=float(F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)))/n
            finally: _carry_load(sv)
        out[d]=(a_/LOG2,b_/LOG2); print(f"  {PILE_DOMAINS[d][:12]:12} empty memory {a_/LOG2:.4f} | own snapshot {b_/LOG2:.4f} | replay gain {(b_-a_)/LOG2:+.4f} BPC")
    print(f"  [replay fidelity] mean gain from the window's own snapshot: {np.mean([b-a for a,b in out.values()]):+.4f} BPC")
    model.train(was); return out
def oracle_final(model,cfg,vals,trs,NDOM,fin,_S):
    """ORACLE LINEAR MEMORY: per domain, the exact pooled least-squares delta-rule memory (and optionally a BPC-optimised one) as the slot's
    initial state, weights frozen; held-out BPC change vs no memory."""
    print("\n=== ORACLE LINEAR MEMORY (final weights frozen; slot initial state = pooled least-squares S* per head) ===")
    sm0_=cfg.slot_mem; cfg.slot_mem=True; old_=dict(model.__dict__.get('_smem',{})); model.__dict__['_smem']={}
    res={}
    try:
        for d in range(NDOM):
            sl_=_S(d); _ORACLE['st']={}; _ORACLE['on']=True; model.eval()
            try:
                with torch.no_grad():
                    for _ in range(int(cfg.oracle_batches)): x,_y=trs[d](); model(x,sl_)
            finally: _ORACLE['on']=False
            model.train(); st_=_ORACLE['st']; _ORACLE['st']={}
            base_=bpc(model,vals[d],sl_); row={'none':base_}
            best_=(base_,None,None)
            for lam in cfg.oracle_lams:
                Sm={}
                for bid,(Ckk,Cvk) in st_.items():
                    dk_=Ckk.shape[-1]; r_=lam*Ckk.diagonal(dim1=-2,dim2=-1).sum(-1)/dk_
                    Sm[bid]=torch.linalg.solve((Ckk+r_[:,None,None]*torch.eye(dk_,device=Ckk.device)).transpose(-1,-2),Cvk.transpose(-1,-2)).transpose(-1,-2)
                for sc_ in cfg.oracle_scales:
                    Sc={k_:v_*float(sc_) for k_,v_ in Sm.items()}
                    model.__dict__['_smem']={sl_:Sc}; b_=bpc(model,vals[d],sl_); row[f'ls{lam:g}x{sc_:g}']=b_
                    if b_<best_[0]: best_=(b_,f'{lam:g} x{sc_:g}',Sc)
            if int(cfg.oracle_gd)>0:                                            # BPC-OPTIMISED initial memory (ceiling of any state)
                Sm={k_:(v_.clone() if best_[2] is not None else torch.zeros_like(v_)) for k_,v_ in (best_[2] or {bid:st_[bid][1] for bid in st_}).items()}
                P_=[torch.nn.Parameter(v_) for v_ in Sm.values()]; ks_=list(Sm.keys())
                rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                for p_,_ in rq_: p_.requires_grad_(False)
                og_=torch.optim.Adam(P_,lr=float(cfg.oracle_gd_lr))
                try:
                    for it_ in range(int(cfg.oracle_gd)):
                        model.__dict__['_smem']={sl_:dict(zip(ks_,P_))}; x,y=trs[d]()
                        l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                finally:
                    for p_,r0_ in rq_: p_.requires_grad_(r0_)
                model.__dict__['_smem']={sl_:{k_:p_.detach() for k_,p_ in zip(ks_,P_)}}; row['gd']=bpc(model,vals[d],sl_)
            if cfg.oracle_persist:                                              # PERSISTENT-READ ORACLE
                model.__dict__['_smem']={}; bestp_=(base_,None,None)
                try:
                    for lam in cfg.oracle_lams:
                        Sm={}
                        for bid,(Ckk,Cvk) in st_.items():
                            dk_=Ckk.shape[-1]; r_=lam*Ckk.diagonal(dim1=-2,dim2=-1).sum(-1)/dk_
                            Sm[bid]=torch.linalg.solve((Ckk+r_[:,None,None]*torch.eye(dk_,device=Ckk.device)).transpose(-1,-2),Cvk.transpose(-1,-2)).transpose(-1,-2)
                        for sc_ in cfg.oracle_pscales:
                            _PMEM['S']={k_:v_*float(sc_) for k_,v_ in Sm.items()}; _PMEM['on']=True
                            b_=bpc(model,vals[d],sl_); row[f'p{lam:g}x{sc_:g}']=b_
                            if b_<bestp_[0]: bestp_=(b_,f'{lam:g} x{sc_:g}',dict(_PMEM['S']))
                    if int(cfg.oracle_gd)>0:                                    # BPC-optimised persistent memory
                        Sm={k_:v_.clone() for k_,v_ in (bestp_[2] or {bid:torch.zeros_like(st_[bid][1]) for bid in st_}).items()}
                        P_=[torch.nn.Parameter(v_) for v_ in Sm.values()]; ks_=list(Sm.keys())
                        rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                        for p_,_ in rq_: p_.requires_grad_(False)
                        og_=torch.optim.Adam(P_,lr=float(cfg.oracle_gd_lr))
                        try:
                            for it_ in range(int(cfg.oracle_gd)):
                                _PMEM['S']=dict(zip(ks_,P_)); _PMEM['on']=True; x,y=trs[d]()
                                l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                        finally:
                            for p_,r0_ in rq_: p_.requires_grad_(r0_)
                        _PMEM['S']={k_:p_.detach() for k_,p_ in zip(ks_,P_)}; _PMEM['on']=True; row['gdp']=bpc(model,vals[d],sl_)
                finally: _PMEM['on']=False; _PMEM['S']={}
                print(f"  {'':12} persistent read: best least-squares {bestp_[0]-base_:+.4f}"+(f" ({bestp_[1]})" if bestp_[1] else "")
                      +(f" | BPC-optimised {row['gdp']-base_:+.4f}" if 'gdp' in row else ""))
            for rk_ in tuple(cfg.oracle_adapter):                               # LOW-RANK MLP ADAPTER ORACLE
                model.__dict__['_smem']={}; gA_=torch.Generator(device='cpu'); gA_.manual_seed(cfg.seed+31+d)
                P_={}
                for b_ in model.blocks:
                    dm_=b_.d_model
                    P_[id(b_)]=(torch.nn.Parameter((torch.randn(int(rk_),dm_,generator=gA_)/math.sqrt(dm_)).to(model.embed.weight.device)),
                                torch.nn.Parameter(torch.zeros(dm_,int(rk_),device=model.embed.weight.device)))
                rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                for p_,_ in rq_: p_.requires_grad_(False)
                og_=torch.optim.Adam([t_ for ab_ in P_.values() for t_ in ab_],lr=float(cfg.oracle_ad_lr))
                _ADPT['P']=P_; _ADPT['on']=True
                try:
                    for it_ in range(int(cfg.oracle_ad_steps)):
                        x,y=trs[d](); l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                    row[f'ad{int(rk_)}']=bpc(model,vals[d],sl_)
                finally:
                    _ADPT['on']=False; _ADPT['P']={}
                    for p_,r0_ in rq_: p_.requires_grad_(r0_)
                print(f"  {'':12} low-rank MLP adapter rank {int(rk_)}: {row[f'ad{int(rk_)}']-base_:+.4f} ({cfg.oracle_ad_steps} steps)")
            res[d]=row
            print(f"  {PILE_DOMAINS[d][:12]:12} no memory {base_:.4f} | "+" | ".join(f"{k_} {v_-base_:+.4f}" for k_,v_ in row.items() if k_!='none')
                  +(f" | best {best_[1]}" if best_[1] is not None else " | none helps"))
        bl_=lambda r_: min(v_ for k_,v_ in r_.items() if k_.startswith('ls'))
        print(f"  [oracle] mean BPC change, best least-squares memory per domain: {np.mean([bl_(r_)-r_['none'] for r_ in res.values()]):+.4f}"
              +(f" | BPC-optimised memory: {np.mean([r_['gd']-r_['none'] for r_ in res.values()]):+.4f}" if int(cfg.oracle_gd)>0 else ""))
        for rk_ in tuple(cfg.oracle_adapter):
            print(f"  [oracle, low-rank MLP adapter rank {int(rk_)}] mean BPC change: {np.mean([r_[f'ad{int(rk_)}']-r_['none'] for r_ in res.values()]):+.4f}")
        if cfg.oracle_persist:
            bp_=lambda r_: min([v_ for k_,v_ in r_.items() if k_.startswith('p')]+[r_['none']])
            print(f"  [oracle, persistent read] mean BPC change, best least-squares: {np.mean([bp_(r_)-r_['none'] for r_ in res.values()]):+.4f}"
                  +(f" | BPC-optimised: {np.mean([r_['gdp']-r_['none'] for r_ in res.values()]):+.4f}" if int(cfg.oracle_gd)>0 else ""))
    finally:
        cfg.slot_mem=sm0_; model.__dict__['_smem']=old_
    return res
def run(cfg):
    if cfg.stream: cfg.infer_ctx=True                  # NO VISIT BOUNDARIES: slots are always inferred
    for k in ('theta_low','theta_high','theta_dist','theta_sigma','theta_reserve','drive_ema','drive_slow','peak_decay','route_width','hebb_rate'): setattr(Config,k,getattr(cfg,k))  # buffers read class attrs
    set_seed(cfg.seed)
    _NOISE['gen']=torch.Generator().manual_seed(cfg.seed+12345)
    for fn in ('_prev_bpc','_plateau_count','_avg_difficulty'):
        if hasattr(train_domain,fn): delattr(train_domain,fn)
    print(f"Run | threshold_mode={cfg.threshold_mode} drive_ref={cfg.drive_ref} use_sens={cfg.use_sens} seed={cfg.seed}")
    PILE_DOMAINS[:]=BASE_DOMAINS+[d for d in cfg.extra_domains if d not in BASE_DOMAINS]   # before the heads are built
    model=Model(cfg).to(DEVICE)
    if getattr(model,'_ng_cur',None) is not None:
        model.__dict__['_ng_cur']=[t_.to(DEVICE) for t_ in model._ng_cur]; model.__dict__['_ng_old']=[t_.to(DEVICE) for t_ in model._ng_old]
    opt=_new_opt(model,cfg)
    for nm_ in ('dream_route','strand_route'):
        r_=getattr(cfg,nm_)
        if r_ is not None:
            if len(r_)!=len(model.blocks)+1: raise ValueError(f"{nm_} needs {len(model.blocks)+1} values (emb, b0..b{len(model.blocks)-1}), got {r_}")
            if any(r_[j]>0 and r_[j+1]<=0 for j in range(len(r_)-1)): print(f"  [route] note: {nm_} {r_}: a part below a zero-weight part gets no dream gradient")
    setup_kernel(model, cfg); setup_compile(model, cfg); _t_run=time.perf_counter()
    _STF['on']=False
    proj_setup(model, cfg)
    _XDG.update(on=cfg.xdg>0,keep=1.0,seed=int(cfg.seed),masks={},cur=None,rows=None,tagon=(cfg.pu_k>0 or cfg.cgate>0 or cfg.film_k>0))
    _FILM['codes']=getattr(model,'film_codes',None)
    _DP2['on']=int(cfg.dp_steps)>=2
    if cfg.pu_k>0: print(f"  [private trunk units] {cfg.pu_k} private MLP units per domain in every block")
    for i_,b in enumerate(model.blocks): b.__dict__['_xdg_i']=i_
    ngp_init(model, cfg); print(f"  [private n-grams] per-domain hashed tables, orders {tuple(cfg.ngp)}, {cfg.ngp_rows} rows each, before each domain's own head")
    for b in model.blocks: b._surp_on=bool(cfg.surprise_mod); b.__dict__['_lean']=bool(cfg.lean_stats)
    _DSPLIT['on']=bool(cfg.dream_mom>0 or cfg.muon_mom_dream>0)
    _init_sd=None
    model._reserve=None
    object.__setattr__(model,'_replay_buffer',[])
    install_cov_hooks(model, cfg)
    kl_log={}; dream_log={}; fd_log={}; fd_snaps={}; battr_log={}
    if cfg.topic_diag: object.__setattr__(model,'_topics',TopicTracker(cfg,DEVICE)); topic_log={}
    cap_s=cfg.d_state*cfg.reserve_mult; cap_m=2*cfg.d_model*cfg.reserve_mult
    def _fx(li,base):                                # 'fresh' extra units held in reserve for later domains
        e=0.0
        return int(round(base*e))
    ns0_=int(round(cfg.d_state*cfg.init_state_mult)); nm0_=int(round(2*cfg.d_model*cfg.init_mlp_mult))
    model._reserve=[Reserve(cfg.d_model, max(0,cap_s-b.d_state)+_fx(li,ns0_), max(0,cap_m-b.mlp_hidden)+_fx(li,nm0_), DEVICE)
                    for li,b in enumerate(model.blocks)]
    print("  [capacity] "+("wider from the start")
                              +": "+" ".join(f"b{li} state {b.d_state}+{R.Wk.shape[0]} MLP {b.mlp_hidden}+{R.min.shape[0]}" for li,(b,R) in enumerate(zip(model.blocks,model._reserve))))
    model._uniq=[Uniq(cfg.d_model, DEVICE) for _ in model.blocks]; model._rgen=torch.Generator().manual_seed(cfg.seed+777)
    model._recruit_log={}
    for b in model.blocks: b._store_uniq=True
    cfg.log_locality=False
    print(f"  [reserve] capacity x{cfg.reserve_mult}: state {cap_s}, MLP {cap_m} per block; initially active "
          f"state {model.blocks[0].d_state}, MLP {model.blocks[0].mlp_hidden}")
    cfg.route_writes=(cfg.route_mode=='write')
    if cfg.unit_budget is None: cfg.unit_budget=cfg.write_budget
    if cfg.channel_budget is None: cfg.channel_budget=cfg.write_budget
    if cfg.channel_gate_mode is None: cfg.channel_gate_mode=cfg.gate_mode
    for b in model.blocks:
        b.route_in_forward=(cfg.route_mode=='forward'); b.write_routed=(cfg.route_mode=='write')
    meta=MetaController().to(DEVICE); meta_opt=torch.optim.Adam(meta.parameters(),lr=cfg.meta_lr)
    npass_=max(1,cfg.n_passes); fresh_=bool(cfg.pass_fresh_data) and npass_>1
    vseed_=(lambda i: 17777+i) if fresh_ else (lambda i: 7777+i); tseed_=(lambda i: 11000+i) if fresh_ else (lambda i: 1000+i)
    shard_=int(cfg.pass_shard_bytes)
    if cfg.no_revisit:                                   # a visit must not run out of fresh windows: steps x batch x (seq_len+1), +1% for the
        shard_=max(shard_,int(cfg.steps_per_task*cfg.batch_size*(cfg.seq_len+1)*1.01))      # document joins
    TRN_=shard_*(npass_ if fresh_ else 1)
    if _stream_bytes is _hf_stream_bytes:            # new domains (fresh-pass data: ALL domains) collected together in one streaming pass
        prefetch_pile([(d,(d,vseed_(i),500_000),(d,tseed_(i),TRN_)) for i,d in enumerate(PILE_DOMAINS) if (fresh_ or d not in BASE_DOMAINS)])
    if fresh_:                                       # UNSEEN TEXT EVERY PASS: disjoint shards, loaders built per pass
        raw_tr_=[stream_bytes(d,tseed_(i),TRN_) for i,d in enumerate(PILE_DOMAINS)]; trp_={}
        print("  [data] fresh text every pass: per domain "+", ".join(f"{d[:10]} {len(r)//npass_/1e6:.1f}MB x{npass_}" for d,r in zip(PILE_DOMAINS,raw_tr_)))
        def train_of(t,p):
            if (t,p) not in trp_:
                for k_ in [k_ for k_ in trp_ if k_[1]<p]: del trp_[k_]               # earlier passes' shards are never used again
                sh_=len(raw_tr_[t])//npass_; trp_[(t,p)]=make_loader(_inject_facts(raw_tr_[t][p*sh_:(p+1)*sh_],cfg,t,p) if cfg.fact_spec else (inf_plant(raw_tr_[t][p*sh_:(p+1)*sh_],cfg,t,p) if cfg.inf_on else raw_tr_[t][p*sh_:(p+1)*sh_]),cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep))
            return trp_[(t,p)]
        trains=None
    else:
        trains=[make_loader((inf_plant(stream_bytes(d,tseed_(i),TRN_),cfg,i,0) if cfg.inf_on else stream_bytes(d,tseed_(i),TRN_)),cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep)) for i,d in enumerate(PILE_DOMAINS)]
        train_of=lambda t,p: trains[t]
        if cfg.fact_spec:                                                        # UPDATE-OR-PROTECT TEST (shared text across passes): a fact domain
            fdom_={int(e[0]) for e in cfg.fact_spec}; ftr_={}                    #   gets its own loader per pass, the same text with that pass's facts
            def train_of(t,p):
                if t not in fdom_: return trains[t]
                if (t,p) not in ftr_:
                    raw_=_inject_facts(stream_bytes(PILE_DOMAINS[t],tseed_(t),TRN_),cfg,t,p)
                    ftr_[(t,p)]=make_loader(raw_,cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep))
                return ftr_[(t,p)]
    vals=[make_loader(stream_bytes(d,vseed_(i),500_000),cfg,fixed_seed=4242+i,span=1+int(cfg.eval_ctx)) for i,d in enumerate(PILE_DOMAINS)]
    NDOM=len(PILE_DOMAINS); schedule=[d for _ in range(max(1,cfg.n_passes)) for d in range(NDOM)]     # REPEATED PASSES over all domains
    if cfg.pass_shuffle:
        _rp=np.random.RandomState(cfg.seed*7919+13); schedule=list(range(NDOM))+[int(d) for _ in range(max(1,cfg.n_passes)-1) for d in _rp.permutation(NDOM)]
        print(f"  [pass shuffle] visit order: "+' '.join(str(d) for d in schedule))
    _t2s=model.__dict__.setdefault('_t2s',{})                      # LABEL-FREE: true domain -> the slot its text was routed to (evaluation only)
    _S=(lambda q: _t2s.get(q,q)) if cfg.infer_ctx else (lambda q: q)
    bpc_m=[[None]*NDOM for _ in range(len(schedule))]; post=[None]*NDOM; prior_buffer=[]; fz_tok={}
    last_visit={}; post_hist={}; pre_visit={d:{} for d in range(NDOM)}; pass_log=[]
    locality_log=[]; prior_imp=[np.zeros(b.mlp_hidden) for b in model.blocks]
    model._orth=None; orth_log={}
    alt_m={}
    pre_lock=post_lock=None
    zeroshot=[None]*len(PILE_DOMAINS)
    _src=_StreamSource(schedule,train_of,cfg,NDOM) if cfg.stream else None; _pend=None; _seen_true=set()   # NO VISIT BOUNDARIES
    for vi,t in enumerate(schedule):                     # v = visit (position in the stream), t = domain (head) being trained
        pass_i=vi//NDOM; seen_d=sorted(set(schedule[:vi+1]))
        if cfg.stream:                                                         # NO VISIT BOUNDARIES: the slot was decided by the detector
            _eo=_src.done
            if _eo: print(f"  [stream] the stream has ended ({vi} detected visits for a schedule of {len(schedule)}): evaluation only"); ts=shist_[-1] if shist_ else 0; _b0=[]
            elif _pend is None: ts,_b0=infer_slot(model,cfg,_src,_src.true_dom,vi)
            else: ts,_b0=_pend
            if not _eo: t=_src.true_dom; _seen_true.add(t); seen_d=sorted(_seen_true|set(schedule[:vi+1]))   # bookkeeping (evaluation only) follows the true text
            _gv=_StreamGet(_src,model,cfg,ts,vi,_b0)
        else:
            _eo=False
            ts,_b0=infer_slot(model,cfg,train_of(t,pass_i),t,vi) if cfg.infer_ctx else (t,[])      # LABEL-FREE: the slot this visit trains
            _gv=_ReplayGet(train_of(t,pass_i),_b0,cfg.infer_batches) if cfg.infer_ctx else train_of(t,pass_i)
        if cfg.infer_ctx and not _eo: _t2s[t]=ts
        shist_=model.__dict__.setdefault('_slot_hist',[]) if cfg.infer_ctx else None
        olds=[]
        for d_ in reversed(shist_ if cfg.infer_ctx else schedule[:vi]):
            if d_!=ts and d_ not in olds: olds.append(d_)
        olds=olds[::-1]                                  # least -> most recently visited (single pass: range(t))
        if cfg.dream_lesion: olds=[d_ for d_ in olds if d_ not in model.__dict__.get('_lesion',set())]   # LESION TEST: no dreams for lesioned slots
        model.__dict__['_self_slot']=None
        if cfg.self_prot_w>0 and ts in (shist_ if cfg.infer_ctx else schedule[:vi]) and ts in model.__dict__.get('_bank',{}) and ts not in olds:
            olds.append(ts); model.__dict__['_self_slot']=ts               # cg15b SELF-PROTECTION: a revisited slot keeps (part of) its own dreams
            print(f"  [self protection] revisited slot {ts}: its own dream bank stays protected at share x{cfg.self_prot_w:g}")
        if (cfg.self_dream_nreal and cfg.real_w_pass and float(cfg.real_w_pass[min(pass_i,len(cfg.real_w_pass)-1)])==0.0
                and ts in schedule[:vi] and ts not in olds): olds.append(ts)   # NO-REAL-TEXT PASS: the visited domain is protected by its OWN dreams too
        object.__setattr__(model,'_olds',olds); object.__setattr__(model,'_visit',vi); model.__dict__['_cur_dom']=int(t)
        if cfg.n_passes>1 and cfg.data_free and not cfg.topic_mode and not _eo: rebuild_old_cov(model, olds)   # protect OTHER domains only
        if vi>0 and t in pre_visit:                                                       # before relearning (repeat visits)
            pre_visit[t][pass_i]=bpc_m[vi-1][t] if (bpc_m[vi-1][t] is not None or t not in schedule[:vi]) else bpc(model,vals[t],_S(t))
        print(f"\n=== Phase {vi+1}: {PILE_DOMAINS[t]}"+(f" (pass {pass_i+1}/{cfg.n_passes})" if cfg.n_passes>1 else "")+f" === [elapsed {(time.time()-_RUN_T0)/60:.1f} min]")
        if cfg.fwt_zeroshot and vi>0 and t not in schedule[:vi] and cfg.scratch_domain is None:              # before ANY training on domain t
            zeroshot[t]=bpc_inferred(model,vals[t],t,t,'bytes',cfg)[0]
            print(f"  [forward transfer] {PILE_DOMAINS[t]} before training (earlier heads, inferred): {zeroshot[t]:.3f} BPC")
        model._curve_val=None
        # PROACTIVE difficulty-based widening: before training a HARD domain, open up
        # the shared manifold proportional to how much harder it is than the easiest
        # domain (measured difficulty). ArXiv (2.66, hardest) gets the most room; easy
        # domains get none. New units unfrozen (for this domain), old units protected.
        easiest=min(DOMAIN_DIFFICULTY.values())
        # every domain gets a minimum fresh block (so later domains write to NEW units,
        # not earlier domains') + extra proportional to difficulty (hard domains more)
        extra=cfg.min_fresh + int((DOMAIN_DIFFICULTY.get(PILE_DOMAINS[t],easiest)-easiest)*cfg.widen_per_difficulty)
        added=0
        while added<extra and model.blocks[0].d_state < cfg.d_state_max and not cfg.reserve_mult:
            model.widen_all(cfg.widen_chunk); added+=cfg.widen_chunk
        if added>0:
            opt=_new_opt(model,cfg)
            print(f"  [proactive widen] {PILE_DOMAINS[t]} difficulty={DOMAIN_DIFFICULTY[PILE_DOMAINS[t]]:.2f} -> +{added} units (d_state={model.blocks[0].d_state})")
        train_domain._prev_bpc=None; train_domain._plateau_count=0   # reset per domain
        if cfg.log_locality:
            W0=[(b.mlp_in.weight.detach().cpu().numpy().copy(), b.mlp_out.weight.detach().cpu().numpy().copy()) for b in model.blocks]
        model.__dict__['_first_visit']=(model._slot_log[-1][3]=='new') if cfg.infer_ctx else (t not in schedule[:vi])
        if (cfg.imp_revisit or cfg.imp_combine=='max') and not _eo and revisit_begin(model,cfg,ts) and cfg.imp_revisit:
            print(f"  [revisit gates] {PILE_DOMAINS[t]}'s own earlier importance taken out for this visit")
        if not cfg.joint_train and not _eo:
            model.__dict__['_s0_train']=None
            _pb=None
            try:
                model.__dict__['_pass_lr']=float(cfg.pass_lr[min(pass_i,len(cfg.pass_lr)-1)]) if cfg.pass_lr else 1.0
                model.__dict__['_dwp']=float(cfg.dream_w_pass[min(pass_i,len(cfg.dream_w_pass)-1)]) if cfg.dream_w_pass else 1.0   # cg14
                if not olds and float(cfg.pass_lr_noold)>0: model.__dict__['_pass_lr']=float(cfg.pass_lr_noold)   # cg13: visits with no old domain
                model.__dict__['_real_w']=float(cfg.real_w_pass[min(pass_i,len(cfg.real_w_pass)-1)]) if cfg.real_w_pass else 1.0
                model.__dict__['_pass_i']=int(pass_i)
                _prb=probe_pre(model,cfg,olds,t,vals) if (vi in tuple(cfg.probe_visits) and olds and cfg.dream_bank and cfg.bank_targets) else None
                if cfg.cd_teacher=='domain' and float(model.__dict__.get('_real_w',1.0))==0.0 and ts in model.__dict__.get('_dom_P',{}):
                    dv_=model.embed.weight.device; model.__dict__['_cd_P']={k_:v_.to(dv_) for k_,v_ in model.__dict__['_dom_P'][ts].items()}
                    print(f"  [teacher] slot {ts}: the conduit teacher is the model saved at the end of this slot's last visit")
                _les_=bool(cfg.trace_lesion) and pass_i>=int(cfg.trace_lesion_pass) and int(t) in tuple(int(z_) for z_ in cfg.trace_lesion)
                if _les_: print(f"  [trace lesion] {_lab(cfg,t)}: visit SKIPPED (no traces, no replay; its old dreams stay)")
                opt=(opt if _les_ else train_domain(model, None, meta_opt, opt, _gv, ts, prior_buffer, cfg))
                if int(cfg.pu_cons_steps)>0 and not _les_ and float(model.__dict__.get('_real_w',1.0))==0.0:   # cg19 CONSOLIDATION PHASE
                    _sd_=set(int(z_) for z_ in schedule[:vi]); pu_consolidate(model,cfg,t,ts,vals,[(d_,_S(d_)) for d_ in range(NDOM) if d_!=t and d_ in _sd_],_S)
                if cfg.inf_on and not _les_: inf_sleep(model,cfg,t,ts,vals,[(d_,_S(d_)) for d_ in range(NDOM) if d_!=t and d_ in set(int(z_) for z_ in schedule[:vi])],_S)   # cg20
                if cfg.cd_teacher=='domain':                                    # PER-DOMAIN TEACHER: save the model as it was at this visit's end
                    model.__dict__.setdefault('_dom_P',{})[ts]={k_:(p_.detach().to('cpu',copy=True) if p_.is_cuda else p_.detach().clone()) for k_,p_ in model.named_parameters()}
                if _prb is not None: probe_post(model,cfg,_prb,vi,vals)
            finally:
                if _pb is not None: cfg.batch_size,cfg.steps_per_task,cfg.dream_batch,cfg.lr,cfg.dream_early_steps=_pb
            if cfg.infer_ctx: _les_ or slot_ref(model,cfg,(_gv.ref_batches or _gv.last) if cfg.stream else _gv.last,ts); shist_.append(ts)
            if cfg.fact_spec:                                                                                # UPDATE-OR-PROTECT TEST: every fact domain's slot
                fd_=sorted({int(e[0]) for e in cfg.fact_spec}); fact_eval(model,cfg,vi,[(_lab(cfg,d_),_t2s[d_]) for d_ in fd_ if d_ in _t2s])
            if cfg.dream_lesion and pass_i>=int(cfg.dream_lesion_pass) and t in tuple(cfg.dream_lesion):   # LESION TEST: after this visit the domain's
                model.__dict__.setdefault('_lesion',set()).add(ts)                                          #   bank is never dreamed again
                print(f"  [lesion] {_lab(cfg,t)} (slot {ts}): its dream bank is switched off for the rest of the run")
            if cfg.stream:
                _pend=_gv.pending
                nb_=[b_ for b_ in _src.bounds if _gv.log and abs(b_-_gv.log[-1][0])<=4*max(1,int(cfg.infer_batches))]
                tr_=model.__dict__.pop('_strace',None)
                if tr_:
                    for b_ in [b_ for b_ in _src.bounds if tr_[0][0]<=b_<=tr_[-1][0]+1]:
                        print(f"  [stream trace] true boundary at batch {b_}: "+' '.join(f"{p_}:{l_:.2f}/{m_:.2f}/{sd_:.2f}/{z_:.1f}/{c_:.1f}" for p_,l_,m_,sd_,z_,c_ in tr_ if b_-4<=p_<=b_+8))
                if cfg.stream_rep: print(f"  [stream rep] visit {vi}: alarm z {_gv.rz:.1f} | largest in-visit z without alarm {_gv.rzmax:.1f} (threshold {cfg.stream_rep_z})")
                print(f"  [stream] visit {vi}: {len(_gv.log)} alarm(s) at stream batches "+' '.join(f'{p_} ({l_:.2f} > {th_:.2f} bits -> slot {sl_})' for p_,l_,th_,sl_ in _gv.log)
                      +(f" | switch {_gv.log[-1][0]-nb_[0]:+d} batches from the true boundary" if (_pend is not None and nb_) else (' | no switch' if _pend is None else ' | switch NOT at a true boundary')))
            if cfg.slot_mem and model.__dict__.get('_smem',{}).get(ts):
                print("  [slot memory] slot "+str(ts)+": mean |S_d| per head, per block: "+' '.join(f"{float(v_.norm(dim=(1,2)).mean()):.2f}" for v_ in model._smem[ts].values())+f" | slots with memory {len(model._smem)}")
            if cfg.ctx_carry and _CARRY['st']:
                print("  [context carry] carried memory at the visit end, mean |S| per row-head: "+' '.join(f"{float(v_.norm(dim=(1,2)).mean()):.2f}" for k_,v_ in _CARRY['st'].items() if not isinstance(k_,tuple)))
            model.__dict__['_s0_train']=None
            if int(cfg.lm_rank)>=0 and float(model.__dict__.get('_real_w',1.0))==0.0: lora_merge(model, cfg, olds, ts)   # REM-like DREAM MERGE
            if cfg.dream_bank and cfg.dream and vi<len(schedule)-1: bank_refresh(model, cfg, ts)   # the domain's dreams (v9h: not after the final visit - never read)
            if float(cfg.sem_share)>0 and cfg.sem_tgt=='student': sem_rescore(model, cfg, ts)
            if cfg.slot_mem and cfg.slot_mem_reenc>0: smem_reencode(model, cfg, ts)   # SLOT MEMORY RE-ENCODING
        object.__setattr__(model,'_live_dom',None)
        # add this domain's examples to the prior buffer (for future forgetting signal) - none when data_free
        for _ in range(0):
            bx,by=train_of(t,pass_i)(); prior_buffer.append((bx,by,t))
        if cfg.data_free and not cfg.topic_mode and not _eo: fold_domain_covariance(model, (ts) if cfg.n_passes>1 else None)                 # running input covariance -> 'old domains'
        for _ in range(cfg.replay_buffer_batches):          # separate, larger REPLAY store (bytes: stored compactly)
            bx,by=train_of(t,pass_i)(); model._replay_buffer.append((bx.to(torch.uint8),by.to(torch.uint8),t))
        # snapshot this domain's usage as "home" (polarization feature = deviation from it)
        with torch.no_grad():
            for b in model.blocks:
                b.unit_home.copy_(b.unit_usage); b.mlp_home.copy_(b.mlp_usage)
        if cfg.log_locality:
            dom_use=[b.mlp_usage.detach().cpu().numpy().copy() for b in model.blocks]
            if t>0:
                ms=[locality_metrics(W0[i][0],b.mlp_in.weight.detach().cpu().numpy(),W0[i][1],
                                     b.mlp_out.weight.detach().cpu().numpy(),dom_use[i],prior_imp[i],cfg.locality_k)
                    for i,b in enumerate(model.blocks)]
                avg={key:float(np.mean([m[key] for m in ms])) for key in ms[0]}
                locality_log.append(avg)
                print(f"  [locality] cluster={avg['cluster']:+.3f} damage_locality={avg['damage_locality']:+.3f} "
                      f"neigh_R2_gain={avg['neigh_r2_gain']:+.3f} own_neigh_corr={avg['own_neigh_corr']:+.2f}")
            for i in range(len(model.blocks)):
                u=dom_use[i]; prior_imp[i]=prior_imp[i]+u/(u.mean()+1e-8)
        if cfg.use_write_threshold and not cfg.topic_mode and not _eo: update_importance(model, cfg)
        if (cfg.imp_revisit or cfg.imp_combine=='max') and not _eo: revisit_end(model,cfg,ts)
        if t==cfg.lock_after_domains-1 and cfg.route_mode=='none':
            model.lock_distinctness()   # no routing: lock is a no-op for computation; skip switch-cost evals
            pre_lock=post_lock=[bpc_m_q for bpc_m_q in [None]*(t+1)]
        elif t==cfg.lock_after_domains-1:
            # ROUTING-SWITCH COST: the lock switches routing from uniform to distinctness-based.
            # Score every trained domain just before and just after it. Eval passes nudge the
            # (not-yet-locked) distinctness scale, so snapshot/restore it: training is unchanged.
            fs=model.feat_scale.clone()
            pre_lock=[bpc(model,vals[q],q) for q in range(t+1)]
            model.feat_scale.copy_(fs)
            model.lock_distinctness()   # freeze reference (= prose cluster over first domains)
            if cfg.route_writes: calibrate_affinity(model, cfg, t+1)
            post_lock=[bpc(model,vals[q],q) for q in range(t+1)]
            for q in range(t+1):
                print(f"  [routing switch] {PILE_DOMAINS[q][:12]:12} pre-lock {pre_lock[q]:.3f} -> post-lock "
                      f"{post_lock[q]:.3f} (switch cost {post_lock[q]-pre_lock[q]:+.3f})")
            print(f"  [distinctness reference LOCKED after {cfg.lock_after_domains} domains - routing now stable, reference=prose cluster]")
        with torch.no_grad():
            for b in model.blocks:
                b.unit_usage.zero_(); b.mlp_usage.zero_()
        end_pass=(vi%NDOM==NDOM-1)
        for prev in (seen_d if (end_pass or not cfg.lean or cfg.fz_diag or cfg.sim_diag) else [t]): bpc_m[vi][prev]=bpc(model,vals[prev],_S(prev))   # lean: all domains at pass ends
        if cfg.head_select_eval:                         # label-free: every seen domain at pass ends (single pass: every visit, or with
            for prev in (seen_d if ((cfg.n_passes==1 and not cfg.diag_light) or end_pass) else [t]):     # diag_light current + end only)
                for mode,r_ in bpc_inferred_modes(model,vals[prev],_S(prev),len(model._slots) if cfg.infer_ctx else len(seen_d),tuple(cfg.head_select_eval),cfg).items():
                    alt_m.setdefault(mode,{})[(vi,prev)]=r_
        post[t]=bpc_m[vi][t]; last_visit[t]=vi; post_hist.setdefault(pass_i,{})[t]=post[t]
        print(f"  post-train BPC: {post[t]:.3f}  (width {model.blocks[0].d_state})"
              +(f" | before this visit {pre_visit[t][pass_i]:.3f}, after previous visit {post_hist[pass_i-1][t]:.3f}" if (pass_i>0 and pass_i in pre_visit.get(t,{}) and t in post_hist.get(pass_i-1,{})) else ""))
        cl=(getattr(model,'_cross_log',None) or {}).get(t)
        if cl: print('  [cross] mean |gain| on each earlier domain\'s specific shunt: '+' '.join(f'{PILE_DOMAINS[j][:6]} {g:.3f}' for j,g in zip(sorted(model._spec)[:len(cl)],cl)))
        if getattr(model,'_spec',None) and t in model._spec:
            nz=sum(int((v!=0).sum()) for v in model._spec[t].values()); print(f'  [specific shunt] {nz/1e6:.2f}M weights kept for this domain')
        dsn=model.__dict__.pop('_dreamstat',None)
        if dsn: dsn={d:[(float(a) if torch.is_tensor(a) else a) for a in v] for d,v in dsn.items()}; dsn={d:v[:4]+[int(round(v[4]))] for d,v in dsn.items()}
        if dsn:
            dream_log[vi]={d:w_[0]/max(w_[1],1) for d,w_ in dsn.items()}
            rbl=[r for r in (getattr(model,'_rb_log',None) or []) if r[0]==t]
            print('  [dream outliers] batches with dream KL > 1 / robust mean KL: '+' '.join(f'{_lab(cfg,d)} {v[4]}/{v[1]} {v[3]/max(v[1],1):.4f}' for d,v in sorted(dsn.items())))
            for d,(kw,txt,lmin,pr) in sorted((model.__dict__.pop('_dworst',None) or {}).items()):
                if kw>1.0:
                    print(f'  [worst dream] {_lab(cfg,d)} batch KL {kw:.1f}, live min log-prob {lmin:.0f}: '+repr(txt[:100]))
                    if pr: print('      max |.| per block (input, gate pre-act, SSM pred, gated, MLP act): '
                                 +' | '.join(f'{a:.0f} {g:.0f} {q:.0f} {z:.0f} {m_:.0f}' for a,g,q,z,m_ in pr[:-1])
                                 +f' || final features {pr[-1][0]:.0f}, logit range {pr[-1][1]:.0f}')
            print('  [dream KL] mean KL to main weights on each old domain\'s DREAMS (weight %.1f): '%cfg.dream_weight
                  +' '.join(f'{_lab(cfg,d)} {w_:.4f}' for d,w_ in sorted(dream_log[vi].items())))
            if cfg.share_mode=='kl':                                         # cg10: next visits' dream shares follow the smoothed dream KL
                SG_=model.__dict__.setdefault('_share_sig',{})
                for d,w_ in dream_log[vi].items():
                    if d==model.__dict__.get('_self_slot'): continue           # cg15b: self-KL includes this visit's learning, not drift
                    SG_[d]=float(w_) if d not in SG_ else 0.5*SG_[d]+0.5*float(w_)
        ks=model.__dict__.pop('_klstat',None)
        if ks: ks={d:[(float(a) if torch.is_tensor(a) else a) for a in v] for d,v in ks.items()}
        if ks:
            kl_log[vi]={d:(w_[0]/max(w_[2],1),w_[1]/max(w_[2],1)) for d,w_ in ks.items()}
            print('  [shunt KL] mean KL to main weights / mean weight per old domain: '+' '.join(f'{_lab(cfg,d)} {a:.4f}/{w:.2f}' for d,(a,w) in kl_log[vi].items()))
        if getattr(model,'_reserve',None):
            ev=model._recruit_log.get(t,[])
            print(f"  [reserve] {PILE_DOMAINS[t]}: activated state {sum(e[1] for e in ev)} / MLP {sum(e[2] for e in ev)} units in {len(ev)} "
                  f"block-events | active per block state "+",".join(str(b.d_state) for b in model.blocks)
                  +" MLP "+",".join(str(b.mlp_hidden) for b in model.blocks)
                  +f" | reserve left state {model._reserve[0].Wk.shape[0]} MLP {model._reserve[0].min.shape[0]} (block 0)")
        # update running difficulty average (for the difficulty-relative widening trigger)
        seen=[post[k] for k in range(t+1) if post[k] is not None]
        train_domain._avg_difficulty=float(np.mean(seen))
        if model._orth is not None and t>0:
            orth_log[t]=orth_summary(model._orth.log, t, cfg.orth_energies)
            if orth_log[t] is not None: print_orth(orth_log[t], PILE_DOMAINS[t], cfg.orth_energies)
        if cfg.topic_diag:
            tr=model._topics.report(vals,t); topic_log[t]=tr
            top=[" ".join(f"t{k}:{int(100*v/max(sum(row),1))}%" for k,v in sorted(enumerate(row),key=lambda z:-z[1])[:3] if v>0) for row in tr['M']]
            print(f"  [topics] {tr['used']} in use | purity {tr['purity']:.2f} NMI {tr['nmi']:.2f} | confidence "
                  +" ".join(f"{c:.2f}" for c in tr['conf'])+" | kept own topic since trained: "
                  +(" ".join(f"{PILE_DOMAINS[d][:6]} {v:.2f}" for d,v in tr['stab'].items()) or "-")
                  +" | head-coverage "+" ".join(f"{v:.2f}" for v in tr['cover'].values()))
            for d,tt in enumerate(top): print(f"     {PILE_DOMAINS[d][:10]:10} -> {tt}")
        rcs=model.__dict__.pop('_recon_stat',None)
        if rcs: model.__dict__.setdefault('_recon_log',{})[t]={li:v[0]/max(v[1],1e-12) for li,v in rcs.items()}
        rsh=model.__dict__.pop('_recon_share',None)
        if rcs: print('  [reconsolidation] mean squared output error on dreams before each correction, relative to the output energy: '
                      +' '.join(f'b{li} {v[0]/max(v[1],1e-12):.4f}' for li,v in sorted(rcs.items()))
                      +(f' | dream tokens corrected (one-sided) {rsh[0]/max(rsh[1],1):.2f}' if rsh else ''))
        ba=model.__dict__.pop('_battr_acc',None)
        if ba:
            battr_log[t]={g:(v[0]/v[2],v[1]/v[2],v[3]/v[2]) for g,v in ba.items()}
            order=[g for g in ['emb']+[f'b{i}' for i in range(len(model.blocks))]+['all'] if g in battr_log[t]]
            print('  [block attribution] this domain\'s learning, each part alone on the main weights (data-free): benefit (current-loss '
                  'drop) / change (old heads\' dream KL) / SIGNED harm (old heads\' loss change on samples from their byte-pair statistics): '
                  +' | '.join(f"{g} {battr_log[t][g][0]:+.4f}/{battr_log[t][g][1]:.4f}/{battr_log[t][g][2]:+.4f}" for g in order)
                  +(''))
        if cfg.n_passes>1 and end_pass:                 # PASS SUMMARY
            fin_=[bpc_m[vi][d] for d in range(NDOM)]; po_=[post_hist.get(pass_i,{}).get(d,post[d] if post[d] is not None else fin_[d]) for d in range(NDOM)]
            fg_=[fin_[d]-po_[d] for d in range(NDOM) if d!=t]
            al_={m:float(np.mean([alt_m[m][(vi,d)][0] for d in range(NDOM)])) for m in cfg.head_select_eval}
            pass_log.append({'pass':pass_i,'final':fin_,'post':po_,'bpc':float(np.mean(fin_)),'forget':float(np.mean(fg_)),'alt':al_,
                             'pre':[pre_visit.get(d,{}).get(pass_i) for d in range(NDOM)]})
            print(f"\n  ===== PASS {pass_i+1} SUMMARY: mean BPC {np.mean(fin_):.3f} | mean forgetting since each domain's visit {np.mean(fg_):+.3f} | label-free "
                  +" ".join(f"{m} {x_:.3f}" for m,x_ in al_.items())+" =====")
            CLp_=[r_ for r_ in model.__dict__.get('_pucons_log',[])]
            if CLp_:
                print(f"    [consolidation] so far: {len(CLp_)} phases | mean gap recovered {100*np.mean([r_[5] for r_ in CLp_]):.0f}% | held-out with private units "
                      f"{np.mean([r_[2]-r_[1] for r_ in CLp_]):+.4f} per phase | old domains {np.nanmean([r_[7]-r_[6] for r_ in CLp_]):+.4f} per phase | "
                      f"retired {sum(r_[8] for r_ in CLp_)}, evicted {sum(r_[9] for r_ in CLp_)}")
            if cfg.trace_lesion and pass_i>=int(cfg.trace_lesion_pass):
                LS_=[int(z_) for z_ in cfg.trace_lesion]; pv_=pass_i-1
                ch_=[(d,fin_[d]-post_hist.get(pv_,{}).get(d,float('nan'))) for d in LS_]
                wt_=[pre_visit.get(d,{}).get(pass_i)-post_hist.get(pv_,{}).get(d,float('nan')) for d in range(NDOM) if d not in LS_ and pre_visit.get(d,{}).get(pass_i) is not None and d in post_hist.get(pv_,{})]
                print(f"    [trace lesion] pass {pass_i+1}, no visit: "+" | ".join(f"{_lab(cfg,d)} {post_hist.get(pv_,{}).get(d,float('nan')):.3f} -> {fin_[d]:.3f} ({c_:+.3f})" for d,c_ in ch_)
                      +f" || the other domains while WAITING for their visit: mean {np.mean(wt_) if wt_ else float('nan'):+.3f} (max {max(wt_) if wt_ else float('nan'):+.3f})")
            for d in range(NDOM):
                pv=pre_visit.get(d,{}).get(pass_i)
                print(f"    {PILE_DOMAINS[d][:18]:18} end of pass {fin_[d]:.3f} | post-train {po_[d]:.3f}"
                      +(f" (pass {pass_i}: {post_hist[pass_i-1][d]:.3f}, {po_[d]-post_hist[pass_i-1][d]:+.3f})" if (pass_i>0 and d in post_hist.get(pass_i-1,{})) else "")
                      +(f" | before relearning {pv:.3f} (forgotten since last visit {pv-post_hist[pass_i-1][d]:+.3f})" if (pv is not None and d in post_hist.get(pass_i-1,{})) else "")
                      +f" | forgetting {fin_[d]-po_[d]:+.3f}")
    alt={}
    for mode,M in alt_m.items():
        L_=len(PILE_DOMAINS)-1; VL=len(schedule)-1; lastd=schedule[-1]
        fin=[M[(VL,q)][0] for q in range(L_+1)]; pst=[M.get((last_visit.get(q,VL),q),M[(VL,q)])[0] for q in range(L_+1)]
        acc=[M[(VL,q)][1] for q in range(L_+1)]
        alt[mode]={'final':fin,'post':pst,'bpc':float(np.mean(fin)),'forget':float(np.mean([fin[q]-pst[q] for q in range(L_+1) if q!=lastd])),'acc':acc}
        print(f"  [inferred heads: {mode}] final BPC {alt[mode]['bpc']:.3f} forget {alt[mode]['forget']:+.3f} | "
              f"domain inferred correctly (token share, final): "+" ".join(f"{PILE_DOMAINS[q][:6]} {acc[q]:.2f}" for q in range(L_+1)))
    print("\n=== REAL FORGETTING ===")
    fg=[]
    for p in range(len(PILE_DOMAINS)):
        final=bpc_m[-1][p]; post[p]=final if post[p] is None else post[p]; f=final-post[p]
        if p!=schedule[-1]: fg.append(f)
        print(f"  {PILE_DOMAINS[p][:12]:12} | post {post[p]:.3f} | final {final:.3f} | forget {f:+.3f}")
    mf=float(np.mean(fg)); mb=float(np.mean([bpc_m[-1][p] for p in range(len(PILE_DOMAINS))]))
    if cfg.inf_on: inf_eval(model,cfg,_inf_store(model),"END OF RUN"); inf_final_bpc(model,cfg,vals,_S)
    print(f"\nMEAN FORGETTING: {mf:+.3f} | MEAN FINAL BPC: {mb:.3f}")
    if cfg.recon_diag: (replay_diag if cfg.recon_mark else recon_fidelity)(model,cfg,vals,NDOM,_S)
    orc_=None
    if cfg.oracle_mem:                                                                # ORACLE LINEAR MEMORY
        trs_=[make_loader(stream_bytes(d,tseed_(i),max(400_000,int(cfg.oracle_batches*cfg.batch_size*(cfg.seq_len+1)*1.2))),cfg) for i,d in enumerate(PILE_DOMAINS)]
        orc_=oracle_final(model,cfg,vals,trs_,NDOM,bpc_m[-1],_S)
    interf=None; hc_=getattr(model,'_hconf_last',None)
    wall_run=time.perf_counter()-_t_run; tl_=getattr(model,'_timing_log',{})
    if tl_: print(f"RUN WALL-CLOCK: {wall_run/60:.1f} min in total (training {sum(v['wall'] for v in tl_.values())/60:.1f} min, of which dream sampling "
                  f"{sum(v['dream'] for v in tl_.values())/60:.1f} min; evaluation and diagnostics the rest) | {np.mean([v['ms'] for v in tl_.values()]):.0f} ms per regular step")
    vl_=getattr(model,'_vlog',None)
    if vl_: print("  [visit length] steps per visit: "+" ".join(f"{PILE_DOMAINS[d][:6]} {L_}" for d,(g_,L_,b_) in sorted(vl_.items()))
                  +f" | total {sum(L_ for g_,L_,b_ in vl_.values())+cfg.steps_per_task*(NDOM-len(vl_))} vs {cfg.steps_per_task*NDOM} budget")
    return {'oracle':orc_,'vlen':vl_,'cov':getattr(model,'_cov_log',{}),'fsplit':getattr(model,'_fsplit_log',{}),'interf':interf,'hconf':(None if hc_ is None else (hc_[0],hc_[1].tolist())),'fdrift':getattr(model,'_fdrift_log',{}),
            'klpos':getattr(model,'_klpos_log',{}),'wall_run':wall_run,'forget':mf,'bpc':mb,'per_domain':[bpc_m[-1][p]-post[p] for p in range(len(PILE_DOMAINS)-1)],'passes':pass_log,'domains':list(PILE_DOMAINS),
            'gut_after_wiki':(bpc_m[1][0]-post[0]) if (len(bpc_m)>1 and bpc_m[1][0] is not None) else None,'locality':locality_log,'orth':orth_log,
            'post':list(post),'alt':alt,'topics':(topic_log if cfg.topic_diag else {}),'kl':kl_log,'dream':dream_log,'fdiag':fd_log,'stale':getattr(model,'_stale_log',{}),'beta':getattr(model,'_beta_log',{}),'timing':getattr(model,'_timing_log',{}),'recon':getattr(model,'_recon_log',{}),'sleep':getattr(model,'_sleep_log',{}),'battr':battr_log,'bw_dyn':getattr(model,'_bw_dyn',None),'tbirths':getattr(model,'_tspawn_log',[]),'tcombine':getattr(model,'_tcombine_log',[]),'tmerges':getattr(model,'_tmerge_log',[]),'treuse':getattr(model,'_tswitch_log',[]),'rollbacks':getattr(model,'_rb_log',[]),'cross':{int(k):v for k,v in (getattr(model,'_cross_log',None) or {}).items()},'zeroshot':zeroshot,'curve':getattr(model,'_curve',{}),'width':model.blocks[0].d_state,'mlp_width':model.blocks[0].mlp_hidden,
            'recruit':getattr(model,'_recruit_log',{}),'merges':getattr(model,'_merge_log',[]),
            'final':[bpc_m[-1][q] for q in range(len(PILE_DOMAINS))],
            # decomposition (lock after domain L): for domains <=L, measure interference from the
            # post-lock score, so the one-off routing switch isn't counted as forgetting
            **({'switch_gut':post_lock[0]-pre_lock[0], 'wiki_on_gut':pre_lock[0]-post[0],
                'wiki_true_post':pre_lock[1] if len(pre_lock)>1 else float('nan'),
                'interf_forget':float(np.mean([bpc_m[-1][q]-(post_lock[q] if q<len(post_lock) else post[q])
                                              for q in range(len(PILE_DOMAINS)-1)]))}
               if (pre_lock is not None and pre_lock[0] is not None) else
               {'switch_gut':0.0,'wiki_on_gut':float('nan'),'wiki_true_post':float('nan'),'interf_forget':mf})}

# ---------------- PROFILING LABELS: named regions for the one-off GPU profile (profile_visit); no effect otherwise ----------------
_PRF={'on':False}
def _rf(name):
    """v9.8: a named region in the one-off GPU profile (no effect otherwise)"""
    return torch.profiler.record_function('R:'+name) if _PRF['on'] else contextlib.nullcontext()
def _labelled(fn,name):
    import functools
    @functools.wraps(fn)
    def g(*a,**k):
        if not _PRF['on']: return fn(*a,**k)
        with torch.profiler.record_function('R:'+name): return fn(*a,**k)
    return g
for _nm in ['_cd_gen', '_cd_encode', '_cd_mis', '_cd_rows', '_cd_cat', '_recon_loss_mark', '_rs_rowloss', '_rs_pick', '_fgram', '_kw_par', '_delta_core', 'muon_prep', 'varpro_old_refit', 'varpro_refit', 'pu_term', 'film_mod', 'ngram_embed', 'ngp_vec', 'priv', 'bank_dream_batch', '_kf_hook', 'apply_update_gate', 'dctl_rate', '_kf_apply', '_ns5_batch', '_ns5_gram_batch', 'muon_apply', 'ngp_step', 'head_center_']:
    if _nm in globals(): globals()[_nm]=_labelled(globals()[_nm],_nm)

if __name__=="__main__":
    # THE WHOLE-PILE RUN (seed 0). The full research file (threshold_lab.py) keeps every tested option and its results.
    NAME='NB4 core x2 (~171M), 1024-byte context, 16 KB steps, DREAM BANK + STORED TARGETS + OPEN-LOOP DREAM RATE, VARPRO REFIT OF EVERY HEAD + MUON (BATCHED) + FISHER-WHITENED TRUNK UPDATES + FiLM DOMAIN CODES + TAIL AVERAGING + KALMAN PROTECTION + AVERAGING PULL, no teacher KL, constant LR, 10 domains, 5 passes x 2000 steps, lean, fast'
    ALLPILE_CONFIG=dict(
        adam_fused=True,
        amp='bf16',
        bank_chunk=3000,
        bank_rec=True,
        bank_targets=32,
        bank_tgt_chunk_big=128,
        bank_tgt_cpu=True,
        bank_tgt_tail='bin',
        batch_size=16,
        block_extra=(0, 0, 0, 0.5),
        block_extra_mode='start',
        block_write=None,
        borrow_check=100,
        borrow_mode='grad',
        borrow_w=1.0,
        channel_thresholds=True,
        cons3=False,
        cov_diag=False,
        cov_skip=True,
        d_model=1088,
        d_state=1088,
        data_free=True,
        diag_light=True,
        dream=True,
        dream_bank=3000,
        dream_batch=16,
        dream_burnin=64,
        dream_cache=True,
        dream_cap=1.0,
        dream_early_mult=1.0,
        dream_grad_floor=0.05,
        dream_graduate=2,
        dream_head_grad='both',
        dream_lazy_rows=True,
        dream_len=1024,
        dream_recency=None,
        dream_stack=True,
        dream_stream=True,
        dream_stream_block=100,
        dream_stream_prompt=64,
        dream_temp=1.2,
        dream_weight=3.0,
        ema_every=4,
        ema_pull_rel=2.2,
        ema_tail=0.25,
        emb_thresholds=False,
        eval_batch=60,
        extra_domains=('Github', 'StackExchange', 'USPTO Backgrounds', 'DM Mathematics', 'EuroParl'),
        fast_dream_chunk=2048,
        fast_dreams=True,
        fast_kl=True,
        fast_teacher=True,
        feat_drift_diag=False,
        film_k=16,
        forget_diag=False,
        gate_every=10,
        gate_ln=True,
        gn_diag=0,
        head_center=True,
        head_dream_k=True,
        head_select_eval=('heads',),
        imp_combine='max',
        init_mlp_mult=4,
        init_state_mult=4,
        kf_dream=0.5,
        kf_dream_every=200,
        kf_dream_stride=3,
        kf_every=10,
        kf_kalman=5.0,
        kf_kalman_max=18,
        kf_lam=3.0,
        kf_share_a=True,
        kf_source='main',
        kl_pos_diag=False,
        lean=True,
        lean_stats=True,
        log_locality=False,
        logit_cap=30.0,
        lr=0.0007,
        lr_min_frac=0.5,
        lr_sched='warmup_once',
        lr_warmup=50,
        lwf_every=2,
        lwf_post_floor=0.1,
        lwf_posterior=True,
        merge_dream_fwd=True,
        mix_beta_max=2.0,
        mix_beta_tok=True,
        mix_chunk='auto',
        mix_conv=0,
        mix_decay=True,
        mix_heads=8,
        mix_local_blocks=(0,),
        mix_local_width=4,
        mix_lowp=True,
        mix_precond=0.5,
        mix_precond_chunk=64,
        mix_precond_detach=True,
        muon=True,
        muon_batch=True,
        muon_gram=True,
        muon_lr=2.0,
        n_passes=5,
        ngp=(2, 3, 4),
        ngp_rows=4096,
        ngram_emb=(2, 3, 4),
        ngram_hashes=2,
        ngram_lr_mult=28.6,
        ngram_protect=2.0,
        ngram_size=(8192, 65536, 131072),
        ngram_sparse=True,
        no_revisit=True,
        noise_diag=250,
        og_common=0.0,
        og_energy=0.35,
        og_in=0.0,
        og_orth='gate',
        og_orth_boost=2.0,
        orth_gate=False,
        pass_fresh_data=True,
        pass_shard_bytes=33200000,
        pmuon=True,
        pos_emb=False,
        profile_visit=2,
        pu_k=256,
        reserve_mult=4,
        route_mode='none',
        samp_bf16=True,
        samp_lowp_mm=True,
        samp_rescore=True,
        samp_window=32,
        seq_len=1024,
        share_mode='sim',
        shunt=True,
        shunt_consol='continuous',
        shunt_every=1000000,
        shunt_lazy=True,
        shunt_mode='all',
        shunt_old_weight=0.0,
        sim_diag=False,
        sleep_eta=0.01,
        sleep_fixed=0.5,
        sleep_kmax=3,
        sleep_kmax_pass=(3, 4),
        sleep_pmax=2.0,
        sleep_pmin=0.5,
        sleep_ref_frac=0.11,
        sleep_rel=3.0,
        stale_diag=False,
        steps_per_task=2000,
        theta_dist='constant',
        theta_high=0.0,
        theta_low=0.0,
        timing_every=100,
        uniq_thr=1000000000.0,
        use_sens=False,
        varpro_batch=True,
        varpro_k=1,
        varpro_lr=0.5,
        varpro_mode='gn',
        varpro_old=True,
        whiten_bf16=True,
        write_gate_open=True,
    )
    for k,v in _CONFIG_DEFAULTS.items(): setattr(Config,k,v)
    cfg=Config(); cfg.seed=0
    for k,v in ALLPILE_CONFIG.items(): setattr(cfg,k,v)
    import json as _json                                             # OVERRIDES: LAB_OVR='{"key": value, ...}' (e.g. the replay check below)
    _OVR=os.environ.get('LAB_OVR','{"n_passes": 3, "steps_per_task": 400, "recon_w": 0.25, "recon_rows": 8, "recon_mark": true, "cd_seg": 8, "recon_diag": false, "real_w_pass": [1.0, 0.0, 0.0], "self_dream_nreal": false, "vp_nreal": false, "cd_w": 1.0, "cd_rows": 16, "cd_chain": 0, "cd_tries": 2, "cd_fixed": true, "cd_inexact_w": 0.0, "cd_par": true, "cd_eps_floor": 0.0001, "cd_temp": 0.7, "cd_ngp_frozen": true, "cd_fac": true, "cd_lr": -1, "cd_gen_chunk": 1024, "cd_inc": true, "cd_greedy": false, "cd_skip_real": true, "cd_recon_w": 0.0, "cd_recon_rows": 8, "cd_fok": true, "cd_fok_far": 0.005, "cd_fok_chk": 0.05, "cd_fok_min": 20000, "cd_fok_1try": 0.9, "cd_runs": true, "cd_run_skip": 7, "rs_sel": true, "rs_frac": 0.5, "rs_pool": 4, "cd_runs_mask": true, "cd_run_excl": 0, "cd_prof": false, "cd_prof_calls": [50, 1650], "cd_mis32": true, "cd_gen_chunk1": 2048, "cd_nosync": true, "cd_reuse": 1, "cd_repair": true, "cd_repair_n": 1, "cd_partial": true, "cd_first_block": true, "cd_floor_lo": 0.25, "cd_floor_hi": 0.4, "cd_floor_every": 4, "pass_shuffle": true, "cd_runs_any": true, "noise_diag": 0, "profile_visit": 0, "head_select_eval": [], "cd_graph": true, "cd_slot": true, "kf_dream": 1.0, "kf_drop_a": false, "cd_graph_rows": 512, "bank_tgt_chunk_big": 128, "dstep": true, "cdet_log": true, "samp_graph": true, "infer_ctx": true, "infer_rep": true, "seed": 1, "share_mode": "kl", "dream_temp_bpc": 0.0, "dream_temp_low": 1.0, "cd_hint_frac": 0.03, "cd_hint_piece": 3, "cd_hint_adapt": 1.25, "cd_hint_min": 0.02, "cd_hint_max": 0.08, "infer_gap_eps": 0.1, "infer_proto_guard": 0.15, "infer_tiebreak": false, "pu_k": 512, "infer_delta_new": 0.16, "infer_proto_gap": 0.25, "cd_evid_w": 0.5, "cd_sig_every": 4, "cd_bytes_all": true, "dream_batch": 32, "nrem_res": 256, "nrem_mix": 0.5, "nrem_mix_sel": "rand", "nrem_own_sel": "rand", "nrem_retire": -1, "nrem_tgt": "bytes", "pu_cons_steps": 100, "pu_cons_lr": 0.3, "pu_cons_rows": 16, "pu_cons_anchor": 0.25, "pu_cons_retire": 0.25, "pu_cons_cap": 128, "pu_cons_eval": 4, "trace_lesion": [], "trace_lesion_pass": 2, "inf_on": true, "inf_probes": 150, "inf_plant_k": 6, "inf_steps": 200, "inf_rows": 32, "inf_rowlen": 256, "inf_lr": 0.1, "inf_mix": 0.5, "inf_df": 40, "inf_maxfan": 6, "inf_budget": 2000, "inf_eval_every": 1, "inf_cons": "mem", "inf_mem_key": "nameb", "inf_mem_beta": 0.5, "inf_mem_iters": 20, "inf_mem_dxb": 262144, "inf_mem_src": 0.5, "inf_mem_src_mode": "add", "inf_mem_prio": 1, "inf_probe_rare": 1, "inf_mem_maint": 200, "inf_audit": 1, "inf_upd_frac": 0.3, "inf_xfer": "none", "inf_eval_ctx2": 1, "inf_splice": 1, "inf_splice_every": 2, "inf_sleep_order": "nrem_rem", "inf_sleep_cycles": 1, "inf_rem_n": 200, "inf_rem_probe_frac": 0.25, "inf_rem_minrep": 1, "inf_rem_stop": 1, "inf_rem_all": 1, "inf_rem_fact_cap": 3000, "inf_eval_qtime": 1, "inf_qt_n": 40, "inf_qt_every": 5, "inf_mem_gate": 3.0, "inf_audit_rule": "recent", "inf_audit_cascade": 1, "inf_audit_debug": 1, "inf_eval_qtime3": 1, "inf_qt_comp": 2, "inf_audit_minrep": 2, "inf_idmem": 1, "inf_id_dim": 4096, "inf_rules": 1, "inf_para_frac": 0.5, "inf_comp_k": 2, "pu_cons_brake": 0.005, "inf_mem_anchor_named": 0.5, "inf_bridge_order": "rare", "inf_name_strict": 1, "inf_eval_ctx": 1, "inf_eval_hard": 1}')
    for k,v in _json.loads(_OVR).items(): setattr(cfg,k,tuple(v) if isinstance(v,list) else v)
    NAME='STAGE C CONDUIT cg20: cg19 (no lesion) + cg20s = cg20r + FOUR GPU FIXES: (1) belief revision at GPU exposure (recency rule needs 2 repeats, not 3; facts are seen 1-2 times) (2) identifier memory 4096 dimensions (1024 overloaded with 3000 facts) (3) RULE LAYER for paraphrase: rules phrase p => phrase q proposed from the fact records, kept only if they save bits on held-out evidence and reach half the going rate, applied when confident (derived facts replayed into the store; question-time paraphrases rewritten), with PARAPHRASE PROBES (facts stated in either or both wordings, competing relations per subject) and a PARAPHRASED one-hop readout (4) CONSOLIDATION BRAKE (a private-into-shared phase that makes old domains or its own domain worse by more than 0.005 BPC is rolled back) + IDENTIFIER MEMORY readout; cg20r = cg20q + FIX: NREM fact replay skips facts the grounding audit superseded (cg20q kept writing the old fact back into the store, so updated facts stayed stale); cg20q = cg20p + STRICT STORE GATE 3.0 (a store correction applies only where its largest logit is at least 3; cut the store cost about 60 pct locally with no recall loss) + RECENCY RULE IN THE GROUNDING AUDIT (for a single-valued relation the most recently stated object wins if it is repeated, at least a quarter of the old count; the old ratio rule left near-ties unresolved forever) + CASCADING RETRACTION (conclusions built on a retracted conclusion are retracted too) + REVISION EVIDENCE readout + MULTI-HOP QUESTION-TIME readout (depth 2 to 5, working memory and identifier lookups, decoys, paraphrased questions, competing relations); cg20p = cg20o + DRIFT-FREE STORE KEY (fixed random code of the last 8 bytes picks the units, the name picks the bank; no trunk features) + NREM FACT REPLAY (each visit up to 3000 one-hop facts, ranked by length over spread, written into the store) + REM LINKING WRITTEN (bridge recalled from the store, bridge-specific check, grounded in recorded facts, any grounded candidate; fixed lookup domain) + QUESTION-TIME CHAINING WITH A WORKING MEMORY readout (prompt facts written into a scratch store, two lookups, new chains and decoys) - skill transfer removed; cg20o = cg20n + SPLICE-AND-APPRAISE readout (true links vs false recombinations: raw confidence, specificity, path check by two own lookups, store lookups, oracle bridge) + SHADOW REM RECOMBINATION after the store writes (bridge recalled from memory, bridge-specific check, grounded in recorded facts; counted only, nothing written); cg20n = cg20m + GENTLE SKILL TRANSFER (worked examples from verified conclusions, lr x0.01, BPC brake; every second chain held out) + SHARPER IN-CONTEXT TEST (distractor fact last, both orders, in-context names as candidates, broken-bridge decoy); cg20m = cg20l + STORE MAINTENANCE (rewrite stored conclusions no longer recalled as the trunk drifts) + GROUNDING AUDIT with changing facts (30 pct of two-hop chains change in pass 2); cg20l = cg20k with the n-gram sparse-gradient FIX (cg19 bug: consolidation accumulated them -> GPU crash) and exposure-first store writes (rare-links-first starved probe conclusions); cg20k = cg20j + ANSWER-OR-ABSTAIN and SHOW-SOURCES readouts + calibration; cg20j = cg20i + guarded fast-store indexing (fix for the cuda illegal-address crash) + RARE LINKS FIRST in store writes + probe names with q/x/z; OFFLINE INFERENCE IN SLEEP (rarest bridges first, budget 2000, strict names: no non-ASCII neighbours, no capitalised common words; one-hop source facts ADDED to the store conclusion writes; unconsolidated conclusions written first; do-no-harm anchors aimed at text with stored names; in-context chaining probes), conclusions consolidated into a NAME-INDEXED FAST STORE (sparse delta-rule logit corrector, weights untouched; inf_cons w restores weight consolidation) (150 planted probe chains, sentence store of real/VERIFIED bytes, bridge-centred chaining with hub abstention, typed conclusions, three-hop via stored conclusions, conclusions consolidated alone into kept verified windows) | cg19: cg17e + PRIVATE-INTO-SHARED CONSOLIDATION on verified windows after every closed-loop visit (100 steps, 0.3x lr, 25% old-domain anchors) + TRACE-LEVEL RETIREMENT (KL < 0.25 x start) + CAPACITY CAP 128 windows per domain + TRACE LESION of Wikipedia and USPTO in pass 3 | '+NAME+' | overrides '+_OVR
    print(f"\n################ {NAME} | seed {cfg.seed} ################")
    r=run(cfg)
    print("\n==================== SUMMARY ====================")
    print(f"final BPC {r['bpc']:.3f} | forgetting {r['forget']:+.3f} | post-train {np.mean(r['post']):.3f}")
    for k,p in enumerate(r.get('passes',[])):
        pre=[x for x in p['pre'] if x is not None]
        print(f"pass {k+1}: end {p['bpc']:.3f} | forgetting {p['forget']:+.4f} | post-train {np.mean(p['post']):.3f}"
              +(f" | before relearning {np.mean(pre):.3f}" if pre else "")+" | label-free "+" ".join(f"{m} {v:.3f}" for m,v in p['alt'].items()))
    dn=r.get('domains',PILE_DOMAINS)
    for d in range(len(dn)):
        print(f"   {dn[d][:18]:18} end of each pass: "+" -> ".join(f"{p['final'][d]:.3f}" for p in r.get('passes',[])))
    tl=r.get('timing',{})
    if tl: print(f"timing: {np.mean([v['ms'] for v in tl.values()]):.1f} ms per regular step | training {sum(v['wall'] for v in tl.values())/60:.1f} min | run {r.get('wall_run',float('nan'))/60:.1f} min")
