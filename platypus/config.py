"""Config: every option with its default; the base domains and their difficulty."""
import math
import torch


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
    block_auto=False                # use the attribution to set each block's write strength automatically: blocks whose damage
    block_write=None                # e.g. (1,1,0.5,0.5): write strength per block once there is something to protect
    forget_diag=False               # at each domain end: which part (backbone / each block / heads / routing) lost each earlier domain
    fd_batches=5
    logit_cap=0.0                   # ARCHITECTURE: >0 soft-caps every head's logits at +-logit_cap (cap*tanh(z/cap))
    gate_ln=False                   # ARCHITECTURE: the SSM output gate reads LayerNorm(x) instead of the raw residual x
    dream_cap=0.0                   # >0: ROBUST dream KL, per-token KL -> cap*log(1+KL/cap) (bounded influence of outliers)
    dream_refresh=50                # every N steps, dream_batch fresh sequences replace the oldest in one old domain's pool
    cross_lr=0.01; cross_min=-1.0; cross_max=2.0
    pg_mode=None                    # POLARISATION gating of weight writes: None | 'weight' | 'proj' | 'proj_uniform' | 'proj_seqnov'
    polar_diag=False                # print, once per domain, whether token angles separate old from new content (no training effect)
    data_free=False                 # NO stored samples: sensitivity from each domain's last live batches, orthogonal
    cov_every=10; cov_tokens=512    # data-free covariance sampling: every N steps, token subsample size
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
    infer_drift_fix=0               # cg20x ROUTING FIX: slot gaps corrected for trunk drift, measured on each slot's own dream-bank rows (see route_fix)
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
    oracle_batches=64
    oracle_lams=(1e-2,1e-1,1.0,10.0)
    oracle_scales=(0.25,0.5,1.0)        #   S* also scaled by each of these (the trunk was trained reading memories of the delta rule's own scale)
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
    recon_w=0.0                     # REPLAY TRAINING (memory as a CONDUIT): with real-text gradient on, recon_rows windows per step are encoded into their
    recon_rows=8                    #   end-of-window delta-rule memory (differentiable) and the model is trained to REGENERATE each window starting from its
    recon_mark=False                # REPLAY TRAINING v2 (harness encoder/decoder): each of recon_rows random cd_seg-byte SEGMENTS is encoded into its
    cd_w=0.0                        # CONDUIT REPLAY (Stage C, the harness recipe): in passes whose real_w_pass is 0, cd_rows random cd_seg-byte segments of the
    cd_rows=16                      #   current real batch are encoded into delta-rule traces by a FROZEN teacher (the visit-start weights), replayed free-running
    cd_seg=64                       #   from the trace (cd_mark, temperature cd_temp), checked for EXACTNESS by re-encoding the replay (trace mismatch <= eps; up to
    cd_tries=4                      #   cd_tries draws for the inexact rows), and the student learns the teacher's trace-conditioned targets reading the replay as
    cd_temp=0.7                     #   ORDINARY text (KL, weight cd_w; inexact rows count 1/4). The weights never get a gradient from the real bytes.
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
    sleep_merge=1                   #   rows per training step = sleep_merge x batch_size (merged batch)
    sleep_dream=True                #   dream rows scaled with the merged batch (same real : dream ratio as a normal step)
    sleep_wake=True                 #   rows not queued train their domain's private rows (head-only backward)
    sleep_score='loss'              #   'loss' = rows ranked by loss; 'mem' = by MEMORY GAIN: loss on the window's second half read WITHOUT the first half minus
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
                                    #   itself, rescaled to Muon's size (no Newton-Schulz orthogonalisation); the per-domain private units (pu_*) and
                                    #   FiLM matrices keep Muon. Local d128: with private units 2.006 -> 1.960 law at the same near-zero forgetting
    muon_ns_steps=5                 # cg12a: Newton-Schulz iterations per Muon step (5 = standard; 3 = cheaper, weak directions amplified ~41x instead of ~490x)
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
    cd_reuse_k=0                    # >0: SPACED REPLAY REUSE (same domain), see cd_reuse_mode
    cd_reuse_lag=15
    cd_reuse_mode='uniform'         #   'uniform' | 'mix' (half by fragility, half uniform) | 'upw' (fragile pieces x cd_reuse_upw) | 'loop' (maintenance loop)
    cd_reuse_upw=2.0
    cd_loop_easy=0.005              # 'loop': degradation (share of positions where the teacher's top byte now differs) <= easy: interval x2; > fail: second chance,
    cd_loop_fail=0.05               #   then dropped; else interval halved; dropped when consolidated (KL < kldrop) or after maxuse re-checks
    cd_loop_maxuse=4
    cd_loop_kldrop=0.02
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
    inf_name_mem=1                  # 10 Nov NAME MEMORY (inference.namemem; on by default): one entry per fact beside the weights, two-part key
                                    #   (subject name + answer slot), SERAC-style slot gate with re-anchoring; acts only in the store-on probe evaluation and its readouts
    inf_nm_max=384                  #   name memory: most entries (newest conclusions up to half, then newest recorded facts)
    inf_nm_selfc=1.0                #   name memory: self-contrast weight when fitting values
    inf_nm_min_seen=2               #   name memory: a recorded fact must have been seen in at least this many sentences (most seen first)
    inf_nm_min_entries=32           #   name memory: stays off until it holds this many facts (its slot gate needs material)
    inf_nm_value='hidden'           #   name memory: 'hidden' (default) = a fitted change to the edited layer's output; 'logit' = the stored answer
                                    #   byte raised to beat the best other byte by inf_nm_margin nats, nothing fitted (local full run, 10 Nov: recall
                                    #   as good or better, but misfires cost 3-4x more on a full memory; staleness is mostly the keys, in both)
    inf_nm_margin=1.0               #   name memory, logit value: margin in nats
    inf_carry=0.0                   # 10 Nov CARRY THE SUBJECT (inference.carry; off): weight of an auxiliary loss teaching the trunk to keep the latest name
                                    #   readable at the bytes after it (InfoNCE among the batch's names, at the edited layer's input)
    inf_carry_dist=32               #   carry: bytes after a name that are trained
    inf_carry_n=128                 #   carry: sampled bytes per batch
    inf_carry_tau=0.1               #   carry: InfoNCE temperature
    inf_carry_dim=128               #   carry: projection size
    inf_carry_every=1               #   carry: apply the loss every this many training steps
    inf_distill=0                   # 10 Nov DISTILLATION IN SLEEP (inference.distill; off): the name memory's facts taught to the trunk's blocks each sleep in
                                    #   self-written wordings (never a test wording), anchors on stored sentences, a brake with rollback
    inf_distill_steps=40            #   distillation: steps per sleep
    inf_distill_rows=16             #   distillation: questions per step
    inf_distill_lr=0.01             #   distillation: learning rate as a multiple of lr (first dose; then a third, a ninth if over the brake; GPU 10 Nov: x0.03 cost +0.098 BPC)
    inf_distill_anchor=1.0          #   distillation: weight of the anchor KL on stored sentences
    inf_distill_arows=16            #   distillation: anchor rows per step
    inf_distill_wake=0.0            # 10 Nov INTERLEAVED DISTILLATION (off): weight of a few question rows about stored facts added to every real-text
                                    #   training step (same backward and optimiser: Fisher whitening, real-text loss and dream rows apply; no burst)
    inf_distill_wake_rows=4         #   interleaved distillation: question rows per training step
    inf_distill_wake_params=''      #   11 Nov: 'mlp_out' = only the blocks' mlp_out weights take the question gradient ('' = every parameter)
    inf_distill_wake_proj=0         #   11 Nov: 1 = ANCHOR PROJECTION (A-GEM): the question gradient's component that would raise next-byte loss on
                                    #   stored real sentences (inf_distill_wake_rows of one slot per step) is removed
    inf_distill_readonly=0          #   11 Nov CONTROL: 1 = the question pool, the never-taught split and every distillation readout as usual, but no question rows trained
    inf_distill_pool_max=0          #   11 Nov CAPACITY: >0 = the question pool takes up to this many facts and conclusions from the store (0 = the name memory's material, cap inf_nm_max)
    inf_distill_conc22=1            #   11 Nov READOUT: conclusions taught vs never taught x closed-form edits in place vs subtracted (+ never taught with both premises taught)
    inf_distill_ever=256            #   11 Nov READOUT: recall over a stable sample of every fact EVER taught (and control ever held), by still/left the pool and age
    inf_distill_wake_ondream=0      #   11 Nov: 1 = question rows only on steps that also train on dream rows, weight x the dream interval (same dose)
    inf_distill_wake_akl=0.0        #   11 Nov: >0 = SOFT ANCHOR: + this x KL(the model at the last sleep || now) on stored real sentences of old slots
    inf_distill_repair=0            #   11 Nov: >0 = REPAIR REPLAY at each sleep: this many steps of that KL on the old slots (previous sleep's predictions)
    inf_distill_repair_lr=0.1       #     its Adam lr = cfg.lr x this
    inf_distill_wake_spaced=0       #   11 Nov: 1 = SPACED REHEARSAL: facts answered in a taught wording at a sleep sampled 1/interval (doubling to 16)
    inf_distill_wake_selfkl=0       #   11 Nov: 1 = facts answered in a taught wording held by KL to their own answer at that sleep, not cross-entropy
    inf_distill_wake_hint=0         #   11 Nov: 1 = HINTED SELF-DISTILLATION instead of question rows: KL(trunk + name memory || trunk) on stored real
                                    #   sentences naming taught subjects, at the positions the memory changes (after arXiv 2610.05200)
    inf_gate_theta=0.0              #   11 Nov RHYTHM GATES (inference.distill): >0 = THETA phase gate, mean consolidation phase length in steps
    inf_gate_theta_jitter=1         #     1 = stochastic phase lengths (exponential), 0 = fixed (control)
    inf_gate_beta=0.0               #   >0 = BETA drift brake strength s: question weight x min(1, (ref / dream-KL ema)^s)
    inf_gate_beta_ref=1.0           #     ref = previous visit's median dream KL x this
    inf_distill_spindle=0           #   1 = SPINDLE coupling: question rows only inside the sleep's repair replay burst (needs inf_distill_repair)
    inf_eval_chunk=0                # 11 Nov: >0 = probe evaluation's beam search and name scoring run in chunks of this many rows (0 = one pass; GPU: 128)
    inf_sleep_phase_free=0          # 11 Nov: 1 = release the CUDA cache after each timed sleep phase (with inf_sleep_phases)
    inf_gauss_drift=0               # 11 Nov READOUT: Gaussian (mean, covariance) drift of each layer's states on fixed stored sentences, sleep to sleep
    inf_distill_brake=0.01          #   distillation: rolled back if the visited or the old domains' BPC rises by more than this
    pu_cons_steps=0                 # cg19 (1) PRIVATE-INTO-SHARED CONSOLIDATION after every closed-loop visit: this many offline steps in which a
                                    #   FROZEN copy of the model (the visit's private units ON) teaches the SHARED block weights to reproduce it with
                                    #   that domain's private units OFF, on the visit's VERIFIED windows (the NREM reservoir; needs nrem_res > 0)
    pu_cons_lr=0.3                  #   lr multiplier (plain Adam over the blocks' shared weights, fresh per phase)
    pu_cons_rows=16                 #   rows per step
    pu_cons_anchor=0.25             #   share of rows from OLD domains' kept verified windows, taught to keep the frozen copy's output (interleaving)
    pu_cons_retire=0.25             # cg19 (3) TRACE-LEVEL RETIREMENT: a window is retired (not kept) once its KL with the private units off falls below
                                    #   this share of its value at the phase start (the shared weights reproduce it: absorbed); <0 = never
    pu_cons_cap=0                   # cg19 (3) CAPACITY CAP: >0 = each domain keeps at most this many verified windows (weakest-first eviction:
                                    #   the most absorbed, lowest remaining KL, go first)
    pu_cons_eval=4                  #   held-out batches per domain for the phase's before/after diagnostics
    trace_lesion=()                 # cg19 (2) TRACE LESION: true domains whose visits are SKIPPED from pass trace_lesion_pass (0-based) on: no traces,
    trace_lesion_pass=2             #   no replay (their old dreams stay) - does the knowledge hold without the hippocampus?
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
                                    #   visit's teaching weight is unchanged (normalised by its own windows)
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
                                    #   x the share of disagreeing positions in its piece (novel pieces trust the evidence more)
    prot_decay=1.0                  # <1: PROTECTION DECAYS WITH DISUSE: an old domain's dream rows are weighted prot_decay ** (visits since its last visit)
    cd_sig_every=0                  # >0: REPLAY-SIGNAL DIAGNOSTIC every this many conduit calls (no training effect): distribution of the learning signal per
                                    #   replayed piece, how concentrated it is, and whether the surprise at experience predicts it (gating before replaying)
    cd_evid_w=0.0                   # EVIDENCE TARGETS (pass-2 closed loop): at positions where the teacher's top byte differs from the VERIFIED replayed byte,
                                    #   the target becomes (1 - w) x teacher + w x that byte (the model's own verified reconstruction, no real text). Fact test:
                                    #   with teacher-only targets a corrected fact never overwrote an entrenched one (self-distillation confirmation bias)
    fact_spec=()                    # UPDATE-OR-PROTECT TEST: synthetic facts ' Name:1234;' injected into training text, entries [domain, pass, version, every_bytes]
                                    #   (version 0 = the original values, 1 = the changed values); evaluated through every fact domain's slot at each visit end
    fact_n=8                        #   number of facts (names and 4-digit values drawn from fact_seed)
    fact_seed=5
    dream_lesion_pass=1             #   (the hippocampal-lesion analogue of Tse et al. 2007: does the domain's knowledge survive without its replay?)
    dream_prio=0.0                  # cg16: PRIORITISED DREAM ROWS within each old domain's bank: this share of rows is drawn in proportion to the row's KL
                                    #   the last time it was used (rows never used: the domain's mean; no extra forward), the rest uniformly. Local dream_test:
                                    #   forgetting -8% (4 rows/step), -2% (every 2nd step), -3% (16 rows), no cost to learning
    dream_prio_sync=8               #   priorities copied to the host every this many dream steps (one sync)
                                    #   bank write only the rest is re-sampled, and ALL rows get fresh targets from the current weights (the GPU writes a
                                    #   domain's bank right after its own visit, so stale pass-1 targets would pull towards the weaker pass-1 model). Local
                                    #   prot2 test (anchored inputs AND targets): drift +0.031 vs +0.119 regenerated, +0.010 fully fixed
                                    #   update because every dream step is clipped to norm 1 (only the real/dream mix changed), so raise the clip instead
                                    #   behaviour there (it acted only once old domains existed and settled at 0.25-0.28 from the second visit)
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

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'bank_roll': 'platypus.dream_bank',
    'dream_cap': 'platypus.dream_bank',
    'dream_refresh': 'platypus.dream_bank',
    'forget_diag': 'platypus.diagnostics',
    'inf_probes': 'platypus.inference.probes',
    'noise_diag': 'platypus.protection',
    'recon_sleep': 'platypus.diagnostics',
    'sim_diag': 'platypus.protection',
    'stale_diag': 'platypus.diagnostics',
}
