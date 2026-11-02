# Platypus

**A language model that learns without forgetting and thinks in its sleep.**

Platypus is a 171-million-parameter byte-level language model that learns ten text domains one after another without forgetting earlier ones. After its first look at the data it learns only from what it can verify, and while it "sleeps" it links facts it met in different places into new conclusions, checks them against their sources and keeps only the ones that hold up. It then dreams: a trained JEPA picks which memories to splice together, the model judges each dream by whether the two halves make each other easier to predict, and what passes is consolidated into its weights under a brake that protects old knowledge. It needs no domain labels, no stored training data and, in its default rule-free mode, no hand-written language rules: entities and relations are found from word statistics, and links are kept only if they pay for themselves in compression. Its memory and sleep cycle also work when attached to a frozen GPT-2.

> Copyright 2026 **CHRISTIAN BIRD**. Licensed under the [Apache License 2.0](#licence).

Current build: **cg26f**.

---

## Contents

- [Why it exists](#why-it-exists)
- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Results](#results)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Reading the logs](#reading-the-logs)
- [Project status](#project-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citing](#citing)
- [Licence](#licence)

---

## Why it exists

Platypus targets four long-standing problems with language models:

| Problem | What goes wrong today |
|---|---|
| **Catastrophic forgetting** | Training on a new domain overwrites the old one. The usual fixes store old data, freeze parameters, or need to be told which task is which. |
| **The two-hop curse** | A model trained on "A works with B" and, separately, "B lives near C" generally cannot say where A's colleague lives without reasoning step by step. |
| **Edits that don't propagate** | Change one fact and the facts that depended on it stay stale (see the MQuAKE and RippleEdits benchmarks). |
| **Confident guessing** | A model answers questions about people who don't exist as fluently as about people who do. It has no signal for "I was never told this". |

## What it does

- **Continual learning without stored data.** Domains are learned in sequence. Old domains are protected by rehearsing the model's own dreams (stored self-generated windows with their top-32 predictions), by curvature-aware (Fisher-whitened) updates, and by small per-domain capacity.
- **A strict closed loop.** After the first pass the model sees no real text again. It learns only from replays of its memory traces that it can verify exactly against what it originally read.
- **Label-free routing.** The domain is never given. The model recognises returning domains by how well it predicts them, corrected for how far each domain's slot has drifted since it was last trained.
- **Rule-free inference in sleep.** After each visit it links facts met in different places. All links live in one sparse table, and every possible two-step link is scored at once (one-scan salience) and promoted by gain × need. It replays salient memories (NREM-like), proposes links from memory and keeps only grounded ones (REM-like), strengthens conclusions it meets again, revises beliefs and retracts conclusions that depended on superseded facts.
- **A fast memory that can abstain.** Conclusions live in an entity-banked, delta-rule logit corrector outside the weights. It is silent for entities it has never seen, so the model can say "I don't know", and it costs almost nothing on ordinary text.
- **Two recall pathways, as in the hippocampus.** Beside the exact store, an overlapping link memory with spelling-aware codes completes damaged cues. A recognition gate uses exact recall when the cue is a known name and pattern completion when it is not.
- **Dreams, judged by the model itself.** NREM-like dreams splice two stored sentences; REM-like dreams pair a recent memory (the cue) with a trace of older knowledge, chosen neither too close nor too far from the cue and different from each other. A dream is kept only if each half makes the other cheaper for the model to predict than random text does. A trained JEPA chooses what to dream, among memories that share a rare name with the cue: 30–40% of its NREM and REM choices are about a gold-linked fact on Wikipedia-style text, against about 1% for random pairs.
- **Systems consolidation into the weights.** Verified conclusions and accepted dreams are replayed into the trunk, with the update steered away from what old domains rely on (their Fisher factors), a brake that rolls back any harmful sleep, and a dose that retries lower and climbs back when it can.
- **Consolidation of shared structure.** After closed-loop visits, what a domain's private units learned is distilled into the shared weights, at a gentle rate and under a brake that rolls back any phase that harms old domains.
- **An identifier memory for exact chains.** Verified facts are bound into a vector-symbolic memory, so multi-step chains need no round trip through text, and broken chains are declined.
- **Rules that pay for themselves.** Equivalences between relation phrases ("answers to" and "reports to") are discovered from the text and kept only if they compress held-out evidence (minimum description length), so paraphrased questions can be answered.
- **Works as a plug-in.** The store and sleep cycle have been attached to a frozen GPT-2 without changing a weight.

## How it works

```
             BY DAY                                         BY NIGHT
 real text ──► slow network ──► next-byte          dream rehearsal ─► verified replay ─► one-scan linking
 (one domain,   (171M params,    prediction ◄──┐                                        & revision
  no labels)     general skill)      ▲          │                                           │
                                     │          └──── corrects ◄──── fast memory ◄──────────┘
                                     │                           (facts by entity, silent when unsure)
                                     │                                      │
                                     └── protected, braked replay ◄── dreams (NREM splices, REM cue x trace)
                                         of verified conclusions        chosen by JEPA, kept only if each half
                                                                        makes the other easier to predict
```

The slow network learns general skill, and only from real text or verified replay. The fast memory holds facts. Sleep connects the two by doing inference offline.

**Architecture (GPU build).** 1024-byte windows; byte plus hashed 2/3/4-gram embeddings; block 0 is a causal local filter; blocks 1–3 are gated delta-rule mixers (8 heads, WY-form parallel solve); residual width 1088; one soft-capped output head per domain. About 171M trunk parameters plus 223M hashed n-gram parameters.

**Key principles**

| Decision | Principle |
|---|---|
| Dream rehearsal plus Fisher-whitened steps | Sequential Bayesian learning under a Laplace approximation: dreams supply the old likelihood, curvature supplies the precision. |
| Learn only from verified replay | Self-training is a fixed-point iteration that converges on its own errors. A truth gate removed 100 of 100 false attractors in a synthetic test. |
| Delta-rule fast memory | Writes are error-driven, so known facts cost nothing; unknown entities have no code, so abstention is exact; a confidence gate keeps the cost on ordinary text near zero. |
| One-scan salience | Every two-step path is scored at once as a sparse product S = W′W′, with log-count links and hub damping, then ranked by gain × need, instead of splicing and checking candidates one at a time. |
| Links that pay their way | Links compete for a fixed pool. The promotion bar rises only when the pool is full, and links pushed out are erased from the store. |
| Recency rule for belief revision | Exponentially decaying supports keep a constant ratio, so ratio tests stall. The most recent repeated statement wins, and retraction cascades through dependants. |
| Exact and overlapping recall | Separation keeps links precise and completion recovers partial cues. A recognition gate chooses between them, as familiarity is thought to do in the brain. |
| Gentle, braked consolidation | Shared structure is cheap to move into the weights and arbitrary bindings are costly, so consolidation runs at 1/100 of the training rate and is rolled back if it harms old domains. |
| Rules by MDL | A rule is kept only if its two-part code saves bits on held-out evidence, judged against the going rate of the rules already paying their way. |
| Dreams judged by mutual compression | Two passages belong together if each makes the other cheaper to predict than chance text does. The model needs no outside verifier; the bar is set by what random pairs achieve in the same sleep. |
| Protected, adaptive replay | Replay buys recall in the weights at a price in forgetting. Steering each update by the old domains' Fisher factors, braking, retrying lower and climbing back keeps the price inside a budget. |
| JEPA as the dream selector | A joint-embedding predictive model (encoder head, EMA target, predictor, SIGReg) learns from links the system has verified itself, then predicts which memories belong together. Its closed-form relative (spectral word codes from the platonic-representation result) seeds the splice proposals. |
| Name index first, JEPA second | Searching the whole store, the JEPA picks same-genre text. Restricting candidates to memories that share a rare name with the cue brings recall; the JEPA then ranks them, and is trained against those same name-sharing alternatives (hard negatives). |

## Results

**Dreams that find related memories (cg26b–cg26e, ten Pile domains, seed 1).** Relatedness is judged against Re-DocRED gold facts planted in the stream.

| | Result |
|---|---|
| NREM dreams (`jepa-name`) related to a gold fact or chain | 11 of 40 and 13 of 37 (ArXiv, PubMed), against 1–2 of 128 for random pairs |
| REM dreams (`cortex-jepa-name`) related | 12 of 30 at PubMed (40%), 7 accepted, against 3 of 128 for the earlier REM matcher and 1 of 128 for random |
| Dreams about the same entity, later domains | NREM 47–71%, REM 8 of 31, 14 of 16, 11 of 12, 7 of 11; random pairs never |
| JEPA held-out partner retrieval | first training sleep (the only clean measurement so far): all linked pairs 0.34–0.41 (chance 0.06), real-text pairs 0.10–0.20 (chance 0.10). Later sleeps reported 0.8–1.0 on real text, but those were inflated: the held-out split was re-drawn every sleep while the heads persisted, so most held-out pairs had been trained on before. Fixed in cg26g/h (a permanent held-out set) |
| The dream judge | known links pass 3 of 5 (+2 bits) early and 16 of 16 (+5 to +6 bits) later; random pairs 1–4 of 128 |
| REM fact inference | no accepted grounded fact scored wrong (about 150 in one run); 177–183 of 197 two-hop chains found |
| Replay into the weights | kept every sleep at ×0.0033; old domains +0.01 to +0.03 per sleep while their BPC kept improving |
| GPU memory | peak 52–57 GB of 95 (during daytime training); sleep 19–51 GB |

Within the name pool the JEPA's ranking so far matches a TF-IDF ranking on names, so most of the gain comes from the name index, and the clean first-sleep evidence for the JEPA generalising on real text is weak. cg26f trains the JEPA against name-sharing alternatives to test whether it can add more. Whether these dreams beat random dreams on two-hop recall and forgetting is not yet tested (the random-dream control is next).

**Dreams and consolidation at full size (cg21b–cg25, ten Pile domains, seed 1).**

| | Result |
|---|---|
| Two-hop recall from the weights after a sleep of replay | rose at every sleep, e.g. 0.12 → 0.38, 0.07 → 0.21, 0.08 → 0.29 (store switched off) |
| Price in forgetting (plain replay, one visit) | ×0.03: +0.40 two-hop for +0.158 old-domain BPC; ×0.01: +0.20 for +0.064; ×0.0033: +0.13 for +0.022 |
| Kept replay | old domains unharmed, at best improved (−0.027) |
| Does it last? | after a full visit of training: 0.20 → 0.20, 0.14 → 0.14, 0.12 → 0.12 (cg24), 0.25 → 0.12 (cg21c) |
| Splice test | known links pass 13–16 of 16 (+4 to +9 bits), random pairs 0–7 of 128 |
| Protected replay | 70% less old-domain damage at one paired visit, no difference at another: not yet consistent |
| Latent consolidation (tried on a copy) | costlier than byte replay (+0.03 to +0.09 BPC) and no two-hop gain |

Transfer of inferred two-hop facts into the weights, lasting across visits, is the first evidence of systems consolidation in Platypus. The numbers are small per sleep and come from one seed. Accepted dreams were mostly unrelated pairs until the name filters of cg25–cg26.

**Rule-free inference at full size (cg20x, ten Pile domains, three passes, planted fact chains).** No relation list and no name patterns:

| | Result |
|---|---|
| Two-hop chains found | 129 of 130 by the end (197 of 197, and 47 of 47 three-hop, by visit 11) |
| One-, two- and three-hop recall with the fast memory | 1.00 / 1.00 / 1.00 |
| Changed facts, one- and two-hop | 1.00 / 1.00, no stale answers (0.45 at visit 15, 1.00 from visit 20) |
| Questions in an unseen wording | 0.91 |
| Answers backed by a stored conclusion | 129 of 129 correct, each citing the true chain |
| Store cost on ordinary text | none measurable |

The previous rule-free build (cg20u) found 11 of 197 chains at the same scale.

**The cg20y run (full size, three passes, sleep inference on).** Mean BPC 1.704, 1.455 and 1.368 after passes 1, 2 and 3; forgetting +0.058 at the end (+0.039 without Wikipedia, see below). Inference stayed exact throughout: two-hop 0.99 and three-hop 1.00 with the store, changed facts 1.00 with nothing stale, 128 of 128 cited sources containing the true chain, and every unknown subject abstained. The store costs +0.004 BPC on ordinary text. Consolidation was kept in every phase except DM Mathematics, and never harmed old domains. Routing was right on 25 of 29 visits. Two errors came from choosing the slot trained one visit earlier; one of them trained USPTO text into Wikipedia's slot, which explains Wikipedia's +0.171 forgetting. cg20z blocks that choice. CLS probes show the inferred facts still live only in the store: switching the store off drops them to 0.02–0.04 for old and recent conclusions alike.

**Continual learning (full size, ten Pile domains).** With dream rehearsal and curvature-aware steps the model reached 1.17 bits per character (BPC) by its third pass, with forgetting of 0.03. In the strict closed loop (no real text after the first pass):

| Second pass | BPC | Forgetting |
|---|---|---|
| Learns nothing new (lower bound) | 1.739 | – |
| Learns only from verified replays | 1.474 | 0.049 |
| Reads real text again (upper bound) | 1.468 | 0.044 |

Verified replay recovered 98% of the gap, with no domain labels. The cg20x run (with sleep inference on) ended at 1.371 BPC, with forgetting of −0.011, +0.075 and +0.064 after passes 1, 2 and 3. Label-free routing was right on 25 of 29 visits; the errors were between the closest pairs of domains (Gutenberg and Wikipedia, GitHub and StackExchange).

**Local tests behind cg20z (five-domain harness).**

| Test | Result |
|---|---|
| Staging v2 (commit a conclusion only when its premises are read again) | 35/35 planted chains, accuracy 1.00; 85% of pooled conclusions are real planted links, against about 70% without it |
| Staging v1 (evidence fading 0.8 per sleep, bar of about 25 bits) | nothing ever passed the bar; 5/35 chains |
| Salience in place of rarity (two versions) | cleaner conclusions, but planted recall fell to 9–16 of 35 |
| Symmetry of links (both directions strong) | recall kept, precision unchanged on its own |
| Routing: "never the slot trained at the last visit", replayed on the cg20y log | fixes visits 14 and 22, changes no other decision |

**Local tests behind cg20y.**

| Test | Result |
|---|---|
| Consolidation, 0.01× rate, 50 steps | 32–38% of the private units' advantage moved into the shared weights per phase, within the brake (0% at 0.3×; 11% at 100 steps; 0% at 200) |
| Two-hop recall with intact names: exact store / recognition gate | 1.00 / 1.00 |
| Two-hop recall with damaged names (dropped letter, swapped letters, lower case): exact store / recognition gate | 0.00 / 0.44–0.69 |
| Routing: strong-prototype rule replayed on the cg20x routing log | fixes 1 of the 4 errors, changes no other decision |

**Local tests behind cg21–cg26 (five-domain harness and offline).**

| Test | Result |
|---|---|
| Link detection on Re-DocRED real text (chain premises found in 6,000 proposals, two seeds) | hybrid TF-IDF + spectral 405 / 411, spectral 376 / 377, TF-IDF 339 / 359, random 2 / 3 |
| Trunk replay of conclusions vs none (small model) | no replay forgets least (−0.018 vs −0.004); dreams only dilute the harm of templated rows |
| JEPA dream selector, first training sleep | held-out partner found 0.06 → 0.81 (chance 0.06) |
| New dream sources (cg26b, cg26c) | forgetting and final BPC unchanged (−0.011 / 2.059 and −0.011 / 2.070, against −0.009 / 2.061 for cg25) |
| JEPA hard negatives (3–5 held-out choices, before the held-out fix) | JEPA 1.00, 0.67, 0.80 against TF-IDF names 0.33, 0.33, 0.60: tiny samples, and possibly inflated by the same held-out leak |
| Strong SIGReg on the trunk (weight 10) | effective rank 6 → 10 over seven sleeps, top direction 37% → 25%, no harm; no retrieval gain yet |
| Activation centring; gentle SIGReg on the trunk | no measurable effect |

**Paraphrase rule layer.** Questions in a wording never seen in the evidence scored 0.93 (0.41 with the rule layer off). All 12 true rules were found and a correlated decoy relation was rejected.

**Frozen GPT-2 (124M) plus Platypus.** 30 candidate answers per question (chance 0.03):

| | GPT-2 alone | Facts in the prompt | GPT-2 + Platypus |
|---|---|---|---|
| Two-step questions | 0.05 | 0.80 | 0.97 |
| Changed facts, two steps | 0.00 | 0.67 | 1.00 |
| Unseen-wording questions | – | – | 0.99 |
| Spotting unknown subjects (AUC) | 0.55–0.65 | 0.67–0.71 | 0.89–0.94 |
| Perplexity on ordinary text | 80.9 | 80.9 | 80.9 |

The inference tests use templated facts about invented people. See [Project status](#project-status) for what has not been shown yet.

## Getting started

Platypus is a single Python script designed to run on one GPU (for example in Google Colab).

**Requirements**

- Python 3.10+
- PyTorch with CUDA
- `datasets` (Hugging Face), for streaming `monology/pile-uncopyrighted`
- `scipy` (sparse matrices for one-scan linking)
- Optional: `flash-linear-attention` for the fast Triton kernels. The script tries to install them and falls back to a pure PyTorch solve if they are unavailable.

**Run**

```bash
pip install torch datasets scipy
python threshold_lean_stagec9_7cg26f_conduit.py
```

In Colab, upload the script, choose a GPU runtime and run it with `!python threshold_lean_stagec9_7cg26f_conduit.py`. If you need credentials (for example a Hugging Face token), keep them in Colab Secrets or environment variables. Never commit them to the repository.

A full run streams ten domains over three passes and takes about four to five hours on a single modern GPU.

## Configuration

Options are set through the `LAB_OVR` environment variable as JSON, on top of the `Config` defaults. Anything you pass overrides the built-in run configuration.

```bash
LAB_OVR='{"n_passes": 3, "steps_per_task": 400, "inf_on": true, "seed": 1}' \
  python threshold_lean_stagec9_7cg26f_conduit.py
```

Environment switches:

| Variable | Meaning |
|---|---|
| `PLATYPUS_RULEFREE=0` | Run the rules setup (relation list and name pattern) instead of the default rule-free mode |
| `PLATYPUS_ROUTEFIX=0` | Turn off the drift-corrected routing and use the original rule |
| `PLATYPUS_DREAMS` | `jepa` (default: dream only what the JEPA selector chose within the name index, NREM and REM), `both`, `nrem`, `rem`, `random` (control: random pairs), `0` (no dreams, no trunk replay) |

Useful options:

| Option | Meaning |
|---|---|
| `n_passes`, `steps_per_task` | Passes over the domain sequence, and training steps per visit |
| `seed` | Random seed |
| `inf_on` | Sleep-time inference and the fast memory |
| `inf_graph` | Linker: 3 = one-scan salience (default in rule-free mode) |
| `inf_replay_gumbel`, `inf_conc_refresh` | Priority-proportional replay; strengthening of conclusions met again |
| `inf_mem_gate` | Confidence gate for applying fast-memory corrections |
| `inf_audit_rule`, `inf_audit_cascade` | Belief revision rule and cascading retraction |
| `inf_idmem`, `inf_id_dim` | Identifier memory and its dimension |
| `inf_rules`, `inf_para_frac` | Paraphrase rule layer, and the fraction of probes given a second wording |
| `inf_lm_side`, `inf_lm_fuzzy`, `inf_dual_probe` | Overlapping link memory beside the store, spelling-aware codes, and the dual-pathway readout |
| `inf_cls_probe` | End-of-run CLS probes |
| `inf_real_sample`, `inf_timing` | Real-text conclusion samples per sleep; sleep timing |
| `pu_cons_lr`, `pu_cons_steps` | Consolidation rate (multiplier on the training rate) and steps per phase |
| `pu_cons_brake` | Rolls back consolidation if it harms old domains by more than this many BPC |
| `infer_drift_clamp`, `infer_proto_strong` | Routing: drift clamped at zero; strong-prototype rule |
| `infer_proto_recency` | Routing: 2 = no rule may pick the slot trained at the previous visit (default) |
| `inf_stage` | Staging buffer: 2 = commit a conclusion only when its premises are read again (default) |
| `inf_cons`, `inf_cons_wlr`, `inf_cons_wlr_max`, `inf_cons_retry`, `inf_cons_minpool` | Trunk replay of conclusions and dreams: starting rate, ceiling, retries after a rollback, minimum pool |
| `inf_cons_protect`, `inf_cons_adapt`, `inf_cons_brake` | Fisher-steered (protected) replay; adaptive dose; the brake |
| `inf_splicemix`, `inf_mix_hybrid`, `inf_mix_sharename`, `inf_mix_nameidx`, `inf_mix_null_q` | Splice dreams: proposer, shared-name requirement, name-indexed pool, chance bar |
| `inf_mix_cortex`, `inf_mix_cxsrc`, `inf_mix_cxsel` | REM dreams: traces per cue, trace source (`bank`, `familiar`, `both`), selection (`top`, `band`, `mmr`) |
| `inf_jd`, `inf_jd_pool`, `inf_jd_drop`, `inf_jd_sigreg` | JEPA dream selector: on/off, pooling, top directions removed, SIGReg weight |
| `inf_mix_jdname`, `inf_mix_cxname`, `inf_mix_benchcue` | JEPA within the name index for NREM and REM dreams; share of cues with gold entities (for measurement) |
| `inf_jd_real`, `inf_jd_hn`, `inf_jd_hnw` | JEPA training: real-text balance; hard negatives per pair and their weight |
| `inf_mix_revisit` | Dreams on revisits, from the verified-replay sentences of the visit |
| `inf_tgrad` | CLS readouts: temporal gradient and store handover (no effect on training) |
| `inf_jepa_probe`, `inf_jepa_geom`, `inf_jepa_next`, `inf_jepa_shadow`, `inf_jepa_consol`, `inf_jepa_every` | Latent readouts (no effect on training) and how often they run |
| `inf_bench`, `inf_ceiling`, `inf_ceiling2` | Re-DocRED real-text benchmark (`embedded` in the GPU file); composition ceiling tests |
| `inf_salience`, `inf_sym`, `inf_mutual`, `inf_iso` | Off by default, for experiments: salience, symmetry, mutual compression, SIGReg on the trunk |

The full list, with a change history, is in `NAME` and `LAB_OVR` near the end of the script.

## Reading the logs

- Per-visit **BPC** and **forgetting** for every domain seen so far
- **Routing**: the slot each visit was assigned to, with the original rule's choice and each slot's drift shown alongside
- **ONE-SCAN SALIENCE**: links in the table, two-step paths scored and links promoted each sleep
- **Inference readouts**: one-, two- and three-hop recall from the weights and with the store; answer-or-abstain; sources; updated facts; store cost on ordinary text
- **IDENTIFIER MEMORY**, **PARAPHRASED** and **RULE LAYER** lines for the identifier memory and the paraphrase rules
- **REAL-TEXT CONCLUSIONS**: a sample of conclusions involving no planted name, with their source sentences, for judging precision on real text
- **inference TIMING**: wall time of each sleep and of its linking step
- **STAGING**: candidates proposed, waiting, and passing the bar each sleep
- **[consolidation]**: gap recovered per phase (summary as a median), and **ROLLED BACK** when the brake undoes a harmful phase
- **CLS PROBES** and **DUAL PATHWAY** at the end: lesion by age, partial cues, schema test, and recall from damaged names by each pathway
- **CONCLUSION REPLAY INTO THE TRUNK**: dose, cost to the visited and old domains, KEPT or ROLLED BACK, retries, protection
- **REPLAY RETENTION**: two-hop from the weights before and right after replay, and the previous sleep's sample after a visit of training
- **MIXING**: the splice test check (known links vs random pairs), proposals and acceptances by source, how many are related (planted chains and Re-DocRED gold), example pairs and cortical traces
- **JEPA DREAMS | TRAIN**: linked pairs, loss, code rank and held-out partner retrieval of the dream selector
- **LATENT READOUT**: geometry, next-span retrieval, views, shadow JEPA and latent consolidation
- **CEILING LONG**: composition learning curve, in and out of distribution, in the weights and in context
- **MIXING | CG26B**, **REM NAME INDEX**, **REVISIT**: benchmark cues, name-indexed candidates and traces, and what a revisit wrote
- **JEPA DREAMS | TRAIN**: also REAL-TEXT held-out and HARD NEGATIVES (JEPA against TF-IDF names)
- **CLS | TEMPORAL GRADIENT** and **STORE HANDOVER**: two-hop with the store off and on by conclusion age; conclusions the weights alone now answer
- **GPU MEMORY**: peak during the visit and the sleep, reserved, device total

## Project status

This is **research code**. It is bulky: it carries options, readouts and experiment paths from many iterations that are switched off in the current configuration. It needs a cleanup (dead options removed, modules split, tests added) and further optimisation before wider use.

Known limits:

- Inference has been tested at scale on planted facts about invented people. On natural text, a first local benchmark (Re-DocRED Wikipedia paragraphs with gold relations) found that word-level linking cannot chain facts stated only once: two single reads never reach the linking threshold. Facts must recur, or links must be found another way (bridge-free mixing and latent "splice" dreams are under test).
- Natural prose is harder than templated sentences: about 0.77 two-hop recall in local tests.
- Pronouns, descriptions and nested clauses are not resolved, so chains through "he" or "she" are not found.
- Consolidation into the weights works at full size but slowly: replay at doses the brake accepts adds only a little two-hop recall per sleep, though what it adds survives later training. Protection against forgetting during replay is not yet reliable from one visit to the next, and the small local model shows no transfer at all.
- The JEPA selector's held-out retrieval figures from cg26b–cg26f were inflated by a leak (the held-out split was re-drawn each sleep while the heads persisted). Clean evidence so far: about 5× chance on linked pairs at the first training sleep, little above chance on real-text pairs. cg26g/h use a permanent held-out set.
- Dreams chosen by the JEPA within the name index are far more often related than random ones, but they are not yet shown to give more two-hop recall or less forgetting than random dreams; the control run is pending. Within the name pool the JEPA's ranking so far matches a simple TF-IDF ranking.
- At the replay dose the brake accepts (×0.0033), later sleeps no longer raise two-hop recall measurably, and transfer kept from earlier sleeps decays in some visits.
- The trunk's passage codes are strongly anisotropic; latent readouts and the JEPA head work around this, but the trunk itself is unchanged.
- Forgetting at full size rises in later passes, and routing confuses the most similar domains.
- There is no fine-tuning baseline and no benchmark result yet.

## Roadmap

1. **Dreams that beat random.** Run the name-indexed JEPA dreams against random-dream and no-replay controls on the same build; show that related dreams give more lasting two-hop recall in the weights.
2. **Free the fast store.** The temporal-gradient and handover readouts are in (cg26f). Next, release conclusions the weights now recall and check they survive without the store.
3. **Real text.** Measure the precision of conclusions drawn from real text, and test on multi-hop question sets and MQuAKE for multi-step edits.
4. **Consolidating structure.** Move relation regularities (which relations chain, which are single-valued, paraphrase rules) into the weights, and show they become independent of the store over time.
5. **Stable domains at scale.** Protect slots from drift, to cut routing errors and later-pass forgetting; test learning rates that follow novelty and consolidation that stops itself at the brake.
6. **References and meaning.** Resolve pronouns and descriptions as rules that must pay for themselves; carry negation, hedging and time on each relation.
7. **Entity resolution.** Merge spelling variants only when the evidence supports it: rare typos of a common name, or variants that share context.

## Contributing

Issues, results and pull requests are welcome. Cleanup work is especially useful: removing dead options, splitting the script into modules and adding tests. By contributing you agree that your contributions are licensed under the Apache License 2.0.

## Citing

If you use Platypus in your work, please cite:

```bibtex
@software{bird2026platypus,
  author  = {Bird, Christian},
  title   = {Platypus: A Language Model That Learns Without Forgetting and Thinks in Its Sleep},
  year    = {2026},
  url     = {https://github.com/chrisbirduk/platypus},
  license = {Apache-2.0}
}
```

## Licence

Copyright 2026 CHRISTIAN BIRD

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Add the full licence text as a `LICENSE` file in the repository root (GitHub offers the Apache 2.0 template when you create the file).
