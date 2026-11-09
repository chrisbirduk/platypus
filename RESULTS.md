# Results log (north-star rungs and the cg27a write side)

Results go here as they come, with the conditions that limit them; negatives stay in. Local = the five-domain harness (35 probes; runs
without an explicit seed are seed 0; local recall differences under about 0.15 are seed noise). GPU = Colab, 130 probes, pass 1 (10 visits).
"Shadow" = a readout that fits a variant on one sleep's conclusions alone, applies it to the pre-edit weights, measures and removes it.

## Words used carefully

- **From the weights**: the trunk's own parameters, fast store off. A closed-form edit to a trunk matrix counts as the weights.
- **Name-addressed binding memory**: a memory read by the subject's name state (the trunk's state over the subject name). It is a
  hippocampus-like binding memory inside the forward pass, *not* the weights; recall through it is not consolidation.
- **Oracle answer positions**: every shadow name-keyed result below is told where the answer starts (teacher-forced, the answer slot
  given). It is a comparison against an oracle-position baseline until the memory fires on its own.

## Rung 3 (amended to the name channel), 9 November 2026

| Result | Numbers | Conditions |
|---|---|---|
| The subject's name state locates its own conclusion across wordings | GPU 0.97 (ceiling 47/48 = 0.98), seed 2 0.98 (ceiling 0.98); local 0.69 and 0.77 (each the ceiling: distinct subjects / conclusions) vs the edit's key at the question end 0.19-0.53; closed-form SFA directions 0.01-0.17 | Readout (`inf_open_diag`); templated held-out wordings; one GPU seed (1), local seeds 0 and 2 |
| **Name-addressed binding memory (one closed-form memory per answer byte, keyed by the name state) produces facts in wordings never written** | Held-out open-ended: GPU 12/80 (0.15) vs 2/80 for the configured edit (both with self-contrast); local 7/80 vs 1/80. First answer byte ranked first: GPU 0.16-0.31 vs 0.00-0.06; local ~0.29 vs ~0.02 | **Shadow, oracle answer positions**, fitted on 8-16 conclusions per sleep, three sleeps each; GPU seed 1, local seed 0. **Seed 2 (GPU, six sleeps, visits 3-9): first byte 0.16-0.31 vs 0.00-0.16, held-out open 0.03-0.16 vs 0.00-0.06**; the first production of a fact in an unwritten wording in this project. Not the weights |
| [own key ; name key] in one edit, and the name key alone | +0.01 to +0.03 answer bytes (never below control, 5 sleeps, 2 seeds); name alone worse (one vector cannot spell an answer) | Shadow |

**Prior art for the mechanism.** Keying a fact by the subject is established: ROME (Meng et al. 2022) locates factual associations in
mid-layer MLPs at the subject's last token and keys its edit there (CounterFact paraphrase 88.6 GPT-2 XL, 93.6 GPT-J); Geva et al. 2023 show
recall as subject enrichment at the subject positions, then attribute extraction by attention at the last position. Platypus's edit was
keyed at the question end (ROME's "late site"), which explains its weak transfer; the name channel finds ROME's site independently in a
byte-level model trained from scratch. What differs: in a pretrained model attention carries the attribute to the answer, so a subject edit
suffices; the name-addressed binding memory writes each answer byte directly, which suggests the young trunk lacks that extraction path. The
ROME-style shadow edit (subject-site, a plain weight edit, no oracle positions) tests whether it does: **it does not.** Solved into the weights it equals 'no edit' in every sleep: GPU seed 2, six sleeps, blocks 1 and 2 of 4 (answer bytes 0.36-0.52 against the same with no edit); local, three sleeps, block 0 of 2 (0.42 / 0.47 / 0.55 vs 0.41 / 0.47 / 0.57). The fit itself shows why: with the best output change at the subject's last byte applied directly (size ~10, 50 steps), the answer improves by ~0.2 bits per byte (3.68 -> 3.46, 3.12 -> 2.88) and answer-byte top-1 by 2-3 points, against 0.61-0.68 for an edit at the answer positions. The young trunk identifies the subject but has almost no path that carries what is attached to it to the answer (ROME's attribute-extraction step). Likely cause, not yet tested: facts planted in one template are stored tied to the wording (Allen-Zhu & Li, Physics of LMs 3.1). (The first version, in code before `a8b7b5f`, edited the last block and was void.)

**Becomes a model result when** the memory fires on its own: a subject register (the most recent name-like span) and an answer-slot
detector ("the next bytes are an object"). The slot signal is what rung 1's code should supply and does not yet (frame-dependent: matches
SFA in the written wording, not held-out). First attempt: a small gate trained on the system's own conclusions, with its false-fire rate on
ordinary text measured against BPC.

## Write side (cg27a), configuration results

| Result | Numbers | Conditions |
|---|---|---|
| Held-out wording two-hop from the weights comes from closed-form edits fitted on three wordings, not from replay wordings | 0.36-0.39 at visit 10 (three edit wordings) vs 0.22 with one edit wording + replay wordings = 0.22 eval-fix (no write side), visit for visit | GPU seed 1; one run per arm; multiple choice among 20. **Seed 2** (self-contrast build): 0.30 at visit 10, 0.33 at the end (seed 1 same build 0.31 / 0.40); visits 6-9 within 0.06 of seed 1 |
| **Self-contrast in the closed-form edit (`inf_cf_selfc`, on by default)** | Edited conclusions, fitted wording, open-ended from the weights: 0.88 / 0.80 / 0.77 at visits 4-6 vs 0.30 / 0.40 / 0.54 without (same seed); held-out 0.06 / 0.24 / 0.32 vs 0.00 / 0.24 / 0.22; all edits kept. End of run: fitted open 0.67 vs 0.54, held-out open 0.25 vs 0.19; 130-probe open two-hop 0.54 vs 0.48, held-out multiple choice 0.40 vs 0.41 (unchanged); BPC 1.731 vs 1.724, forgetting -0.002 vs -0.012; real forgetting (per domain, mean of 10) +0.012 vs +0.005, largest USPTO +0.109 vs +0.064 and FreeLaw +0.022 vs -0.004: a possible cost, to check on seed 2 | **Real build** (with competitor contrast), GPU seed 1, one run per arm, slot given, edited conclusions only. Shadow (no competitor contrast), 7 sleeps: fitted open 0.17 -> 0.59, held-out MC 0.51 -> 0.65. Local: no effect, no harm (2 seeds) |
| Greedy decoding loses names the model ranks well | Held-out misses: true name more probable than the greedy output in 0.88; rank among 200 probe names median 6 | GPU end of run; constrained decoding is reported beside greedy, never as the headline |
| Temporal gradient under store lesion | Two-hop on the system's conclusions, store off: concluded before visit 8 0.91 (n 47), from visit 8 0.50 (n 20); end of run, split at visit 7: 0.88 (n 56) / 0.51 (n 69), and with self-contrast 0.87 (n 53) / 0.59 (n 70); seed 2 (self-contrast) 0.91 (n 53) / 0.53 (n 68), and its store lesion split by exposure: older edited 0.96 (n 23), older not edited 0.78 (n 18), recent edited 0.67 (n 9), recent not edited 0.50 (n 26); sleeps in the edit 0 / 1-2 / 3+: 0.61 (n 44) / 0.67 (n 9) / 0.96 (n 23) - so the age effect holds among conclusions never edited (0.78 vs 0.50), and re-solved edits add to it; replay count is not yet split; store lesion split at visit 8, with self-contrast 0.86 (n 42) / 0.71 (n 24) (store on 1.00 / 1.00), so self-contrast narrows the gap by holding recent conclusions better; build without closed-form edits (1 Nov): 0.02 / 0.04 | **Two GPU runs of one seed (1)**, near-identical builds, so not independent; age confounded with exposure (sleeps, replays, re-solves since concluding); needs the exposure control and a second run before it is a claim. Local mixed (n 5-9) |
| Weights hold the system's conclusions, not the source facts | Never-edited conclusions 0.21 open-ended (fitted wording); source one-hop open-ended from the weights ~0.00 | GPU seed 1 |

## Slow features (north-star rungs 0-2)

| Result | Numbers | Conditions |
|---|---|---|
| Rung 0: closed-form SFA ties the trained JEPA on held-out-wording retrieval | 0.15 vs 0.12, 0.15 vs 0.10, 0.20 vs 0.12 (seeds 0, 2, 1); written wording JEPA ahead (0.38-0.42 vs 0.07-0.15); names alone 0.70-0.90 | Local; GPU pending |
| Rung 1: online slow head (**amended 9 Nov, 17:05: rung 1 is now slow features re-solved each sleep in closed form, as the latent judge already does; online rules parked**) | Not passed after five versions. v3 reaches closed-form slowness at 16 bytes (0.811 vs 0.814) but not its retrieval; v4 (sentence pairs) ties SFA top-1 in the written wording on two seeds (0.07, 0.15), not held-out (0.05 vs 0.12-0.15). Three alternative rules, seed 0 (top-1/top-5, written | held-out; SFA 0.07/0.35 | 0.12/0.35): SFA from running statistics 0.05/0.17 | 0.05/0.20; IncSFA-style two-stage (GHA whitening + minor components) 0.10/0.20 | 0.03/0.12 (its first version diverged to NaN, and NaN scores ranked first gave a false 1.00: both fixed); LPL 0.00/0.10 | 0.00/0.03 (codes collapsed, units correlated ~0.5). None passes held-out | Local, stop-gradient (no BPC cost) |
| Rung 2: latent judge (SFA cosine) | Offline: AUC 0.94 / 0.75 vs byte judge 0.76 / 0.48. In real sleeps (`SHADOW3`, visits 2-5): known-linked controls vs random 0.94-1.00 vs byte judge 0.84-0.96, at ~1 ms vs 17-21 ms per pair | Local seed 0; the controls share the bridge name by construction (shared rare name 1.00), 1-7 per sleep: an easy test |

## Timescale bank

| Result | Numbers | Conditions |
|---|---|---|
| New-domain detection: slowest band + router loss | AUC 0.90 / 0.99 (200 / 1600 bytes) vs router loss 0.83 / 0.93, slowest band 0.84 / 0.97 | Local, two seeds, leave one domain out. The rule cascade's own `[change detection]` (fires on every switch, no false alarms in the logs) was not scored on the same windows: that comparison is still owed |
| Sticky router (posterior carried across windows) | Tokens routed right inside a domain 0.90 -> 0.96; mixture BPC 2.3387 -> 2.3343; slightly worse just after a switch; the reset gates catch 42-90% of switches at 2% false resets but do not recover BPC after a switch | Local, seed 2, compared with the current head-loss router on the same streams (not with the cascade) |
| Routing known domains by prototypes | No band combination beats the head-loss router (0.93-1.00 per window); equal-weight combinations hurt; Fisher weighting best of the band combinations (0.76 / 0.88 / 0.93) | Local, two seeds |

## Negatives (kept as they are)

- **Forgetting detection from prototype drift fails.** Rank correlation of drift with the per-domain BPC rise over 35 points: trunk +0.20 to +0.28 (relative to the bank -0.00 to +0.29); in pass 1 drift tracks learning (-0.76). Slow-head drift relative to the bank is weakly positive on every run (+0.18 to +0.39), a hint only. Probe- or residual-based detection stays.
- **Forgetting detection from the model's surprise on its own stored sentences fails**: -0.27 / -0.36 (seed 0), -0.32 / -0.45 (seed 2), all points / pass 2. Replay protects exactly the stored text.
- **Rung 2b is negative locally**: first-byte surprise is a capital-letter detector (AUC 0.90 overall, 0.48-0.50 within capitalised words); entity codes are no more stable across occurrences (0.43-0.56); relations on Re-DocRED at chance (substitution 0.45, compression 0.50, latent judge 0.50-0.57). The reader remains the ceiling on all natural-text results; the name-channel key only helps where a name has been found.
- **Case factoring** (`case_factor`, c/C share a letter table): BPC -0.008 and -0.002 (two seeds, noise), entity test unchanged.
- **Stale keys**: prototypes recorded at a domain's first visit route at chance by the end (0.20-0.26). An anchor term on the slow head (`slow_anchor` 1) did not hold its reference codes (cosine to the previous visit -0.04 to 0.30), interim, local seed 0.
- **Six wordings** in the edit add nothing over three (GPU, five sleeps).
