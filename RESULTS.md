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
ordinary text measured against BPC. **First gates (local, two seeds, 6 sleeps, no oracle, firing on 1% of ordinary bytes):** SERAC-style gate (system-written forms, hard negatives after names, recency, two layers) first answer byte 0.07-0.25 vs told-the-slot 0.10-0.38 and linear gate 0.00-0.28; ties or beats the linear gate in 5 of 6 sleeps; open-ended 0.06-0.09 in three sleeps; BPC on ordinary text +0.001 to +0.003; still fires early in 50-69% of questions. Parked as a readout: the fast store already answers 0.97 in unseen wordings. **GPU (seed 3, eight sleeps):** SERAC-style first byte 0.03-0.39 (mean 0.17) vs told-the-slot 0.09-0.50 (mean 0.23) and the configured edit 0.03-0.31 (mean 0.12); open-ended 0.00-0.22 (two sleeps at 0.22, equal to told-the-slot); fires at the slot 0.72-1.00; BPC on ordinary text +0.0000 to +0.0006. Linear gate mean 0.10; WISE-style 0.00-0.06 with 46-123 fires per 1000 bytes.

## Two relations per subject, the two-part key memory, and the JEPA, 10 November 2026

Local, seeds 0 and 2, end of pass 1 (readouts on the trained model; nothing changes the weights). Material: each subject's planted fact
("X works with Y") and one of its conclusions ("So X is connected to Z") - two relations per subject, so a key that knows only the subject
is capped at 0.50. Held-out questions: SAME RELATION in a new frame ("It is on record that X works with") and PARAPHRASED RELATION
("X is a colleague of", "We read that X is a colleague of").

| Result | Seed 0 | Seed 2 | Conditions |
|---|---|---|---|
| Retrieval, NAME key only | 0.50 / 0.50 | 0.50 / 0.50 | same relation / paraphrased; the subject is always right, the relation is a coin toss |
| Retrieval, edit key at the question end | 0.42 / 0.27 | 0.60 / 0.24 | the relation is always right, the subject often wrong |
| **Retrieval, edit key + NAME (the two-part key)** | **0.94 / 0.89** | **0.95 / 0.80** | 26 and 21 subjects |
| Retrieval, JEPA trained code + NAME | 0.73 / 0.51 | 0.50 / 0.56 | LeJEPA objective (see below) |
| Open-ended recall, no memory | 0.08 / 0.00 | 0.07 / 0.00 | |
| Two-part key memory, told the answer slot | 0.87 / 0.53 | 0.71 / 0.31 | one entry per fact; value = that fact's own output change, fitted on written wordings only |
| Name-only memory, told the answer slot | 0.50 / 0.36 | 0.45 / - | |
| **Two-part key memory END TO END: SERAC slot gate + re-anchoring, no oracle** | **0.87 / 0.46** | **0.71 / 0.30** | equals the told-the-slot result on both seeds; the gate's latest firing up to the slot is AT the slot 0.80-1.00 |
| Its cost on held-out ordinary text (gate + a stored name present) | 0.0 fires / 1000 bytes, BPC +0.0000 | 0.1 / 1000, BPC -0.0011 | without the stored-name condition: 8-11 fires / 1000, BPC +0.005 to +0.011 |
| Other gates, end to end (first firing, no re-anchoring) | AND 0.48 / 0.01; SERAC 0.38 / 0.01; P(capital next) 0.29 / 0.01 | AND 0.50 / 0.00 | each fires too early (SUM, SERAC) or too rarely (AND, P(capital): the small model gives a capital only 0.25 at paraphrased slots) |

**Reading.** WHERE (the slot) is solved locally by the SERAC gate once each firing restarts the memory (re-anchoring is causal: the
memory at each byte comes from the latest firing at or before it). WHICH is solved by the two-part key in the relation's own wording (0.88
right entry on seed 0) but weaker with a paraphrased relation on seed 2 (0.37). The VALUE transfers partly: even the true entry at the true
slot gives 0.66-0.68 open-ended with a paraphrased relation. This is recall from a memory beside the weights, not consolidation.

**The JEPA under the default LeJEPA objective.** Only the head g is trained: the target head stays at its random initialisation and the
predictor is never trained, so earlier "JEPA target" rows measured the JEPA's pooled input under a random projection, and "JEPA predicted"
rows measured noise (relabelled, b81c80a). The trained code does not carry the relation (above). As a link operator by plain similarity:
next planted fact top-1 0.33 / 0.34 (chance 0.002; symbolic lookup 1.00), links it trained on 0.38 / 0.38, links it never saw 0.10 / 0.12;
two hops 0.06 / 0.00. Update dependents: 0 and 2 testable updates, too few. A transition-operator test needs `inf_jd_obj=hybrid`.

## The name memory as a model option (inf_name_mem, on in cg27a), 10 November 2026

Local, seed 0, full run, against the same run without it (JR0): **weights only identical** (forgetting -0.005, final BPC 2.411, every
store-off line identical over 10 visits); store on 2.416 vs 2.415. Held-out recall through the model's own forward, memory off -> on, visit 10
(entries = newest recorded facts + conclusions): conclusions in the standard held-out wordings 0.01 -> 0.20 (88), facts in a new frame
0.06 -> 0.60 (48), facts with a paraphrased relation 0.00 -> 0.07 (15). Misfires: text naming stored subjects 15.6 fires / 1000 bytes,
BPC +0.02 there; ordinary text 0. Rebuild 2-6 s per sleep; 15-25% of values refitted per sleep (drift).

| Change | Result | Conditions |
|---|---|---|
| GPU visit 1, newest 384 recorded facts | 0.02 -> 0.02 | the newest were prose co-occurrences seen once (5335 names in the store) |
| Facts chosen by sightings (>= 2, most seen first) | planted share 0.80-0.82; own wording 0.64-0.83, new frame 0.55-0.82 | local short run; few entries (11-33) |
| ... its misfires | 37-54 fires / 1000 bytes, BPC +0.24-0.40 on text naming stored subjects; store-on forgetting -0.174 vs -0.168 | a gate trained on few facts; hence inf_nm_min_entries=32 |
| Gate: the CHOSEN entry's own name must match (not any stored name) | conclusions 0.18, facts 0.53, paraphrased 0.05; misfires 5.1 / 1000, BPC +0.009 (was 15.6, +0.020) | full-run model, visit 10 |
| + question part of the chosen entry > tq (own non-slot bytes, 90th pct) | conclusions 0.09, facts 0.53, paraphrased 0.00; misfires 1.1, +0.003 | conclusions lose half |
| tq = 10th pct of own slots / 99th pct of own non-slot bytes | conclusions 0.03 / 0.01, facts 0.50 / 0.31; misfires 0.5 / 0.0 | |

**Does the answer-slot state carry the subject?** (full-run model, both layers, ridge map from the slot state to the name state fitted on
the system's own written texts, 324 subjects, half used to fit.) Subjects the map was fitted on: 0.10-0.47 (facts, new frame, layer 1
best); subjects it never saw: 0.01-0.05 (chance about 0.006); paraphrased relation 0.00-0.02; the register 1.00, the slot state with no
map 0.00-0.01. **No.** The slot state carries no general, linearly readable subject; the map memorises subjects. The name register cannot
be replaced by a learned projection of the slot state; the trunk would have to learn the path (distillation) or the network needs a
mechanism that carries the name forward.

## The name memory on the GPU, the value type, and keeping it out of training (10 November 2026, evening)

**GPU, seed 4, facts chosen by sightings (run 19:32)**, memory off -> on, through the model's forward:

| Sleep | Conclusions, standard held-out wordings | Facts, new frame | Paraphrased relation | Own wording | Misfires, text naming stored subjects | Values kept / refitted |
|---|---|---|---|---|---|---|
| 1 | - | 0.00 -> 0.00 | 0.00 -> 0.00 | 0.00 -> 0.00 | +0.0002 BPC | - |
| 2 | 0.00 -> 0.17 (6) | 0.00 -> 0.21 | 0.00 -> 0.33 | 0.00 -> 0.67 | +0.007 | 0 / 76 |
| 5 | 0.33 -> 0.62 (84) | 0.00 -> 0.85 | 0.00 -> 0.26 | 0.06 -> 0.94 | +0.018 | 219 / 44 |
| 6 | 0.28 -> 0.60 (96) | 0.02 -> 0.75 | 0.00 -> 0.21 | 0.21 -> 0.98 | +0.016 | 265 / 26 |
| 7 | 0.50 -> 0.69 (96) | 0.02 -> 0.67 | 0.00 -> 0.33 | 0.19 -> 0.98 | +0.019 | 301 / 37 |
| 8 | 0.36 -> 0.60 (96) | 0.02 -> 0.92 | 0.00 -> 0.09 | 0.27 -> 0.92 | +0.031 | 250 / 24 |

Ordinary text 0-0.5 fires / 1000 bytes, at most +0.0004 BPC. 9-12 s per rebuild. Planted share of entries 0.60 -> 0.67-0.69 (facts),
0.86-0.96 (conclusions). **Caveat:** in this run the memory still reached training (below), so its "off" (from the weights) column is not clean.
Sleep GPU peak 70-72 GB vs 40-51 GB without the memory - not from the fitting batch (8 vs 24 facts gives the same 70 GB at sleep 2); a
per-stage readout (`[NAME MEMORY | GPU]`) is in the next build.

**Value type (hidden fitted change vs output-level boost).** Local full run, seed 0, visits 6-10: output-level recall as good or better
(conclusions 0.49 vs 0.41 fresh at visit 10; paraphrased 0.28-0.38 vs 0.15-0.34), both lose much between rebuilds (facts new frame stale
0.33 / 0.31 at visit 8 vs fresh 1.00 / 0.98 - mostly key drift), misfire cost 3-4x higher for the output-level boost on a full memory
(+0.06-0.14 vs +0.01-0.04 BPC on text naming stored subjects). **GPU (run 21:39), sleep 1:** output-level own wording 1.00, new frame 0.81,
paraphrased 0.84 (19) vs 0.00 everywhere for the hidden value; sleep 2 fresh: own 0.98 vs 0.65, paraphrased 0.56 vs 0.33; stale (sleep 1's
memory after visit 2): 0.31 / 0.37 vs 0.00 / 0.00; misfire cost +0.021 vs +0.006. The gate variant (the chosen fact's own name) kept recall
with about 25% fewer misfires.

**Keeping the memory out of training.** The memory acted wherever the fast store acts; sleep steps that switch the store on to decide
(REM recombination, need scoring, store writes, self-taught scoring) carried it into training: a local run parted from the run without it
at visit 3 (dream mixing), final weights-only BPC 2.412 vs 2.411. Scoped to the store-on probe evaluation and its readouts: a short run with
the memory on from 8 facts then matched the run without it (weights only -0.168 / 2.981, store on -0.168 / 2.982, every store-off and
training line), apart from one near-tie in the dream mixing (2 of 440 candidates swap labels; nothing downstream changes).

**Carry the subject (`inf_carry` 0.1).** Local (seed 0, fixed run): training task 0.04-0.14 over visits 2-8; slot readout unseen subjects
0.00-0.10 at visits 3 and 6 (chance 0.03-0.05) - no signal; recall from the weights 0.00. The local model is too small and trains too
briefly (about 95 steps per visit). **GPU (seed 4, run 22:24):** training task 0.30 (last 50: 0.72) at visit 2, 0.85 (0.94) at visit 3.
Slot readout at sleep 3, UNSEEN subjects, layer 3 (top): paraphrased relation 0.56 (27), conclusions in held-out wordings 0.42 (12), facts in
a new frame 0.14 (70); layer 2: 0.26 / 0.33 / 0.06; chance 0.015; no map 0.00-0.08. No GPU control without the carry loss yet (the
0.01-0.05 baseline is local, 2 layers). From the weights, fixed set: 0.03 (edited 0.12 -> 0.00 without the closed-form edits). The sleep
GPU peak was already 70-72 GB before the name memory ran (not the memory).
**NOTE (10 Nov, 23:00): the carry loss is starting to compete with language modelling.** Post-train BPC (store on) against the seed-4 run
without it: +0.002, +0.014, **+0.041** at visits 1-3 (seed noise about 0.017) - the cost grows as the task is mastered (0.94). Weight 0.1
may be too strong once the task is learned: lower it, apply it every few steps (`inf_carry_every`), or anneal it when the task accuracy
passes a level.

## Distillation in sleep (burst): the dose curve (10 November 2026, late)

GPU, seed 4, cg27a + carry 0.1 + `inf_distill` 1 (40 steps x 16 questions per sleep in seven self-written wordings, trunk blocks only,
plain Adam, anchor KL on 16 stored sentences per step, brake 0.01 with dose retries; a fixed fifth of the facts never taught). Recall from
the trunk (memory and store off) in the TEST wordings, measured at each dose before any rollback:

| Sleep | Dose (x lr) | Answer loss, taught wordings | BPC visited | BPC old | Taught / control (facts) | Brake |
|---|---|---|---|---|---|---|
| 1 | 0.03 (first run) | 2.16 -> 0.89 | +0.098 | - | (measured after rollback) | over |
| 1 | 0.01 | 2.00 -> 1.09 | +0.057 | - | 0.00 / 0.00 | over |
| 1 | 0.0033 | 2.00 -> 1.41 | -0.009 | - | 0.00 / 0.00 | kept |
| 2 | 0.01 | 2.14 -> 0.89 | +0.097 | +0.108 | 0.00 / 0.00 | over |
| 2 | 0.0033 | 2.14 -> 1.02 | +0.081 | +0.049 | 0.00 / 0.00 | over |
| 2 | 0.0011 | 2.14 -> 1.09 | +0.029 | +0.017 | 0.00 / 0.00 | over (all rolled back) |

**Reading.** The trunk fits the taught wordings (loss falls at every dose) and NOTHING transfers to the test wordings at any dose, even
those costing 0.06-0.11 BPC: recall gained per unit of BPC lost is zero at sleeps 1-2. The burst is expensive out of proportion to what it
teaches (sleep 2: a ninth of the dose still +0.029 / +0.017). Caveat: at sleeps 1-2 the carry path was only forming (carry task 0.30 at
visit 2, 0.94 by visit 3 in the carry run), so sleeps 3-4 are the fair test of this run. Why the burst is close to a worst case (review, 10
Nov): a fresh plain Adam over every block parameter outside the Fisher-whitened update; 40 consecutive steps of same-shaped rows; thin
anchors (a KL on 16 stored sentences). **Next:** interleaved distillation (`inf_distill_wake`): a few question rows in every real-text
training step, same backward and optimiser as the real text; retention measured after each visit, taught vs control. "Works" = taught
facts above the never-taught fifth, in test wordings, from the trunk alone with the closed-form edits removed, at a cost that passes the
brake - and still there after the next visit's ordinary training.

## Interleaved distillation: first transfer above control (11 November 2026, 00:30, local)

Local short run, seed 0, carry 0.1 + `inf_distill_wake` 0.25 (4 question rows about stored facts in each real-text training step, seven
self-written wordings, run forward + backward after the real step's backward so the gradients pass through the same optimiser step; a
fixed fifth of the facts never taught). Recall from the trunk (memory and store off) in the TEST wordings (a new frame, a paraphrased
relation), measured at each sleep after the visit's ordinary training:

| After visit | Taught facts | Never-taught control |
|---|---|---|
| 7 | 0.08 (12) | 0.00 (5) |
| 8 | **0.38 (13)** | 0.00 (11) |
| 9 | **0.26 (19)** | 0.00 (11) |
| 10 | **0.26 (27)** | 0.00 (13) |

Conclusions 0.00 (4). Facts are not touched by the closed-form edits, so this is the trunk. The sleep burst on the same local model gave
0.00. **Cost (short run, weights only):** forgetting -0.091 / final BPC 3.033 vs -0.168 / 2.981 without (carry + burst: -0.151 / 2.981) -
+0.05 BPC; short runs have 24 steps per visit, so the questions are a large share of training and the cost may be inflated. Full runs at
weights 0.25 and 0.1 next (recall gained per unit of BPC). Earlier fix: a forward between the real forward and its backward changed a
tensor in place (autograd error); the question rows now run after the real backward.

### Dose curve, full runs (11 November 2026, 02:20, local; both runs stopped by a container restart during visit 9 of 10)

Seed 0, carry 0.1 in all three; reference CR1 = carry 0.1 without distillation. Recall from the trunk in the test wordings after each
visit; control = the never-taught fifth.

| After visit | 0.25: taught facts | 0.25: conclusions | 0.1: taught facts | 0.1: conclusions | Control facts (both) |
|---|---|---|---|---|---|
| 2 | 0.58 (12) | - | 0.58 (12) | - | 0.00 (4) |
| 4 | 0.42 (50) | 0.25 (4) | 0.26 (50) | 0.25 (4) | 0.00 (12) |
| 6 | 0.44 (45) | 0.12 (8) | 0.38 (45) | 0.12 (8) | 0.00 (18) |
| 8 | 0.47 (43) | 0.14 (14) | 0.30 (43) | 0.00 (14) | 0.00 (29) |
| 9 | **0.49 (39)** | 0.10 (20) | 0.13 (38) | 0.15 (20) | 0.00 (37); conclusions 0.00 (8) |

**Cost.** BPC right after training each domain is unchanged (within 0.012 of CR1 at every visit). The cost is in retention: a
domain's BPC when it is revisited, before retraining, is higher than in CR1 and **grows rather than fades**:

| Revisit (visits 6-9) | CR1 | 0.25 | 0.1 |
|---|---|---|---|
| visit 6 | 2.626 | 2.654 (+0.028) | 2.645 (+0.019) |
| visit 7 | 2.578 | 2.613 (+0.035) | 2.606 (+0.028) |
| visit 8 | 2.520 | 2.559 (+0.039) | 2.544 (+0.024) |
| visit 9 | 3.077 | 3.125 (+0.048) | 3.108 (+0.031) |

End of pass 1 (all domains): mean BPC 2.729 / forgetting -0.074 (CR1); 2.766 / -0.031 (0.25); 2.755 / -0.043 (0.1). Seed noise is
about 0.017.

**Reading.** Taught facts beat control at both doses, from the trunk, and at 0.25 recall holds near 0.45-0.49 as the taught set
grows; at 0.1 it falls to 0.13 by visit 9. Recall per unit of BPC (visit 9, revisit cost) is about 10 at 0.25 (0.49 / 0.048) and 4 at
0.1 (0.13 / 0.031), so 0.25 is the better dose. But the cost does not pass the consolidation test: it grows over the revisits instead
of fading. The question rows are displacing older domains between visits. Next fixes from the critique: restrict the question
gradient to the kv or middle blocks, or project it away from the anchors. Conclusions stay low (0.10-0.15 on 8-20) and are not
separated from control (0.00 on 8).

### Where the question gradient goes (11 November 2026, 07:45, local short runs)

Seed 0, carry 0.1, interleaved weight 0.25, short mode; weights only.

| Short run | Taught facts after visits 7-10 | Control | Forgetting | Final BPC |
|---|---|---|---|---|
| No distillation | - | - | -0.168 | 2.981 |
| All parameters (DWS2) | 0.08, 0.38, 0.26, 0.26 | 0.00 | -0.091 | 3.033 (+0.052) |
| All parameters + anchor projection, A-GEM (DWP) | 0.08, 0.31, 0.32, 0.26 | 0.00 | -0.103 | 3.029 (+0.048) |
| **mlp_out weights only** (DWM, `inf_distill_wake_params=mlp_out`) | 0.08, 0.15, 0.16, 0.11 | 0.00 | -0.140 | **2.991 (+0.010)** |

Restricting the question gradient to the blocks' mlp_out weights cuts the cost about 80% (final BPC within seed noise) and halves
recall: recall per unit of BPC about 11 vs 5. Most of the damage, and most of the recall, came through parameters every domain shares
(embeddings, shared n-gram tables). The anchor projection conflicts on 57-70% of steps but removes only about 1% of the question
gradient (4.5% at most): one averaged anchor direction cannot protect drift spread over many directions.

### Protection on top of mlp_out-only writes (11 November 2026, 09:10, local short runs)

Same settings (seed 0, carry 0.1, weight 0.25, mlp_out only). Replay-only repeat of the reference: -0.156 / 2.976 (no distillation
earlier: -0.168 / 2.981), so differences of about 0.01 are run-to-run noise.

| Add-on | Taught facts after visits 7-10 | Control | Forgetting | Final BPC |
|---|---|---|---|---|
| none (DWM) | 0.08, 0.15, 0.16, 0.11 | 0.00 | -0.140 | 2.991 |
| **spaced rehearsal** (DWSP, `inf_distill_wake_spaced`) | 0.08, 0.15, **0.21, 0.19** | 0.00 | -0.141 | **2.991** |
| self-KL for known facts (DWK) | 0.08, 0.15, 0.11, 0.11 | 0.00 | -0.141 | 2.989 |
| questions on dream steps only (DWO) | 0.08, 0.05, 0.00 | 0.00 | -0.158 | 2.981 |

Spaced rehearsal: about 70% more recall at the end at the same cost (the trunk answers 3-4 of 12-19 taught facts in a taught wording;
intervals spread to 16). Dream steps only: free but starved (5 question steps per visit in short runs). Self-KL: no gain.

### Gaussian drift readout (11 November 2026, local short runs)

`inf_gauss_drift`: W2^2 (Bures) between this sleep's and the last sleep's Gaussian of each layer's residual stream on fixed stored
sentences of old domains, relative to the spread. Output layer, per sleep (mean-shift part in brackets):

| Visit | Replay only (GR) | mlp_out writes (DWO / DWSP) | All-parameter writes (GA) | No replay (GN) |
|---|---|---|---|---|
| 6 | 0.0254 | 0.0249 / 0.0251 | 0.0267 | 0.0260 |
| 7 | 0.0258 (0.012) | 0.0254 / 0.0253 | 0.0288 (0.015) | 0.0294 |
| 8 | 0.0239 (0.012) | 0.0232 / 0.0231 | 0.0265 (0.015) | 0.0277 |
| 9 | 0.0194 | 0.0194 / 0.0184 | 0.0200 | 0.0226 |
| 10 | 0.1925 | 0.1740 / - | 0.1428 | 0.1765 |

Runs are identical through visit 5 (no writes; replay barely acts in 24-step short visits). From visit 6-9 the two things that raise
forgetting raise drift: no replay +14-16% (forgetting -0.026 / 3.081 vs -0.156 / 2.976 at the end), all-parameter writes +3-12%, mostly
the mean; mlp_out writes stay at the replay-only level. Visit 10 jumps about tenfold in every run and does not follow the pattern
(another cause dominates). Suggestive at this scale; the full-run pair with and without replay is the proper test.

## Width and replay (11 November 2026, local full runs, seed 0)

Forgetting at each revisit (BPC before retraining minus after the previous visit; visits 6-9):

| Run | Revisits | Pass-1 mean forgetting | Final forgetting / BPC |
|---|---|---|---|
| Width 64, replay (CB0) | -0.066, -0.028, -0.034, -0.173 | -0.076 | -0.005 / 2.415 |
| Width 32, replay | -0.073, -0.038, -0.029, -0.137 | -0.059 | (stopped at visit 9) |
| Width 64, no replay | +0.180, +0.055, +0.015, +0.098 | +0.207 | +0.071 / 2.484 |
| Width 32, no replay | +0.392, +0.284, +0.177, +0.484 | +0.319 | (stopped at visit 9) |

Width 32: trunk 0.1M + shared n-gram tables 6.6M (width 64: 0.3M + 13.1M). Without replay, doubling the width removes about a third of
the forgetting; replay removes all of it at either width (slight backward transfer). Replay, not parameter count, is the protection.
The width-32 runs were stopped by container restarts during visit 9 (an idle container is reclaimed after about an hour).

## Interleaved distillation on the GPU (11 November 2026, seed 4, pass 1)

Three runs on the same seed and schedule: **cg27a_distillm** (question rows, mlp_out weights only, pool = the name memory's material,
capped at 384; stopped in visit 6), **cg27a_distillcap** (the same with the pool taken from the store up to 2000; stopped after visit 6),
**cg27a_distillctl** (the control: identical to distillcap, pool and readouts included, but no question rows trained; visits 1-5).
Recall is from the trunk (memory and store off), open-ended (greedy answer exact), in two wordings never taught.

**Facts.** 32-fact sample, taught vs never-taught:

| After visit | distillm | distillcap | control (no question rows) |
|---|---|---|---|
| 2 | 0.29 (45) vs 0.00 (15) | 0.33 (43) vs 0.00 (15) | 0.00 (45) vs 0.00 (15) |
| 3 | 0.62 (42) vs 0.02 (41) | 0.46 (39) vs 0.02 (41) | 0.00 (42) vs 0.00 (41) |
| 4 | 0.64 (39) vs 0.00 (37) | 0.51 (39) vs 0.03 (37) | 0.03 (39) vs 0.00 (37) |
| 5 | 0.71 (38) vs 0.05 (42) | 0.64 (42) vs 0.07 (42) | 0.02 (42) vs 0.00 (42) |

Fact recall from the weights is entirely the question rows' doing (control 0.00-0.03 on the same pool).

**Capacity and accumulation (distillcap, ever-taught ledger).** Pool 68 / 164 / 255 / 307 / 406 taught after visits 1-5 (the name memory
holds 0 / 78 / 198 / 311 / 382). Recall over a stable sample of everything ever taught: 0.12 / 0.31 / 0.42 / **0.55** after visits 2-5
(the youngest bin includes facts that entered at that sleep and were not yet taught); by age after visit 5: taught 2-3 visits ago
**0.73** (143 questions), 4+ visits ago **0.69** (61); never-taught control 0.03 / 0.05 / 0.06. Recall holds and rises while the pool
grows six-fold past the memory's cap; facts taught early are not lost as new ones arrive.

**Conclusions (2 x 2: closed-form edits in place -> subtracted).**

| | Taught | Never taught | Never taught, both premises taught |
|---|---|---|---|
| distillcap, visit 4 | 0.48 -> 0.08 (50) | 0.25 -> 0.00 (8) | 0.17 -> 0.00 (6) |
| distillcap, visit 5 | 0.69 -> **0.15** (68) | 0.19 -> **0.00** (16) | 0.07 -> 0.00 (14) |
| control, visit 4 | 0.26 -> 0.00 (50) | 0.00 -> 0.00 (8) | 0.00 -> 0.00 (6) |
| control, visit 5 | 0.40 -> 0.00 (68) | 0.25 -> 0.00 (16) | 0.21 -> 0.00 (14) |

Conclusions in the weights come from the closed-form edits (both rows fall to ~0 without them in the control); with the question
rows, taught conclusions keep 0.08-0.15 without the edits against 0.00 never taught - a small distillation effect. No composition from
taught premises.

**Cost.** Post-train BPC of the visited domain, distillcap vs control: 2.020 / 2.009, 2.083 / 2.078, 2.031 / 2.033, 1.666 / 1.662,
1.574 / 1.577 (visits 1-5) - none. Old domains' BPC at each sleep (fast store on), distillcap vs control: 1.999 / 1.976, 2.021 / 2.002,
1.977 / 1.959, 1.898 / 1.878 (sleeps 2-5) - **+0.018 to +0.023, steady**. End-of-run forgetting not measured (both runs stopped).

**Probe recall from the weights** (standard 134 probes, store off), distillcap vs control: visit 4 one-hop 0.37 / 0.15, open one-hop
0.19 / 0.00, two-hop 0.08 / 0.10, open two-hop 0.04 / 0.02; visit 5 one-hop 0.40 / 0.19, open 0.31 / 0.00, two-hop 0.14 / 0.16, open
two-hop 0.08 / 0.06. The one-hop rise is the distillation's; the two-hop rise over the run (0.03 -> 0.16-0.22) happens without it
(conclusion replay and the closed-form edits). **Replay retention** (two-hop from the weights on the last sleep's replayed sample after a
visit): with distillation 0.75 / 0.88 / 0.96 (visits 4-6), control 0.67 / 0.83 (visits 4-5) - conclusion replay sticks without question
training; the question rows add about 0.05 at most (the 'access' reading is not supported for retention).
**Gaussian drift** of old domains (output layer) identical with and without the question rows (0.31 / 0.29, 0.30 / 0.29, 0.22 visits 2-4).

**Memory.** The 70 GB sleep peak was the probe evaluation's beam search (134 probes x 26 first-step beams of long prompts in one pass);
chunked (`inf_eval_chunk 128`; local: evaluation lines identical over 7 visits) the sleep peak is 49-50 GB, now set by dream mixing.

**Spaced rehearsal on the GPU** (cg27a_distillsp = distillcap + `inf_distill_wake_spaced 1`, facts only; conclusions rehearsed every
step). Sleep 4 against distillcap: facts still in the pool 0.42 / 0.42 (320 questions), facts taught 2-3 visits 0.55 / 0.54, taught facts
after the visit (32-fact probe) 0.57 / 0.51, control facts 0.03-0.04 / 0.03-0.05; 190 of 282 taught facts known in a taught wording, mean
rehearsal interval 2.61 (known facts skipped). Conclusions taught 0.60 / 0.48 with the edits, 0.08 / 0.08 subtracted (edits, as before).
Drift of old domains (output layer) 0.215 / 0.202. Visit 5 probes: one-hop 0.41 / 0.40, open one-hop 0.28 / 0.31, paraphrased 0.23 / 0.34
(n 44). Sleep peak 50.0 GB. **Same recall with known facts skipped**: the local gain (0.19 vs 0.11 taught) does not appear on the GPU
by sleep 4; the saving is in rehearsal rows, which matters as the pool grows.
Sleep 5 (pool 405): facts still in the pool 0.51 / 0.55, taught 2-3 visits 0.65 / 0.73, 4+ visits 0.66 / 0.69 (61 questions), taught
facts after the visit 0.71 / 0.64, control 0.05 / 0.06; mean interval 3.83, 26 facts at 16; drift 0.182 / 0.186. **Verdict: no gain,
slightly lower for facts 2-3 visits old**; not carried into the next build (cg27a_distillrp: distillcap + repair replay).

**Second seed** (cg27a_distillcap5 = distillcap + chunked evaluation, SEED 5; same options). Seed 5 vs seed 4 (capacity): sleep 3 facts
still in the pool 0.32 / 0.31, taught 2-3 visits 0.53 / 0.59, taught facts after the visit 0.47, control 0.00 / 0.03; sleep 4 still in the
pool 0.39 / 0.42, taught at least one visit 0.51 (distillsp 0.50), 2-3 visits 0.50 / 0.54, after the visit 0.44 / 0.51, control 0.01 / 0.05;
visit-4 probes open one-hop 0.24 / 0.19, paraphrased 0.36 / 0.27. **The fact-recall result reproduces on a second seed** (sleep 2 lower:
0.20 vs 0.33 for facts taught one visit). Conclusions lag only because seed 5's first closed-form edit came a sleep later (sleep 4).
Sleep 5 (pool 412): taught 4+ visits **0.71 / 0.69**, 2-3 visits 0.61 / 0.73, still in the pool 0.46 / 0.55, taught at least one visit
0.59, after the visit 0.67 / 0.64, control 0.01 / 0.06; visit-5 probes one-hop 0.42 / 0.40, open 0.30 / 0.31, paraphrased 0.39 / 0.34;
drift 0.194 / 0.186; sleep peak 49.8 GB. Conclusions 2x2: never taught with both premises taught 0.42 -> 0.08 with the edits subtracted
(12 questions, 6 conclusions: one answered; conclusion replay still runs for them) - not evidence of composition at this n.

**Composition from distilled facts, held-out chains** (cg27a_distillbr2: distillcap + inf_kgx_hold 4 + synthetic bridge anchoring on half
the held-out chains' bridges; seed 4; stopped at visit 6). The first build (cg27a_distillbr, real descriptors) anchored 1-3 facts per sleep:
the planted bridge names occur only in their own chain, so no true descriptor exists. In distillbr2, 51 held-out bridges, 16 anchored with
an invented attribute ('Fazora keeps the amber ledger.'), 21 taught facts anchored by visit 6. COMPOSITION PROBE (chains of facts taught
for a visit, never concluded): visit 5: 5 chains, 3 with both premises answered, composed 0.00; **visit 6: 7 chains, 4 with both premises
answered, composed 0.00** (anchored: 2 chains, premises not both known; not anchored: 3 chains, 2 known, 0.00). Conclusions 2x2, never
taught with both premises taught: 0.45 -> **0.05** with the edits subtracted (20 questions; conclusion replay still trains them). Fact
recall unchanged (4+ visits 0.68-0.72, 2-3 visits 0.65-0.71, control 0.01-0.04). **The trunk does not compose two distilled facts**,
even when it answers both premises; the anchoring arm reached too few testable chains (2) to judge.

**Repair replay on the GPU** (cg27a_distillrp = distillcap + `inf_distill_repair 160`, seed 4): **a clear failure, stopped at sleep 2.**
The repair burst took the old slot's KL to the previous sleep's predictions from 0.062 to 0.0035 (3 s) - a near-complete revert of
the mlp_out weights - and took the visit's fact learning with it: taught facts right after the visit 0.31 (before the repair), facts
taught one visit 0.06 after it (distillcap 0.33, distillsp 0.33 at the same point), control 0.00. Locally (40 steps, small pools) the
KL fell only 0.033 -> 0.023 and recall was too low to show the loss. The question rows and the repair act on the same weights, so the
repair cannot tell fact learning from drift. **Source in context** (same run): sleep 1 answer bytes 0.47 -> 0.49 with the sentence, exact
0.00 -> 0.00; sleep 2 taught facts, taught wording, exact 0.11 -> **0.02** with the sentence (it hurts), answer bytes 0.57 -> 0.59;
never-taught 0.43 -> 0.49 (n 33). The trunk does not read the answer from its context window (1024 bytes): no teacher for prompt
distillation at this scale.

### Rhythm gates and other add-ons (11 November 2026, local short runs, mlp_out writes)

| Add-on | Taught at the end | Forgetting | Final BPC |
|---|---|---|---|
| none (DWM) | 0.11 | -0.140 | 2.991 |
| repair replay at each sleep (40 KL steps) | 0.11 | **-0.156** | 2.996 |
| spindle: questions only inside the sleep burst | 0.00 | -0.087 | 3.068 |
| beta drift brake (throttled 17-87% of steps, gate mean 0.52-0.98) | 0.11 | -0.142 | 2.988 |
| theta, stochastic phases | 0.11 | -0.161 | 2.976 |
| theta, fixed phases | 0.07 | -0.138 | 2.993 |
| hinted self-distillation, all parameters (pool 3-9 sentences) | 0.11 | -0.128 | 2.999 |
| soft anchor: KL to the old slots' predictions during the visit, x1.0 (DWA) | **0.22** | -0.144 | 2.988 |
| hinted self-distillation, mlp_out only (HM; pool 5-9 sentences) | 0.11 (n 9) | -0.152 | 2.979 |
| soft anchor, SEED 1 (DWA1) vs plain distillation seed 1 (DWM1) | 0.14 vs 0.08 (n 50) | -0.020 vs -0.023 | 2.913 vs 2.912 |

**Soft anchor on the GPU** (cg27a_distillak = distillcap + inf_distill_wake_akl 1.0 + held-out chains; seed 4; stopped after sleep 5), against
distillcap: sleep 3 still in the pool 0.29 / 0.31, 2-3 visits 0.62 / 0.59, drift 0.265 / ~0.30; sleep 4 0.40 / 0.42, 0.56 / 0.54, drift
0.198 / 0.202; **sleep 5 still in the pool 0.43 / 0.55, 2-3 visits 0.49 / 0.73, 4+ visits 0.65 / 0.69, after the visit 0.48 / 0.64, drift
0.163 / 0.186**. Locally (two seeds) a small recall gain at the same cost; on the GPU a TRADE: about 13% less old-domain drift for less fact
recall - the anchor resists the very directions the facts are written in (null-space probe). Not carried forward.

**JEPA readouts on the GPU** (same run, sleeps 2-5): HYBRID retrieval (sentences naming A, ranked by the JEPA key) 0.92-1.00 written,
0.77-0.85 held-out wording (name lookup alone 0.55); one-hop with the generic-hybrid sentence in context 0.64-0.81 vs weights alone 0.49-0.73
(+0.08 to +0.15). Two-hop by retrieval no better than its random-bridge control (0.62 vs 0.65 at sleep 5, n 26). Latent judge with the name
masked: masked controls vs same-topic pairs 0.25-0.36 (no relational signal beyond names). LATENT CHAINING: P(e_A, r) -> e_B memorises
(1.00 on its facts, 0.13-0.19 on never-taught facts, chance 0.05), P twice 1.00 on the unconcluded chains (two lookups, n 1-5); **Q (composition
learned from 20-37 concluded chains) 1.00 on its training chains, 0.00 on every unconcluded chain (n 1-5)**. JEPA is a strong memory key;
no sign of learned inference structure.
Sleeps 6-8 of the same run (more chains): COMPOSITION PROBE (trunk) 0.00 on 7 / 11 / **18 chains (15 with both premises answered)**; Q 0.29 (n 7,
65 training chains) / 0.09 (n 11, 81) / **0.06 (n 18, 144)** - chance; two-hop by two JEPA retrievals 0.57 vs 0.33 random-bridge control (n 40) /
0.70 vs 0.58 (n 50) / **0.76 vs 0.53 (n 76)**; one-hop with the generic-hybrid sentence 0.73 -> 0.81 / 0.72 -> 0.86 / 0.75 -> 0.83 (n 62-100). The
trunk and the learned composer do not compose; chaining two retrievals does, by about +0.2 over its control.

### JEPA mathematics, toy test (tools/toy_jop.py, 11 November, seed 0, 1500 steps)

A world WITH composition rules (family tree: grandparent = parent o parent; org chart: skip_manager = manager o manager) plus two random
relations; 600 entities, linear relation operators P_r on entity codes, InfoNCE + isotropy (JEPA-style); half of the grandparent and
skip-manager facts held out (true answer among all 600, chance 0.0017).

| | composed P_r P_r, held-out | two lookups | learned composite operator, held-out | schema: nearest single operator to P_r P_r |
|---|---|---|---|---|
| TIED (one space, as JEPA) | **1.00 / 1.00** | 1.00 / 1.00 | 0.40 / 0.51 | parent o parent -> **grandparent** (0.73 vs 0.91); manager o manager -> **skip_manager** (0.62 vs 1.03) |
| UNTIED (keys vs values, as attention) | **0.00 / 0.00** | 1.00 / 1.00 | 0.27 / 0.26 | parent o parent -> parent; manager o manager -> manager (wrong) |
| TIED + 150 trained two-step paths | 1.00 / 0.99 | 1.00 / 1.00 | **0.78 / 0.66** | grandparent (0.56 vs 0.83); skip_manager (0.56 vs 0.99) |

(grandparent / skip_manager.) With one embedding space the relation operators compose algebraically (no intermediate lookup), the
composite relation partly LEARNS the rule for people never seen with it (0.40-0.51; 0.66-0.78 with a few trained paths), and the nearest
operator to each product names the right composite relation - **rules discovered from facts alone**. With separate key and value spaces
(attention-like) the algebra breaks (0.00) and the schema is wrong. Caveats: random relations also find a 'nearest' operator (lives_near o
lives_near -> lives_near, 0.79 vs 1.16) - a schema needs a null (shuffled facts) and a functional check before it is trusted; and the planted
facts in Platypus have no composition rules (every chain is arbitrary, the conclusion generic), which is why the GPU composer Q could only
memorise. Testing this in Platypus needs planted facts that obey rules.

**Rule world on by default, local short-run check (11 November, seed 0, default build with inf_rule_world 12):** 17-97 rule-world facts planted per domain, no errors; pass 1 3.649 / -1.420, pass 2 **3.008 / -0.139** against the defaults check without it 2.997 / -0.150 and the retrieval check 2.992 / -0.139 - within seed noise (0.017). The rule world is safe as a default.

### Rule discovery against a null, and transfer to new families (tools/toy_rules.py, 11 November, seed 0, 1500 steps)

40 families and 40 org charts of 7 people (child-of, grandchild-of; supervised-by, overseen-by), plus works_with (one-to-many, 3 each, no
rule); 560 entities, chance 0.0018. 30% of the families and charts are NEW: only their one-hop facts are seen, none of their composites. A
rule counts only if the product's nearest operator beats the same distance in 3 shuffled-fact nulls by 0.05 and is neither factor.

| | child o child | sup o sup | grand o grand, over o over, child o sup, works_with o works_with | new groups: grandchild / overseen, discovered rule applied (compose, snap, compose) | composed without snap | learned composite operator directly |
|---|---|---|---|---|---|---|
| clean | **grand** 0.74 vs null 1.27 (z +18) | **over** 0.79 vs 1.27 (z +77) | no rule (all within 0.01-0.08 of the null) | **1.00 / 1.00** (n 48 / 48) | 0.94 / 0.94 | **0.00 / 0.00** |
| 10% wrong parents | **grand** 0.84 vs 1.26 (z +17) | **over** 0.79 vs 1.27 (z +45) | no rule | **0.85** / 1.00 | 0.83 / 0.98 | 0.00 / 0.00 |

The null does its job: the two true rules stand far outside it, and every relation with no rule (including the one-to-many works_with and
the cross-domain child o sup) is rejected - the caveat from toy_jop.py is answered. The discovered rule TRANSFERS: applied by composition to
families and charts never seen with a composite fact, it gets every grandchild and skip-level right (0.85 with 10% corrupted parents, close to
the 0.81 expected from two noisy hops). The learned composite operator, by contrast, does not transfer at all (0.00): a relation learned as a
lookup is memorised per entity; only the composed operators generalise. Inference here comes from discovering which relations compose and
then composing - not from training the conclusion.

### The reader against gold relations, and a frame-JEPA probe (tools/reader_audit.py, tools/frame_probe.py; 11 November, local)

READER AUDIT on the embedded Re-DocRED benchmark (4,726 gold Wikidata relations, ~2,000 sentences; window frequencies estimated from
English word frequencies): every clause read gives 1,252 records, of which 163 have both arguments benchmark entities and 37 of those
(0.23) a gold relation; **gold pairs covered 36 of 4,726 (0.8%)**; the first record per sentence (what the evidence record keeps) 0.7%;
after the name-memory / distillation filters 116 records, 17 gold pairs (0.4%); facts must be seen twice (benchmark documents appear
once): about 0. Causes: the arguments are the clause's two RAREST words (918 of 2,864 entities have no word rare enough; rare common words
become arguments: 'microscopic' ... 'spinal'), multi-word names are split (1,771 of 2,864), the relation must lie between the arguments
with no punctuation, one record per sentence, single capitalised words only downstream, latest object only, seen twice. Only 12% of the
gold pairs have both entities named in one sentence (the rest need coreference / cross-sentence reading).

FRAME PROBE: a frame encoder (word embeddings, BiGRU, the states at the two masked fillers -> a relation code -> an operator) trained
WITHOUT labels on 80,551 frames from local Wikipedia (the operator maps one filler's code to the other's, InfoNCE); tested on 724
benchmark frames with the gold entities masked - is the nearest other frame of the same gold relation type?

| majority type | untrained encoder | bag of connector words | trained frame code (3 epochs) | trained + connector words |
|---|---|---|---|---|
| 0.24 | 0.39 | **0.44** | 0.40 | 0.47 |

The connector words carry the relation (0.44 against 0.24); the frame-JEPA objective as built does NOT learn it (0.40, level with
untrained; linear probe 0.54 against 0.56). v1 (a free code per filler span) stayed at chance: most spans occur once. Likely causes: the
partner filler's identity is predicted mostly by topic / co-occurrence, not by the relation (the latent judge's failure again); the
capitalised-run scaffold splits names, so many training frames are name-internal ('Church' of 'Scientology'); 10 MB on two CPUs.

### Real-text replay, false memories, the trunk's confidence and the alignment control (cg27a_rtx, cg27a_rtx16, cg27a_replaynoal; seed 4; 12 November evening)

All three = cg27a_replay (source replay of planted facts) plus: `cg27a_rtx` - the sentences of HALF the benchmark documents read replayed as
plain text, 4 per question-row step (the other half read once: the control), no real-text view alignment; `cg27a_rtx16` - the same, 16 per
step; `cg27a_replaynoal` - no real-text replay and NO view alignment (inf_distill_align 0; the alignment control). cg27a_rtx16 and
cg27a_replaynoal were stopped at visit 4 (the author: enough to choose the next runs).

REAL-TEXT RECALL from the weights (true object among up to 10 same-type names; 'prior' = the same question with a fresh subject):

| visit | rtx: replayed / prior | rtx: control / prior | rtx16: replayed / prior | rtx16 DIAGNOSTICS, replayed named (48): storage / own-frame / template / prior | rtx16 control named: own-frame / prior |
|---|---|---|---|---|---|
| 2 | 0.12 / 0.12 | 0.05 / 0.03 | 0.19 / 0.19 | 0.00 / 0.12 / 0.06 / 0.08 | 0.15 / 0.08 |
| 3 | 0.14 / 0.14 | 0.14 / 0.12 | 0.17 / 0.16 | 0.02 / 0.12 / 0.08 / 0.06 | 0.08 / 0.08 |
| 4 | 0.27 / 0.27 | 0.12 / 0.09 | 0.30 / 0.27 | **0.06 / 0.23 / 0.08 / 0.08** | 0.04 / 0.04 |
| 7 | 0.19 / 0.14 | 0.08 / 0.06 | - | - | - |

**First sign of storage of real sentences (rtx16, visit 4): replayed documents asked in their own sentence's wording 0.23 vs 0.08 (about
11 of 48 vs 4), never-replayed documents at their prior; not in another wording (0.08).** One visit. cg27a_rtx's gap over the prior
(+0.05 at visit 7) is within noise: cg27a_replaynoal, which replays no real text, moved +0.05 at visit 3. Real-text replay loss after ~15
readings a sentence: 0.97 (rtx16) vs 0.95 (rtx); planted source sentences 0.4-0.6. Cost: BPC within 0.006 of cg27a_rtx, under a minute a
visit for 4x the replay. Local safety (short run, 4x replay): final BPC 2.996 / forgetting -0.158 vs 3.003 / -0.149.

FALSE MEMORY PROBE (from the trunk; (a) a fresh name with a known relation, (b) a known subject with a relation it has no fact for;
greedy answer is a STORED name) and the trunk's confidence (first answer byte) on known facts it answers right vs never-read questions:

| visit | (a) rtx / rtx16 / replaynoal | (b) rtx / rtx16 / replaynoal | AUC known-right vs never-read: rtx16 / replaynoal |
|---|---|---|---|
| 2 | 0.09 / 0.12 / 0.16 | 0.19 / 0.19 / 0.22 | 0.70 / 0.59 |
| 3 | 0.61 / 0.61 / 0.71 | 0.58 / 0.61 / 0.65 | 0.71 / 0.73 |
| 4 | 0.69 / 0.83 / 0.83 | 0.47 / 0.56 / 0.59 | **0.84 / 0.82** |
| 7 | 0.93 / - / - | 0.47 / - / - | - |

**The trunk learns to answer any question in the taught form with a stored name** - in every build, with or without real-text replay, so
the question rows (or planted replay) cause it, not real text; leakage of a subject's other-relation object 0.00. **Its confidence
separates what it knows from what it never read better as training goes on** (AUC 0.59-0.73 -> 0.82-0.84); not enough to abstain on (the
store stays the gatekeeper). Tested next: refusal rows (cg27a_refuse) and the ABSTENTION readout.

ALIGNMENT CONTROL (facts only read, from the weights; ledger): never asked 0.00 / 0.03 / 0.06 at visits 2-4 without alignment
(replaynoal) vs 0.00 / 0.04 / 0.02 (rtx) and 0.09 at visit 4 (cg27a_replay); taught 0.22 / 0.43 / 0.43 vs 0.25 / 0.43 / 0.44. No
difference through visit 4; the decisive visits were not run.

### cg27a_link: store linking and prequential bits (GPU, seed 4, 13 November; complete, 10 visits)

cg27a_rtx16 + inf_store_link 1 + inf_preq 1. BPC within seed noise of cg27a_rtxt at every visit (largest +0.016 at visit 6); local safety
identical to the baseline (-0.158 / 2.996). Linking feeds only S_['relc'] (readouts), so nothing in training changes.
- **Letter links: the 1.00 'same entity' printed at every visit is an artefact.** The check counted a link right when the two names
  shared ANY distinctive benchmark word - true of every tail link by construction ('Smith' / 'John Smith'). Strict check (both names gold
  mentions of one entity; added 13 November, compress.py): **0.58 (31)** on 50 benchmark documents locally against 1.00 (18) lenient;
  offline on the 500 dev documents the same letter rule is 0.51 (964 pairs). Letter links are roughly half right, not all right.
- **Compression links: 0 of 6** on benchmark entities (lenient check), 565 links by visit 5; median gain 4-5 bits.
- Relation substitution AUC 0.598 / 0.642 / 0.606 at visits 2 / 4 / 6 (best same-type candidate 0.153 vs chance 0.086 at visit 6): weak
  and flat, not rising.
- Prequential bits on held-out text of the domains read: all bytes 2.023 -> 1.881 (visits 1-5); known names 2.2-2.3 vs unknown 3.5-5.1
  bits per byte. NEXT-domain figures are invalid in this run (read through an untrained slot; fixed for cg27a_rtxw).
- Gate (threshold from the previous sleep), never-read false vs ungated: 0.40 / 0.35 / 0.27 / 0.10 vs 0.63 / 0.65 / 0.55 / 0.56 at visits
  3-6; right overall 0.41 / 0.33 / 0.50 / 0.34 vs ungated 0.44 / 0.39 / 0.53 / 0.53. Smoothed threshold: false 0.40 / 0.43 / 0.33 / 0.16,
  right 0.41 / 0.36 / 0.52 / 0.39 - steadier.
- **Visits 7-10.** Smoothed gate right / never-read false: 0.61 / 0.24, 0.61 / 0.19, 0.67 / 0.23, **0.67 / 0.12** (ungated 0.73 / 0.64,
  0.69 / 0.58, 0.78 / 0.66, 0.73 / 0.58): at visit 10 the trunk keeps 92% of its right answers and drops 79% of its false ones. Known-fact
  confidence AUC 0.87 / 0.90 / 0.88 / **0.94**; known right 0.73 / 0.69 / 0.78 / 0.73 (cg27a_rtxt 0.59 / 0.53 / 0.62 / 0.64). Relation
  substitution 0.616 at visit 8. Prequential bits on read domains 1.80 at visit 10, known names 1.94 vs unknown 3.36.
- **End of run: forgetting +0.088, mean final BPC 1.818** (cg27a_rtxt +0.077 / 1.808; without real-text replay +0.026 / 1.767);
  Gutenberg +0.238, StackExchange +0.336. Linking cannot touch training, so the +0.01 against cg27a_rtxt is GPU run-to-run variation; the
  real-text replay cost stands (cg27a_rtxw tests the fix).

### Rules from real facts: the miner, not the amount of text (tools/dataset_volume.py, tools/count_miner.py, tools/count_miner_reader.py; 13 November)

Re-DocRED train + dev (3,553 documents, 93,278 gold facts, local copy); 20% of facts held out; proposals checked against all gold facts on
pairs Re-DocRED annotates (both entities in one document).

| Documents | _rm_mine (learned snap): rules / proposals gold / writes | COUNTING miner (n >= 20, Wilson z 4 >= 0.7): rules / proposals gold | held-out recovered |
|---|---|---|---|
| 262 | 3 / 0.18 / 0 | 1 / 0.42 | 15 |
| 800 | 2 (both spurious) / 0.00 / **9, all wrong** | 4 / 0.35 | 83 |
| 1,600 | 1 / 0.04 / 0 | 16 / 0.34 | 219 |
| 3,553 | out of memory (8 GB) | **36** / 0.33 | 603 |

- **_rm_mine does not scale to real facts** (thousands of names): fewer rules with more text, and at 800 documents a spurious rule passed on
  4 of 4 and would have written 9 wrong facts. Its n >= 3 at z 1.96 is far too lenient against thousands of candidates. Rule world only.
- **Exact chain counts find real rules, more with more text**: a region's country from the places in it (0.88 of 992), siblings share a
  father (0.93 of 119), a track's label and performer from its album (0.91-0.92 of 158-177). 40 s, little memory.
- **But proposals are 0.33 gold** (0.49 when 3+ rules agree, 0.41 when every chain ends on the same name): rules that hold 0.88-0.96 on
  subjects with a known answer are far weaker on subjects without one. Proposals are hypotheses, never writes.
- **Through the reader: no rules at all.** The reader's records on the 3,053 train documents cover 2,903 of 85,932 gold facts (**3.4%**);
  with gold relation types on those records 0 rules; on its 65,153 raw records (300 frames) 0; on 181,872 benchmark + Wikipedia records,
  looser bound, 4 punctuation 'rules'. **The reader's fact recall is the blocker for real-text inference.**

### Rescuing compression linking (tools/link_rescue.py, 13 November; Re-DocRED dev 500 documents, stand-in trunk 192 wide, 6,000 steps)

The stand-in's link AUC is 0.56 (the model's 0.74-0.84): compare rows with each other. Candidates: the 12 nearest earlier mentions; pairs
letters do not link (118,346; 3.3% the same entity).

| Policy | Links | Same entity | Right, of them HARD (no shared word) |
|---|---|---|---|
| chance | 118,346 | 0.033 | 3,952 / 727 |
| top 1.5% by gain (the GPU's link rate) | 1,776 | 0.075 | 133 / 83 |
| calibrated on letter links (50th-90th percentile) | 8,571-52,090 | 0.040-0.049 | - |
| one earlier mention per later mention, margin >= 2 bits, capitalised | 509 | 0.191 | 97 / 62 |
| the same + top 1.5% | 182 | **0.264** | 48 / 35 |

- Calibration on letter links fails here: letter links are only 0.51 right offline, so they are poor positives.
- **Competition (each later mention picks one earlier mention, by a clear margin) lifts precision 3.5x over the fixed-rate policy and 8x
  over chance**, and most of what it finds are hard pairs letters can never find. Still far from writable: a vote, or a hypothesis.

### Learning what to extract from the trunk's feedback (tools/extract_learn.py, tools/extract_gain.py, tools/extract_curve.py; 13 November)

The idea: over-generate structural pieces of wording (every name pair in a sentence with the words between: 78,784 candidates in Re-DocRED
train + dev, 24% related) and let the trunk's feedback teach a learner which are needed. Feedback (label-free) = bits a record saves the
stand-in trunk (192 wide, 6,000 steps) where its names meet again in another document, beyond what the names alone save. Learner: ridge
regression on features any reader has (the words between, hashed; first / last word; length; punctuation; how often the wording recurs).
Tested on 710 held-out documents (15,609 candidates, 24.2% related).

| Extractor | AUC | top 10%: precision / recall | top 25%: precision / recall |
|---|---|---|---|
| random | 0.50 | 0.25 / 0.10 | 0.24 / 0.24 |
| closeness of the names | 0.54 | 0.24 / 0.10 | 0.27 / 0.28 |
| **learned from trunk feedback (beyond the names alone; 5,254 records)** | **0.55-0.56** | **0.33-0.35 / 0.14** | **0.31 / 0.32** |
| pooled per exact wording, same feedback | 0.52 | 0.38 / 0.16 | 0.26 / 0.27 |
| learned from the RAW record gain | 0.38-0.40 | 0.09-0.11 / 0.04 | 0.14-0.15 / 0.15 |
| ceiling A: same learner, gold labels, same 5,254 records | 0.74 | 0.50 / 0.21 | 0.45 / 0.47 |
| ceiling B: same learner, gold labels, all 63,000 training candidates | 0.81 | 0.71 / 0.29 | 0.55 / 0.57 |

- **Trunk feedback pooled by a learner is a real but weak teacher**: top-10% precision 0.33-0.35 against 0.24 base - about 40% of the way
  to gold labels on the same records. The per-record signal it learns from is AUC 0.54.
- **The raw gain teaches the wrong thing** (0.38-0.40, below chance): records of names that merely co-occur (lists, repeated sets of
  names) save the trunk the most bits. Feedback must be taken beyond what the names alone are worth.
- A LIST-STYLE null ('A , B') removes most of the signal (per record 0.52, top 10% 0.26); dropped.
- **The per-record signal does not grow with training in a small trunk** (extract_curve, trained on this text: 0.48 / 0.51 / 0.52 / 0.52 at
  0 / 1,000 / 3,000 / 6,000 steps). Whether a stronger trunk gives a stronger signal is the GPU question (cg27a_xgain; the model's trunk
  saves 2-3 bits per related mention, the stand-in 0.06-0.12).
- AUC helpers corrected the same day: tied scores now share their mean rank. Before, discrete scores (closeness, wording frequency, letter
  triples) read below 0.5; e.g. closeness 0.49 -> 0.59 in tools/extract_gain.py. Earlier letter-triple HARD figures (0.17-0.35) were biased.

### How good must premises be? Record-level compression into types (13 November, evening)

**Premise threshold** (tools/premise_threshold.py; the store's precision SET by keeping half the related records plus unrelated ones; A-C
proposed through a shared name from another document, as tools/integrate_test.py; checked where Re-DocRED annotates the pair; base 0.09):

| premise precision | A-C proposals | gold | with 2+ bridges | gold |
|---|---|---|---|---|
| 0.24 (store as is) | 39,241 | 0.10 | 11,300 | 0.08 |
| 0.38 | 20,911 | 0.13 | 3,930 | 0.11 |
| 0.50 | 13,369 | 0.18 | 1,831 | 0.18 |
| 0.60 | 8,709 | 0.23 | 790 | 0.28 |
| 0.70 | 6,011 | 0.31 | 359 | 0.43 |
| 0.80 | 4,628 | 0.35 | 251 | 0.52 |
| 0.90 | 3,446 | 0.42 | 180 | 0.64 |
| 1.00 | 2,796 | 0.51 | 130 | 0.75 |

- Inference precision rises smoothly with premise precision; corroboration (2+ bridges) helps only above about 0.5 and amplifies noise below.
- A store chosen by QUALITY does better than random premises of the same precision: the consistency-teacher learner's top 5% of records
  (premise precision 0.55-0.56) gives A-C gold **0.33-0.34** (480-510 proposals; 3.7x base), against 0.18-0.23 for random premises at 0.5-0.6;
  top 10% (0.46): 0.28 (about 2,500 proposals); top 25% (0.36): 0.165. Recall claim only (proposals checked against gold; nothing written).

**Record-level MDL co-clustering** (tools/record_mdl.py; the 78,784 store records as (name, wording, name); a three-way block model, names
into K classes and wordings into M classes by information-theoretic co-clustering; MDL over data + block table + assignments; no labels):

| variant | MDL choice | bits a record (none: 33.6) | name classes: purity / NMI (best K) | relation cue AUC (gold type pair 0.80) | block score premise precision top 10% (learner 0.45) |
|---|---|---|---|---|---|
| random starts | K 4, M 32 | 33.46 | 0.43-0.50 / 0.02-0.18 | 0.56-0.61 | 0.27-0.35 |
| split warm starts | K 4, M 32 | 33.43 | 0.43-0.50 / 0.11-0.16 | 0.59-0.61 | 0.29-0.31 |
| started from context types (tools/name_types.py) | K 6, M 32 | 33.46 | 0.48-**0.60** / 0.15-**0.25** | **0.64** (K 6 M 32 pair + wording class) | 0.27-0.29 |
| fitted on the learner's top 10% (cleaner store, 0.46), split | K 4, M 8 | 25.75 (none 25.4) | 0.43-0.48 / 0.11-0.16 | 0.53-0.59 | 0.27-0.39 |
| fitted on the top 10%, from context types | K 6, M 8 | 26.51 (none 25.3) | 0.43-0.51 / 0.13-0.21 | 0.55-0.60 | 0.18-0.40 |

- **Not positive.** Classes save 0.1-0.2 bits a record at best on the whole store and none on the clean store once the table is paid for.
  The best relation cue (0.64) is level with words + context types (0.62-0.67) and far from the gold type pair (0.80). Block scores do not
  raise premise precision, and combined with the learner they do not help it (top 5%: 0.38-0.56 against the learner's 0.54). A-C from
  block-scored stores: gold 0.02-0.10 (one set of 267 at 0.20).
- **Why: data per name** (tools/record_mdl_freq.py). 56% of names have 1-2 records, 40% have 3-9, 5% have 10+ (43% of records). Names
  with 10+ records get purity **0.58-0.64, NMI 0.30-0.36** (context-type start); names with 3-9 get 0.37-0.51 / 0.07-0.16; names with 1-2
  are near chance (0.28-0.39 / 0.02-0.07). Compression finds types where each name has enough records, so the limit is the amount of text
  per name, not the compression principle. The Re-DocRED store is too small for most names, matching tools/name_types.py (T2a, 10 MB too few
  contexts).

**More text per name: the volume curve** (tools/distant_records.py, record_mdl.py RM_EXTRA; DocRED's distant training set, 101,873 Wikipedia
documents with no overlap with Re-DocRED, read into store records the same way; its labels are not used; split warm starts, 2 restarts, K
4-64, M 32/128; evaluation unchanged, on Re-DocRED gold; one run each):

| store | records | Re-DocRED names with 10+ records | name classes: purity / NMI (best) | relation cue AUC (best; gold types 0.80) | learner + block association, premise precision top 5% / 10% (learner alone 0.54 / 0.45) |
|---|---|---|---|---|---|
| Re-DocRED only | 79k | 5% | 0.50 / 0.16 | 0.61 | 0.48-0.53 / 0.42-0.45 |
| + 10k distant documents | 286k | 12% | 0.56 / 0.20 | 0.61 | 0.46-0.55 / 0.38-0.45 |
| + 30k | 704k | 21% | 0.67 / 0.32 | 0.65 | 0.50-0.55 / 0.42-0.49 |
| + 101,873 (all) | 2.2M | **33%** | **0.72 / 0.37** | **0.68** (K 4 M 128 name-class pair + wording class; pair alone 0.66) | **0.60 / 0.52-0.53** |

- **Positive: compression finds types once names have enough records**, and the relation cue improves with text, monotonically from 10k.
  All-text classes are recognisable: cities, countries, US states, years, nationalities, given names, numbers, software products, works
  (MDL choice K 16, M 128). By records per name (all text, K 16): 1-2 records purity 0.41 (majority 0.27), 3-9 0.57, 10-49 0.62, 50+ 0.70.
- **Premise precision:** at full volume the block association adds to the label-free learner for the first time (top 5% 0.54 -> 0.60, top
  10% 0.45 -> 0.53, held-out documents). Below full volume it does not. One run; the gain is at the target's lower edge (0.6).
- **But inference from that store gets worse:** keeping the top 10% by learner + association (premise precision 0.53) gives A-C gold 0.20,
  below the learner's own top 10% (0.46 precision, A-C 0.28). Type association favours records of relations that do not compose through a
  shared name (e.g. nationality and country pairs), so premise precision alone does not predict inference (as the premise threshold test's
  quality-vs-random gap also showed, in the other direction).
- Trend: about +0.03 relation-cue AUC per tripling of text from 10k documents; on that trend the gold type cue (0.80) is several orders of
  magnitude of text away, so types from volume alone close part of the gap, not all of it.

**Keying the compression by a longer-term context** (tools/doc_topics.py: every Re-DocRED and distant document clustered by its words, tf-idf,
SVD 64, k-means 8 / 32, no labels; record_mdl.py RM_CTX: the block table becomes [document context, A class, wording class, B class] and
every record is coded GIVEN its document's context; full store, K 4-32, M 32/128, split starts, 2 restarts; one run each):

| full store (2.2M records) | data bits a record given context (best) | name types purity / NMI | relation cue AUC | learner + association, premise precision top 5% / 10% |
|---|---|---|---|---|
| unkeyed | 36.7 | **0.72 / 0.37** | 0.68 | 0.60 / 0.52-0.53 |
| keyed, 8 contexts | 36.7 (table 8x larger) | 0.64 / 0.31 | 0.67 (0.65 without the context cue) | 0.61 / 0.50 |
| keyed, 32 contexts | 36.6 (table 32x larger) | 0.59 / 0.29 | 0.65 | 0.53 / 0.47 |
| unkeyed classes + same-context cue (RM_CTXEVAL, 8) | - | 0.72 / 0.37 | **0.70** (top-1 0.23 vs 0.26 without) | 0.60 / 0.51-0.53 |

- **Not positive as a key.** Given the context, records cost at most 0.1-0.4 bits less, less than the larger table costs, so MDL rejects it;
  the name types loosen (the context absorbs what the classes did: in a sports document who-plays-for-whom is already likely), more with
  finer context. Relation cue, premise precision and A-C inference (0.27-0.29 at 10% kept, within run noise) do not improve.
- **Slightly positive as a separate cue:** the document context beside the unkeyed types lifts the relation AUC 0.68 -> 0.70, the best
  label-free figure so far, though top-1 falls (0.26 -> 0.23). Context is better used next to the types than as the key the types are
  learned under.

### Premises that chain: composability learned from later confirmation (tools/chain_bootstrap.py, 13 November, late)

Re-DocRED store (78,784 records), A-C proposed through a shared name from another document (2.2M proposals; 80,185 verifiable, gold
0.108); half the pairs (by hash) learn which premise KINDS compose - kinds = (wording class, bridge side) pairs from record-level compression
(K 16, M 128), or the name-type triple - from label-free CONFIRMATION (A and C later read together in one sentence of another document);
policies applied to the other half (40,013 verifiable, gold 0.108). Both runs were stopped by the machine's memory before hop 2 (inference
built on inference) and, at 30k, before trunk congruence.

| policy (test pairs) | Re-DocRED only | + 30k distant documents (confirmations 0.2% -> 1.7%) |
|---|---|---|
| all proposals | 0.108 | 0.108 |
| 2+ / 3+ bridges | 0.082 / 0.069 | 0.082 / 0.069 |
| composability (wording kinds), top 25% / 10% | 0.073 / 0.061 | 0.072 / 0.067 |
| composability (name-type triple), top 25% / 10% | - | 0.053 / 0.029 |
| **confirmed by later reading** (the inference waits for the text) | **0.170** (2,304) | **0.143** (9,022) |
| stand-in trunk congruence (gain from A, top 3 of 10; cg27a_rti's check) | **AUC 0.445**: passed 0.078 vs failed 0.124 | (not reached) |
| two keys (composability AND congruence) | 0.035 | - |

- **Not positive: composability learned from confirmation selects WORSE than taking every proposal**, more so with the coarser type-triple
  kinds. A sentence that names A and C together is mostly co-occurrence (lists of places, people of one team), so the kinds that get
  'confirmed' are the kinds that co-occur, not the kinds that relate - the co-occurrence trap again (compression gain, usefulness, record
  surprise). Bridges also lower the rate here (0.082 / 0.069): with premise precision about 0.24, corroboration amplifies co-occurrence (as
  tools/premise_threshold.py: bridges help only above about 0.5).
- **The stand-in trunk's check points the wrong way** (AUC 0.445, about 3.4 standard errors below 0.5): 'A helps the trunk predict C' favours
  wrong inferences. cg27a_rti uses this check to choose writes; cg27a_rtiw (check as a readout only, every 2+ bridge inference written) is the
  fallback if the GPU trunk's AUC is at or below 0.5 at visit 2.
- **Positive, modest: waiting for the text helps** - inferences later read together in a sentence are right 0.14-0.17 (1.3-1.6x base).
- **Confirmation in a RELATIONAL wording fixes the direction (CB_CONF=rel; the later sentence's record among the consistency learner's top
  quarter, label-free; Re-DocRED only):** composability learned from it now selects ABOVE base - wording kinds 0.113 / 0.115 (top 25% / 10%),
  name-type triples **0.115 / 0.124** (4,001 inferences; about 3 standard errors over 0.108) - where co-occurrence confirmation gave 0.029-0.073.
  Small but the sign flips: relational wording is the judge that separates relating from co-occurring. As a direct filter it is strong but
  rare: inferences later read together in a relational wording are right **0.359** (487; 3.3x base), with 2+ bridges **0.387** (287) - close
  to direct extraction, so their value is in predicting before the text arrives and in teaching which kinds compose.
- **Bootstrapping rounds (CB_ROUNDS 4, CB_GROW 2%)**: seeds = 461 train inferences confirmed in a relational wording (gold 0.360); each round
  the 2% of train proposals (21,969) with the most composable type triple join the confirmed set, composability re-learned. TEST composability
  top 10%: **0.123 -> 0.135 -> 0.134 -> 0.135 -> 0.141** (base 0.108; 4,001 inferences, about 1.3x base by round 4); the added batches are
  right 0.116-0.159 (no collapse, so drift is limited, but each batch is far below the seeds; the confirmed set dilutes 0.36 -> 0.15).
  **Modestly positive**: coverage grows 190-fold while test precision creeps up.
- **Hop 2 (CB_HOP2 lite; inference built on inference)**: from accepted inferences right 0.360 (relational confirmation, 948), 0.141
  (composability top 10%) or 0.133 (a random sample), the hop-2 inferences through stored records are right **0.143 / 0.137 / 0.148** (10-26k
  verifiable; 2+ routes 0.12-0.15). **Not positive: hop-2 precision does not inherit the premise's** - an accurate untyped inference
  ('A is connected to C') carries no relation, so chaining from it is as good as chaining from any co-occurrence. Building on inference needs
  typed (relation-identified) inferences.
- Discrepancy to resolve: on the GPU (distant set, Wikidata labels) 2+ bridges RAISE A-C precision (0.23 vs 0.11-0.13 for all); on
  Re-DocRED gold they lower it. Likely the distant labels reward transitive chains (located in, P131) that Wikidata labels densely.

### GPU, 13 November night: cg27a_prem (surprise-gated store + premises at scale, visits 1-6) and cg27a_rti (end-to-end inference, visits 1-4); seed 4

cg27a_prem - SURPRISE GATE (the model change): admits 0.49-0.50 of real-text sentences each visit (about 45 s); among sentences joining two
benchmark entities, gold-related pairs admitted vs other co-occurrences: 0.88 vs 0.51 (1.71; 65 related), 0.53 vs 0.46 (1.16; 141),
0.97 vs 0.80 (1.21; 71), 0.97 vs 0.77 (1.26; 59), 0.85 vs 0.53 (1.60; 53) at visits 1-5 (visit 6, Github: no benchmark text). **Positive,
5 of 5 visits above 1** (confound not excluded: related sentences may be more surprising for other reasons). Real-text recall at the
relation prior every visit, as earlier builds; gated answers right when answered 0.63 -> 0.88 (visits 3-6).
PREMISES AT SCALE (readout; Wikidata labels for the distant set):

| visit | store (distant docs) | name types purity K16 / NMI K4 | relation cue (gold types) | A-C 2+ bridges right: random / trunk surprise / type association | trunk surprise AUC |
|---|---|---|---|---|---|
| 2 | 20k | 0.60 / 0.20 | 0.57 (0.73) | 0.18 / 0.13 / 0.06 | 0.52 |
| 4 | 41k | 0.65 / 0.28 | 0.59 (0.75) | 0.19 / 0.12 / 0.08 | 0.51 |
| 6 | 61k | 0.66 / 0.33 | 0.63 (0.78) | 0.23 / 0.13 / 0.16 | 0.51 |

Types and the relation cue rise with reading as offline (positive); the trunk's surprise on single records carries no signal; no
label-free single-premise policy beats random premises for inference (type association improves as types improve).

cg27a_rti - END-TO-END REAL-TEXT INFERENCE (premises at random, 2+ bridges, the trunk's check, write 'A is connected to C'): the CHECK is
at chance (AUC 0.512 visit 2, 0.491 visit 4; passed right 0.374 / 0.319 vs failed 0.364 / 0.335 - the 600 best-bridged inferences are right
about 0.33-0.37 by Wikidata labels). WRITES: 181 at visit 2, 370 by visit 4 (a fifth held back). RECALL FROM THE TRUNK of the visit-2
writes (store off), taught vs held back: visit 3 RIGHT 0.07 (96 questions) vs 0.03 (32), WRONG 0.13 (172) vs 0.07 (42); **visit 4 RIGHT
0.09 vs 0.03, WRONG 0.02 vs 0.00** - right inferences retained, wrong ones fading (about 2.5 standard errors; visit 3 showed the opposite).
Provisional until visits 6-10.
**Visits 5-6 (writes of visits 2 + 4):** check AUC 0.478 at visit 6 (third time at chance; passed and failed both right 0.37). Recall, taught
vs held back: visit 5 RIGHT 0.04 (194) vs 0.06 (52), WRONG 0.06 (360) vs 0.00 (106); **visit 6 RIGHT 0.09 vs 0.02, WRONG 0.06 vs 0.01**.
Written inferences reach the weights (taught above held back for both), but right is not reliably above wrong (0.09 vs 0.06, about 1.3
standard errors; the order changes between visits) - the visit-4 'fading' does not repeat. Consistent with tools/false_fade.py: right
inferences are kept only when later reading supports them, and here the trunk never reads the documents they came from. Next build: the
trunk reads the source documents, recall split by write age, a third held back.

### Do wrong inferences fade while right ones stay? (tools/false_fade.py, 13 November, late; stand-in trunk, two seeds)

120 RIGHT inferences (A and C gold-related in a Re-DocRED training document) and 120 WRONG ones (A with a same-type name from another
document, never related), taught as 'A is connected to C.' (300 steps, 8 fact rows + 24 text windows), then 600 steps of reading in two
arms. Recall: C first among 10 same-type names after 'It is on record that A is connected to ' (chance 0.10).

| | seed 0 right / wrong | seed 1 right / wrong |
|---|---|---|
| before teaching | 0.28 / 0.10 | 0.22 / 0.12 |
| taught | 0.39 / 0.25 | 0.32 / 0.33 |
| + reading the EVIDENCE documents (those naming the pairs) | **0.39 / 0.11** | **0.33 / 0.21** |
| + reading OTHER documents | 0.27 / 0.10 | 0.19 / 0.13 |

**Positive (both seeds): wrong inferences fade whatever is read; right ones keep their taught gain (+0.11 above before-teaching in both
seeds) only when later reading supports them, and fall back without it.** Continued reading filters written inferences - a plausible
mechanism for the GPU cg27a_rti visit 4 pattern (right 0.09 vs wrong 0.02). In cg27a_rti the trunk never reads the distant documents the
inferences come from, which may be why its weights-level signal is faint: the next build should have the trunk read them. Small trunk;
about 2-3 standard errors per seed.

**Caveat added 14 November (tools/evidence_infer.py v2, below):** in this test the right pairs' evidence documents name A and C *together*
(the relation is stated, not inferred), and the wrong inferences' C is never read at all. Both favour the right C by familiarity and by
direct statement. With novel pairs and the familiarity matched, the effect does not appear (next section).

### Universal schema and typed, coupled inference (tools/typed_infer.py, 13 November, late; Re-DocRED + 30k distant documents)

UNIVERSAL SCHEMA (Riedel et al. 2013): PPMI over 15,285 wordings x 609,864 ordered pairs (32,662 seen 2+ times), SVD 64, 64 latent
relations (k-means). RELATION IDENTITY on held-out gold-related records (18,000 pairs): wording class (record compression) AUC 0.561;
**universal-schema wording vector 0.658, + compressed name-type pair 0.669**; latent relation (equal) 0.562; gold type pair 0.803. As good
as the best label-free cue so far at this volume (record compression at 101k documents 0.68) - pairs recur too rarely for the factorisation
to do more here; it should gain with volume.

TYPED, COUPLED INFERENCE: each premise typed by its latent relation and the bridge's side; a COMPOSITION TABLE (premise relations -> the
relation A and C are later read in, in a relational wording) learned on train pairs (462 confirmed); TYPE COUPLING (NELL): the predicted
relation must have been seen with A's and C's name types. Test pairs (40,013 verifiable, gold 0.108):

| policy | inferences | gold | TYPED right (relation too) |
|---|---|---|---|
| all proposals | 40,013 | 0.108 | 0.045 |
| typed composability top 25% / 10% | 10,028 / 4,277 | 0.132 / 0.148 | 0.084 / 0.098 |
| type coupling only | 14,550 | 0.104 | 0.058 |
| **typed composability top 10% AND type coupling** | **2,257** | **0.160** | **0.116** |

**TYPED HOP 2** (3,000 accepted typed inferences, hop-1 gold 0.160, + stored records, composition-gated again): **0.196** (896; ungated
0.121) - against untyped hop 2 at 0.14 whatever the premise. **Positive: typed inferences compose** - hop 2 precision now exceeds hop 1, the
first time inference built on inference has gained rather than lost. Typed composability is the best label-free premise policy so far
(0.160, 1.5x base; untyped bootstrapped 0.141, untyped type triples 0.124). Small absolute numbers; one run.

**Replication at full scale (101,873 distant documents; same test pairs, Re-DocRED gold), 23:30:** universal schema 0.654, + name-type pair
0.665 (no gain over 30k: 0.658 / 0.669; recurring pairs 118,900); typed composability top 25% / 10% **0.129 / 0.138**, + type coupling
0.123 / **0.140**; typed hop 2 composability-gated **0.149** (909; hop-1 0.144). **Robust part: typed composability selects about 1.3x
base (0.13-0.16 in both runs). Not robust: type coupling (adds 0-0.01) and the typed hop-2 gain (0.196 at 30k, 0.149 at full - about level
with untyped hop 2).** Single runs; treat the 30k figures as the upper end.

Relational scorer port check (tools/relscore_check.py; held-out Re-DocRED gold, base 0.242): the package's in-store scorer top 5% 0.497
(AUC 0.603) vs the offline learner 0.541 (0.627) - the port is sound. Its weak benchmark figures in the readout came from a tokenisation
mismatch (benchmark records split on spaces, DocRED's with punctuation as tokens); fixed in platypus/inference/premises.py.

### Inference as compression of the fact table (tools/fact_compress.py, 14 November, morning; Re-DocRED train + dev store, two held-out splits)

The record compressor codes p(A | class A) and p(B | class B) independently: which names go together - the facts - is never compressed, and
an inference is exactly a redundancy among facts (A-B, B-C make A-C cheap). Here each name gets a low-rank CODE of the store's name x name
record graph (PPMI, truncated SVD; no labels). 400 documents are held out: their records are removed, and so is any record anywhere pairing
two names of one held-out document - nothing in the store states any candidate pair. Candidates: names co-occurring in a held-out document,
both still in the store; gold = related in that document.

| pairs never stated together | base | bridges (common neighbours), top 5% | Adamic-Adar AUC / top 5% | CODE 128 AUC / top 5% | **CODE 512 AUC / top 5%** |
|---|---|---|---|---|---|
| with a chain A-B-C, split 0 | 0.094 | 0.196 | 0.619 / 0.243 | 0.684 / 0.291 | **0.726 / 0.365** |
| with a chain, split 1 | 0.086 | - | 0.610 / 0.240 | 0.707 / 0.294 | **0.746 / 0.323** |
| no chain, split 0 | 0.073 | 0.061 | 0.500 / 0.061 | 0.549 / 0.124 | **0.555 / 0.153** |
| no chain, split 1 | 0.063 | - | 0.500 / 0.043 | 0.552 / 0.115 | **0.538 / 0.118** |
| all pairs, split 0 | 0.081 | 0.131 | 0.555 / 0.177 | 0.612 / 0.221 | 0.618 / 0.281 |

VOLUME (same 15,103 candidate pairs, store + 30k distant documents, 625k more records): CODE 512 AUC 0.618 -> **0.698**, top 5% 0.281 ->
0.297, top 25% 0.141 -> 0.169; Adamic-Adar flat (0.555 -> 0.559). More reading improves the compressed code and not chaining. (At 128
dimensions with the larger candidate set the distant records diluted it: the code needs room - 512 dims, 32-bit, fits in 8 GB alone.)
Popularity (degree product) is below chance (AUC 0.39-0.46); weighting records by the relational-wording learner changes nothing.
**Positive, two splits: the compressed fact table infers.** Where a chain exists, the top 5% by code are right 0.32-0.37 - 3.8x base and
level with waiting for later text to confirm (0.36, which needs the pair stated) - with nothing in the store stating the pair; where no
chain exists, chaining is at chance by definition and the codes still find related pairs at about 2x base. Inference grows with the size
of the code (32 -> 512). Earlier label-free inference filters reached 0.13-0.15 (typed composability). Caveats: gold is 'related', not the
relation; Re-DocRED only (the 30k distant-document run needs less memory: running at 128 dimensions).

### GPU cg27a_rcode3: writing where the code and chaining agree (seed 4, 14 November, visits 2-5)

One change against cg27a_rcode2: the 100 held-out pairs ranked highest by the rank sum of the code (512 dims) and Adamic-Adar are written
(cg27a_rcode2: the 200 most similar codes). Visit 2 (345k records, 86,965 candidates never stated together, labelled 0.095), labelled share
of each rule's top 100 / 200: code 0.19 / 0.32, Adamic-Adar 0.41 / 0.415, **code + Adamic-Adar 0.66 / 0.60**; **the 100 written right
0.66** (cg27a_rcode2 at the same readout: 0.32). Recall from the trunk at visit 3 (one visit of teaching), test wording never taught:
open-ended taught right 0.22 / wrong 0.27 vs held back 0.00 / 0.05; rank among itself and 9 held-back answers (the corrected readout,
chance 0.10) taught 0.61 / 0.54 vs held back 0.18 / 0.10. (cg27a_rcode2 at the same point: open-ended 0.15 / 0.12.)
**Positive: where the code and chaining agree, two thirds of what is written is right (base 0.095), double the code alone, and it is taken
into the weights faster (twice the rows per inference).** False memories are now about a third of what is written, not two thirds.

Visits 4-5. The second batch (written at visit 4, 200,000 candidates scored, labelled 0.087) is right 0.64 (cg27a_rcode2's visit-4
batch: 0.255). Recall from the trunk in the test wording, taught vs held back, right / wrong:

| | written at visit 2: taught | held back | written at visit 4: taught | held back |
|---|---|---|---|---|
| open-ended exact, visit 4 | 0.50 / 0.56 | 0.06 / 0.00 | - | - |
| **open-ended exact, visit 5** | **0.52 / 0.54** | **0.12 / 0.00** | **0.33 / 0.23** | **0.03 / 0.00** |
| (cg27a_rcode2, visit 5) | 0.37 / 0.41 | 0.00 / 0.00 | 0.00 / 0.08 | 0.00 / 0.00 |

Rank among itself and 9 held-back answers (corrected readout, chance 0.10), all batches: visit 4 taught 0.88 / 0.88, held back 0.24 / 0.10;
visit 5 taught 0.82 / 0.66, held back 0.22 / 0.15 (n 36 / 20). Cost: held-out bits on the domains read, visits 1-5, 2.0263 / 2.0209 /
2.0053 / 1.9186 / 1.8533 against cg27a_rcode2's 2.0314 / 2.0265 / 2.0127 / 1.9263 / 1.8710 - none (the gap is already 0.005 at visit 1,
before any inference is written: run-to-run variation, not a gain).
**Positive, one seed: two thirds of what is written is right in both batches (0.66, 0.64), and recall is faster and higher than in
cg27a_rcode2 (half the inferences, twice the rows each).** The weights still take in right and wrong alike (0.52 / 0.54) - selectivity
lives in the store's choice, not in the trunk. Held-back RIGHT answers are ranked above chance at every visit (0.18 / 0.24 / 0.22, against
0.10 / 0.10 / 0.15 for held-back wrong ones; small n): either the trunk generalises a little from the taught inferences or right answers
are more familiar names - not yet separated.

**Second seed (cg27a_rcode3_s5, SEED 5, visits 1-4).** The store side is the same as seed 4 (same documents, same held-out hash: the same
86,965 candidates and the same 100 writes, right 0.66), so what replicates is the trunk. Written at visit 2, taught vs held back, right /
wrong: open-ended visit 3 0.28 / 0.38 vs 0.00 / 0.00 (seed 4: 0.22 / 0.27 vs 0.00 / 0.05); **visit 4 0.67 / 0.69 vs 0.16 / 0.00** (seed 4:
0.50 / 0.56 vs 0.06 / 0.00). Rank (corrected): visit 3 taught 0.52 / 0.42 vs held back 0.19 / 0.00; visit 4 0.78 / 0.71 vs 0.19 / 0.10.
**Replicates: the written inferences reach the weights in an untaught wording, right and wrong alike, held back far lower.** Held-out
bits, visits 1-4: 2.0102 / 2.0248 / 2.0189 / **1.9571** (seed 4: 2.0263 / 2.0209 / 2.0053 / 1.9186) - the visit-4 gap is mostly
Gutenberg (2.161 vs 2.021; other domains +0.006 to +0.018), a domain whose cost already swings between runs (cg27a_rcode2 seed 4: 2.061
at visit 4, 2.110 at visit 5); without a same-seed control it is a watch item, not a measured cost. Held-back right at 0.16-0.19 fits
tools/gen_diag.py: a prior from reading (the held-back pairs' documents are kept out of the store, not out of the text the trunk reads).
Visit 5 (seed 5 vs seed 4): visit-2 batch taught 0.73 / 0.67 vs held back 0.16 / 0.00 (0.52 / 0.54 vs 0.12 / 0.00); **visit-4 batch, one visit
of teaching, 0.49 / 0.31 vs 0.03 / 0.05** (0.33 / 0.23 vs 0.03 / 0.00); rank taught 0.68 / 0.64 vs held back 0.14 / 0.15. Held-out bits
1.8905 (seed 4: 1.8533); the domain just read matches the earlier seed-5 build cg27a_replay_s5 (post-train PubMed 1.678 vs 1.679, FreeLaw 1.583
vs 1.585), the gap is in older domains (Gutenberg 2.160 vs 2.039, PubMed 1.725 vs 1.687, Wikipedia 2.064 vs 2.038) - seed or change cannot
be told apart without a same-seed run without the inferences.
Visit 6 (seed 5): a third batch written, right 0.55 (the readout's top 100 is 0.71, but pairs already written are skipped, so the
batch reaches further down the ranking). Visit-2 batch kept: taught 0.71 / 0.65 vs held back 0.16 / 0.00; **visit-4 batch after two visits
0.66 / 0.46 vs 0.03 / 0.15**; rank taught 0.85 / 0.58 vs held back 0.17 / 0.25 - right inferences now held more firmly than wrong ones
(consistent with tools/gen_diag.py: the reading prior already favours right answers and adds to the teaching). Held-out bits 1.9166
(cg27a_rcode2 seed 4 at visit 6: 1.8986; Gutenberg 2.178 vs 2.101).

### Does more reading flag a wrong inference? (tools/fact_compress.py FC_SAVE / FC_LOAD, 14 November; Re-DocRED gold, two splits)

The store writes its top 200 by code + Adamic-Adar after reading 5,000 distant documents (right 0.37 / 0.405), reads 30,000 and re-ranks
the same 200 (every candidate pair is still kept out of the store). Right vs wrong:

| | split 0 | split 1 |
|---|---|---|
| still in the top 200 after more reading: right / wrong | 0.78 / 0.60 | 0.73 / 0.61 |
| AUC of the stage-1 rank alone (within the 200) | 0.685 | 0.646 |
| AUC of the rank CHANGE | 0.575 | 0.530 |
| change within the lower / upper half of the stage-1 rank | 0.522 / 0.602 | 0.505 / 0.608 |
| a newly read document names the pair: right / wrong | 0.99 / 0.96 | 0.99 / 0.94 |

**Weak.** Most of the drop-out is the weakly ranked writes falling away, which the first ranking already knew; among the confident half,
falling rank flags wrong writes at about 0.6. Co-occurrence in new reading says nothing (nearly every pair turns up). Re-ranking is a
mild check on confident writes, not a retraction trigger.

### Codes from compression, read as tokens (tools/compose_synth.py CS_PCODE 2, 14 November; synthetic, 3 x 192, seed 0)

The person and city codes are no longer handed out: they are k-means clusters (40) of a 16-dim SVD of the PPMI of the fact graph (G + G^2;
person-city and city-country links, plus CS_NOISE random person-city links per person), appended as a token after each name. No
citizenship is ever stated for anyone (CS_IMPL 0). Citizenship of held-out people (chance 0.10) after 6,000 steps, and of a new person in
an old city taught 'lives in' only (300 steps):

| noise | country purity of the people's clusters | held-out citizenship (steps 1k-6k) | new person, 300 steps |
|---|---|---|---|
| 0 | 1.00 | **0.75** (0.83 - 0.69) | **0.66** |
| 1.0 | 0.52 | **0.32** (0.39 - 0.24) | **0.32** |
| 2.0 | 0.31 | 0.16 (0.12 - 0.20) | 0.14 |

**Positive: codes made by compressing the fact table, not handed out, are read by the trunk in one hop - it infers a composite it was
never told, in proportion to how clean the code is (about 60-75% of the code's own purity).** Noisy codes carry correspondingly less;
the trunk does not clean them up. Real-text codes (Re-DocRED) are the next test (tools/codetok_diag.py).

### GPU cg27a_rcode4: write 200 by code + resource allocation, 16 rows a step (seed 4, 14 November, visits 1-2)

Visit 2 (the same store and candidates as cg27a_rcode3), labelled share of the top 100 / 200 / 400 of all 86,965 candidates: code 0.19 /
0.32 / 0.345; Adamic-Adar 0.41 / 0.415 / 0.367; code + Adamic-Adar 0.66 / 0.60 / 0.565; **code + resource allocation 0.59 / 0.59 / 0.578**;
the 200 written right 0.590 (118 right inferences, against cg27a_rcode3's 66). **The offline edge of resource allocation does not appear
at GPU scale** (level at 200, +0.013 at 400, -0.07 at 100): the gain is from writing more, at about the precision code + Adamic-Adar would
give for 200. Held-out bits visits 1 / 2: 2.0237 / 2.0271 (cg27a_rcode3 seed 4: 2.0263 / 2.0209).
Visit 3 (one visit of teaching): open-ended taught right / wrong 0.14 / 0.10 vs held back 0.00 / 0.02 (cg27a_rcode3: 0.22 / 0.27 vs
0.00 / 0.05); rank taught 0.43 / 0.37 vs held back 0.11 / 0.11 (cg27a_rcode3: 0.61 / 0.54 vs 0.18 / 0.10); dose loss 1.234 -> 0.354
(cg27a_rcode3: 1.059 -> 0.253). Rows per taught inference are the same as cg27a_rcode3 (about 47), yet each is learned more slowly with
twice as many in play - so the number of right inferences recalled is about the same (rank: 0.43 x 81 = 35 vs 0.61 x 49 = 30). Held-out
bits visit 3: 2.0110 (cg27a_rcode3: 2.0053) - no cost from the 16 rows.
Visit 4: the visit-2 batch, after two visits of teaching, open-ended taught right / wrong 0.43 / 0.36 vs held back 0.01 / 0.00 (cg27a_rcode3:
0.50 / 0.56); **rank taught 0.89 / 0.87 vs held back 0.11 / 0.29** (cg27a_rcode3: 0.88 / 0.88 vs 0.24 / 0.10) - caught up per inference with
1.65x as many taught (rank: about 72 right inferences recalled vs 43). The second batch (visit 4) right 0.59 again (code + RA top 200 0.63,
code + AA 0.635). Held-out bits visit 4: 1.9246 (cg27a_rcode3: 1.9186). The dose loss is a mean over its rows, so 16 rows at weight 1
pushed each row half as hard as 8 did: learning was slower (visit 3), not capped (visit 4). inf_rti_rows_ref (added) restores the push.

### Is relation orthogonal to the trunk's main structure? NEGATIVE here (tools/relation_ortho.py, 14 November; stand-in trunk 3 x 192)

The stand-in trunk reads the held-out Re-DocRED relation records (names masked; 600 queries x 30 candidates; same relation 0.084); its state
averaged over the wording between the names, per layer. Relation identity (nearest candidate of the same relation / AUC) with the top k
principal directions of ORDINARY TEXT removed, and the share of the between-relation variance (20 commonest relations) in those directions:

| layer | k = 0 | 5 | 10 | 20 | 50 | 100 | between-relation vs all variance in the top 5 / 10 / 20 directions |
|---|---|---|---|---|---|---|---|
| 1 | 0.133 / 0.539 | 0.155 / 0.523 | 0.160 / 0.532 | 0.150 / 0.533 | 0.157 / 0.572 | 0.128 / 0.551 | 0.79 vs 0.69 / 0.86 vs 0.79 / 0.92 vs 0.88 |
| 2 | 0.140 / 0.531 | 0.147 / 0.528 | 0.143 / 0.516 | 0.143 / 0.516 | 0.140 / 0.518 | 0.135 / 0.519 | 0.40 vs 0.30 / 0.60 vs 0.46 / 0.68 vs 0.58 |
| 3 | 0.142 / 0.519 | 0.148 / 0.519 | 0.135 / 0.519 | 0.140 / 0.520 | 0.142 / 0.517 | 0.115 / 0.515 | 0.58 vs 0.40 / 0.70 vs 0.57 / 0.81 vs 0.71 |

**Negative: removing the main structure does not reveal a relation signal** (AUC 0.52-0.57 at every k; removing the records' own top
directions makes it worse), and **what differs between relations lies MORE in the top directions than the average variance does**
(ratio 1.05-1.44), not orthogonal to them. In this small trunk relations are barely represented at all over the wording (AUC ~0.53, frame
words alone 0.63). Caveat: 3 x 192, wiki-trained; a 171M trunk may hold relations differently - the same measurement as a GPU readout
would settle it for the real model.

### Is there a tipping point? Recycling inferences as premises (tools/fact_compress.py FC_RECYCLE, 14 November; two splits)

Re-DocRED gold + 30k distant documents, codes 256. The first batch (top 200 by code + Adamic-Adar) is right 0.395 / 0.385. 200 inferences
from the top 2,000 are added to the store graph as if read, at a SET precision; codes and chaining are recomputed; the NEXT 200 written
(the added pairs excluded) against the same next batch without recycling, split 0 / split 1:

| recycled at | next batch with recycling | without | change |
|---|---|---|---|
| the actual first batch (0.40 / 0.39) | 0.200 / 0.185 | 0.195 / 0.170 | +0.005 / +0.015 |
| 0.6 | 0.315 / 0.270 | 0.305 / 0.270 | +0.010 / 0.000 |
| 0.7 | 0.300 / 0.290 | 0.305 / 0.280 | -0.005 / +0.010 |
| 0.8 | 0.305 / 0.270 | 0.305 / 0.275 | 0.000 / -0.005 |
| 0.9 | 0.270 / 0.240 | 0.270 / 0.245 | 0.000 / -0.005 |
| 1.0 | 0.280 / 0.245 | 0.280 / 0.245 | 0.000 / 0.000 |

**No tipping point through the store: even perfectly right recycled inferences do not change the next batch (within +-0.015).** 200
edges among some 600,000 read name pairs do not move the codes or the chaining scores; next-batch pairs touching a recycled name are right
more often (0.45-0.49 at precision 1.0 vs 0.07 for the rest), but they were ranked there anyway. What limits the next batch is DEPLETION
(the first batch takes the best pairs: 0.39 -> 0.17-0.31) and what the store has READ, not its premises. A self-reinforcing loop would
have to run through reading (more and better records), not through recycling conclusions.

### Contradiction by surprise? Confirmation by prediction (tools/surprise_trigger.py, 14 November; stand-in trunk, two seeds)

The stand-in trunk is taught 120 RIGHT claims (gold relation phrase, gold object), 120 WRONG ones (same phrase, a random object of the
right type) and leaves 120 untaught, in two arms: the gold phrase (typed) or 'is connected to'. Later reading: each pair's real Re-DocRED
sentence (A before C, the TRUE object in the document's words). Change in the trunk's surprise at the true object after teaching (bits),
seed 0 / seed 1:

| arm | right | wrong | control | AUC wrong vs right | flagged above the control's 90th percentile: wrong / right |
|---|---|---|---|---|---|
| gold phrase (typed) | -7.42 / -9.07 | +1.04 / +0.52 | +0.72 / +1.24 | **0.797 / 0.816** | 0.13 / 0.01, 0.09 / 0.02 |
| 'is connected to' | -7.30 / -8.57 | +0.83 / +0.29 | +0.75 / +1.26 | **0.805 / 0.800** | 0.12 / 0.01, 0.08 / 0.01 |

**Positive, but as CONFIRMATION, not contradiction: a right taught claim makes the true object far cheaper when later text names it
after A (-7 to -9 bits); a wrong one leaves it where an untaught pair is (+0.3 to +1.0 vs +0.7 to +1.3).** Typing does not matter here:
the trunk learns 'C follows A' whatever phrase it was taught. The trigger it suggests: a written inference whose object the trunk does not
predict when later text names A with a name of the object's class is unconfirmed. It applies where wrong claims have a true competitor
(functional relations - the subset compression can type, tools/compress_typing.py); it cannot catch associations that later text keeps
pairing (the GPU's wrong writes, tools/trigger_retest.py), which the trunk would 'confirm' too.

### GPU cg27a_rasym: the asymmetry write filter (seed 4, 14 November, visits 1-6)

cg27a_rcode5 + inf_rcode_asym 1.0 (write only pairs with |log degree ratio| > 1). Visit 2 (the same store and candidates as cg27a_rcode5),
labelled share of the top 100 / 200 / 400: code + RA 0.59 / 0.59 / 0.578 overall vs **0.67 / 0.68 / 0.642 among asymmetric pairs**; code +
AA 0.66 / 0.60 / 0.565 overall vs **0.74 / 0.74 / 0.662 among asymmetric pairs**. **The 200 written right 0.680 (cg27a_rcode5: 0.590):
the filter replicates at GPU scale, +0.09.** Code + Adamic-Adar would have done better still under the filter (0.74). Held-out bits
visits 1 / 2: 2.0145 / 2.0206 (cg27a_rcode5 2.0241 / 2.0322).
Visit 3: rank taught 0.56 / 0.66 vs held back 0.08 / 0.18; open-ended 0.09 / 0.07 (cg27a_rcode5 0.14 / 0.10), dose loss to 0.350 - slower
in the first visit. **Visit 4: the visit-2 batch open-ended 0.73 / 0.74 vs held back 0.03 / 0.03 (cg27a_rcode5: 0.54 / 0.49); rank 0.94 /
0.96 vs 0.12 / 0.18. The second batch (visit 4) right 0.66 (cg27a_rcode5: 0.59).** Held-out bits 2.0167 / 1.9183 at visits 3 / 4
(cg27a_rcode5 2.0137 / 1.9258). More right inferences written AND recalled better after two visits, at no cost.
Visits 5-6 against cg27a_rcode5 at the same visits:

| | cg27a_rasym | cg27a_rcode5 |
|---|---|---|
| batches written right (visits 2 / 4 / 6) | **0.680 / 0.660 / 0.670** | 0.590 / 0.590 / 0.570 |
| visit 6, open-ended, taught right / wrong vs held back | **0.68 / 0.69** vs 0.03 / 0.00 | 0.40 / 0.39 vs 0.02 / 0.00 |
| visit 6, by batch (visit 2; visit 4), taught right | **0.74; 0.62** | 0.44; 0.37 |
| visit 6, rank, taught right / wrong vs held back | 0.95 / 0.99 vs 0.21 / 0.05 | 0.85 / 0.77 vs 0.14 / 0.14 |
| visit 6, taught facts answered right (false memory probe c) | 0.52 | 0.36 |
| held-out bits visits 4 / 5 / 6 | 1.9183 / 1.8624 / 1.8840 | 1.9258 / 1.8645 / 1.8980 |

**Positive through visit 6: the precision gain holds on every batch (+0.08 to +0.10, the third batch not falling), and recall from the
weights is about 1.7 times cg27a_rcode5's - about 91 right inferences recalled per batch against 47.** Wrong writes are recalled as
strongly (0.69), so false memories also rise (about 45 per batch against 32); the right-to-wrong ratio among recalled inferences improves
from 1.5 to 2.0. Held-back controls stay at 0.00-0.03. Bits slightly better at every visit (within seed noise). The candidate table again
favours code + AA under the filter: top 200 0.740 / 0.700 / 0.730 against code + RA 0.680 / 0.670 / 0.700 (visits 2 / 4 / 6).

### GPU cg27a_rcode5: the dose-fixed write path through pass 1 (seed 4, 14 November, visits 1-10)

cg27a_rcode4 + inf_rti_rows_ref 8 (the dose loss weighted by rows / 8). Four batches of 200 written (visits 2 / 4 / 6 / 8): right 0.59 /
0.59 / 0.57 / **0.425** - the best pairs run out by the fourth. Recall from the weights in the test wording (never taught), by batch,
taught vs held back, right / wrong:

| visit | batch of visit 2 | batch of visit 4 | batch of visit 6 | batch of visit 8 | rank, taught vs held back (right / wrong) |
|---|---|---|---|---|---|
| 4 | 0.54 / 0.49 vs 0.05 / 0.07 | - | - | - | 0.94 / 0.93 vs 0.11 / 0.25 |
| 6 | 0.44 / 0.40 vs 0.04 / 0.00 | 0.37 / 0.38 vs 0.00 / 0.00 | - | - | 0.85 / 0.77 vs 0.14 / 0.14 |
| 8 | 0.57 / 0.55 vs 0.01 / 0.02 | 0.56 / 0.52 vs 0.00 / 0.00 | 0.42 / 0.34 vs 0.00 / 0.00 | - | 0.86 / 0.76 vs 0.21 / 0.13 |
| 9 | 0.51 / 0.47 vs 0.01 / 0.05 | 0.47 / 0.40 vs 0.02 / 0.00 | 0.38 / 0.29 vs 0.01 / 0.00 | 0.21 / 0.21 vs 0.02 / 0.00 | 0.80 / 0.65 vs 0.19 / 0.10 |

**Every batch is kept through pass 1 (the first at about 0.5 open-ended seven visits after it was written), held back stays at 0.00-0.05,
and from visit 6 right inferences are held more firmly than wrong ones (rank 0.80-0.86 vs 0.65-0.77).** Held-out bits visits 1-9:
2.0241 / 2.0322 / 2.0137 / 1.9258 / 1.8645 / 1.8980 / 1.8720 / 1.8076 / 1.7635 (cg27a_rcode2 to visit 6: ... 1.8710 / 1.8986) - no cost
visible. A readout error late in the run ([NAME MEMORY READOUT] stale unavailable, ValueError, 4 times; caught, the run continued; fixed
in cg27a_rasym).
Visit 10: batches 0.42 / 0.36, 0.42 / 0.38, 0.31 / 0.25, 0.28 / 0.26 (taught right / wrong) vs held back 0.00-0.04; rank 0.79 / 0.66 vs
0.22 / 0.10. **End of pass 1 (weights only): forgetting +0.037, final BPC 1.779** - against cg27a_replay seed 4 +0.026 / 1.767 (no
real-text replay, no written inferences) and the real-text replay builds before the Wikipedia-slot fix +0.077-0.088 / 1.807-1.818. Per
domain: Gutenberg +0.068, StackExchange +0.090, USPTO +0.094, PubMed +0.044, FreeLaw +0.041; Wikipedia -0.034, ArXiv -0.017, EuroParl 0.000.
The whole write path costs about +0.011 forgetting / +0.012 BPC against the cleanest seed-4 run, most of the real-text replay cost gone.

### An exclusivity trigger, and asymmetry as a write filter (tools/exclusive_trigger.py, 14 November; the GPU write path, distant set)

400 written by the package's own path at 20,375 distant documents (right 0.578), judged on documents 20,375-40,749. The subject S is the
name with the lower store degree; the object O's CLASS is a k-means class of the names' compressed codes (no labels). Trigger: in later
reading S is tied to a competitor of O's class at least as often as to O, or O's share of S's later partners of that class is below 0.25.

| writes | n (wrong) | competitor >= object: flagged wrong, LR | object's share < 0.25: flagged wrong, LR |
|---|---|---|---|
| all | 400 (0.42) | 0.51, 1.4 | **0.58, 1.9** |
| subject read again with that class (2+) | 346 (0.40) | 0.51, 1.6 | **0.58, 2.0** (AUC of the share 0.636) |
| asymmetry > 1 (subject clearly more specific) | **200 (0.28)** | 0.18, 0.6 | 0.12, 0.3 |

**Weak positive for the trigger: AUC 0.64 and a likelihood ratio of about 2 (the re-mention trigger: 0.52, about 1) - not enough to
retract on at a 0.4 wrong base (the flagged set would be about 58% wrong).** Restricting to functional-like writes REVERSES it.
**Unexpected and more useful: asymmetry selects.** Writes whose subject is clearly more specific than the object are wrong 0.28 against
0.42 overall (200 of 400): a label-free WRITE filter, not a retraction trigger. One write set, one seed; to be checked on more writes and
on Re-DocRED gold before it is built.
Checked on Re-DocRED gold (tools/fact_compress.py FC_ASYM, + 30k distant documents, two splits): the top 200 by code + Adamic-Adar are
right 0.395 / 0.395; among them, pairs with |log degree ratio| > 1 right 0.672 / 0.667 (dropped 0.256 / 0.261), > 2 right 0.750 / 0.783;
**writing the top 200 among pairs with |log degree ratio| > 1: right 0.585 / 0.525 against 0.395** (top 400: 0.515 / 0.453 against 0.30).
**Positive, two splits and the GPU path agree: asymmetry is a label-free write filter worth about +0.13 to +0.19 precision.**

### Typing from the literature: typed predicates, entailment back-off, relation discovery (tools/typed_research.py, 14 November; two splits)

Re-DocRED gold + 30k distant documents. T1 types RECORDS READ in held-out documents (the pair stated, in its own wording); T2 also types
pairs NEVER STATED in the store (an inference). Probe as tools/compress_typing.py; 'coverage' = share typed at 80% precision. Split 0 / 1:

| typing a READ record by (commonest relation 0.32 / 0.31) | exact: typed, right | exact coverage | family coverage |
|---|---|---|---|
| **the wording alone** | 0.85 / 0.83, **0.518 / 0.516** | **0.149 / 0.204** | **0.445 / 0.468** |
| latent wording cluster (universal schema) | 1.00, 0.392 / 0.382 | 0.036 / 0.000 | 0.221 / 0.203 |
| compressed class pair alone | 0.41 / 0.38, 0.421 / 0.446 | 0.001 / 0.016 | 0.111 / 0.147 |
| TYPED PREDICATE (class A, wording, class B) | 0.17 / 0.14, 0.581 / 0.704 | 0.000 / 0.105 | 0.125 / 0.122 |
| typed predicate + ENTAILMENT BACK-OFF (directional inclusion) | 0.18 / 0.16, 0.548 / 0.682 | 0.000 / 0.095 | 0.121 / 0.126 |
| T2 encoder (relation discovery by reconstructing arguments) | 1.00, 0.386 / - | 0.000 / - | 0.000 / - |

| typing a NEVER-STATED pair (commonest relation 0.476) | exact right | coverage |
|---|---|---|
| class pair alone | 0.445 | 0.002 |
| T2 decoder (the latent relation that best links A and C) | 0.475 | 0.000 |

(Split 1's T2 was killed for memory; one split.)
**Negative for the literature methods at this volume: typed predicates are precise but sparse (only 14-17% of held-out records have one
seen 3+ times), entailment back-off rescues 1%, and joint relation discovery types worse than the raw wording and no better than the
commonest relation for unstated pairs.** **What does type: the wording a pair is READ in** - 0.52 exact, a sixth to a fifth of records at
80% precision, nearly half by family. Typing comes from a sentence that states the pair, not from inference; a typed inference has to be
confirmed by such a sentence (predict, then type on confirmation, as cg27a_rtc confirmed pairs).

### A three-times longer-trained stand-in trunk (18k steps; 14 November, night)

The stand-in trunk (3 x 192, wiki + Re-DocRED dev) trained 18,000 steps instead of 6,000: 1.70 bits per byte against 1.92. The same tests:

| test | 6k trunk | 18k trunk |
|---|---|---|
| relation identity over the wording, layers 1 / 2 / 3 (nearest same / AUC; chance 0.084) | 0.133 / 0.539, 0.140 / 0.531, 0.142 / 0.519 | 0.137 / 0.539, 0.132 / 0.533, 0.128 / 0.524 |
| teaching, rank after teaching: dose x4 / x16 / read too | 0.44 / 0.74 / 0.53 | **0.61 / 0.82 / 0.64** |
| the same after 300 steps of other text | 0.37 / 0.47 / 0.40 | 0.39 / 0.51 / 0.40 |
| confirmation by prediction, AUC typed / untyped (seed 0) | 0.797 / 0.805 | 0.792 / 0.762 |

**CORRECTION (15 November, 00:10): the 18k trunk is OVERFIT, not better trained.** Its 1.70 is on its own training text (wiki.bin is
10.8 MB; 18k steps x 32 x 128 bytes is about 7 passes over it). On Re-DocRED documents it never read it is WORSE: 2.503 / 2.493 bits per
byte against the 6k trunk's 2.246 / 2.235 (tools/keyed_units.py, the two document sets). So the table compares a fitted trunk with an
overfit one: faster learning of taught facts with the same retention, no relation structure and no better confirmation is what
memorisation would give, and it says nothing about a trunk trained longer on FRESH text. The question 'does more training on new text
help' stays open; a GPU run with fresh data per pass (pass_fresh_data 1) is the test that can answer it.

### Sparse memory finetuning inside the trunk (tools/sparse_mem.py, 14-15 November; 6k stand-in trunk, Lin et al. 2025 at small scale)

Block 2's MLP replaced by a product-key memory (16,384 slots, top 16 per byte), 2,500 steps on wiki: held-out wiki 2.055 bits per byte
(2.272 with the memory switched off: the model uses it; 11,812 slots used by background text). (A first version that added the memory
BESIDE the MLP left it unused - equal bits with it off - and was stopped.) Then 120 never-read Re-DocRED facts in batches of facts only
(no replay, the paper's stress test), set A then set B, 150 steps each; recall in the test wording (rank among 10) and bits added on
held-out wiki / dev text:

| method | lr | A after A | B after B | A after B | forgetting after A -> after B (wiki) |
|---|---|---|---|---|---|
| full | 3e-5 / 1e-4 | 0.70 / 0.95 | 0.77 / 0.95 | 0.67 / 0.78 | +0.46 -> +0.76 / **+1.17 -> +1.59** |
| LoRA | 3e-4 / 1e-3 | 0.50 / 0.87 | 0.58 / 0.92 | 0.42 / 0.67 | +0.54 -> +0.86 / +1.43 -> +1.97 |
| MLP_OUT (the GPU build's choice) | 3e-5 / 1e-4 | 0.35 / 0.57 | 0.40 / 0.62 | 0.37 / 0.50 | +0.13 -> +0.22 / +0.37 -> +0.61 |
| all memory values | 1e-3 / 1e-2 | 0.58 / **0.93** | 0.58 / **0.98** | 0.55 / **0.90** | **+0.07 -> +0.17** / +1.01 -> +1.60 |
| SPARSE memory, TF-IDF top 256 | 1e-3 / 1e-2 | 0.30 / 0.30 | 0.33 / 0.33 | 0.30 / 0.32 | +0.000 -> +0.001 / +0.007 -> +0.020 |
| SPARSE memory, top 1,024 | 1e-2 | 0.47 | 0.58 | 0.43 | +0.035 -> +0.072 |

**As the paper: dense finetuning on facts alone is catastrophic (+1.2 to +2.0 bits), and memory-only updates forget far less at the same
learning rate. But sparse memory finetuning as built here barely LEARNS (rank 0.30-0.58 against 0.93 for all memory values): the TF-IDF
selection over the whole batch picks slots used by the rare frame words, not where the answer is predicted.** Second round queued: slots
chosen on the answer bytes only, and more of them (1,024 / 4,096). Note the key memory outside the trunk (next section) learns the same
facts at rank 0.93-1.00 with no trunk forgetting by construction.

**Round 2 (15 November): slots chosen by TF-IDF counted only where the ANSWER bytes are predicted:**

| sparse memory | lr | A after A | B after B | A after B | forgetting after A -> after B (wiki) |
|---|---|---|---|---|---|
| answer bytes, top 1,024 | 1e-2 | **0.78** | **0.77** | **0.72** | **+0.07 -> +0.15** |
| answer bytes, top 1,024 | 3e-2 | **0.87** | **0.92** | **0.87** | +0.26 -> +0.44 |
| answer bytes, top 4,096 | 1e-2 / 3e-3 | 0.83 / 0.57 | 0.92 / 0.68 | 0.75 / 0.53 | +0.36 -> +0.67 / +0.07 -> +0.18 |
| whole batch, top 4,096 | 1e-2 | 0.82 | 0.88 | 0.78 | +0.31 -> +0.56 |

**Positive: choosing slots where the answer is predicted fixes the learning, and sparse memory then beats every dense method at equal
forgetting** - at about +0.15 bits it learns 0.78 (all memory values 0.58, mlp_out 0.35 at +0.22; full fine-tuning needs +0.76 for 0.70),
and at 0.87-0.92 it costs +0.44 bits against full fine-tuning's +1.59 for 0.95 (3.6 times less). The paper's ordering holds at this
scale; the absolute forgetting is still large because the batches are facts only, with no replay. Open-ended exact recall stays near 0
for every in-trunk method (multi-word real names, byte by byte), while the key memory decodes them at 0.80-0.98.

**Round 3, the GPU's setting (15 November): facts MIXED with ordinary text, 8 fact rows + 24 wiki rows a step, 400 steps a set (about 53
rows per fact):**

| method | lr | A after A | B after B | A after B | forgetting after A -> after B (wiki / dev) |
|---|---|---|---|---|---|
| full | 3e-5 | 0.50 | 0.52 | 0.43 | +0.012 -> +0.018 / +0.005 |
| MLP_OUT (the GPU build's choice) | 3e-5 / 1e-4 | 0.32 / 0.37 | 0.35 / 0.42 | 0.32 / 0.37 | +0.009 / +0.020 |
| all memory values | 1e-3 | 0.50 | 0.50 | 0.47 | -0.001 |
| **SPARSE memory, answer slots, top 1,024** | 1e-2 | **0.70** | **0.73** | **0.62** | +0.019 -> +0.029 / +0.033 |
| **SPARSE memory, answer slots, top 1,024** | 3e-2 | **0.88** | **0.87** | **0.68** | +0.064 -> +0.079 / +0.098 |
| SPARSE memory, answer slots, top 1,024 | 1e-1 | 0.88 | 0.88 | 0.67 | +0.21 -> +0.22 |

**Positive in the GPU's setting: with text mixed in, every method forgets little, and sparse memory learns far more for it - 0.70 at
+0.03 bits where full fine-tuning reaches 0.50 at +0.02 and the GPU build's mlp_out only 0.37 at +0.02; 0.88 at +0.08.** A candidate
for a GPU build: a product-key memory layer in the trunk, question rows writing only the answer slots (TF-IDF on the answer bytes), lr
about 1e-2.
Seed 1 replicates (different facts and order): full 0.47 / 0.52 / 0.38 (+0.005), mlp_out 0.40 / 0.42 / 0.33 (+0.009), sparse answer
slots lr 1e-2 **0.72 / 0.70 / 0.58** (+0.023), lr 3e-2 **0.88 / 0.87 / 0.60** (+0.071).
**Refresh (SM_REPLAY 0.25: while B is taught, a quarter of the fact rows re-teach A; lr 1e-2, seeds 0 / 1):** A after B **0.70 / 0.68**
(without refresh 0.62 / 0.58), B after B 0.70 / 0.68 (0.73 / 0.70), forgetting +0.030 / +0.025 (+0.029 / +0.023). **Refresh holds the
old facts at their learned level for a small cost to the new ones and none in text** - as in the keyed memory. The GPU build already
re-asks earlier written inferences across visits, which is this refresh.

**Exact retraction inside the trunk? (SM_RETRACT, mixed setting, lr 3e-2):** after B, 30 facts of A retracted by restoring the answer slots
their questions use most to their pre-teaching values. Rank (before teaching 0.30):

| slots restored | retracted | rest of A | B | wiki bits added |
|---|---|---|---|---|
| top 1,024 | 0.60 -> **0.30** | 0.70 -> 0.63 | 0.87 -> **0.65** | +0.079 -> +0.068 |
| top 256 | 0.60 -> 0.50 | 0.70 -> 0.70 | 0.87 -> 0.82 | +0.079 -> +0.077 |

**Not clean: inside a byte-level trunk the answer slots are SHARED between facts (the bytes of one answer's name are predicted by slots
other answers use), so restoring enough slots to remove the facts takes bystanders with them (B -0.22).** The key memory, addressed by
(subject name, wording), retracts exactly with bystanders held (0.03-0.04). For the design: facts and inferences that may need retracting
belong in the KEYED memory; the in-trunk sparse memory is for absorbing what is settled, at low forgetting.

### A key-to-key sparse memory addressed by name + JEPA wording key (tools/jepa_mem.py, 14 November, night; seed 0)

The author's design: a sparse memory (Lin et al. 2025) whose address is (the subject's NAME KEY, a JEPA WORDING KEY of the name-masked
question prefix) and whose VALUE is a name key, so a read can be the next read's query; bytes only at the end (nearest known name).
Frozen stand-in trunk; product keys 64 x 64 = 4,096 slots, top 8 per read; wording head trained only on the system's seven frames (the
test wording never seen); writing changes only the slots a fact reaches (the trunk is never touched, so no text forgetting by
construction). Lifelong order in one memory: A (60, = sparse_mem's) -> B (60) -> M (40 subjects x 2 relations) -> 60 chains (both hops
taught, never the chain). Recall in the test wording, open (nearest of 2,987 known names) / rank (among 10 of the type), 6k / 18k trunk:

| | K-JEPA | K-RAW (no JEPA training) | K-NAME (no wording half) |
|---|---|---|---|
| A after A | 0.88 / 0.93, **0.98 / 1.00** | 0.93 / 1.00, 0.97 / 1.00 | 0.57 / 0.73, 0.55 / 0.63 |
| B after B | 0.82 / 0.95, 0.90 / 0.98 | 0.88 / 0.95, 0.90 / 1.00 | 0.23 / 0.37, 0.25 / 0.37 |
| M (two relations per subject) | 0.65 / 0.82, **0.91 / 1.00** | 0.94 / 0.99, 0.85 / 0.96 | 0.09 / 0.21, 0.06 / 0.16 |
| chain hops | 0.75 / 0.90, 0.89 / 0.97 | 0.86 / 0.97, 0.82 / 0.98 | 0.06 / 0.17, 0.07 / 0.15 |
| **TWO-HOP, the chain never taught: decoded** | 0.58 / 0.72, **0.78 / 0.87** | 0.73 / 0.88, 0.65 / 0.82 | 0.02 / 0.10, 0.02 / 0.03 |
| two-hop, latent (no decoding between hops) | 0.27 / 0.47, 0.42 / 0.62 | 0.32 / 0.53, 0.25 / 0.52 | 0.00 / 0.15, 0.02 / 0.12 |
| at the end (after 340 more facts): A / M rank | 0.77 / 0.81, 0.93 / 0.99 | 0.88 / 0.96, 0.92 / 0.95 | - |

Wording head: the test wording's relation found by the training frames' keys 1.00 (raw state 0.81 / 0.86); a PARAPHRASED relation phrase
('passed away in' for 'died in') only 0.38 for both.
**Positive: a sparse key memory learns real facts in an unseen wording almost perfectly, keeps most through 340 later facts, separates
two relations of one subject only with the wording half, and COMPOSES: two-hop answers for chains never taught 0.72-0.88 (rank) with the
first hop decoded to a known name.** The JEPA head helps on the better trunk (M 1.00, two-hop 0.87) but not on the 6k one; decoding
between hops beats latent chaining (the decode cleans the key). The gap: paraphrased relation phrases (0.38) - the wording key has to
learn synonymy from text, not from our frames. Seeds, abstention, retraction and capacity queued.

**Seeds and new readouts (15 November, 00:07; seeds 0 / 1 on both trunks, and a 128 x 128 = 16,384-slot memory; JEPA key / raw key,
rank in the test wording):**

| | 6k s0 | 6k s1 | 18k s0 | 18k s1 | 18k, 16k slots |
|---|---|---|---|---|---|
| M (two relations per subject) | 0.84 / 1.00 | 0.96 / 0.96 | 1.00 / 0.96 | 0.95 / 0.95 | 0.91 / 0.97 |
| two-hop, never taught, decoded | 0.73 / 0.87 | **0.97** / 0.85 | 0.87 / 0.83 | 0.83 / 0.75 | 0.88 / 0.80 |
| RETRACT 30 of A: retracted (open) | 0.53 -> 0.33 | 0.63 -> 0.33 | 0.60 -> 0.33 | 0.70 -> 0.40 | 0.67 -> 0.27 |
| RETRACT: largest bystander change (open, rest of A / B / M) | 0.04 | 0.02 | 0.00 | 0.00 | 0.04 |
| ABSTAIN AUC (best cosine, taught vs never-taught subjects) | 0.61 / 0.70 | 0.72 / 0.65 | 0.62 / 0.68 | 0.66 / 0.60 | 0.72 / 0.62 |
| CAPACITY: set A after 600 more facts (about 980 in all) | 0.68 / 0.85 | 0.70 / 0.75 | 0.80 / 0.75 | 0.77 / 0.53 | **0.87** / 0.72 |
| paraphrased relation phrase, set A | 0.43 / 0.70 | 0.45 / 0.55 | 0.53 / 0.63 | 0.60 / 0.62 | 0.62 / 0.55 |

**Replicated on every seed and trunk: facts learned in an unseen wording (rank 0.93-1.00), two relations of one subject separated by the
wording half (the name-only memory 0.11-0.21), and composition for chains never taught (0.73-0.97 decoded). Retraction is CLEAN - the
retracted facts lose about half their open recall and no bystander moves by more than 0.04 - but partial at 30 steps.** Abstention from
the memory's confidence is weak (0.60-0.72). A memory 4x larger keeps old facts better as more are written (A 0.87 vs 0.68-0.80). The JEPA
key and the raw key are level overall (the frame-trained head does not beat the raw state on its own); the open gap is paraphrase
(0.43-0.70), which the text-trained key (K-TEXT, queued) addresses.

Keyed units on the (overfit) 18k trunk, gate, on top of NAME: DATE +5.4 / +4.6, NUMBER +0.8 / +0.6, FACT +1.2 / -0.04, PLACE -0.2 / -0.2
(bits per 1,000 bytes, late, first / second set): the same ranking as on the 6k trunk (DATE and NUMBER replicate, FACT unreliable, PLACE
negative); the larger savings reflect the worse trunk (NAME alone 80-83 vs 30-33).

**Retraction by ERASING SLOTS (15 November; 6k trunk, seeds 0 / 1 / 2): the slots that a retracted fact's own training-frame queries
reach are set to zero - no gradient, no search.** Open recall in the test wording, retracted 30 of set A -> after; largest bystander
change (open, rest of A / B / M):

| key | retracted | largest bystander change | slots erased |
|---|---|---|---|
| JEPA key | 0.57 -> **0.00**, 0.67 -> **0.03**, 0.57 -> **0.00** | 0.04, 0.04, 0.03 | 295-324 |
| raw key | 0.73 -> 0.00, 0.50 -> 0.00, 0.50 -> 0.00 | 0.04, **0.12**, **0.10** | 762-858 |
| (gradient, -log(1 - p), 30-150 steps) | 0.53-0.73 -> 0.23-0.47 | 0.00-0.04 | - |
| (gradient, margin 0.1, 150 steps) | 0.57 -> 0.23, 0.67 -> 0.37 | 0.02 | - |

**Positive: erasing a fact's slots retracts it completely while bystanders hold - and the JEPA key does it with 2.5 times fewer slots and
no collateral damage, where the raw key spreads facts wider and costs bystanders up to 0.12.** Gradient unlearning in the memory is
clean but leaves a third to a half. Retraction is exact because every write has an address - the property a dense trunk lacks.
**With the memory nearly full (JM_RX_LATE: after about 2,400 facts; 6 runs - 4,096 slots without / with 25% refresh, 16,384 with refresh;
seeds 0 / 1):** the retracted facts' open recall falls to **0.00 in all six** (from 0.27-0.53); bystanders' rank moves by at most 0.06,
open by at most 0.07 in five runs and 0.16 once (the rest of A, 16k slots seed 0). **Erasure stays exact under load.**
(Note, 15 Nov: the RANK readout counted a tie as a hit, and an erased read (all zeros) ties every candidate, so ranks AFTER erasing
were inflated - e.g. 'rank 0.43' for erased facts. Fixed in tools/jepa_mem.py and inference.kmem (the answer must score strictly above
every other candidate); the OPEN numbers used above are unaffected.)
Third seed of the full test (6k s2): the same pattern (two-hop decoded 0.65 / 0.75, M 0.89 / 0.86, name only 0.15).

**Capacity, 2,000 more facts (15 November; 6k trunk, JEPA key, seeds 0 / 1; about 2,380 facts in all, no replay):** rank in the test
wording of the OLDEST and NEWEST batch of 200 at the end, and of set A (the first facts):

| slots | oldest batch | newest batch | set A | slots written |
|---|---|---|---|---|
| 4,096 (64 x 64) | 0.57 / 0.59 | 0.83 / 0.95 | 0.62 / 0.65 | 3,377 / 3,466 |
| 16,384 (128 x 128) | **0.76 / 0.65** | 0.94 / 0.91 | **0.77 / 0.65** | 7,008 / 7,307 |

**Graceful decay with age, no collapse; four times the slots keeps the oldest facts better on seed 0 (+0.19) and a little on seed 1
(+0.06).** At about 2,400 facts the small memory is nearly full (83% of slots written). Lin et al. use 1M slots; a lifelong store needs
the memory to scale with what it holds, and older facts need refresh (replay of memory writes, as sleep replays the store).

**Refresh by replay (JM_REPLAY: a share of every write batch re-writes earlier facts; 4,096 slots, seeds 0 / 1):**

| replay | oldest batch | newest batch | set A |
|---|---|---|---|
| none | 0.57 / 0.59 | 0.83 / 0.95 | 0.62 / 0.65 |
| 25% | **0.78 / 0.77** | 0.82 / 0.93 | **0.77 / 0.72** |
| 50% | **0.79 / 0.83** | 0.81 / 0.92 | **0.78 / 0.75** |

**Positive: re-writing earlier facts flattens the age curve - the oldest batch +0.18 to +0.24 at no cost to the newest - as much as four
times the slots did.** Refresh is the memory's version of sleep replay; a quarter of each write batch is enough.

**Abstention by teaching 'nothing here' (JM_NULL 8 / 32 rows a step: reads for inventory names never used as subjects pushed to zero;
6k, seeds 0 / 1):** abstention AUC by the read's norm 0.56-0.57, by the best cosine 0.59-0.62 (without: 0.52-0.55 / 0.61-0.72), while
recall falls (B at the end 0.42-0.53 against 0.88-0.92; two-hop 0.58-0.77 against 0.65-0.97). **Negative: unknown names reach the same
sub-keys as known ones, so pushing them to zero erases real facts.** Abstention should come from an explicit index of the subjects the
memory holds (the name gate / the store's name inventory: is this subject's key within a small distance of a written one?), not from the
memory's reads.

**K-TEXT: a wording key learned from REAL TEXT only (15 November; 20,000 name pairs stated in 2+ different wordings within a sentence,
50k distant documents, InfoNCE; the frames and the test wording never seen).** Relation of the test wording / of a PARAPHRASED relation
phrase recognised by the frames' keys: K-TEXT 0.90 / **0.52**, 0.90 / 0.38, 0.95 / 0.33 (18k s0, 18k s1, 6k s0) against the frame-trained
head 1.00 / 0.38, 1.00 / 0.33, 1.00 / 0.43. In the memory it is WORSE than the frame-trained key on everything (two-hop 0.55-0.72 vs
0.83-0.87; set A at the end 0.55-0.70 vs 0.92-0.93; paraphrased wording 0.40-0.58 vs 0.53-0.60) - it spreads facts over more slots
(2,364-2,935 vs 1,716-1,759) and interferes more. **Negative as built: co-occurrence of a name pair in two sentences is too weak and too
noisy a synonymy signal (most pairs are 'X, Y' lists and loose associations), and it costs the sharpness the memory needs.** A head trained
on text pairs AND frames (K-BOTH) keeps the test wording at 1.00 and reaches 0.48 / 0.33 on paraphrase (6k s0 / s1, frame head 0.43 / 0.29).
In the memory (rank): two-hop decoded 0.88 / 0.62 (frame-trained JEPA key 0.73 / 0.97), paraphrased wording, set A 0.65 / 0.43 (JEPA 0.43 /
0.45), set A after 600 more facts 0.72 / 0.42 (JEPA 0.68 / 0.70). **Mixed across the two seeds, no reliable gain: the paraphrase gap stays
open.** Synonymy needs a cleaner signal than name-pair co-occurrence - e.g. sentences the store already verified as stating the same fact,
or the trunk's own predictions (the wording after which the object is equally expected).

### GPU cg27a_kmg: the gated memory (seed 4, 15 November, visits 1-3)

Visit 2: taught facts WITH 0.42 / 0.50 vs trunk 0.10 / 0.48 (gate open 0.42 - the memory had written 60 subjects); HELD-BACK facts 0.00 /
0.37 = trunk 0.00 / 0.37 (gate open 0.07); never-read names: gate open 0.00. Visit 3: taught inferences RIGHT **0.57 / 0.97** vs 0.15 / 0.52
(gate open 1.00), WRONG 0.59 / 0.95 vs 0.15 / 0.62; facts 0.27 / 0.68 vs 0.25 / 0.63; HELD-BACK facts 0.00 / **0.50 = trunk 0.50** (gate open
0.27); never-read names 0.00. **The gate removes the cost (cg27a_kmi: held back 0.63 vs 0.80) and keeps the gains; the memory abstains on
never-read names.** COUNTER-WRITE (oracle, on a copy, support driven to zero): WRONG support +0.31 -> -0.20 bits but rank 0.92 -> 0.90 -
**the TRUNK has learned the wrong inferences too (0.62 alone): retraction in the memory cannot undo the trunk's copy.** Hence cg27a_kmc:
the trunk is taught an inference only once judged and kept. Bits 2.0342 / 2.0434 / 2.0319 (four identical-training runs at visit 1 span
2.0145-2.0342: run-to-run noise about 0.02).
Visit 4 (stopped there for cg27a_kmc): facts WITH **0.52 / 0.85** vs trunk 0.42 / 0.75 (gate open 0.78); held back 0.03 / 0.57 vs 0.07 /
0.60 (gate open 0.37 - a small leak where the subject has other facts); paraphrased 0.10 / 0.57 = trunk; never-read names 0.00; COUNTER-WRITE
(oracle, copy): WRONG support -0.97 -> -2.04 bits yet rank 0.55 -> 0.57 - the trunk alone ranks the taught wrong inferences 0.92. Bits
1.9398 (cg27a_kmi 1.9354, cg27a_kmem 1.9185: kmi and kmg agree, both 0.017-0.021 above kmem - borderline).

**Package smoke for cg27a_kmc** (tools/kmi_smoke.py KM_KMC=1): counter-write with a MARGIN (support below -1 nat per byte) took WRONG support
+5.7-5.8 -> -4.9 to -7.0 bits per byte (rank 0.86-1.00 -> 0.00-0.20), RIGHT and facts unchanged; consolidation masked 27 of 60 inference rows
(unjudged or flagged) from the trunk; the 7 retracted were counter-written out of the real memory.

### GPU cg27a_kmc: consolidation through the memory (seed 4, 15 November, visits 1-2 so far)

Visit 2 (the memory as cg27a_kmg; the 200 written inferences are held from the trunk until judged): taught facts WITH 0.40 / 0.50 vs trunk
0.10 / 0.47 (gate open 0.42; cg27a_kmg 0.42 / 0.50 vs 0.10 / 0.48); held back 0.00 / 0.33 = trunk (gate open 0.07); never-read names gate
open 0.00; inferences not yet in the memory (gate open 0.00 - they are written this visit and learned next). CONSOLIDATION: 200 awaiting
judgement, the trigger has nothing to judge yet (expected: it judges from visit 4). COUNTER-WRITE WITH THE MARGIN (oracle, on a copy): WRONG
rank 0.15 -> **0.00**, support -0.05 -> **-4.34 bits per byte** (cg27a_kmg's support-to-zero loss stopped near 0); RIGHT 0.20 -> 0.97 (the
re-write teaches them on the copy); facts 0.53 -> 0.42 (a cost to bystanders, -0.07 bits support). At this visit the memory had not yet
learned the inferences, so the counter-write suppresses rather than removes; the test that matters is visit 3-4. Bits 2.0325 / 2.0324
(cg27a_kmg 2.0342 / 2.0434, cg27a_kmi 2.0292 / 2.0304, cg27a_kmem 2.0277 / 2.0266: within noise).
Visit 3: **the trunk has NOT learned the unjudged inferences - RIGHT 0.00 / 0.08, WRONG 0.00 / 0.10 alone (cg27a_kmg at visit 3: 0.15 / 0.52
and 0.15 / 0.62)** - while the model answers them through the memory: RIGHT **0.32 / 0.93**, WRONG 0.36 / 0.87 (gate open 1.00 / 0.92).
Consolidation holds them out of the trunk exactly as designed. Taught facts WITH 0.42 / 0.77 vs trunk 0.27 / 0.73; held back 0.03 / 0.53 =
trunk (gate open 0.23); never-read names 0.00. COUNTER-WRITE (oracle, copy; now the memory holds the inferences): WRONG rank 0.92 -> **0.56**,
support +1.63 -> -1.44 bits; RIGHT 0.95 -> 0.97; facts 0.72 -> 0.72 (cg27a_kmg: wrong stayed 0.90, the trunk held them). Better than kmg, no
bystander cost, but only partial removal at 60 steps. Bits 2.0201 (cg27a_kmg 2.0319, cg27a_kmi 2.0203, cg27a_kmem 2.0105).
Visit 4 - THE FIRST VERDICTS: the trigger [signal docs] judged the visit-2 writes at AUC **0.637** (offline 0.639); flagged 40, wrong 0.35
(unflagged 0.24); 11 retracted (and counter-written out of the memory), 12 flagged taught kept as control. CONSOLIDATION: 160 judged and KEPT
(right **0.76**; written 0.74) now go to the trunk; 40 flagged never will (right 0.65). **The trunk receives 122 right and 38 wrong; an
oracle would pass 122 and 0** - 14 of 52 wrong ones kept out at the price of 26 right ones held back: the judge purifies only a little, as
offline. A further 200 written at visit 4 now wait in the memory alone. The readout's inference groups now mix the visit-2 writes (learned)
and the visit-4 writes (not yet learned): RIGHT WITH 0.02 / 0.52 vs trunk 0.00 / 0.13 (gate open 0.57), WRONG 0.00 / 0.53 vs 0.00 / 0.07 -
a readout artefact, not a loss (a by-age split is needed). Facts WITH **0.57 / 0.90** vs trunk 0.37 / 0.88; held back 0.07 / 0.47 vs trunk
0.03 / 0.57 (gate open 0.30: the small leak, as cg27a_kmg); paraphrased 0.12 / 0.68 = trunk; never-read names 0.00. Counter-write (copy):
WRONG 0.50 -> 0.45, RIGHT 0.55 -> 0.75 (mixed groups). Bits 1.9438 (cg27a_kmg 1.9398, cg27a_kmi 1.9354, cg27a_kmem 1.9185).
Visit 5 - THE TRUNK LEARNS THE KEPT INFERENCES: trunk alone, taught inferences RIGHT 0.17 / 0.48 (visit 3: 0.00 / 0.08), WRONG 0.07 / 0.40
(the 38 wrong ones the judge kept are now taught too); WITH the memory 0.10 / 0.88 and 0.05 / 0.68 (groups mix kept, flagged and awaiting -
the by-status readout is added for the next build). Facts WITH 0.52 / 0.77 vs trunk 0.47 / 0.73 (the trunk catching up); held back 0.00 /
0.60 vs 0.10 / 0.63 and paraphrased 0.12 / 0.62 vs 0.20 / 0.68 - **a small cost now where the gate opens (0.33 / 0.68)**; exact (greedy)
recall WITH the memory below the trunk's on inferences (0.10 vs 0.17); memory loss rising 0.144 -> 0.169 with 9181 of 16384 slots used -
the memory is filling (cg27a_kma doubles it to 65,536). Never-read names 0.00. Counter-write (copy): WRONG 0.57 -> 0.57 - the trunk now holds
the kept wrong ones. Bits 1.8910 (cg27a_kmi 1.8788, cg27a_kmem 1.8624). **The memory series' bits gap to cg27a_kmem grows from visit 4
(kmc +0.005 / +0.006 / +0.010 / +0.025 / +0.029; kmi +0.002 / +0.004 / +0.010 / +0.017 / +0.016)** - common to kmi, kmg and kmc, so not
consolidation; cg27a_kmgctl (memory frozen, every computation kept) separates the memory's learning from the extra computation.

### GPU cg27a_kma: the keyed memory addressed at every byte of any text (seed 4, 15 November, visits 1-2)

Visit 2: **reading the memory on ALL held-out text costs nothing - bits +0.0000 / +0.0001 per domain (trunk 2.0358), the memory addressed at
59-84% of bytes but the gate open at 0.0-0.1 per 1000** (it opens only where a written subject meets written slots). Recall as cg27a_kmc
with the in-context key: taught facts WITH 0.42 / 0.45 vs trunk 0.07 / 0.42 (kmc 0.40 / 0.50 vs 0.10 / 0.47); BARE wording ('A r ', no frame)
0.45 / 0.45 vs 0.20 / 0.45; paraphrased 0.15 / 0.30 vs 0.03 / 0.25; held back = trunk (gate open 0.03); never-read names 0.00. Wording head
(in context, 16 relations) loss 0.00. Counter-write (copy): WRONG -> 0.00, RIGHT 0.12 -> 0.95 (re-written), facts kept. Trunk bits 2.0375 /
2.0358 (cg27a_kmc 2.0325 / 2.0324). (Visit 2's two read domains printed the same bits with the memory, 2.0359: at visit 3 they differ - a
coincidence.)
Visit 3: **the in-context address gives far better EXACT recall** - taught inferences RIGHT WITH **0.68 / 0.98** vs trunk 0.00 / 0.10
(cg27a_kmc 0.32 / 0.93), WRONG 0.67 / 0.90 vs 0.00 / 0.13; taught facts WITH **0.60 / 0.80** vs trunk 0.28 / 0.78 (kmc 0.42 / 0.77); BARE
wording 0.53 / 0.78 vs 0.30 / 0.80; paraphrased 0.07 / 0.38 vs 0.03 / 0.35; held back 0.03 / 0.53 = trunk 0.00 / 0.53 (gate open 0.03);
never-read names 0.00; 15,016 of 65,536 slots used. ALL TEXT: bits WITH the memory +0.0007 / **+0.0030** / +0.0003 (Gutenberg / Wikipedia /
ArXiv; gate open 0.6 / 4.7 / 0.4 per 1000 bytes), next domain +0.0003 - a small cost where the gate opens on Wikipedia (where the facts'
names live), no gain yet on known names. Counter-write (margin, copy): WRONG 0.92 -> 0.77 (cg27a_kmb's byte edit replaces it). Trunk bits
2.0161 (cg27a_kmc 2.0201, cg27a_kmi 2.0203).

### GPU cg27a_kmf: readouts - the ledger, memory feedback, memory-only two-hop (seed 4, 15 November, visits 1-2)

Visit 2: INFERENCE LEDGER on the real writes - **the write score's AUC 0.504 (the offline subset's 0.70 did not hold: the top 200 scores
are nearly tied, 1.99), bridges 0.563**; premises are often DATES (Nashville ~ Tennessee: '1964, 1963, 1920'). MEMORY-ONLY TWO-HOP: 0
chains - the written inferences ('connected to', subject = the alphabetically first name) rarely chain with each other or with the taught
facts (planted names): the readout needs symmetric chains over inferences (fix for the next build). Memory as kmt (facts WITH 0.40 / 0.50
vs trunk 0.05 / 0.45; never-read 0.00; bits WITH the memory +0.0000-0.0001). Trunk bits 2.0296 / 2.0292.
Visits 3-4: **MEMORY FEEDBACK FROM READING - negative as a judge: AUC 0.623 / 0.466 (bits saved), 0.538 / 0.518 (confirmations -
contradictions), coverage 0.39 / 0.49; selection Spearman -0.10 / +0.07.** And telling: on real spans naming an inference's subject the
memory HURTS - bits saved on the second name: inferences RIGHT -721 / -3112, WRONG -352 / -880 (178 / 369 spans; about -8 bits a span): an
UNTYPED 'connected to' answer is the wrong continuation in almost every real sentence about the subject, right or wrong as a link. Untyped
inferences should not be read on general text at all (and they feed the all-text cost: Wikipedia +0.0072 at visit 4). RETRACTED group
(visit 4, before this visit's byte edit): WITH the memory 0.91 vs trunk 0.09 - the edit runs after the readout; visit 5 shows it. Ledger:
write score 0.495, bridges 0.474. Memory-only chains 0 (as predicted). Facts WITH 0.80 / 0.92 vs trunk 0.45 / 0.90; never-read 0.00.

### GPU cg27a_kmt: stricter gate (0.85) + relation types from the GPU trunk (seed 4, 15 November, visits 1-2)

RELATION TYPES FROM THE TRUNK (visit 2, 60,000 premises records, 6,595 with one Wikidata relation): the factorial on the GPU trunk is
ADDITIVE (R^2 0.97; variance subject / relation / object 0.60 / 0.20 / 0.18) with the relation's subspace apart from the names' (overlap
0.11 / 0.13, subject-object 0.12; random 0.01). Types (64) against Wikidata: surface wordings 0.352 / 0.273 (purity / NMI), raw 0.355 /
0.288, names out 0.382 / 0.305, type-masked 0.491 / 0.424, **MTB key type-masked 0.562 / 0.534** - the offline ranking reproduced at GPU
scale. Rules over the types: 8-21 rules, typed proposals right 0.00-0.23 - **rule discovery fails at scale too.** TWO-HOP through the memory
(10 chains): WITH the memory first hop exact 0.40, C rank 0.90; trunk alone 0.20, 0.90 (true B 0.90 / 0.80) - no gain yet (few chains;
both hops taught to the trunk too). Memory as cg27a_kmb (facts WITH 0.42 / 0.48 vs trunk 0.18 / 0.43; never-read 0.00; bits WITH the
memory +0.0000 at visit 2, gate open 0.0 per 1000). Trunk bits 2.0160 / 2.0278.
Visit 3: **THE STRICTER GATE CUTS THE ALL-TEXT COST ~7x - Wikipedia +0.0023 (gate open 2.4 per 1000) against cg27a_kmb's +0.0173 (10.6) at
the same visit**; Gutenberg +0.0006, ArXiv +0.0004, next domain +0.0005; at 0.95: Wikipedia +0.0014 (open 1.0). Recall nearly kept: taught
inferences RIGHT WITH 0.65 / 0.98 vs trunk 0.00 / 0.10 (kmb 0.68 / 0.98); facts WITH 0.50 / 0.72 vs trunk 0.27 / 0.70 (kmb 0.55 / 0.78);
BARE wording 0.53 / 0.78 vs 0.40 / 0.75; held back = trunk (gate open 0.03); never-read names 0.00. Byte edit (copy): WRONG 0.92 -> 0.08,
RIGHT 0.97 kept, facts kept. TWO-HOP (30 chains): WITH the memory C rank 0.90 = trunk 0.90 (first hop exact 0.37 vs 0.23) - the trunk
knows these chains' facts too, so no gain shows. Learning speed 0.48-0.51 (negative). Trunk bits 2.0182.
Visit 4: **THE GATE HOLDS AS THE MEMORY FILLS - Wikipedia +0.0034 (gate open 4.2 per 1000) against cg27a_kmb's +0.026 (17.2) at visit 4**;
other domains +0.0003-0.0008; at 0.95 Wikipedia +0.0020 (2.0). Taught facts WITH **0.82 / 0.93** vs trunk 0.38 / 0.87 (best so far); BARE
wording 0.78 / 0.90 vs 0.57 / 0.90; held back 0.07 / 0.57 vs 0.07 / 0.60; never-read names 0.00. Judge as kmb (AUC 0.636; 160 kept, 40
flagged, 11 retracted - the RETRACTED group is empty at this readout: the byte edit acts after it; visit 5). Byte edit (copy): WRONG 0.55 ->
0.05. TWO-HOP (30 taught chains): WITH 0.93 vs trunk 1.00 (the trunk knows these chains). RELATION TYPES (documents < 40,749): MTB key
type-masked 0.565 / 0.544 (wordings 0.339 / 0.267) - stable; rules over types still fail (typed right 0.00-0.21). Trunk bits 1.9425.
Visit 5: the trunk learned the KEPT inferences (trunk alone RIGHT 0.90, WRONG 0.76; flagged 0.00 / 0.14; awaiting 0.20). Facts WITH 0.72 /
0.78 vs trunk 0.37 / 0.73; never-read names 0.00. ALL TEXT: **Wikipedia +0.0076 (gate open 10.7 per 1000) - the cost creeps up again as the
memory fills (visits 3-5: +0.0023, +0.0034, +0.0076), still 4x below cg27a_kmb's +0.033; at 0.95: +0.0032 (open 4.2)**; others +0.0004-0.0018.
The RETRACTED group read 0 at visits 4-5: retracted inferences leave the taught pool, which the readout sampled - fixed (taken from the ledger)
for cg27a_kmf. Two-hop (taught chains) WITH 0.87 vs trunk 0.97. Byte edit (copy) WRONG 0.95 -> 0.23 (the kept wrong ones are in the trunk).
Trunk bits 1.8869.

### GPU cg27a_kmb: byte-edit retraction (seed 4, 15 November, visits 1-3)

Visit 3 (the memory now holds the visit-2 inferences): **BYTE EDIT (oracle, on a copy): WRONG rank 0.92 -> 0.03, RIGHT 1.00 -> 0.97, facts
0.70 -> 0.70, undone exactly** (24 answer bytes still above the margin) - against the margin counter-write at the same point, 0.92 -> 0.56
(cg27a_kmc) and 0.92 -> 0.77 (cg27a_kma). **The GPU's partial retraction is solved.** LEARNING SPEED AS A JUDGE (Go-CLS; 136 inferences
learned 4+ times, right 0.71): memory loss at first learning AUC 0.514, now 0.489, the drop 0.483, the trunk's own loss at first 0.503 -
**negative: how fast the memory learns an inference says nothing about whether it is right.** Recall as cg27a_kma (facts WITH 0.55 / 0.78
vs trunk 0.30 / 0.77; inferences RIGHT 0.68 / 0.98 vs 0.00 / 0.08; never-read names 0.00; awaiting inferences WITH 0.97 vs trunk 0.13).
ALL TEXT: **bits WITH the memory +0.0044 / +0.0173 / +0.0007 (Gutenberg / Wikipedia / ArXiv; gate open 3.0 / 10.6 / 0.8 per 1000), next
domain +0.0029 - a real cost on Wikipedia, larger than cg27a_kma's +0.0030 at the same visit (gate open 4.7)**: the gate (0.7) opens on
written subjects in ordinary Wikipedia text and the read hurts there (known names +0.014). The stricter-gate readout (0.85 / 0.95) was
added after kmb started; it rides on the next build. Trunk bits 2.0157 (cg27a_kma 2.0161).

Visit 4 (cg27a_kmb): trigger AUC 0.636 (as kmc); KEPT 160 (right 0.76), flagged 40, 11 retracted - the byte edit acts on the real memory
after this readout. BY STATUS (WITH the memory vs trunk, rank): KEPT right 0.97 vs 0.33, KEPT wrong 0.97 vs 0.07, FLAGGED right 1.00 vs 0.00,
FLAGGED wrong 0.86 vs 0.14 (not yet edited), AWAITING (written at visit 4) 0.10 vs 0.07 - the visit-2 writes 1.00 vs 0.20. Taught facts WITH
**0.78 / 0.93** vs trunk 0.38 / 0.90; BARE wording 0.75 / 0.88 vs 0.47 / 0.82; held back 0.07 / 0.53 vs 0.07 / 0.60; never-read 0.00.
Byte edit (copy): WRONG 0.50 -> 0.03, RIGHT 0.60 -> 0.57, facts 0.85 -> 0.85. Learning speed AUC 0.50-0.51 again (negative). ALL TEXT: bits
WITH the memory **+0.026 on Wikipedia (gate open 17.2 per 1000; known names +0.026)**, +0.0045 / +0.0013 / +0.0033 on the others - the cost
grows as the memory fills (cg27a_kmt's stricter gate). Trunk bits 1.9239 (cg27a_kmc 1.9438, cg27a_kmem 1.9185).

Visit 5 (cg27a_kmb): **the trunk learned the KEPT inferences and not the flagged ones** - trunk alone, KEPT right 0.87 / KEPT wrong 0.83
(visit 4: 0.33 / 0.07), FLAGGED right 0.20 / wrong 0.29, awaiting 0.27 - consolidation as designed (and the judge lets 38 wrong ones
through). WITH the memory: facts 0.70 / 0.78 vs trunk 0.53 / 0.73; inferences RIGHT 0.67 / 0.97 vs 0.22 / 0.47; never-read names 0.00. The
readout's FLAGGED groups mixed retracted with flagged controls (still taught, kept in the memory) - a RETRACTED group is added for
cg27a_kmt. Byte edit (copy): WRONG 0.95 -> 0.20 (292 bytes left above the margin: the kept wrong ones are in the trunk now), RIGHT 0.95,
facts 0.75. Learning speed 0.48-0.52 (negative, third time). ALL TEXT: **Wikipedia +0.033 (gate open 23.9 per 1000)**, Gutenberg +0.009,
others +0.001-0.004 - growing each visit with the memory (kmt). Trunk bits **1.8570** (cg27a_kmem 1.8624, cg27a_kmi 1.8788, cg27a_kmc 1.8910):
no trunk cost - the earlier gaps were run-to-run noise.

### Reconsolidation-style retraction: only the wrong answer's bytes, only the slots it reads (inf_kmc_mode byte; tools/kmi_smoke.py, 15 November)

After reconsolidation (an update of the retrieved trace at prediction error) rather than extinction (a competing memory): the slots each
retracted answer byte reads in its TRAINING frames are retrieved and only that byte's logit is lowered in them (the name's bytes, not the
shared final '.'; one change per slot and byte - the largest asked, not the sum) until the memory's support is below -1 nat; every edit is
recorded and can be UNDONE. Package smoke (tiny model, 300 memory steps), rank among 10 in the TEST wording (never edited) WITH the memory:

| | WRONG (retracted) | RIGHT (siblings: same relation, shared wording slots) | facts | undo |
|---|---|---|---|---|
| margin counter-write, 60 steps (cg27a_kmc) | 1.00 -> 0.20; 0.86 -> 0.00; **after the trunk moved 0.65 -> 0.45** | 0.97-1.00 | 1.00 | - |
| margin, 200 steps | 1.00 -> 0.15; 0.86 -> 0.00; after the move 0.65 -> 0.30 | 0.97-1.00 | 1.00 | - |
| BYTE EDIT (first version: summed, final '.' included) | -> 0.00 (support -52 to -80 bits) | 0.97 -> 0.70-0.75 | 1.00 | exact |
| BYTE EDIT (wide: 64-slot neighbourhood) | -> 0.00-0.05 | 0.97 -> 0.33-0.47 | 0.97 | exact |
| **BYTE EDIT (name bytes, one change per slot and byte)** | **1.00 -> 0.05; 0.86 -> 0.00; after the move 0.65 -> 0.10** | **0.93-0.95** | **1.00** | **exact** |

**Positive: the byte edit removes the retracted inference to the trunk's level - also after the trunk moved, where the margin counter-write
reproduced the GPU's partial removal (0.45) - at a cost of 0.02-0.04 to siblings, no cost to facts, and is exactly reversible** (forgetting as
lost access: a retraction by the weak judge - 65% of the GPU's flagged inferences were right - can be undone if later reading supports it).
The first version over-shot (the same slots edited once per frame) and lowered the final '.' every answer shares; the 64-slot version
reaches unseen wordings but costs siblings too much.

### The GATED integrated memory read on ALL text (tools/int_mem.py IM_GATE=1, 15 November; stand-in trunk, mixed setting)

As the integrated test above (the memory read inside the forward pass at every byte with a subject), but at evaluation the read counts only
where its address was written (the subject is one the fact rows wrote for AND half the read weight is on slots they wrote):

| lr | A after A (open / rank) | B after B | A after B | paraphrased | forgetting wiki / dev, after A -> after B | ungated (same lr) |
|---|---|---|---|---|---|---|
| 0.03 | 0.40 / 1.00 | 0.53 / 1.00 | 0.27 / 1.00 | 0.02 / 0.71 | **+0.000 -> +0.000 / +0.011 -> +0.015** | +0.27 -> +0.53 / +0.34 -> +0.62 |
| 0.1 | **0.69** / 1.00 | **0.77** / 1.00 | 0.35 / 1.00 | 0.01 / 0.71 | **+0.001 -> +0.001 / +0.026 -> +0.036** | +0.78 -> +1.29 / +1.03 -> +1.55 |

**Positive: the gate removes the cost of reading the memory on ordinary text - wiki +0.000-0.001 bits (from +0.5 to +1.3) and dev text
+0.015-0.036 (from +0.6 to +1.5) - with recall unchanged (rank 1.00, exact names 0.69-0.77, paraphrase 0.71).** The residual dev cost: the
gate's subject index also took in capitalised frame words ('From', 'Notes') and the answers themselves, since every position of a fact row
wrote; restricting writes to the answer bytes should close it. The memory can be read on all text.

### The gate and the counter-write in the package (tools/kmi_smoke.py KM_PRESET=cg27a_kmg, 15 November)

Tiny untrained Model, fabricated pool, 300 memory steps: GATE open on taught facts and inferences 1.00, on held-back facts 0.15-0.17, on
questions about never-read names 0.00. COUNTER-WRITE (oracle, on a copy, 60 steps): driving the memory's SUPPORT for the wrong answers
(bits per answer byte it adds) to zero took WRONG from +5.8-6.0 to -2.7 to -3.6 (rank 1.00 -> 0.30-0.50) with RIGHT inferences and facts
unchanged; an unlikelihood loss reached only +2.3. Built as cg27a_kmg.

### Label-free triggers from CONSISTENCY (tools/stab_trigger.py, 15 November; the GPU path's 200 writes, 0.740 right, distant labels)

Research routes (semantic entropy / SelfCheckGPT: stability under resampling; CAGED / CKRL / KGTtm: agreement across views and path
evidence; temporal stability as evidence accumulates). AUC right vs wrong, and the lowest 20% flagged (wrong among them; base 0.26):

| signal | AUC | wrong among flagged | kept right |
|---|---|---|---|
| (restatement: later documents stating the pair, tools/trigger_retest.py) | 0.639 | 0.35 | 0.76 |
| 1 BOOTSTRAP STABILITY: mean / worst percentile over 5 resamples (70% of documents) / minus spread | 0.547 / 0.528 / 0.504 | 0.30 / 0.25 / 0.18 | 0.72-0.75 |
| 2 VIEWS at write time: code / Adamic-Adar / resource allocation / 3-hop paths | 0.399 / 0.606 / 0.632 / 0.631 | 0.22 / 0.30 / 0.40 / 0.40 | 0.73-0.78 |
| 2 the WORST view (all must support) / minus the views' spread | 0.623 / 0.607 | 0.40 / 0.40 | 0.775 |
| 3 TEMPORAL: percentile after the next 2/10 of reading (indirect support only) / its change / 3-hop change | 0.612 / 0.608 / 0.596 | 0.32 / 0.29 / **0.43** | 0.75-0.78 |
| combined: bootstrap worst + view worst + temporal change | 0.628 | 0.40 | 0.775 |

(The resample and later tables are 256-dimensional - memory; the write-time rule at 256 dims scores 0.572.) **Negative as a trigger: no
consistency signal passes AUC 0.64; the best flag sets hold wrong 0.40-0.43 against 0.26 (kept precision 0.74 -> 0.78).** Bootstrap stability
is no help (the written pairs are stable whether right or wrong); path support (resource allocation, 3-hop) and its growth carry a little.
Caveat: AUCs against the DISTANT labels (Wikidata), which agree with human labels only 0.85 (tools/trigger_retest.py part 2); offline against
Re-DocRED human labels the restatement trigger reached 0.72-0.73 - part of the ceiling here may be the labels.

### Are the triggers weak, or are the labels? (tools/human_trigger.py, 15 November; Re-DocRED writes, HUMAN and distant labels side by side)

The same signals on the Re-DocRED writes of tools/fact_compress.py (200 per split; human-labelled right 0.37 / 0.41; distant labels agree with
the human ones 0.85 / 0.87). Later reading = distant documents 0-30,000; path views on the distant graph with every record stating a written
pair left out (indirect support only). AUC against HUMAN | DISTANT labels, split s0 / s1:

| signal | HUMAN | distant |
|---|---|---|
| restatement: documents stating the pair | 0.702 / 0.707 | 0.698 / 0.736 |
| restatement: short-gap records (3 or fewer words between) | **0.719 / 0.714** | 0.693 / 0.734 |
| path: Adamic-Adar | 0.645 / 0.608 | 0.661 / 0.643 |
| path: resource allocation | **0.745 / 0.713** | 0.768 / 0.738 |
| path: 3-hop | 0.481 / 0.481 | 0.481 / 0.542 |
| temporal: change of resource allocation (documents 10,000 -> 30,000) / of 3-hop | 0.741 / 0.696; 0.487 / 0.490 | 0.765 / 0.709; 0.493 / 0.548 |
| combined: restatement + RA + 3-hop / all | 0.677 / 0.658; 0.681 / 0.654 | 0.689 / 0.707; 0.692 / 0.698 |

Flagging the lowest 20% of short-gap restatement or resource allocation: wrong among flagged 0.80-0.86 (human base 0.59-0.63).
**The labels do not hide the trigger: on the same writes the human and distant AUCs agree within about 0.03 (human slightly lower).** The
signals reach 0.70-0.75 here and 0.61-0.64 on the GPU path's own writes because the WRITES differ: the GPU path selects its writes by code +
Adamic-Adar, so the path evidence is already spent in the selection and what remains to separate them is weaker. The 0.64 ceiling on the
GPU path is therefore real. Rank-sum combinations are worse than the best single signal (3-hop is noise here and drags them down).

### Choose with one signal, judge with another (tools/select_judge.py, 15 November; the GPU path's write step, 200 writes at document 2/10, judged on 2/10 -> 4/10)

| writes chosen by | written right | best judge (AUC) | TO THE TRUNK with the lowest 20% flagged | oracle |
|---|---|---|---|---|
| the code alone | 0.475 | restatement + RA 0.631-0.634 (restated later: right 0.38 / wrong 0.15) | 0.53 (160 kept); kept only if restated: 0.69 (52 kept) | 95 right |
| **code + Adamic-Adar (GPU now)** | **0.740** | RA 0.654, restatement + RA 0.652, documents 0.639 | **0.762-0.772** (160 kept) | 148 right |
| code + resource allocation | 0.680 | restatement + RA 0.640 | 0.725-0.730 | 136 right |

**Negative: choosing by the code alone and keeping path support for the judge loses** - the writes fall from 0.74 to 0.475 right and no
judge recovers it (best end to end 0.53, or 0.69 on only 52 kept). Path evidence does more good choosing than judging; the current
pipeline (code + AA, judged by documents) is the best of these, 0.77 to the trunk. The judge ceiling on these writes is ~0.65 whatever
the signal. Further gains must come from a new kind of evidence (the Go-CLS learning-speed readout rides on cg27a_kma), not a re-split.

### Does the relation word give a direction of its own? (tools/rel_direction.py, 15 November; 18k stand-in trunk, d 192)

The author's hypothesis: in a plain pooled (JEPA-style) representation of 'A r C', the relation adds a direction roughly orthogonal to what
names and numbers add. PART A, factorial (16 subjects x 12 relation phrases x 16 objects, and x 5 years), trunk states pooled over the
sentence (mean; middle / last block):

| | variance share subject / RELATION / object / year | additive R^2 | subspace overlap RELATION-subject / RELATION-object / subject-object / RELATION-year (random ~0.04) |
|---|---|---|---|
| middle block, mean | 0.40 / 0.22 / 0.38 | 1.00 | 0.12 / 0.11 / **0.42** |
| last block, mean | 0.33 / 0.32 / 0.33 | 0.99 | 0.17 / 0.18 / **0.32** |
| middle, with years | 0.37 / 0.21 / 0.39 / 0.02 | 0.99 | 0.12 / 0.13 / 0.44 / **0.05** |
| last, with years | 0.31 / 0.31 / 0.33 / 0.03 | 0.98 | 0.18 / 0.18 / 0.34 / 0.21 |

**Positive for the hypothesis: the pooled representation is almost exactly ADDITIVE (R^2 0.98-1.00) - subject + relation + object (+ year)
- and the relation's effects lie in a subspace largely apart from the names' (overlap 0.11-0.18, against 0.32-0.44 between subject and
object, which share a 'name' subspace) and from the year's (0.05 in the middle block).** (At the relation phrase's last byte the state is
~all relation: 0.99-1.00 of the variance.)
PART B, real records ('A <words> C', one Wikidata relation, the 12 commonest; evaluation only): 5-nearest-neighbour relation accuracy of the
pooled states 0.50 / 0.52 (chance 0.08); with the NAME subspace of part A projected out **0.54 / 0.54**, k-means NMI 0.27 -> 0.30 (last
block); projected ONTO part A's 8-dim relation subspace 0.23-0.31 (12 synthetic phrases span too little of real relations). Relation
type is readable from plain representations of real records, and removing the names' direction helps; the relation subspace has to be
learned from real text (not from 12 phrases) before it can type records label-free.

### A LEARNED gate (tools/int_mem.py IM_LGATE=1, 15 November; 18k stand-in trunk, the integrated memory, lr 0.1, fact rows write)

A small network on the read's features (written-slot weight, subject written, read confidence, address scores, byte offset) gates the read,
trained by the trunk's own loss on fact rows AND ordinary wiki rows (+0.02 x the mean gate: a price for opening), against the fixed gate:

| gate | A after A | B after B | A after B | paraphrased | bits wiki / dev (after B) | never-read names open |
|---|---|---|---|---|---|---|
| fixed (subject written, half the weight on written slots) | 0.81 / 1.00 | 0.89 / 1.00 | 0.56 / 0.98 | 0.03 / 0.66 | -0.000 / +0.016 | 0.00 |
| LEARNED | 0.79 / 1.00 | 0.92 / 1.00 | 0.52 / 0.98 | 0.03 / 0.65 | -0.000 / +0.015 | 0.00 |

**Neutral: the learned gate matches the fixed one (recall, abstention) but does not lower the cost on dev text (+0.015 vs +0.016).** The
features it sees cannot tell a helpful read from a harmful one here; on the GPU the cost is traced to the untyped inferences' slots
(cg27a_kmf memory feedback), which the gate-rule readout of cg27a_kmr measures directly. Keep the fixed gate.

### A discriminating test: relational support or only lists? (tools/readiness.py part 1, 15 November; the GPU path's 200 writes)

After an outside review: judge an inference by the plausible ALTERNATIVE that explains the same evidence - our errors are mostly associated
pairs without a relation, so: is the pair ever linked by a RELATIONAL wording (1-8 words, not only ',' / 'and' / 'or' / '(' / '-', no 2+
commas in the span), or only by lists? Records of documents < 4/10 (the judge's horizon); AUC right vs wrong (distant labels score only):

| signal | AUC |
|---|---|
| **relational records of the pair** | **0.697** |
| share of the pair's records that are relational | 0.625 |
| bridges linked relationally on both sides | 0.598 |
| not list-only | 0.597 |
| bridges that are entities (no dates / numbers) / entity bridges linked relationally | 0.581 / 0.592 |
| all bridges (comparison) | 0.594 |
| (the current judge: documents stating the pair) | 0.639 |

**Positive: counting only RELATIONAL records of the pair beats the current judge (0.697 vs 0.639)** - and LIST-ONLY pairs (co-occur,
never relationally: 14% of the writes) are right only 0.48 against 0.78 for the rest. The alternative explanation the errors share -
'they appear in the same list' - is testable label-free. Discounting date / number bridges did not help (0.581 vs 0.594). Parts 2-3
(readiness, name merging) were killed for memory alongside a training job; rerun later.
**Confirmed through the GPU's own code path** (tools/rel_judge_confirm.py: premises._rcode_step writes at visit 2, premises._rret_judge with
inf_rret_sig 'rel' at visit 4): judge AUC 0.688 (relational records 0.697; since the write 0.675; documents 0.639 on the same batch);
FLAGGED 40 now wrong **0.45** (the documents judge: 0.35), unflagged 0.21 (0.24); list-only pairs 25, right 0.44. The visit-4 batch (judged at
visit 6) ran out of memory locally; cg27a_kmr's visit-6 judge line measures it on the GPU.

### Do the existing weights say an inference is false? (tools/weights_judge.py, 15 November; 18k stand-in trunk, the GPU path's 200 writes)

1 TRUTH DIRECTION (Marks & Tegmark 2023; CCS): a probe on read spans vs the same spans with the second name swapped (same entity type) -
held-out accuracy **0.477 (chance 0.50): the stand-in trunk cannot tell a read fact from a swapped one at all**; on the writes, probe AUC
0.448-0.453, the trunk's likelihood 0.539. 2 INTERFERENCE (3 SGD steps on the inference, MLPs; loss on what was read about A and C): AUC
0.489 - learning ANY inference LOWERS that loss (right -0.361, wrong -0.375 nats: shared names and wordings); gradient cosine 0.452 (+0.73
both). 3 HELD-OUT GAIN (Go-CLS validation; Mattar & Daw's gain: loss on text read AFTER the write naming A or C): AUC 0.500 (+0.278 vs
+0.284 nats). **Negative - but the prerequisite is missing: a trunk that cannot distinguish read facts from swapped ones (0.48) has no
knowledge to judge with.** These signals need a trunk that knows facts: repeated on the pretrained trunk (B) and to be measured on the
GPU trunk.

### Rules by composing relation maps between name codes (tools/rel_ops.py, 15 November; after Hernandez et al., ICLR 2024)

Types as rel_mtb's best (type-masked Matching-the-Blanks key, 64 types over 30,000 records); name codes from the store's record graph
(PPMI-SVD 128, 167,022 names); per type and direction a ridge map code(A) -> code(C) (128 with 40+ records; held-out cosine median 0.52,
best 0.71). RULES: of 16,128 compositions W(t2) W(t1), 1,482 match some W(t3) above a shuffled-type null's 99th percentile (0.719; top
0.859) - but every top rule is 'X then <co-mention> => X' (',' / 'and' / '-' types: siblings P3373, spouses P26): the co-occurrence codes
put related names NEAR each other, so the maps are close to a scaled identity and composing with one changes nothing. TYPING the GPU path's
200 writes (0.74 right): the best map's type is the pair's relation 0.22 (0.30 among pairs with one); as a JUDGE, best map fit AUC 0.478,
best composed-rule fit 0.498. **Negative: relation maps over CO-OCCURRENCE codes are near-identity - rules found are trivial, typing and
judging at chance.** Maps need name codes that carry attributes (what a name IS), not only who it appears with.
With ATTRIBUTE codes instead (RO_ATTR=1: a name's code = the mean Matching-the-Blanks key of the relation types it takes part in, as head
and as tail, + its entity type; 30,677 names): maps fit far better (held-out cosine 0.915) but mostly because the entity types are easy;
170 rules above the null (top 0.643), again 'X then Y => X' ('author'^-1 then 'from'^-1 => 'author'^-1); typing the writes 0.28 (0.38
among pairs with a relation); judge AUC 0.502 / 0.537. **Still negative: label-free rule discovery from this corpus has now failed four
ways (chains, shared pairs, maps over co-occurrence codes, maps over attribute codes).**

### Typed entailment graph from the records' name pairs (tools/ent_graph.py, 15 November; after Hosseini et al., TACL 2018)

Typed predicates (wording + the names' entity types) with 5+ name pairs, documents 0 -> 2/10: 3,579 predicates over 146,183 pairs, but
only 5,504 predicate pairs share ANY name pair. Paraphrase groups (inclusion both ways >= 0.3 / 0.5): 3 / 2 groups of 2+ predicates -
punctuation variants (') for the' ~ '( mlb ) for the'); the 'types' are then the predicates themselves (purity 0.79 over 3,576 types,
covering 0.53 of the labelled records: not comparable). ENTAILMENT edges (one way): 285 / 97, specific -> generic wording (', d.c. ,' =>
','; 'in washington ,' => 'in'); new typed facts right 0.095 / 0.161 against the stated predicate's own 0.224 / 0.351. **Negative: the
distant set (Wikipedia leads) rarely restates one name pair in different wordings - entailment by argument-pair inclusion needs the
redundancy of many sources reporting the same events (news, as Hosseini et al.).** The representation (MTB key) generalises across pairs;
counts over pairs cannot.

### Relation types with the names MASKED and a Matching-the-Blanks key (tools/rel_mtb.py, 15 November; 18k stand-in trunk, 30,000 real records)

After Peng et al. (EMNLP 2020: models lean on names; mask them) and Baldini Soares et al. (ACL 2019, Matching the Blanks: two sentences of
the same name pair are positives). Pooled states (middle + last block) of each record, 64 k-means types, scored against Wikidata only
(3,228 records with one relation); the MTB key: a 2-layer head, InfoNCE over 15,660 name pairs stated in two different documents.

| record read as | purity | NMI |
|---|---|---|
| surface wordings (baseline) | 0.343 | 0.290 |
| 'A <words> C.' (names) | 0.377 | 0.318 |
| 'someone <words> something.' (names MASKED) | 0.432 | 0.378 |
| MTB key, masked | 0.397 | 0.359 |
| 'a person <words> a place.' (names masked, ENTITY TYPES as placeholders) | 0.486 | 0.463 |
| **MTB key, type-masked** | **0.574** | **0.568** |

**Positive: masking the names, putting their entity types in their place and training a Matching-the-Blanks key label-free types real
records far better than their wordings - purity 0.34 -> 0.57, NMI 0.29 -> 0.57 (twice).** Rules over the types remain unusable on 30,000
records (1-6 rules, typed proposals right 0.00-0.05): typing is no longer the bottleneck - rule discovery by counting chains is (sparsity).
Next: typed entailment graphs (argument-pair inclusion; Hosseini et al. TACL 2018) and rules by composing relation operators.

### Relation types from the representation, then rules over them (tools/rel_types.py, 15 November; 18k stand-in trunk, 30,000 real records)

Label-free: real records 'A <words> C.' (documents 0 -> 2/10) as the trunk's pooled state; k-means into 64 types; AMIE rules over the
types. Scored only: type purity / NMI against Wikidata (3,228 records with one relation), and typed proposals (the head type's commonest
relation holds for x-z).

| types from | purity | NMI | rules | typed proposals | TYPED right | untyped right |
|---|---|---|---|---|---|---|
| the 63 commonest SURFACE wordings | 0.343 | 0.290 | 0 | 0 | - | - |
| raw pooled states | **0.397** | 0.322 | 5 | 140 | 0.400 | 0.414 |
| names out (name subspace learned from REAL per-name means) | 0.361 | 0.283 | 10 | 295 | 0.119 | 0.180 |
| names out (name subspace from a SYNTHETIC factorial, 8 dims) | 0.394 | **0.337** | 5 | 50 | **0.440** | 0.500 |
| the same, 16 dims | 0.397 | 0.336 | 9 | 150 | 0.287 | 0.320 |

**Partly positive: the representation types real records better than their wordings (purity 0.40 vs 0.34, NMI 0.32-0.34 vs 0.29), and
removing a synthetic name subspace helps a little; but a name subspace learned from real per-name means is confounded with relations
(overlap with the wording subspace 0.61 - a name's records share its relations) and hurts. Rules over the types are too sparse on 30,000
records and typed precision 0.20-0.44 is far below the untyped path's 0.74 - not usable at this trunk and sample size.** The stand-in trunk
is small (d 192); the test belongs on the GPU trunk, over all records, as a readout.

### Typed real-text inference by rule mining over surface wordings (tools/typed_rules.py, 15 November; distant documents 0 -> 2/10)

AMIE-style closed-path rules r1(x,y) & r2(y,z) => r3(x,z) over typed records (the 150 commonest specific wordings between two names, 1-6
words apart, typed by the names' entity types), label-free; proposals = body pairs never stated; scored against Wikidata (evaluation only):

| relations | typed facts | rules kept | proposals | untyped right | TYPED right (the head's relation) |
|---|---|---|---|---|---|
| full wording | 54,059 | 12 (paraphrase pairs) | 24 | 0.42 | 0.42 |
| last word | 124,251 | 102 (sentence templates: 'born' TIME & 'in' LOC => TIME 'in' LOC) | 13,554 | 0.04 (best bin 0.21) | 0.04 (best bin 0.30) |
| last word, entities only, body facts from different documents | 67,031 | 17 | 1,609 | 0.10 | 0.08 |
| full wording, entities only, different documents | 24,389 | 5 | 128 | 0.16 | 0.09 |

**Negative: rules over SURFACE wordings do not give typed inference** - surface wordings are not relations (one relation has many
wordings, one wording - 'in', 'of' - many relations), so the rules found are paraphrase pairs or one sentence's template, and the typed
proposals are right 0.04-0.42 against the untyped path's 0.74. Typed inference needs records typed by RELATION, not by wording.

### Invariance across environments as the judge (tools/env_judge.py, 15 November; after Lopez-Paz, 'Against All Likelihood', arXiv 2611.09385)

Environments made without labels: the 40,749 documents read by 4/10 clustered by the names they mention (TF-IDF -> SVD 64 -> k-means, K =
8 / 16 / 32). For the GPU path's 200 writes (code + AA, 0.740 right; code + RA, 0.680): environments restating A-C later, environments with
a bridge path A-x-C inside them, the share of environments mentioning both in which they are linked (invariance), environments mentioning both.

| signal (AUC right vs wrong; code + AA / code + RA) | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| environments restating A-C later | 0.566 / 0.578 | 0.544 / 0.578 | 0.541 / 0.557 |
| environments with a path A-x-C inside | 0.545 / 0.551 | 0.504 / 0.544 | 0.508 / 0.552 |
| invariance share (linked wherever both occur) | 0.504 / 0.479 | 0.471 / 0.449 | 0.443 / 0.412 |
| environments mentioning both | 0.533 / 0.563 | 0.509 / 0.578 | 0.523 / 0.589 |

To the trunk with the lowest 20% flagged: 0.725-0.756 (code + AA) - no better than flagging nothing (0.740) and below the current judge (0.772).
**Negative: invariance across label-free document environments is WORSE than the counts it was meant to replace (documents 0.639, resource
allocation 0.654), and the invariance share is below chance.** The errors here are not repeated falsehoods but ASSOCIATED pairs without the
relation (wrong = no Wikidata relation between names that do co-occur): such pairs are linked in many environments, often more than right
ones. Invariance of an association cannot tell 'related' from 'holds this relation' - a typed claim is needed before invariance can judge it.

### Typing written inferences on confirmation (tools/type_confirm.py, 15 November; the GPU path, code + AA under the asymmetry filter)

200 writes (0.740 right), typed by later reading (1-6 words between the names, either order): 0.94 typed - but the commonest typed wording is
',' (122 of 187; 'X, Y' lists and appositions), then 'and' (8), 'in' (5). TYPING ACCURACY (evaluation only: the wording's commonest
Wikidata relation is the pair's own) 0.48. Signals (AUC vs distant labels): later records stating it 0.601, wording functional 0.541,
specific 0.489, not contradicted later 0.512 (108 of 187 'contradicted' - commas are 'functional' and everything is contradicted).
**Negative: in this text, written pairs are restated mostly as lists, so later reading neither types them reliably nor contradicts the
wrong ones. The trigger for untyped writes is not reachable from co-mention; retraction needs a different signal.**

### GPU cg27a_kmi: the integrated keyed memory (seed 4, 15 November, visits 2-5)

The model's answer = trunk + memory, test wording, open / rank among 10 (WITH vs trunk alone):

| visit | taught facts | held-back facts | taught inferences RIGHT | WRONG | held-back inferences |
|---|---|---|---|---|---|
| 2 | **0.38** / 0.47 vs 0.07 / 0.45 | 0.00 / 0.33 vs 0.00 / 0.40 | (written at this readout) | | |
| 3 | 0.32 / 0.70 vs 0.30 / 0.62 | 0.00 / 0.43 vs 0.03 / 0.47 | **0.42 / 0.93** vs 0.05 / 0.50 | 0.54 / 0.92 vs 0.08 / 0.59 | 0.00 / 0.13 vs 0.00 / 0.10 |
| 4 | 0.35 / 0.73 vs 0.43 / 0.73 | 0.03 / 0.53 vs 0.07 / 0.63 | 0.33 / 0.53 vs 0.32 / 0.50 (half not yet taught) | 0.42 / 0.53 vs 0.30 / 0.47 | 0.00 / 0.17 vs 0.03 / 0.23 |
| 5 | **0.53** / 0.80 vs 0.43 / 0.78 | 0.10 / **0.63 vs 0.80** | **0.67 / 0.98** vs 0.35 / 0.83 | 0.68 / 0.95 vs 0.15 / 0.85 | 0.00 / 0.10 vs 0.00 / 0.20 |

(At visits 2 and 4 the sampled 'taught' inferences include the ones written at that readout, taught only from the next visit.) Paraphrased
facts 0.15 / 0.35 -> 0.10 / 0.65 (trunk 0.00 / 0.30 -> 0.12 / 0.72). Memory steps 399 a visit, 98-99% of question rows addressed; slots used
9,515 of 16,384 by visit 5. The readout memory with TRUNK name codes on the same visits: inferences 0.40 / 0.72 -> 0.14 / 0.33 (crowding, as
cg27a_kmem). Bits 2.0292 / 2.0304 / 2.0203 / 1.9354 / 1.8788 against cg27a_kmem's 2.0277 / 2.0266 / 2.0105 / 1.9185 / 1.8624 (+0.002 to
+0.017; the trunk's training is meant to be identical - within seed noise but growing: to check whether the memory's extra forwards have
side effects).
**Positive: the hashed key holds - taught inferences 0.67 exact / 0.98 rank at visit 5 against the trunk's 0.35 / 0.83, facts +0.10 exact.
The cost: where nothing was written, the memory's read still adds logits - held-back facts lose rank (0.80 -> 0.63) and paraphrases
(0.72 -> 0.65). The memory needs a GATE: read only when its address is one it has written.** Wrong inferences are recalled as strongly
(0.68 / 0.95): retraction still matters, and its trigger is not solved.
Visit 6 (a write visit: the sampled 'taught' inferences include the 200 written at this readout): facts WITH **0.63 / 0.90** vs trunk 0.60 /
0.82; inferences RIGHT 0.43 / 0.68 vs 0.28 / 0.60, WRONG 0.47 / 0.72 vs 0.40 / 0.68; HELD-BACK facts 0.00 / 0.47 vs 0.07 / 0.60 (the cost
again); paraphrased 0.10 / 0.57 vs 0.07 / 0.62. Memory: 3,591 steps, 98% of rows addressed, 10,490 of 16,384 slots used. Bits 1.9100
(cg27a_rasym 1.8840 at visit 6; cg27a_kmem stopped before visit 6). The readout memory with trunk codes: inferences 0.21 / 0.57.

### Retraction for the integrated memory: trigger and precise erase (15 November, local)

**Trigger, on the CURRENT GPU writes** (tools/trigger_retest.py with code + AA under the asymmetry filter, distant set, the GPU path:
200 writes, 0.740 right; judged on the next 2/10 of the documents): later reading states 94% of the pairs again (right 0.95 / wrong 0.92).
AUC against the distant labels: documents stating the pair again **0.639** (bottom 20% wrong 0.35, overall 0.26), records 0.638, short-gap
records 0.607, bridges at write 0.563, the rret trigger (short + PMI) 0.528, predictive compression 0.504-0.594, popularity 0.360 (i.e.
popular names lean wrong). **Negative for untyped links: the best label-free signal flags wrong writes at 0.35 against 0.26 - retracting the
flagged fifth would raise the kept writes from 0.74 to about 0.76 right while losing 26 right ones.** Untyped claims are not contradicted
by real text; a trigger needs TYPED claims (the earlier offline surprise test reached 0.80 with the pair's own source sentence).

**Precise erase: one wording, two objects** (tools/jepa_mem.py JM_M2; 60 subjects 'is connected to' two names - the GPU's case; seeds 0 / 1):

| | retracted (open) | SIBLING: same subject, same wording (open) | other facts |
|---|---|---|---|
| ERASE the slots | 0.40 -> 0.00, 0.30 -> 0.00 | **0.47 -> 0.00, 0.45 -> 0.00** | unchanged |
| COUNTER-WRITE (unlearn the retracted, re-write the sibling, 60 steps) | 0.40 -> **0.00**, 0.30 -> **0.00** | 0.47 -> **0.95**, 0.45 -> **0.85** | unchanged |

**Positive: when two facts share an address, erasing kills both; a counter-write removes the retracted one (open 0.00) and KEEPS the
sibling (it even sharpens), bystanders untouched.** The precise erase is solved; the trigger for untyped links is not.

### The keyed memory INTEGRATED into the forward pass (tools/int_mem.py, 15 November; 6k stand-in trunk, mixed setting, seed 0)

Address from the model's own context at every byte (subject = latest completed capitalised run; wording = the bytes since, through a
JEPA head on the trunk's middle states; offset = answer byte), the read added to the last residual; only the memory values learn, from
the fact rows (wiki rows read but do not write). 105 facts (52 + 53), recall THROUGH THE WHOLE MODEL in the test wording:

| lr | A after A (open / rank) | B after B | A after B | paraphrased | forgetting wiki / dev (after B) |
|---|---|---|---|---|---|
| (trunk alone) | 0.00 / 0.27 | | | 0.00 / 0.28 | |
| 0.03 | 0.40 / **1.00** | 0.53 / **1.00** | 0.27 / **1.00** | 0.02 / **0.72** | **+0.53 / +0.62** |
| 0.1 | **0.69** / 1.00 | **0.77** / 1.00 | 0.35 / 1.00 | 0.01 / 0.74 | **+1.29 / +1.55** |

ERASE (the slots read where 30 facts' answers are predicted): retracted rank 1.00 -> 0.27-0.30 (the trunk's own level), bystanders' rank
0.92-1.00. **Positive on learning - rank 1.00 and EXACT names 0.69-0.77 through the model's own bytes (every in-trunk method ~0), paraphrase
0.72-0.74 - but reading the memory on ORDINARY TEXT costs +0.5 to +1.5 bits** (wiki subjects and wordings reach written slots; erasing
the 30 facts' slots alone gives back a third of it). So the GPU build reads it on the question rows only (cg27a_kmi).

**GPU cg27a_kmem, visits 4-5 (the readout memory, trunk name codes):** taught facts 0.45 / 0.61 -> 0.59 / 0.74; inferences RIGHT 0.20 /
0.43 -> **0.05 / 0.16** while the trunk, taught them, reached rank 0.88 / 0.75; two-hop 0.09 / 0.15-0.20; persistent facts written at
visit 2: 0.49 -> 0.20 -> 0.32 open. ERASE still exact (WRONG -> 0.00, facts unchanged). Bits 1.9185 at visit 4 (cg27a_rasym 1.9183).
**The subject half of the address - the trunk's name code - drifts and crowds as the trunk learns the inferences; the integrated build
uses a NAME-IDENTITY HASH (byte 3-grams) instead** (offline: memory loss 1.05 vs 1.17 with trunk codes, 6x cheaper).
Package smoke (tools/kmi_smoke.py, tiny untrained Model, 300 memory steps): the model's rank WITH the memory 0.97-1.00 vs trunk 0.05-0.15,
held back unchanged, still 0.95-1.00 after the trunk moved; exact names need more steps than this budget (0.00-0.02).

### GPU cg27a_kmem: the keyed memory at GPU scale (readout) + rasymaa's code + AA (seed 4, 15 November, visits 1-2)

Keyed memory (name x JEPA wording key, 128 x 128 slots; inference.kmem), recall in the TEST wording, open / rank among 10 (chance 0.10):
taught facts **0.92 / 0.94** at visit 1, **0.72 / 0.83** at visit 2 (held back 0.00 / 0.23, 0.03 / 0.09; the trunk answers taught facts
right 0.06 open at visit 2, false-memory probe c); real-text inferences RIGHT **0.57 / 0.81**, WRONG 0.51 / 0.69 (held back 0.00 / 0.18,
0.00 / 0.23); TWO-HOP 0.19 / 0.44 (32 chains; first hop decoded right 0.50). Wording head on the GPU trunk: the test wording's relation
0.95 / 0.98, a PARAPHRASED relation 0.76 / 0.88 (better than offline) - but paraphrased recall of facts only 0.21-0.26 / 0.25-0.34
(inferences via 'is tied to' 0.50 / 0.71). ERASE (oracle, the WRONG inferences): 0.51 -> **0.00**, facts 0.72 -> 0.73, RIGHT inferences
0.57 -> 0.42 (same subject, same wording - as warned). PERSISTENT: facts written at visit 1 read at visit 2 with the drifted keys **0.73 /
0.88** (0.96 / 1.00 when written). Code + AA wrote **0.740** right at visit 2 (cg27a_rasym 0.680), as its readout predicted. Bits 2.0277 /
2.0266 (cg27a_rasym 2.0145 / 2.0206; visit 1 precedes any change: seed noise). **Positive so far: at GPU scale the memory holds the real
facts in an unseen wording far better than the trunk, survives one visit of key drift, and erases exactly.**

### Keyed units beyond names: which pay for themselves in bits? (tools/keyed_units.py, 14 November; Re-DocRED text, 6k stand-in trunk)

The author's idea: 4-6 generic KEYED units like NAME, indexable, extracted from local context, each with its own memory, kept only if they
measurably improve language modelling. Counted extractors (no learning yet), each with a keyed continuation memory (the bytes after each
firing, stored under its key); when the key fires again, the stored continuations consistent with the bytes read so far predict the next
byte, mixed with the trunk at lam = n / (n + 2). Prequential on the last 600 of 3,053 documents (630,697 bytes; trunk alone 2.246 bits per
byte); memories filled from every earlier document. Bits saved per 1,000 bytes:

| unit (key) | alone | on top of NAME (fixed mixture) |
|---|---|---|
| **NAME** (a run of capitalised words) | **29.99 (1.34% of the trunk's bits)** | - |
| FACT (a name + the 1-4 words after it) | 9.79 | **-1.77** |
| PLACE (a locative word + a name) | 3.57 | -0.50 |
| DATE (the sentence's last name, where a year or month starts) | 2.97 | **+1.29** |
| NUMBER (the sentence's last name + the word before a number) | 3.05 | +0.30 |
| ALL five | 29.31 | -0.68 |

**NAME carries almost all of it. Every other unit saves bits alone, but only DATE (+1.29) and NUMBER (+0.30) add anything on top of NAME;
FACT and PLACE are mostly NAME's predictions again and, through the fixed mixture, make it worse.** The mixture is the weak point, not the
units: a unit with many stored continuations gets a high weight whether or not they are right.

**The feedback gate (same run, second version).** A softmax over the trunk and each firing unit; each unit's logit comes from a table keyed by
(unit, consistent continuations, bytes into the continuation, the trunk's entropy at that byte), starts at the fixed rule and takes one
online gradient step on -log p after each byte is scored (prequential). Bits saved per 1,000 bytes, all 600 documents / last 400:

| | fixed mixture | learned gate (all / last 400) | on top of NAME with the gate (all / last 400) |
|---|---|---|---|
| NAME | 29.99 | **32.29 / 33.04** | - |
| NAME+FACT | 28.22 | 32.76 / 33.75 | **+0.47 / +0.72** |
| NAME+PLACE | 29.48 | 32.14 / 32.82 | -0.15 / -0.21 |
| NAME+DATE | 31.27 | 34.35 / 35.15 | **+2.05 / +2.11** |
| NAME+NUMBER | 30.29 | 32.84 / 33.64 | **+0.54 / +0.60** |
| ALL five | 29.31 | **35.06 / 36.10 (1.61% of the trunk's bits)** | **+2.76 / +3.06** |

**Positive: with trust learned from the bits each unit saves, three units add beyond NAME (DATE, NUMBER, FACT), all five together save
36.1 bits per 1,000 bytes against NAME's 30.0 under the fixed rule (+20%), and the gain grows after warm-up.** PLACE stays slightly
negative (its predictions are NAME's) and is the one to drop.

**Second document set (Re-DocRED test, 500 documents, 520,299 bytes; memory from all 3,053 training documents).** On top of NAME with the
gate, all / last two-thirds, bits per 1,000 bytes (first set's last 400 for comparison):

| | test set | first set |
|---|---|---|
| NAME alone: fixed rule / gate (late) | 28.37 / 32.01 | 29.99 / 33.04 |
| + DATE | **+1.65 / +1.80** | +2.11 |
| + NUMBER | **+0.58 / +0.57** | +0.60 |
| + FACT | -0.50 / -0.48 | +0.72 |
| + PLACE | -0.24 / -0.28 | -0.21 |
| **+ DATE + NUMBER** | **+2.29 / +2.46** | **+2.71** |
| + all four | +1.36 / +1.54 | +3.06 |

**Replicated: the gate (+12-13% for NAME alone), DATE and NUMBER. Not replicated: FACT (helps on one set, hurts on the other). PLACE
hurts on both.** The kept design is NAME + DATE + NUMBER behind the learned gate: 34.5-35.8 bits saved per 1,000 bytes (late), about 1.55%
of the trunk's bits, against 28.4-30.0 for NAME under the fixed rule (+19-20%). Counted extractors throughout; learned extractors next.

### An AMIE-style functional contradiction trigger (tools/exclusive_trigger.py, 14 November; GPU write path, distant set, split 0)

AMIE (Galarraga 2013): a wording is FUNCTIONAL if each subject has one object under it (functionality >= 0.8 over 5+ records; 3,328
wordings). A written inference A-C is CONTRADICTED if A later appears under a functional wording with an object other than C. 400 writes
at document 20,375, judged on documents 20,375-40,749:

| writes | judged | contradicted: wrong | not contradicted: wrong | likelihood ratio |
|---|---|---|---|---|
| all | 336 (wrong 0.40) | 70: 0.49 | 0.37 | 1.4 |
| asymmetry > 1 (functional-like) | 166 (wrong 0.24) | 20: 0.35 | 0.23 | 1.7 |

**Weak: a functional contradiction is only 1.4-1.7 times likelier on a wrong write and fires on a fifth of them.** Confirmation by
prediction (tools/surprise_trigger.py, AUC 0.80) stays the best retraction trigger found.

### Type constraints from the class-pair table (tools/fact_compress.py FC_LCWA, 14 November; GPU write path, distant set, two splits)

The type-constrained knowledge-graph literature (Krompass 2015, local closed-world assumption): keep a candidate inference only if its
name-class pair (compressed classes of A and C) is one the store often links. Precision of the top 200 / 400 written by code + AA, split 0 / 1:

| filter | top 200 | top 400 |
|---|---|---|
| none | 0.395 / 0.395 | 0.302 / 0.297 |
| class-pair PMI above its 25th percentile | 0.375 / 0.400 | 0.282 / 0.297 |
| class-pair PMI above its 50th percentile | 0.345 / 0.360 | 0.263 / 0.270 |
| asymmetry > 1 alone | 0.585 / 0.525 | 0.515 / 0.453 |
| asymmetry > 1 + class-pair PMI above its 25th percentile | 0.590 / 0.535 | 0.522 / 0.448 |

**Negative: class-pair PMI does not separate right from wrong candidates (AUC 0.535 / 0.534); the filter costs precision on its own and
adds nothing to the asymmetry filter.** The compressed classes are too coarse to act as types (as compress_typing found for exact
relations); asymmetry stays the only label-free write filter that works.

### Can compression statistics type an inference? (tools/compress_typing.py, 14 November; Re-DocRED gold + 30k distant documents, two splits)

Gold-related pairs of held-out documents with every record pairing them removed (3,186-3,536 test pairs, never stated in the store).
Label-free signals; a probe maps each signal's value to the gold label it most often has on kept documents. 'Coverage at 0.8' = the share
of test pairs the signal's most confident predictions type at 80% precision. Split 0 / split 1:

| target (baseline) | best signal: right | coverage at 0.8 precision |
|---|---|---|
| exact relation (commonest, P131: 0.363 / 0.363) | class pair + asymmetry 0.410 / 0.443 | **0.000-0.014 for every signal** |
| family: location, person, organisation, work, time (location: 0.610 / 0.594) | path wording 0.675 / 0.662 (on 67% of pairs); class pair + asymmetry 0.613 / 0.614 | class pair + asymmetry 0.395 / 0.416 |
| functional or not (functional share 0.488 / 0.480) | **class pair + asymmetry 0.746 / 0.774**; asymmetry alone 0.717 / 0.737 | **0.608 / 0.747** |
| direction, location pairs | **the more specific name (lower store degree) is the head 0.753 / 0.756** | - |

Signals: compressed name-class pair; ASYMMETRY (log of A's store degree over C's); latent relations (universal schema) of the bridge records
and their composition; the pair's own later records (only 4-8% of held-out pairs have one); the wordings' functionality (AUC 0.536-0.546).
On Re-DocRED alone (no distant documents) family looked better (coverage 0.66); the noisier, larger store lowers it.
**Mixed. The exact relation cannot be typed without labels; the family barely beats 'always location'. But two coarse properties can:
whether the inferred relation is FUNCTIONAL (0.75-0.77, two thirds of pairs typed at 80% precision) and, for location-like pairs, its
DIRECTION (0.75) - both mostly from how general one name is against the other in the store.** That is enough for a weaker trigger than
typing: a confidently functional inference A -> C is contradicted when later reading ties A as strongly to another name of C's class.

### Why the retraction trigger failed on the GPU (tools/trigger_retest.py, 14 November; the package's own code path, distant set)

PART 1 reruns the GPU path (_load_distant, _rcode_step at 20,375 documents, seed 4: 200 written, right 0.590 - as on the GPU) and judges the
writes on documents 20,375-40,749. The trigger reproduces: AUC 0.518, bottom 20% wrong 0.38 (overall 0.41). Its parts against the distant
labels:

| signal | AUC | bottom 20% wrong (overall 0.41) |
|---|---|---|
| documents / records / short-gap records stating the pair | 0.58 / 0.58 / 0.58 | 0.57 / 0.57 / 0.53 |
| document PMI | **0.43** (reversed) | 0.28 |
| the names' popularity (low = popular) | **0.41** (popular pairs are more often right) | 0.28 |
| the write-time code; bridges at write | 0.39; 0.55 | 0.28; 0.50 |
| **predictive compression** (A's later partners coded with C's old neighbourhood, over a same-degree control) | **0.50** | 0.50 |

93% of the writes are stated again in the next documents (right 0.97, wrong 0.88): the signal is nearly saturated. Among these writes the
popular pairs are the labelled ones, so PMI - which discounts popularity - points the wrong way and cancels the weak count signal.
Predictive compression: the inference saves bits on the names' later partners for 97% of writes, right and wrong alike (AUC 0.50; on the
Re-DocRED writes 0.49 / 0.53 against human labels) - the 'wrong' writes are related in the store's sense too.

PART 2, the Re-DocRED writes under both truths: human and distant labels agree on 0.85 / 0.87; distant labels add 0.12-0.14 'right' and
almost never drop a human-right pair (0.01). **The labels are not what broke the trigger; the writes are** - pairs of popular, often
co-mentioned entities from the distant documents, where re-mention says little and popularity tracks the label.

**What this means.** A 'wrong' write here is mostly a pair the text keeps putting together that has no labelled relation: compression
cannot tell it apart, because it does compress. Telling such pairs apart needs WHICH relation (typed inferences a later sentence can
confirm or contradict), not more evidence that the names are associated. Best untyped trigger left: raw re-mention counts (AUC 0.58,
likelihood ratio about 1.7) - too weak to retract on.

### Refusal again: familiarity rows and a name gate (tools/toy_refuse.py, 14 November; toy, 3,000 steps, two seeds)

Taught wording (the toy cannot answer in the held-out one); refusal on fresh names from the stored names' syllables (never read); facts
only read answered right; known facts right. NAME GATE (no training): abstain when the trunk's surprise at the subject's name (bits a byte
after 'Some say that ') is above the 95th percentile of known names'. Seed 0 / seed 1:

| condition | refuse never-read | only-read right | known right | name gate: AUC read vs fresh; abstains never-read / only-read |
|---|---|---|---|---|
| none | 0.00 / 0.00 | **0.62 / 0.78** | 1.00 / 1.00 | 0.87 / 0.93; **0.48 / 0.50**, 0.06 / 0.02 |
| refusal rows, same syllables | 0.85 / 0.97 | 0.15 / 0.12 | 0.97 / 0.94 | 0.75 / 0.79; 0.30 / 0.28, 0.15 / 0.09 |
| + 12 wordings | 0.87 / 0.80 | 0.05 / 0.09 | 0.99 / 0.99 | 0.95 / 0.96; 0.98 / 0.90, 0.33 / 0.13 |
| refusal + FAMILIARITY rows ('Have I read about A? yes / no') | 0.82 / 0.75 | 0.38 / 0.21 | 0.99 / 0.99 | 0.71 / 0.82; 0.25 / 0.50, 0.10 / 0.15 |
| familiarity rows only | 0.00 / 0.00 | 0.47 / 0.55 | 0.96 / 0.91 | 0.82 / 0.93; 0.50 / 0.77, 0.19 / 0.09 |

In the held-out wording every refusal-trained arm refuses KNOWN facts too (0.54-0.81): the refusal is tied to taught frames. **Familiarity
rows halve the damage of refusal rows to facts only read (0.15 / 0.12 -> 0.38 / 0.21) but do not remove it, and cost reading on their own
(0.47 / 0.55). The name gate, with no training, abstains on half of never-read names at a 2-6% cost to facts known or only read, and leaves
learning from reading untouched.** For the GPU: no refusal training; the name gate at answer time beside the confidence gate.

### GPU cg27a_rret (= cg27a_rcode5 + retraction; seed 4, 14 November, visits 1-3): the dose fix

Retraction cannot act before the first judgement (visit 4), so visits 1-3 test the dose scaling alone (inf_rti_rows_ref 8: each of 16 rows
pushes as one of 8). Visit 3, the visit-2 batch after one visit of teaching: dose loss 1.171 -> 0.234 (cg27a_rcode4 1.234 -> 0.354;
cg27a_rcode3 1.059 -> 0.253); **rank taught right / wrong 0.54 / 0.57 vs held back 0.11 / 0.11** (cg27a_rcode4 0.43 / 0.37; cg27a_rcode3
0.61 / 0.54); open-ended 0.14 / 0.10 (unchanged from cg27a_rcode4; cg27a_rcode3 0.22 / 0.27). Held-out bits 2.0235 / 2.0263 / 2.0125
(cg27a_rcode4 2.0237 / 2.0271 / 2.0110). **The fix restores most of the per-inference learning (loss and rank near cg27a_rcode3's, with
twice as many taught: about 44 right ranked first vs 30), at no cost; open-ended exact recall has not caught up at one visit.**

Visit 4. **The trigger does not transfer to the GPU: AUC 0.517; the 40 flagged are wrong 0.38 against 0.42 unflagged** (200 judged on 20,374
documents read since; 20 with no confirming record). 23 flagged taught: 8 retracted (3 wrong), 15 kept as control - too few, and no more
wrong than the rest. Offline (Re-DocRED human labels, 0.6 wrong) the same score had AUC 0.72-0.73; here truth is the distant set's
Wikidata labels and the writes are pairs from held-out distant documents - which difference breaks it is not yet known. Recall with the
dose fix after two visits: **rank taught 0.97 / 0.90 vs held back 0.19 / 0.25** (cg27a_rcode4 0.89 / 0.87; cg27a_rcode3 0.88 / 0.88);
open-ended 0.47 / 0.39 vs 0.01 / 0.00 (cg27a_rcode4 0.43 / 0.36). Second batch right 0.59. Held-out bits 1.9236 (cg27a_rcode4 1.9246).

### Codes as tokens on real text: NEGATIVE (tools/codetok_diag.py, 14 November; stand-in trunk, Re-DocRED, two seeds)

Each name followed by its compression cluster token (128 k-means clusters of the store's co-mention graph codes) in the kept documents'
sentences and in 'A is connected to C' rows; held-out pairs (gold-related in held-out documents, never sharing a kept sentence), rank of the
true partner among 10 of its type and popularity quintile (the code itself, no trunk: 0.30 / 0.28). Step 1,500, seed 0 / seed 1:

| arm | held-out (never stated) rank | wiki bits |
|---|---|---|
| no tokens | 0.370 / 0.390 | 2.021 / 2.018 |
| real code tokens | 0.455 / 0.460 | 2.019 / 2.015 |
| **the same tokens shuffled across names** | **0.470 / 0.465** | 2.017 / 2.017 |

**Negative: the gain over no tokens is the added token, not the code** - shuffled tokens, which carry no relational information, match or
beat the real ones at every checkpoint (step 500: 0.445 / 0.425; 1,000: 0.495 / 0.445). A plausible reading: any fixed token after a name
marks the name's end and sharpens the candidate scoring. With real-text codes this noisy (0.28-0.30 on their own), and a trunk that reads
a code only as well as the code is clean (tools/compose_synth.py CS_PCODE 2), codes as tokens do not carry inference on real text here.

### A retraction trigger from later reading (tools/confirm_trigger.py, 14 November; Re-DocRED gold, two splits)

The 200 writes of each split (code + Adamic-Adar after 5,000 distant documents), judged on distant documents 5,000-30,000, label-free. AUC
right vs wrong, and the least confirmed 20% as the retraction set (wrong share; likelihood ratio = wrong caught / right lost):

| signal | AUC | bottom 20%: wrong, LR (split 0 / 1) | at the GPU's base (a third wrong) |
|---|---|---|---|
| re-ranking by the store's own score (test A) | 0.65-0.69 | 0.80 / 0.68, 2.3 / 1.4 | 0.42-0.55 |
| documents stating the pair | 0.71 | 0.85 / 0.82, 3.3 / 3.2 | 0.62-0.63 |
| short-gap records (3 or fewer words between) | 0.71-0.72 | 0.82 / 0.88, 2.8 / 4.8 | 0.59-0.71 |
| document PMI of the pair | 0.64-0.65 | 0.93 / 0.85, 7.2 / 3.9 | 0.67-0.79 |
| **short-gap records + PMI (rank sum)** | **0.72-0.73** | **0.88 / 0.93, 4.1 / 8.4** | **0.68-0.81** |
| the names' popularity | 0.46 | - | - |

**Positive: what later reading says about the pair itself - stated close together, more often than the names' frequencies predict -
flags wrong inferences far better than re-ranking (likelihood ratio 4-8 against about 2).** At the GPU's base the retraction set would be
about 70-80% wrong. Small samples (40 flagged a split); the labels are incomplete, so some 'wrong' writes are true.

### Which selection rule writes the most right inferences: more signals (tools/fact_compress.py FC_SELECT 2, 14 November; two splits)

Store of Re-DocRED + 30,000 distant documents, codes 256 dims, all candidates never stated together (base 0.095 / 0.086). Share labelled
related among the top N written, split 0 / split 1:

| rule | top 50 | top 100 | top 200 | top 400 |
|---|---|---|---|---|
| code alone | 0.40 / 0.28 | 0.43 / 0.26 | 0.38 / 0.305 | 0.33 / 0.28 |
| code + Adamic-Adar (cg27a_rcode3) | 0.60 / 0.58 | 0.52 / 0.52 | 0.395 / 0.395 | 0.30 / 0.30 |
| **code + resource allocation** | **0.62 / 0.52** | **0.53 / 0.54** | **0.465 / 0.445** | **0.40 / 0.37** |
| code + Adamic-Adar + L3 (3-paths) | 0.64 / 0.64 | 0.50 / 0.44 | 0.385 / 0.38 | 0.33 / 0.315 |
| bagged code (3 fits on 80%) + Adamic-Adar | 0.64 / 0.58 | 0.51 / 0.50 | 0.375 / 0.375 | 0.30 / 0.28 |

**Positive, both splits: resource allocation (shared neighbours weighted 1/degree, not 1/log degree) in place of Adamic-Adar keeps
precision as more is written - level at the top 100, +5 to +7 points at 200, +7 to +10 at 400.** The store could write two to four times
as many inferences at nearly the same precision. 3-paths and bagged codes add nothing.

### Unlearning a wrong taught inference (tools/retract_diag.py, 14 November; stand-in trunk 3 x 192, two seeds)

The trunk is taught 120 right pairs, 120 wrong ones (A with a random name of the answer's type) and 120 bystanders in the system's
question frames (about 18 rows each), then 200 retraction steps with the right ones rehearsed. Rank of the taught answer among it and 9
of its kind (chance 0.10), seed 0 / seed 1:

| | WRONG | RIGHT | BYSTANDER (taught, then left alone) | wiki bits |
|---|---|---|---|---|
| before teaching | 0.15 / 0.10 | 0.27 / 0.31 | 0.33 / 0.32 | 1.900 |
| taught | 0.42 / 0.40 | 0.50 / 0.59 | 0.68 / 0.55 | 2.024 / 2.014 |
| STOP (no longer taught) | 0.25 / 0.25 | 0.58 / 0.69 | 0.50 / 0.42 | 1.982 / 1.978 |
| NEUTRAL answer ('no one on record') | 0.25 / 0.25 | 0.63 / 0.66 | 0.54 / 0.46 | 2.013 / 2.008 |
| UNLIKELIHOOD, weight 1 | 0.00 / 0.01 | 0.68 / 0.65 | 0.43 / 0.49 | 2.034 / 2.010 |
| NEUTRAL + UNLIKELIHOOD | 0.00 / 0.00 | 0.73 / 0.72 | 0.47 / 0.50 | 2.065 / 2.056 |
| **UNLIKELIHOOD, weight 0.1** | **0.01 / 0.01** | 0.64 / 0.71 | **0.55 / 0.53** | 2.005 / 1.990 |
| UNLIKELIHOOD, weight 0.3 | 0.02 / - | 0.65 / - | 0.51 / - | 2.012 / - |

**Positive: unlikelihood on the wrong answer's bytes (with the right ones rehearsed) removes a taught wrong inference in both seeds;
stopping or a neutral answer only halves it.** Collateral damage at weight 1 is seed-dependent (bystanders -7 / +7 points against STOP);
at weight 0.1 none in either seed (bystanders 0.55 / 0.53 against STOP's 0.50 / 0.42), text cost +0.01 to +0.02 bits against STOP. The mechanism is settled; the trigger (which writes are
wrong) is not.

### Held-back right inferences ranked above chance: prior, not generalisation (tools/gen_diag.py, 14 November; stand-in trunk)

GPU cg27a_rcode3 ranks held-back RIGHT answers first at 0.18-0.24 (held-back wrong 0.10-0.15). Stand-in trunk, 4 x 90 pairs (taught right
/ wrong, held back right / wrong), rank among held-back answers as the GPU readout (in brackets: among 9 of its type):

| | taught R | taught W | held back R | held back W |
|---|---|---|---|---|
| before teaching | 0.13 (0.24) | 0.08 (0.12) | **0.17 (0.33)** | 0.03 (0.11) |
| after teaching (36 rows per taught pair) | 0.43 (0.64) | 0.40 (0.69) | **0.19 (0.29)** | 0.03 (0.09) |

**Negative for generalisation:** held-back right answers are preferred BEFORE any teaching, and teaching other pairs leaves that
unchanged. The trunk's reading already favours fitting answers; the GPU's held-back 0.22 is that prior, not inference learned from the
taught pairs.

### Two hops answered by reading a code in one (tools/compose_synth.py CS_PCODE 1, 14 November; synthetic, 3 layers x 192, seed 0)

Every person also carries a code summarising their neighbourhood - their city's country's code, right 80% of the time, as a compressor
of the fact table would give it ('Pebel Belsavo-qa'). Held-out people are never told their citizenship (chance 0.10; ceiling about 0.80):

| | citizenship of held-out people (never taught) | new person in an old city, taught 'lives in' only (300 steps) |
|---|---|---|
| no person codes (earlier run) | 0.12 | 0.10-0.11 |
| **person codes + 400 worked citizenship examples** | **0.75-0.80** (from step 1000) | **0.70-0.75** |
| **person codes, NO citizenship examples at all** | **0.50-0.66** (0.57 at the end) | **0.58-0.60** |

Without any citizenship ever stated, the trunk reads the person's code and answers with the country it learned only from CITY codes
('Velmora-qa lies in C'). Arms whose people carry no code stay at chance (0.08-0.12) whatever else is known. **Positive: what the trunk
cannot compose (two hops), it reads in one hop from a code that already holds the composite.** Caveat: the code here is a token close to
the answer (80% informative); real compressed codes are 512-dim vectors and carry 'related', not which relation - whether they can be
attached to names as tokens the trunk reads (e.g. the nearest of k code clusters) is the next test.

### GPU cg27a_rcode2: the compressed fact table's inferences reach the weights (seed 4, 14 November, visits 2-5)

One change against cg27a_rcode: the written inferences get their own question rows (8 a step; about 900-1,900 rows in play). Same
store result (visit 4, with a chain: code AUC 0.674 vs Adamic-Adar 0.568; the 200 written right 0.255-0.32). Recall from the trunk in
the test wording 'It is on record that A is connected to ' (never taught), inferences written at visit 2:

| | taught RIGHT | taught WRONG | held back RIGHT | held back WRONG |
|---|---|---|---|---|
| open-ended exact, visit 3 | 0.15 (92) | 0.12 (164) | 0.00 (36) | 0.02 (108) |
| **open-ended exact, visit 4** | **0.40** | **0.38** | **0.03** | **0.00** |
| rank among 10 written answers, visit 3 | 0.41 (46) | 0.32 (82) | 0.11 (18) | 0.04 (54) |
| **rank, visit 4** | **0.78** | **0.84** | **0.06** | **0.11** |

(cg27a_rcode, the same inferences at about one row a visit: 0.00-0.03 open-ended. The rank readout's distractors were partly taught
answers, which pushes the held-back figure below chance; fixed for the next build.) Cost: held-out bits on the domains read, visits
2-4, 2.0265 / 2.0127 / 1.9263 against cg27a_rcode's 2.0281 / 2.0129 / 1.9293 - none.
**Positive, one seed: inferences drawn from real text without labels or rules - pairs never stated together anywhere in the store,
chosen by compressing the fact table - are recalled from the weights in a wording never taught, 0.40 open-ended (0.78 ranked) against
0.00-0.03 (0.06-0.11) for the held-back third.** **But right and wrong ones alike (0.40 vs 0.38; 0.78 vs 0.84):** the weights take in
whatever they are taught, so about two thirds of what is written - the wrong inferences - becomes false memories at the same rate.
Selectivity must come before the write (the code's top, confirmation by later reading) or after it (retraction when text contradicts).
Visits 5-6: the visit-2 inferences are KEPT and rise (open-ended taught right / wrong 0.37 / 0.41 at visit 5, 0.46 / 0.40 at visit 6; held
back 0.00-0.02); the visit-4 batch follows the same curve (0.00 / 0.08 after one visit, 0.17 / 0.23 after two). Rank recall at visit 6:
taught 0.84 / 0.81, held back 0.08 / 0.06. Held-out bits at visits 5 / 6: 1.8710 / 1.8986 against cg27a_rcode's 1.8661 / 1.8978 (+0.005,
+0.001).

### Popularity-matched check, and which relation (tools/fact_compress.py, tools/rel_codes.py; 14 November, Re-DocRED gold, two splits)

POPULARITY MATCHED (AUC within quintiles of the degree product): with a chain, code-512 0.709 / 0.706 vs Adamic-Adar 0.691 / 0.700 (raw
0.726 / 0.746 vs 0.619 / 0.610) - much of the code's raw AUC lead came from chaining being misled by popular names; at matched popularity
they nearly tie, while at the top of the ranking (what is written) the code stays ahead (0.32-0.37 vs 0.24 in the top 5%). With no chain
the code holds (0.553 / 0.536 matched; chaining 0.500).
WHICH RELATION (gold-related held-out pairs, 1,217-1,371; base: the commonest gold relation, P131, 0.53-0.55; oracle: the commonest for the
pair's gold entity types, 0.64-0.68): a DistMult code per name over 64 label-free latent relations 0.31-0.44 (and it does not even rank
relatedness: AUC 0.41-0.47); a neighbour vote in code space (the relation A holds with names whose codes are like C's) 0.39-0.49, most
confident quarter 0.49-0.60. **Negative:** label-free latent relations are too noisy (relation identity AUC 0.66) to say which relation
better than guessing the commonest one. The compressed code says THAT, not WHICH.

### Why written real-text inferences are not recalled (tools/teach_diag.py, 14 November; stand-in trunk, 120 real Re-DocRED pairs)

GPU cg27a_rcode visits 3-7: written inferences recalled from the trunk at 0.00-0.03 (open-ended exact), taught or held back, right or
wrong, while the code kept its lead in the store (visit 6, 969k records: with a chain AUC 0.659 vs Adamic-Adar 0.565, top 5% 0.268 vs
0.158; written right 0.285). On the GPU each written inference gets about one question row a visit (4 rows a step from one random slot,
shared with that slot's other facts) and is never read in any text. Taught offline the same way (the system's seven frames, test wording
'It is on record that A r '), recall measured open-ended exact AND as the answer's rank among 10 of its kind (chance 0.10; 0.28 before):

| condition | rows per fact | open-ended | rank, taught | rank, +300 steps of other text |
|---|---|---|---|---|
| GPU-like dose, never-read names, 'is connected to' | 2.5 | 0.00 | 0.34 | 0.29 |
| dose x4 | 10 | 0.00 | 0.44 | 0.37 |
| dose x16 | 40 | 0.01 | **0.74** | **0.47** |
| familiar names (read by this trunk), 2.5 / 10 rows | 2.5 / 10 | 0.00 | 0.31 / 0.39 | 0.32 / 0.30 |
| the gold relation's phrase, 10 rows | 10 | 0.00 | 0.46 | 0.36 |
| the source sentence also read, 10 rows | 10 | 0.00 | 0.53 | 0.40 |

**Two causes:** (1) the readout - open-ended exact recall cannot see real multi-word names (0.01 while rank is 0.74); planted names are short
invented strings; (2) the dose - about one row a visit is far too little. Familiarity and the relation phrase add little; reading the
sentence some. Next build (cg27a_rcode2): the written inferences' own question rows, 8 a step, and a rank-recall readout.

### GPU cg27a_rcode: the compressed fact table at scale (seed 4, 14 November; visit 2 readout, distant labels)

Store graph 152,606 names from 344,600 records; 2,045 distant documents held out (their records and every record pairing two of their
names removed); 86,965 candidate pairs never stated together, labelled 0.095. The same readout, same candidates, at two code sizes:

| | Adamic-Adar AUC / top 5% / 25% | CODE 128 AUC / top 5% / 25% | **CODE 512 AUC / top 5% / 25%** |
|---|---|---|---|
| with a chain (labelled 0.119) | 0.587 / 0.246 / 0.156 | 0.593 / 0.224 / 0.172 | **0.682 / 0.330 / 0.225** |
| all pairs (0.095) | 0.587 / 0.192 / 0.140 | 0.579 / 0.186 / 0.134 | **0.625 / 0.278 / 0.161** |
| no chain (0.074) | 0.500 / 0.070 / 0.072 | 0.509 / 0.120 / 0.080 | 0.535 / 0.129 / 0.077 (popularity 0.587) |
| the 200 written | - | right 0.270 | **right 0.320** (3.4x base; cg27a_rti about 0.18) |

**Replicates at scale:** with 512 dimensions the compressed fact table beats chaining clearly on pairs with a chain (AUC 0.68 vs 0.59; top
quarter 0.225 vs 0.156) and writes inferences right 0.32 of the time; at 128 dimensions it only ties (the dilution seen offline). Codes in
12 s. Where no chain exists, popularity wins on the distant labels (they favour well-known names; on Re-DocRED gold popularity was below
chance), so that comparison needs gold or a popularity-matched control. Consolidation readouts from visit 3.

### Does reading the premises make a novel inference stick? (tools/evidence_infer.py, 14 November, overnight; stand-in trunk, two seeds)

120 SUPPORTED pairs (A-B gold in one Re-DocRED document, B-C gold in another, A and C never in the same document anywhere) and 120
UNSUPPORTED pairs (A with a C of the same type, no chain, never together). Recall: C first among 10 same-type names, in two wordings never
taught; GAIN FROM A ranks C by how much A (against 'someone') helps predict it, which familiarity cannot move (chance 0.10 throughout).

| after 600 steps | seed 0 supported / unsupported (wording 1, wording 2, gain) | seed 1 |
|---|---|---|
| v1: READ the evidence (unsupported: A's documents only) | 0.21, 0.23 / 0.11, 0.11 | 0.22, 0.21 / 0.15, 0.14 |
| **v2: READ the evidence (unsupported: A's and C's documents)** | **0.18, 0.18, 0.07 / 0.11, 0.15, 0.06** | **0.15, 0.19, 0.12 / 0.13, 0.13, 0.10** |
| v2: taught, then evidence | 0.23, 0.23, 0.07 / 0.20, 0.20, 0.07 | 0.19, 0.22, 0.16 / 0.19, 0.18, 0.09 |
| v2: taught, then other text | 0.14, 0.16, 0.12 / 0.07, 0.07, 0.13 | 0.16, 0.14, 0.15 / 0.12, 0.12, 0.12 |

**Negative.** v1's apparent inference from reading (0.12 -> 0.21-0.23) was mostly familiarity: once the unsupported C is read as often,
supported leads by 0.00-0.07 (about one standard error) and the gain from A stays at chance. Reading the premises does not put the A-C
link in this trunk's weights (consistent with open question 7: the trunk does not compose at this scale), and reading the evidence after
teaching does not keep supported inferences above unsupported ones. For GPU cg27a_rti2 this predicts no right-over-wrong separation
from evidence replay alone; its written inferences are familiarity-matched by construction (the B-C premise names C for right and wrong
inferences alike), so it is a clean test of the same question at scale.

### Standard baselines on the same trunk and schedule (14 November, overnight; local five-domain harness, 2 passes, seed 0; README open question 1)

platypus/baselines.py (off by default): `bl_plain 1` switches the system's own protections off (dreams, Fisher-whitened Muon, current-batch KL,
n-gram protection, reconstruction, conduit replay, write gates, the sleep layer); on top of that, online EWC (`bl_ewc 100`), stored-example
replay (`bl_er 1`, 16 raw batches kept per visit) or per-episode adapters (`bl_adapter 16`, trunk frozen after the first visit). The system
reference (SYS) runs the identical options with its protections on. SYS was stopped by the 2-hour job limit in phase 9 of 10, so the fair
comparisons are pass 1 (all five domains seen once) and the pass-2 revisits it reached.

| after pass 1 | mean BPC | forgetting since each visit | raw text kept |
|---|---|---|---|
| **system** | **2.797** | **-0.035** | none |
| stored-example replay | 2.821 | +0.199 | 16 batches per domain |
| EWC | 3.019 | +0.509 | none |
| plain fine-tuning | 3.287 | +0.785 | none |
| per-episode adapters | 3.285 | 0.000 | none |

Pass-2 revisits, BPC on the domain just before it is revisited minus just after its pass-1 visit (forgetting over the intervening domains),
then BPC after the revisit:

| | Wikipedia | PubMed | FreeLaw |
|---|---|---|---|
| system | -0.057 -> 2.518 | -0.007 -> 2.397 | -0.006 -> 2.193 |
| stored-example replay | +0.277 -> 2.521 | +0.014 -> 2.220 | +0.009 -> 2.080 |
| EWC | +0.651 -> 2.491 | +0.201 -> 2.187 | +0.138 -> 2.049 |
| plain fine-tuning | +0.856 -> 2.509 | +0.348 -> 2.218 | +0.232 -> 2.090 |

Final (after pass 2, all domains): plain 2.592 (forgetting +0.292), EWC 2.502 (+0.179), replay 2.481 (+0.135), adapters 3.172 (0.000); the
system's final figure is missing (stopped in phase 9). What this shows, on one seed and a small schedule: the system does not forget (the
only run below zero on every revisit) and matches stored-example replay on mean BPC after pass 1 without keeping raw text; it pays in
plasticity - each domain is learned less well at its visit (ArXiv 2.94 vs 2.57 for plain fine-tuning; PubMed revisit 2.40 vs 2.22), so on
the final figure it may trail replay and EWC. Not yet shown: the final figure, a second seed, the ten-domain GPU schedule.

Second seed of the baselines (seed 1): pass 1 mean BPC / forgetting - plain 3.400 / +0.924, EWC 3.102 / +0.611, replay 2.812 / +0.193,
adapters 3.289 / 0.000; final - plain 2.679 (+0.395), EWC 2.572 (+0.294), replay 2.498 (+0.094), adapters 3.159 (0.000). Seed spread on
the final figure is up to 0.09 BPC (plain), 0.02 for replay: the system's 0.02 lead over replay after pass 1 is within noise; its
forgetting gap (-0.035 against +0.19-0.20 for replay on both seeds) is not.

CONSOLIDATION OR REPLAY COUNT (README open question 3; new readout inf_tgrad_reps, JEPA-dreams run, two sleeps in pass 2): store-off
two-hop recall regressed on the conclusion's age and its trunk replay rows (standardised): age +0.27 / +0.14, log replays -0.13 / -0.05
(n 27 / 49; age and replays correlate +0.50 / +0.71). Recall from the weights follows age rather than replay count - consistent with
consolidation, but too few conclusions to rely on.

JEPA vs random dreams (README open question 2): runs are identical through pass 1 (the dream choice acts from pass 2). On the revisits
reached, JEPA dreams 2.517 / 2.377 against no dreams in the sleep replay (the system reference) 2.518 / 2.397: within noise (0.017).
The random-dreams arm could not reach pass 2 within the 2-hour job limit and was stopped: open.

### Second seeds of the label-free results (14 November, overnight; SPLIT_SEED 1 and 2 = new held-out documents and new train / test pair halves)

| result (held-out Re-DocRED gold) | seed 0 (13 Nov) | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| label-free extractor (consistency teachers -> learner), AUC / precision top 10% / 25% (base 0.24) | 0.627 / 0.45 / 0.38 | 0.617-0.625 / 0.42-0.45 / 0.36-0.37 | 0.625-0.634 / 0.46-0.47 / 0.37-0.39 | **robust** |
| universal schema, relation identity AUC (30k distant) | 0.658 | 0.649 | 0.658 | **robust** |
| typed composability top 10%, gold (base 0.105-0.108) | 0.148 | 0.122 | 0.126 | weaker: 1.15-1.4x base |
| + type coupling, top 10% | 0.160 | 0.120 | 0.130 | not robust |
| typed hop 2, composability-gated (vs hop 1 of the accepted) | 0.196 (0.160) | 0.174 (0.124) | 0.176 (0.135) | hop 2 above hop 1 in all three at 30k (not at full scale: 0.149 vs 0.144) |
| A-C confirmed by later reading in a relational wording | 0.359 | 0.363 | 0.360 | **robust: 3.4x base** |
| 2+ bridges | 0.082 | 0.077 | 0.077 | **below base in all three** |
| learned composability (wording kinds) top 10% | 0.115 | 0.097 | 0.094 | no gain |
| stand-in trunk congruence (gain from A, top 3 of 10) | 0.078 | 0.087 | 0.087 | below base in all three (the trunk's check points the wrong way) |
| hop 2 from confirmed inferences (hop 1 0.36-0.37) | - | 0.149 | 0.147 | inference on inference falls back to near base |

The one strong label-free filter for inferences is still waiting for the text: an A-C proposal that a later sentence states in a relational
wording is right 0.36 of the time against 0.105 for all proposals, on three splits. Everything computed from the premises alone (bridges,
composability, the trunk's congruence) is at or below base. Note for the GPU builds: cg27a_rti requires 2+ bridges; on Re-DocRED that
selects below base (0.077-0.082 vs 0.105), although on the distant set at GPU scale it was the best of the policies tried (0.178-0.188).

### Brain-inspired tests, round 2 (13 November, afternoon)

- **Name types without labels** (tools/name_types.py; names x contexts / own words / shape, PPMI, SVD, k-means): purity against gold types
  0.45-0.49 (NMI 0.13-0.29); as the relation cue (tools/relation_free.py RF_TYPES) the learned type pair gives AUC 0.50-0.57 alone and
  0.62-0.67 with the words between (words alone 0.63, name shapes 0.66; gold type pair 0.79-0.83). Not positive: these features do not recover
  the types that matter.
- **Spreading activation for inference** (tools/spread_infer.py; 80,185 verifiable A-C): top 10% gold - resource allocation 0.153, Adamic-Adar
  0.077, RA weighted by the learned extractor 0.218, + rarest bridge 0.221, three-step paths (L3, sample) 0.215 vs sample base 0.115; single
  rarest bridge 0.211. Not positive: spreading over the name graph adds nothing over the rarest shared name.
- **Event boundaries from trunk surprise** (tools/event_bounds.py; 400 unseen documents, 16,343 name pairs in one sentence, 22% related):
  boundaries at surprise peaks above the 75th percentile (22 per 100 words) keep related pairs together more than unrelated ones AT THE SAME
  DISTANCE (ratio 1.14 / 1.23 / 1.32 for pairs 1-3 / 4-7 / 8-20 words apart; random boundaries 0.94-0.99, fixed spacing 0.97-1.10,
  punctuation 0.80-0.93); a segment holding a pair is related 0.291 vs base 0.223. **Modestly positive**: the trunk's surprise marks where
  fact-bearing units start and end, beyond distance and punctuation.
- **JEPA codes on name-masked records** (tools/relation_free.py RF_JEPA=3000: a 2-layer encoder, 40% of the words masked, predictor to an EMA
  target encoder's states, L1, no negatives): relation identity AUC 0.568, nearest same relation 0.152 (chance 0.083); + name shape 0.677.
  Below plain n-grams of the same masked record (0.665). The L1 rose during training (0.85 -> 0.95): the target drifted; not positive here.
- **Overlapping replay** (tools/overlap_replay.py; the stand-in trained 1,500 more steps on pairs of passages from different documents that
  name the same entity, vs random pairs): relation identity from the trunk's state 0.581 (overlap) vs 0.578 (random) vs 0.582 (start); held-out
  bits per byte 2.444 (overlap) vs 2.256 (random) vs 2.192 (start). Not positive: replaying overlaps side by side taught no relational
  structure here and fitted the repeated names (held-out worse).
- **Schema-congruent consolidation** (tools/schema_consol.py; 100 new facts '<invented person> was born in <place>' mixed into ordinary text;
  recall = true object first of 10 objects of its arm): congruent (real places that follow 'born in') vs incongruent (organisations / works)
  at steps 50 / 100 / 200 / 400 - seed 0: 0.23 / 0.24 / 0.10 / 0.06 vs 0.05 / 0.05 / 0.05 / 0.09; seed 1: 0.14 / 0.19 / 0.29 / 0.35 vs
  0.07 / 0.10 / 0.05 / 0.16; held-out bits per byte the same in both arms (+0.05). Weakly positive and unstable (seed 0 collapses after
  step 100); the arms' distractor sets differ (places vs organisations), so difficulty may account for part of it.
- **Replay by need** (tools/paraphrase_test.py PT_NEED; 4 extra biography copies for the half of the read-only people who will be asked vs 2
  for every read-only person, same total), at step 4,000: the asked people in NEW question forms 0.33 vs 0.22; in the taught forms 0.55 vs
  0.61 (150 items each, about +-0.04). Need-weighted replay helps where generalisation is needed, at a small cost in the familiar form.
- **Prototype errors** (tools/paraphrase_test.py PT_ZIPF; values Zipf-skewed, the commonest 29% of facts): of the model's wrong answers about
  read-only people in new question forms, 0.14 (step 2,000: 0.15) are the commonest value - about 2.7x an even spread over the 19 wrong
  values (0.05), about half of a pure frequency guess (~0.3). A real but partial pull towards the prototype (Spens & Burgess): some false
  answers are typical values, which the gate must catch.
- **Usefulness as the teacher** (tools/usefulness.py; held-out records whose names meet again in another document, 1,370, related 0.304; LATER
  BENEFIT = bits the record saves the stand-in trunk on the second name there). Each policy keeps a quarter:

  | Policy | kept related | later benefit, raw | beyond the names alone |
  |---|---|---|---|
  | random | 0.313 | +0.089 | -0.021 |
  | closeness | 0.371 | +0.102 | -0.008 |
  | consistency-teacher learner | 0.576 | +0.057 | +0.006 |
  | trunk-feedback learner | 0.585 | +0.052 | +0.013 |
  | both (rank sum) | 0.596 | +0.046 | +0.006 |
  | **learned from raw later benefit** | **0.067** | **+0.160** | -0.035 |

  Optimising raw later benefit keeps almost no real relations (0.067): what saves the most bits later is names that merely co-occur. Usefulness
  must be measured beyond co-occurrence (there the relation-keeping policies lead, weakly). On records whose names recur, both label-free
  learners keep 0.58-0.60 related, about twice the base rate.

### The surprised store: keep what the trunk does not already predict (tools/surprise_store.py, 13 November; SLIMM)

The stand-in trunk (192 wide, 6,000 steps; never saw these documents) reads 600 Re-DocRED training documents (4,798 sentences, 2.16 bits per
byte); every record (two names in one sentence, words between) gets the trunk's surprise; 'already knows' = the true second name costs the
fewest bits among up to 10 names of the document (related records 0.161 vs chance 0.101: this trunk knows little).

| Keep a quarter of 12,459 records by | related | recall | of kept related facts, trunk does NOT know | NEW true facts per 100 kept |
|---|---|---|---|---|
| random | 0.246 | 0.238 | 0.876 | 21.6 |
| **surprise over the record** | **0.292** | **0.282** | **0.926** | **27.0** |
| surprise on the second name | 0.250 | 0.242 | 0.966 | 24.2 |
| surprise on the words between | 0.241 | 0.233 | 0.856 | 20.6 |
| predictable record (least surprise) | 0.229 | 0.221 | 0.714 | 16.3 |
| closeness of the names (no trunk) | 0.284 | 0.275 | 0.865 | 24.6 |

- **Surprise-gated storage keeps 25% more new true facts per record than random (27.0 vs 21.6) and 66% more than keeping the predictable
  (16.3)**: it keeps more real relations and, among them, more the trunk does not already know.
- Sentences: the most surprising quarter hold 1.25 related records per sentence, the most predictable 0.55 (partly because surprising sentences
  hold more names). Surprise on the second name alone best avoids storing what the trunk knows (0.966) but does not pick relations.
- The stand-in knows few facts (0.16), so 'already known' filtering matters little here; the model's trunk knows more (known-fact recall
  0.53-0.78 on the GPU), where this gate should save more.

### Brain-inspired tests: relation identity from name-free codes; inference at reading time (13 November)

**Relation identity** (tools/relation_free.py): records with both names masked ('[X] , the capital of [Y]', plus 3 words each side), Re-DocRED
held-out documents, 600 queries x 30 candidates from other documents, 187 gold relation types.

| Cue | nearest candidate same relation (chance 0.08) | AUC same vs not |
|---|---|---|
| frame words (tf-idf) | 0.16-0.19 | 0.63-0.64 |
| word + character n-grams of the masked record, untrained | 0.20-0.23 | 0.65-0.67 |
| label-free encoder (InfoNCE: same two names in another document = positive) | 0.16 | 0.54-0.56 |
| stand-in trunk's state over the masked record | 0.17 | 0.58 |
| (earlier) trunk substitution, tools/relsub.py | - | 0.58-0.60 |
| name SHAPE pair alone (digits / capitalised) | 0.16 | 0.67 |
| frame words + shape pair | 0.21 | 0.73 |
| **GOLD entity-type pair alone** (PER-LOC, ORG-TIME ...) | **0.38** | **0.83** |
| frame words + gold type pair | 0.37 | 0.85 |

- **The types of the two names carry most of relation identity** (0.83 alone), as unsupervised relation extraction found (Tran, Le &
  Ananiadou 2020). Learning name types without labels is the most promising route to relation identity.
- Same-name pairs as label-free positives teach the wrong thing (0.54): only 29% of such pairs share a relation.

**Inference at reading time** (tools/integrate_test.py): every name pair in a sentence is a stored record; a record B-C recalls the A-B records
of other documents and proposes A-C (hub names in 200+ documents skipped). 2.2M proposals, 80,185 verifiable (A and C named in one document).

| Proposals | gold-related |
|---|---|
| base rate: name pairs in one document | 0.092 |
| all verifiable proposals | 0.108 |
| top 10% by the rarest bridge | 0.211 |
| top 10% by the learned extractor (both records) | 0.186 |
| 2+ / 3+ / 5+ distinct bridges | 0.081 / 0.067 / 0.062 (more bridges = hub-like names, worse) |
| **ceiling: both premise records truly related** | **0.480** (6,856 proposals; neither related: 0.073) |

- **Integration without rules is sound when the premises are true**: two true records sharing a name give a true A-C relation half the time,
  5x the base rate, with no rule and no relation label. The bottleneck is premise quality (extraction), not the inference step.

### cg27a_xgain: extraction gain from the model's trunk (GPU, seed 4, 13 November; visits 2-6)

cg27a_rtxw + inf_extract_gain 1 (readout only). Candidate records = gold-mention name pairs in one sentence of the benchmark documents read;
gain = bits a record saves the trunk where its names meet again, beyond the names alone. Closeness column from the pre-fix (tie-biased) AUC.

| Visit | records (related) | words between AUC | other-document targets | whole record | names only | top quarter related |
|---|---|---|---|---|---|---|
| 2 | 119 (0.31) | 0.48 | 0.58 (64) | 0.43 | 0.43 | 0.28 |
| 4 | 301 (0.32) | 0.48 | 0.50 (210) | 0.50 | 0.50 | 0.29 |
| 6 | 444 (0.32) | 0.54 | 0.55 (315) | 0.50 | 0.47 | 0.35 |

- The model's trunk saves 1.2-1.7 bits per record, equally for related and unrelated pairs; what the relating words add is at most weakly
  related to relations (0.54 at visit 6, about one standard error above chance) - the stand-in trunk's level (0.54). Not a usable teacher alone.
- Safe: BPC within 0.011 of cg27a_rtxt at visits 1-6; the gate cuts never-read false answers 0.60 -> 0.29 and 0.64 -> 0.20 (visits 5, 6).
- Domain-specific counting for the consistency teacher (tools/consistency_raw.py CR_PERDOM=1): Re-DocRED + WikiText 0.547 -> 0.563; all
  domains still chance (0.49); WikiText alone 0.53. The teacher works when it counts over text very like the target (Re-DocRED itself: 0.59).
- ENTITY NEIGHBOURHOOD (CR_NEIGH=K: count only over the K pool documents sharing the most distinctive names with each held-out document,
  names by capitalisation; pool = every domain): K 5 / 20 / 50 -> AUC 0.51-0.54, top 10% 0.31-0.35 (all domains unselected 0.49 / 0.29;
  Re-DocRED alone 0.59 / 0.45); without the Re-DocRED training documents in the pool, K 20: 0.53-0.54. Neighbourhoods pull in PubMed and law
  documents through shared capitalised words. A small gain over mixing, far short of a matched collection.

### The store teaching itself what to extract: consistency teachers (tools/consistency_teacher.py, 13 November; no trunk)

Same candidates and held-out documents as tools/extract_learn.py (15,609 held-out candidates, 24.2% related). Label-free teachers counted on the
training documents: T1 the same names and wording in another document; T2 FUNCTIONAL wording (a subject joined to one object); T3 SELECTIVE
wording (the shapes of the names it joins); T4 records in a chain of a rule the counting miner verifies over the candidates themselves.

| Extractor (held-out documents) | AUC | top 10%: precision / recall | top 25%: precision / recall |
|---|---|---|---|
| random | 0.50 | 0.25 / 0.10 | 0.24 / 0.24 |
| learner from TRUNK feedback (tools/extract_learn.py) | 0.55-0.56 | 0.33-0.35 / 0.14 | 0.31 / 0.32 |
| T2 + T3 applied directly (wordings seen in training) | 0.54 | 0.41 / 0.17 | 0.30 / 0.31 |
| **learner from the combined consistency teacher** | **0.63** | **0.45 / 0.19** | **0.38 / 0.39** |
| + self-training (the learner's kept records recount T2 / T3), round 1 | **0.64** | 0.46 / 0.19 | 0.39 / 0.40 |
| rounds 2-3 | 0.63 | 0.45 / 0.18 | 0.38 / 0.39 |
| ceiling: same learner, gold labels, all training candidates | 0.81 | 0.71 / 0.29 | 0.55 / 0.57 |

- **Consistency is a better teacher than the trunk's prediction**: AUC 0.63 vs 0.55; top quarter 0.38 related vs 0.31 (base 0.24). On
  training records the strongest single teacher is FUNCTIONAL wording (top quarter 0.42); corroboration (0.26) and rule chains (0.26; the
  counting miner finds 1 rule, punctuation, over candidate records) add little.
- Self-training helps once (+0.01 AUC) and then plateaus.
- Roughly: keeping the top quarter captures ~40% of the within-sentence related pairs (~8% of all gold facts) at 0.38-0.39 precision,
  against the current reader's ~3% at ~0.23.
- **More text, better teacher** (CT_FRAC: share of the training documents read; same held-out test): learner AUC 0.49 / 0.56 / 0.60 / 0.63,
  top 25% 0.24 / 0.31 / 0.33 / 0.37 at 10% / 25% / 50% / 100% (6,500 / 16,000 / 32,000 / 63,000 training candidates); functional-scored
  wordings 8 / 17 / 20 / 43. About +0.03-0.04 AUC per doubling, not levelling off.
- **On raw text, as the store sees it** (tools/consistency_raw.py: names by capitalisation, no gold mentions; same held-out test): counted over
  the Re-DocRED documents alone AUC 0.59, top 25% 0.35 (gold mentions 0.63 / 0.37); **adding other text makes it worse** - + WikiText 0.55 /
  0.29; + every domain the model reads (10.5M words) 0.49 / 0.25. The volume gain holds for more of the SAME kind of text only; counted over
  everything the model reads, this teacher is near chance unless it counts per domain.
- **Do rules come out of what it keeps?** (tools/kept_rules.py; counting miner, n >= 20, Wilson z 4 >= 0.7) Top 10% / 25% / 50% kept: 3,208 /
  6,407 / 10,685 of 93,331 gold facts covered (precision 0.45 / 0.36 / 0.30). With GOLD relation types on the kept related records: 0 / 0 /
  **1 rule** (a region's country from the places in it, 0.90 of 102; its proposals 0.74 gold - facts the extractor missed). With the
  WORDING as the relation (what the store has): 0 rules at every share. All gold facts: 136 rules. Inference by counting needs both several
  times this coverage and relation identity; relation identity is the larger gap.
- **The trunk's feedback adds nothing to consistency** (tools/extract_combo.py): consistency alone AUC 0.628 / top 25% 0.375; + trunk
  feedback at weight 1 / 2 / 4: 0.622 / 0.595 / 0.577 (top 25% 0.363 / 0.352 / 0.338).

### Several wordings: does a read-only fact become answerable? (tools/paraphrase_test.py, 13 November; byte LM 128 wide, 3 layers)

200 invented people (birth city, university, employer; 20 values each, chance 0.05), each biography read 5 times; questions taught about half
(A), the other half (B) only read. Multiple choice by bits. SINGLE: one fixed wording; MULTI: 5 wordings per attribute, shuffled order.

| After 10,000 steps | SINGLE | MULTI |
|---|---|---|
| storage (biography wording), B | 1.00 | 0.93 |
| B, read only, the taught question forms | 0.64 | 0.60 |
| A, question forms never trained | 0.47 | 0.58 |
| **B, read only, question forms never trained** | **0.30** | **0.39** |

- A trunk-sized byte LM taught questions about some people answers about people it only READ (0.60-0.64): learning from reading is not an
  architectural limit; the model's 0.12-0.19 never-asked recall reflects how real facts are seen (once or twice, long sentences, pronouns).
- The weak cell is NEW QUESTION FORMS (0.30-0.47): the model binds question shapes to stored facts. Several fact wordings lift it modestly
  (+0.09-0.11) and leave read-only extraction in the taught forms unchanged; plateau by step 4,000 in both.

### cg27a_rtxt: title for pronoun in real-text replay, to visit 10 (GPU, seed 4, 13 November)

cg27a_rtx16 + inf_distill_rtx_title 1 (pronoun-led replayed sentences replayed with the document's title for the pronoun). BPC within 0.006 of
cg27a_rtx16 throughout; peak GPU memory 55.5 GB; the only errors the old NAME MEMORY READOUT 'stale' (caught; also in earlier runs).

| visit | named: storage / own / another wording / prior | PRONOUN-LED: storage / own / ANOTHER wording / prior | control named: own / prior |
|---|---|---|---|
| 5 | 0.02 / 0.27 / 0.21 / 0.17 | 0.02 / 0.15 / 0.15 / 0.15 | 0.10 / 0.04 |
| 6 | 0.12 / 0.33 / 0.21 / 0.21 | 0.02 / 0.19 / **0.17 / 0.08** | 0.06 / 0.10 |
| 7 | 0.08 / 0.27 / 0.15 / 0.17 | 0.04 / 0.31 / **0.23 / 0.19** | 0.08 / 0.08 |
| 8 | 0.06 / 0.35 / 0.12 / 0.17 | 0.06 / 0.29 / **0.21 / 0.17** | 0.08 / 0.04 |
| 9 | 0.10 / 0.46 / 0.33 / 0.31 | 0.08 / 0.38 / **0.21 / 0.15** | 0.08 / 0.12 |
| 10 | 0.08 / 0.27 / 0.19 / 0.17 | 0.04 / 0.44 / **0.25 / 0.10** | 0.12 / 0.15 |

**Real sentences stored (named storage 0.06-0.12 from visit 6, control 0; replicates cg27a_refuse) and known in their own wording; named
facts do not reach other wordings (+0.02 at most). Pronoun-led facts replayed with the title DO: above their prior in another wording at
visits 6-10 (+0.09, +0.04, +0.04, +0.06, +0.15; 48 sentences each).** One run.

**END OF RUN - A REAL COST: forgetting +0.077, final BPC 1.807 (weights only), against cg27a_replay seed 4 +0.026 / 1.767 (same seed, no
real-text replay; seed 5 +0.034 / 1.775).** Per domain: Gutenberg +0.224 (vs +0.009), StackExchange +0.312 (vs +0.083); the rest within
about 0.03. The benchmark documents (Wikipedia text) are split across domains 0-4 and replayed through THOSE domains' slots, 16 sentences a
step at every later visit - Gutenberg's slot replays 53 Wikipedia documents. The 16-per-step real-text replay was never run to the end
before (cg27a_rtx / rtx16 stopped at visits 7 / 4) and the local safety runs cannot show it (local replay starts at visit 6, tiny text).
Not attributable to the title change itself (it only rewrites pronoun-led sentences); a cg27a_rtx16 run to visit 10 would separate them.

GATED ANSWERS (threshold from the previous sleep, out of sample): never-read given a stored name ungated -> gated 0.54 -> 0.45, 0.65 -> 0.25,
0.55 -> 0.39, 0.59 -> 0.24, 0.65 -> 0.34, 0.58 -> 0.14, 0.64 -> 0.15, **0.57 -> 0.12** (visits 3-10); known right overall 0.36 -> 0.36, 0.39 ->
0.22, 0.52 -> 0.47, 0.58 -> 0.47, 0.59 -> 0.52, 0.53 -> 0.44, 0.62 -> 0.41, **0.64 -> 0.52**; the threshold climbed 0.14 -> 0.53 with the
trunk's confidence. Confidence AUC known-right vs never-read 0.68 -> **0.90** (visit 10). Learning from reading: never asked 0.16 at visit 10.
COMPRESSION GAIN, links: 0.770 / 0.782 / 0.796 / 0.839 / 0.823 (visits 2-10; same entity 3.9 vs 0.3 bits at visit 8); units: PMI 0.59-0.64,
binding gain 0.41-0.47 (none). Real-text relational module: 19,390 facts, 0 rules.

### Rules from real text with today's reading pieces - and the oracle (tools/realtext_mine.py, 13 November)

262 benchmark + 630 local Wikipedia documents, 181,872 records; the relational module (_rm_mine, every guard) over the 60 commonest frames.
| variant | facts mined | testable candidate chains | rules |
|---|---|---|---|
| R0 the reader as it stands (title rule for pronouns included) | 12,333 | 91 (best: glue words, 'from' o 'and' -> 's' 0.25 of 4) | 0 |
| R1 + linking mentions (3,678 short mentions joined to an earlier full name; errors like 'London' -> 'College London') | 12,103 | 45 | 0 |
| R2 + merging wordings that connect the same name pairs | - | **0 merges** (no two frames share 2+ pairs at Jaccard >= 0.15) | 0 |
| **ORACLE: gold relation types as frames** | **678** (records on gold pairs; 54 types: P131 148, P19 57, P27 56, ...) | **0** (no chain with 3+ observed composite facts) | **0** |

**Even with relation identity solved, this much text gives no rule**: the reader puts only 678 records on gold pairs, too few for any chain
to be observed three times. The blocker on real text is the NUMBER of facts read - reader recall (5.5% of gold relations) times corpus size
- before relation identity. Merging wordings by shared pairs found nothing for the same reason (pairs rarely recur across documents).

### cg27a_rtxt local safety (13 November): title for pronoun in real-text replay (inf_distill_rtx_title 1)

Short local run (RTXTL2; the first, RTXTL, was cut at visit 10 by a container restart - visits 1-9 within 0.001 BPC of RTX16L): final BPC
**2.998 / forgetting -0.156** vs RTX16L 2.996 / -0.158 (seed noise ~0.017); FAILURES none, no errors; 221 pronoun-led sentences replayed with
the title at the last visit. Safe.

### Overnight offline tests (12-13 November; tools/overnight/; Re-DocRED dev, 500 documents; stand-in byte LM 128 wide, 4,000 steps, 2.10 bits/byte, and the same network untrained)

**Compression gain, stand-in trunk (tools/compress_gain.py):** units - binding gain at joins inside capitalised runs AUC 0.437 (untrained
0.501, surprise 0.513; 8,702 joins), spans unchanged (F1 0.696 vs 0.698). Links - 119,310 candidate pairs (12 nearest earlier mentions),
4,439 same entity: raw gain AUC **0.533** (HARD 0.485, VERY HARD 0.477) vs untrained 0.514 / 0.524 / 0.553; string features 0.895
(HARD 0.446); string + gain 0.941 vs string + untrained gain 0.946. **At this size the trunk's compression gain carries no linking signal
- unlike the GPU trunk (0.756 all, 0.61 -> 0.68 on pairs sharing no word, visits 2 -> 4).** The linking signal appears with a stronger
trunk; the larger stand-in (stage C) tests the trend.

**One relation, many wordings - by substitution (tools/relsub.py):** 3,262 one-sentence gold-relation records (90 types); 600 queries
against 30 random candidates each. Best candidate of the same gold type: frame words 0.125, substitution with the trained trunk 0.050,
untrained 0.060 (chance 0.060). AUC same type vs not: substitution trained **0.584**, untrained 0.503, frame words 0.368 (tf-idf cosine is
mostly 0, so its ranking is ties), words + trained 0.609. **A weak signal at this size: the trained trunk ranks same-relation frames a little
higher overall but never picks the right one first; frame words still pick better.**

**Pronoun reference by substitution (tools/pronoun_ref.py):** 122 pronoun-led sentences with a single proxy-gold antecedent among 2-8 earlier
entities (He 55, It 36, She 15, ...; chance 0.205). TITLE (the document's first entity) **0.549**, most mentioned 0.369, type rule 0.410,
most recent 0.148; substitution with the trained trunk 0.197, untrained 0.213 (He 0.11, She 0.07, It 0.25). **The stand-in trunk cannot
resolve pronouns; the title heuristic resolves more than half** (Wikipedia openings are about their title).

**The same two tests on a LARGER stand-in (192 wide, 6,000 steps, 1.91 bits/byte vs 2.10):** relation identity by substitution - AUC 0.578
(was 0.584), best candidate 0.062 (chance 0.060), untrained 0.512; pronoun reference - substitution 0.238 (was 0.197), untrained 0.238,
title 0.549. **No gain from the larger stand-in on either.** Its trunk-guidance and compression runs: boundaries unchanged (surprise 0.559; sentence-initial
0.862 vs the corpus statistic 0.978); linking by states still no better than untrained (0.930 vs 0.934; HARD 0.682 vs 0.702); units by
binding gain 0.423 (no signal); **links by compression gain 0.564 (HARD 0.574, VERY HARD 0.583) vs the small stand-in's 0.533 / 0.485 /
0.477 and untrained 0.501 / 0.530 / 0.543.** Across three trunks the linking signal from compression gain rises with the trunk: 2.10 bits/byte
0.53 -> 1.91 bits/byte 0.56 -> the GPU trunk 0.76 (pairs sharing no word 0.49 -> 0.57 -> 0.61-0.68). Nothing else does.

**Overnight summary.** (1) Compression gain is the one trunk signal that improves with the trunk, and only for linking mentions; not for
units, relations or pronouns. (2) Name units: counts (PMI 0.73) beat every trunk measure (0.42-0.59), but capitalised runs are already
right where it matters (span F1 unchanged). (3) Relation identity by substitution: weak (AUC 0.58), not improved by a larger stand-in.
(4) Pronouns: the stand-ins are at chance; the document's title resolves 0.55. (5) Refusal rows: the syllable shortcut explains cg27a_refuse;
same-generator names make refusal real but suppress facts only read (0.62 -> 0.15), so the confidence threshold is the safer route to
abstention. (6) The default readouts (compression gain, probe 64, gated answers) are safe: identical BPC locally.

**Refusal: shortcut or transfer? (tools/toy_refuse.py, toy byte transformer, 3,000 steps, seed 0):** facts about invented people, question
rows for 3/4 of them (the rest only read), refusal rows about fresh names answered 'unknown'. Greedy answers; refusal rates on fresh names
from the stored names' syllables / from a third syllable set / on KNOWN facts; known right; only-read right:

| condition | TAUGHT wording: refuse fresh (stored syl.) / fresh (third) / known | known right | ONLY-READ right | HELD-OUT wording: refuse fresh (stored) / (third) / known |
|---|---|---|---|---|
| none | 0.00 / 0.00 / 0.00 | 1.00 | **0.62** | 0.00 / 0.00 / 0.00 |
| distinct syllables (as cg27a_refuse) | **0.03** / 0.50 / 0.00 | 0.99 | 0.55 | 0.15 / 0.55 / 0.16 |
| same syllables | **0.85** / 0.87 / 0.01 | 0.97 | **0.15** | 0.73 / 0.90 / 0.60 |
| same + 12 wordings | 0.87 / 0.95 / 0.01 | 0.99 | **0.05** | 0.87 / 0.83 / 0.60 |

**The shortcut is real**: with refusal names from their own syllables, the model refuses names that SOUND unfamiliar (third set 0.50) but
almost never a never-read name that sounds like a stored one (0.03) - what cg27a_refuse showed (3-5%). With names from the same generator,
refusal becomes genuine (0.85 on never-read names, 0.01 on known facts). **But it then refuses facts it only READ: only-read recall
0.62 -> 0.15 (0.05 with more wordings)** - the rows teach 'a name I was not ASKED about -> unknown', not 'a name I never READ'. In the
held-out wording it also refuses known facts it cannot answer there (0.60; known right there only 0.06-0.30). Explicit refusal rows
therefore conflict with learning from reading; the confidence threshold (no training change) does not teach this confusion.

**Local check of the new default readouts (GATEL = the 4x real-text run + compression gain every 2nd visit, probe 64, gated answers):**
final BPC 2.996 / forgetting -0.158 - identical to the run without them (readouts only); no errors, FAILURES none. GATED ANSWERS (threshold
calibrated at the previous sleep, applied to the new questions) at visits 8-10: never-read questions given a stored name 0.35 / 0.28 / 0.31
ungated -> **0.17 / 0.15 / 0.15** gated; known facts right overall 0.18 / 0.16 / 0.14 ungated -> 0.05 / 0.11 / 0.08 gated (the small local
model knows few facts, so the gate costs much of what it knows; the GPU, where known-right confidence is 0.59-0.66, is the real test).

### Refusal rows, replay by storage check, abstention, compression gain on the real trunk (cg27a_refuse to visit 6, cg27a_rtxsp to visit 4; seed 4; 12 November, night; both stopped by the author)

Both = cg27a_rtx16 + ONE change: `cg27a_refuse` - a fifth of the question rows ask about FRESH names never read (taught wordings), answered
'unknown' (inf_distill_refuse 0.2); `cg27a_rtxsp` - real-text replay probability ~ the trunk's own loss on each sentence squared
(inf_distill_rtx_space 2), with the COMPRESSION GAIN readout. Local safety (short runs): final BPC 3.000 / forgetting -0.154 (refuse) and
3.000 / -0.153 (rtxsp, alpha 1) vs 2.996 / -0.158; GPU BPC within 0.006 of cg27a_rtx16; no errors.

REAL TEXT, replayed documents, named sentences (48): storage / own wording / ANOTHER wording (template) / fresh-subject prior

| visit | cg27a_refuse | cg27a_rtxsp | cg27a_rtx16 | control documents (refuse): own wording / template / prior |
|---|---|---|---|---|
| 3 | 0.02 / 0.19 / - / 0.06 | 0.02 / 0.12 / - / 0.10 | 0.02 / 0.12 / 0.08 / 0.06 | 0.06 / - / - |
| 4 | 0.04 / 0.27 / - / 0.12 | 0.08 / 0.35 / 0.23 / 0.23 | 0.06 / 0.23 / 0.08 / 0.08 | 0.06 / - / - |
| 5 | 0.02 / 0.27 / - / 0.17 | - | - | 0.06 / - / - |
| 6 | **0.19 / 0.33 / 0.17 / 0.10** | - | - | 0.10 / 0.08 / 0.06 |

**At visit 6 real sentences are STORED (9 of 48 completed; control 0) and answered in a wording they never used above the prior (0.17
vs 0.10; pronoun-led 0.12 vs 0.06; control at its prior)** - the planted world's order (storage, own wording, new wording) on real text,
one visit, small numbers. Replay weighted by the trunk's loss lowered that loss fast (median 1.89 -> 1.06 by visit 4, hardest quarter
-1.4 then -0.5 nats; ~0.33 of replay to the hardest quarter) but did not store faster by visit 4: loss-hard sentences are not the
unstored ones.

FALSE MEMORIES AND ABSTENTION (from the trunk; 64 questions per group): fresh name answered with a stored name - refuse 0.56 / 0.71 /
0.56 / **0.69** at visits 3-6 vs cg27a_rtx 0.61 / 0.69 / 0.60 / **0.87**; known subject, missing relation 0.47 / 0.58 / 0.34 / **0.33**
vs 0.58 / 0.47 / 0.48 / 0.44. Explicit 'unknown' in the untaught test wording only 0.03-0.05 (refusal-row loss 0.26 -> 0.05): refusal
does not transfer as a word, but false answers fall. **A confidence threshold keeping 75% of the known facts answered right leaves 0.12-0.18
false answers (0.45-0.64 without; refuse, visits 4-6) and 0.23-0.25 without refusal rows (rtxsp)**; confidence AUC known-right vs
never-read 0.86-0.88 (refuse), 0.80-0.84 (rtxsp). Over-refusal of known facts 0.00-0.02. No cost to learning: taught 0.57-0.60, never
asked 0.09-0.12 vs cg27a_rtx 0.57-0.60 / 0.05-0.09.

COMPRESSION GAIN ON THE REAL TRUNK (rtxsp; Re-DocRED gold mentions of the domains read): links - an earlier mention saves bits on a later
one of the same entity (1.88 vs 0.53 bits), AUC all pairs 0.756 / 0.758 at visits 2 / 4; pairs sharing NO word **0.611 -> 0.684**, very
hard 0.622 -> 0.657, where letter overlap is 0.17-0.22; combined with letters 0.897 (visit 2) vs letters alone 0.862; antecedent choice
worse than letters (0.59 vs 0.88). Units - binding gain at joins of capitalised words 0.492 / 0.516 (surprise 0.571 / 0.591): no signal.
Offline, counts: PMI separates a name's continuation from a boundary at 0.730 (strict MDL with a spelling cost 0.496), but splitting
capitalised runs by it does not change exact-span F1 (0.698; tools/mdl_units.py).

### Finding names: a text JEPA vs byte surprise (tools/toy_jname.py, tools/realtext_names.py; 12 November)

TOY (synthetic documents, unseen test names; EMA-target JEPA; AUC name vs other words): training wordings, first mention - byte surprise
1.00, low JEPA surprise 0.68, gap 0.96; held-out wording - byte 0.52, JEPA 0.60, gap 0.55; swap test 0.39-0.53 (fails); identity of a name
across documents: byte LM hidden state 0.66, JEPA latent 0.46 (chance 0.02). (The LeJEPA variant was cut off by the time limit.)
REAL TEXT (byte LM 2.12 bits/byte and a LeJEPA-style JEPA, trained on local Wikipedia + the benchmark; 39,163 words of the benchmark
documents, 25% part of a gold entity name): **capitalised-not-first 0.93, byte surprise 0.69, low JEPA surprise 0.48 (no signal), gap
0.65**; on the 2,355 gold-name words capitalisation cannot see (lower-case or sentence-initial): byte surprise 0.58, JEPA 0.46, gap 0.55.
**The JEPA does not find names on real text**; byte surprise carries a weak signal for the names capitalisation misses (many of those
'names' are common nouns inside entity names: 'University', 'Party'). Name finding beyond capitalisation needs another route.

### Can the trunk guide the store? Boundaries and mention linking (tools/trunk_guides.py, 12 November; Re-DocRED dev, 500 documents, 13,189 gold mentions)

After EM-LLM (episode boundaries at the model's own surprise) and hippocampal indexing (memory keyed by the cortex's states). Stand-in trunk:
a byte LM trained here on local Wikipedia + the documents (128 wide, 3 layers, 4,000 steps, 2.10 bits/byte); control: the same network
untrained. Thresholds and combinations fitted on documents 1-250, scored on 251-500.

| test | baseline (the reader's own cues) | trained trunk | untrained network | combined |
|---|---|---|---|---|
| joins inside capitalised runs: same mention vs boundary (AUC) | - | 0.565 | - | split threshold fitted: none (exact-span F1 0.698 unchanged) |
| sentence-initial capitalised word starts a mention (AUC) | corpus mid-sentence capitalisation **0.978** | 0.864 | - | 0.970 (worse than the baseline) |
| same-entity mention pairs, all (AUC) | words + letter triples 0.904 | 0.927 | **0.937** | string + trunk 0.949, string + untrained 0.952 |
| pairs sharing no word (1,221) | 0.472 | 0.684 | **0.719** | 0.739 vs 0.757 |
| pairs sharing no word and few letter triples (932) | 0.322 | 0.612 | **0.649** | 0.669 vs 0.692 |
| antecedent right (1,759 mentions) | 0.932 | 0.906 | 0.902 | 0.936 vs 0.941 |

**At this scale the trunk adds nothing the store's own cues or plain byte form do not already give.** Boundaries: its surprise barely
separates a name's continuation from a new mention, and the corpus capitalisation statistic beats it on sentence-initial names. Linking:
the TRAINED states never beat the UNTRAINED network - what the states carry is byte form (shared capitals, as in acronyms), not learned
identity; the remaining hard pairs need world knowledge ('AGF' / 'Asian Games Federation', 'Osaka Bay' / its Japanese name, 'Fatih Terim'
/ 'Imparator'). Not tested: pronouns and descriptions ('she', 'the company') - Re-DocRED annotates named mentions only. Caveats: a small
stand-in trunk (the GPU trunk reaches ~1.7 bits/byte); the guidance would have to grow with the trunk, and with 'guidance only adds' the
floor stays at today's reader.

### Reference ceiling on real text (tools/reference_ceiling.py, 12 November; Re-DocRED benchmark documents, names = fast-channel fillers)

Gold relations whose two names are found in ONE sentence: no reference 435 (9.2%); + the title in sentences with a third-person pronoun or
opening with frame material **614 (13.0%)**; + the title in every sentence 622 (13.2%); + the previous sentence's subject 537 (11.4%); both
618. Upper bound - both names found anywhere in the document: **857 (18.1%)**, 41% of them with the title as one argument. Reference by
the title lifts the one-sentence ceiling by 40% (9 -> 13%); cross-sentence pairs beyond it add at most 5 points; **the larger limit is
recognising the names at all** (82% of gold relations have a name the fast channel never isolates as a filler: common-noun entities,
dates and numbers, lower-case names).

### Toy: paraphrase alignment closes the paraphrase gap (tools/toy_para_align.py, seed 0, 6,000 steps)

3 relations x 2 phrasings; every fact declared in phrasing a, 20% also in phrasing b; 420 of 600 taught as questions in phrasing a only.
Unseen question prefixes:

| alignment | taught: phrasing a / PARAPHRASE b | never taught: phrasing a / PARAPHRASE b |
|---|---|---|
| off | 0.92 / 0.76 | 0.67 / 0.41 |
| as built (question a -> declaration a, taught) | 0.92 / 0.83 | 0.82 / 0.61 |
| **+ paraphrase (question b -> declaration a, taught)** | 0.93 / **0.93** | 0.82 / **0.82** |
| + paraphrase, never-asked facts too (label-free) | 0.92 / 0.91 | 0.80 / 0.80 |

Aligning the paraphrased question of TAUGHT facts with the stored wording's declaration removes the paraphrase gap for facts never taught
(0.61 -> 0.82, equal to the stored wording): the paraphrase is learned as a relation-level mapping, not per fact. Aligning the never-asked
facts as well adds nothing here. For Platypus: the relational module's paraphrase merges (always correct on the GPU) give the pairs.

### Real-text views for label-free alignment (tools/realtext_views.py, 12 November; Re-DocRED benchmark documents)

A view = (question, declaration) cut from one sentence by the fast channel: subject (the title supplied for pronoun- or frame-led
sentences) + the frame before an object filler. 5,696 views in 262 documents (21.7 per document): adjacent 2,991, anchored with the
title supplied 1,986, anchored and named 719. **Gold relations with a view: 342 of 4,726 (7.2%)** - the same ceiling as the reader (one
sentence holds both names), 126 of them only through a title-supplied view; only 34 have views in 2+ frames. **Paraphrase load: a relation
type arrives in about as many frames as it has relations** (median 1.00 distinct frames per relation; located-in 63 frames / 73 relations,
birthplace 23 / 34: 'was born in', ', born in', '( born in', 'in [F] ,' ...). Label-free alignment on ordinary text is bounded by one-sentence
views (~7%) unless cross-sentence reference grows, and its test questions in another wording need frames merged into relation types.
Label-free alignment does not need the view to be a TRUE relation (it aligns a question with what the text says), so the reader's precision
matters less here than its coverage.

### Reader tests 4-6 (tools/reader_tests3.py, 12 November; subject-anchored + cleaned records, local Wikipedia + the benchmark, Re-DocRED gold)

Reference (M2 + M3, benchmark records): gold rate 0.43, 342 gold relations covered.

| test | result |
|---|---|
| M5 relation properties as filters: type signature / functional / irreflexive (name fragments) / all three | 0.44 (random drop 0.44) / 0.43 (87 "functional" frames, all glue: '(', 'in the') / **0.45 at no coverage loss** / 0.46 at 297 covered |
| M4 tensor factorisation as a filter (keep the best-explained 75 / 50 / 25%): no constraints | 0.44 / 0.45 / 0.43 |
| ... + inverse and paraphrase constraints (19 paraphrase, 1 inverse pair mined) / + tied to frame clusters | 0.43 / 0.45 / **0.47** ; 0.43 / 0.45 / **0.48** (155 covered) |
| M4 completion (each title's best unread object per frame, among its document's fillers) vs a random document filler | 0.22 / 0.30 / 0.33 vs 0.16-0.19; 7-9 new gold relations |
| M6 rules: anchored frames as read / pooled into 40 clusters / cluster x type pair | **0 rules** in each; 93-95% of candidates have no observed chains; best testable predicts 0.12-0.33 |

Constraints help only at heavy filtering (+0.05 at a quarter of the records); completion beats chance but adds <10 relations; real text
still yields no rules at this scale. The name-fragment filter is the one free gain. Reader work stops here (see the alignment route).

### Option B, stages 1-2 offline: relation codes from a byte LM's states, rules over them (tools/relcodes.py; 12 November) - NO GO at this scale

130,798 records (anchored + cleaned adjacent; local Wikipedia + the benchmark), 97,322 distinct frames. A byte LM (2.12 bits/byte) reads
'Someone <frame> '; a projector trained contrastively on 66,301 same-name-pair positives (Matching-the-Blanks on the model's states). The
contrastive loss barely moves (5.29 -> 5.10; chance ~5.5): same-pair records rarely share a relation (list items, co-mentions).
STAGE 1 (425 benchmark gold-pair records; nearest record shares a gold relation type): majority 0.20, **frame words 0.54**, raw LM state
0.41, **trained relation code 0.40**, code + words 0.55; 48 clusters, gold-type purity 0.36. Clusters are sorted mostly by the frame's last
word ('in' / 'was born in' / 'died in'; 'to' / 'returned to' / 'moved to'). STAGE 2: rules over the 48 clusters - 9,431 candidates, 1,403
testable (vs 49 over frame strings: pooling does gather evidence), functional accuracy median 0.03, best 0.37 of 27 - **0 rules**; over frame
strings 0 rules (best 0.22). The learned code does not beat frame words; neither supports a rule. Needs a stronger trunk (or more text)
and cleaner positives before option B can go further.

### POWERED STEP 4: inferred facts become the trunk's own knowledge when taught (cg27a_step4p, seed 4, visits 1-7; 12 November)

cg27a_replay + the rule world x4 (48 families, 48 org charts) and HALF the relational module's written facts held back. Written facts: 120
by visit 7, **precision 1.00, 117 of them composites never planted anywhere**. From the trunk (store and memory off), test wording, facts
written at an earlier sleep, **closed-form edits SUBTRACTED**:

| visit | TAUGHT | HELD BACK | trunk's answers to held-back facts (right / middle hop / other; family size 6, chance ~0.17) |
|---|---|---|---|
| 4 | **0.61** (18) | 0.12 (25) | 4 / 1 / 20 of 25 |
| 5 | **0.50** (28) | 0.13 (30) | 6 / 0 / 24 of 30 |
| 6 | **0.56** (32) | 0.18 (34) | 8 / 0 / 26 of 34 |
| 7 | **0.44** (45) | 0.14 (50) | 5 / 0 / 45 of 50 |
| 8 | **0.56** (55) | 0.17 (65) | 10 / 0 / 55 of 65 |

**Facts the system INFERRED, never stated in its text, are recalled from its weights when taught (0.44-0.61), and held-back ones stay at
the guessing level (0.12-0.18, chance ~0.17)** - visit 7: 20 of 45 vs 7 of 50 (edits subtracted), a gap the small runs could not show
(withdrawn above at n 2-6). The held-back answers are wrong names, not the middle hop: no sign of the trunk composing on its own (as
before). SAFETY: the local short run FAILS (mean BPC 3.210 / forgetting -0.058 vs 3.021 / -0.127 for the same build at normal rule-world size: 4x the planted template facts are a large share of the small local text); on the GPU, where a domain's visit reads ~6.5 MB, post-train BPC per visit is within +0.01 of cg27a_replay (2.021 / 2.081 / 2.034 / 1.675 / 1.587 / 2.085 / 1.802 / 1.413 vs 2.031 / 2.078 / 2.028 / 1.667 / 1.574 / 2.065 / 1.798 / 1.400) - a scale artefact locally, not a GPU cost. Taught recall dipped at visit 7 as the pool grew and recovered at visit 8 (0.56, 31 of 55 right vs held back 10 of 65). 148 written by visit 8, all correct, 144 never planted. Gutenberg's reliance in this run 0.19-0.22 at visits 5-8 (cg27a_replay 0.26-0.30 on the same seed and replay): the erosion varies run to run. One seed.

### Seed 5 REPLICATES learning from reading (cg27a_replay_s5, visits 1-10; 12 November)

The same build as cg27a_replay on seed 5 (no model change). Never-asked facts (read only, replayed, never questioned), from the weights:

| visit | own sentence completed (control) | test-wording questions, ledger (seed 4) | after-visit probe (seed 4) | taught, ledger (seed 4) | Gutenberg reliance (seed 4) |
|---|---|---|---|---|---|
| 4 | 0.53 | 0.09 (0.09) | 0.14 (0.14) | 0.54 (0.46) | 0.29 (0.21) |
| 6 | 0.57 | 0.12 (0.03) | 0.18 (0.10) | 0.62 (0.60) | 0.30 (0.31) |
| 8 | 0.59 | 0.25 (0.14) | 0.29 (0.23) | 0.69 (0.72) | 0.25 (0.27) |
| 10 | **0.81** | **0.30** (0.21) | **0.52** (0.31) | 0.69 (0.72) | 0.34 (0.38) |

**Two seeds agree: replay makes facts the model only read answerable from its weights** - 0.21-0.30 (ledger, ~70 facts) and 0.31-0.52 (after-visit
probe, ~35) at visit 10, against 0.01-0.03 in every run without replay; storage 0.69-0.81. Seed 5 is ahead of seed 4 throughout. Taught
facts 0.69-0.72 on both. Gutenberg's shared-trunk reliance rises on both seeds (0.34 / 0.38 at visit 10) - replay's cost, still without a
same-seed no-replay run past visit 4. Post-train BPC per visit: 2.027 / 2.067 / 2.011 / 1.679 / 1.585 / 2.076 / 1.799 / 1.397 / 1.122 / 1.685.
Alignment of taught facts was on in both runs; its share is the open question (cg27a_replaynoal).

### Label-free alignment and the two diagnostics (cg27a_alignlf, seed 4, interim visits 1-7; 12 November)

cg27a_replay + half the never-asked facts aligned (question -> stored declaration, no labels), the other half the pure control.

| visit | ALIGNED, never asked: questions | PURE CONTROL: questions | aligned / pure: own sentence | ledger, all never-asked |
|---|---|---|---|---|
| 3 | 0.08 (24) | 0.00 (21) | 0.18 / 0.06 | - |
| 4 | 0.09 (43) | 0.03 (35) | 0.39 / 0.33 | - |
| 5 | 0.13 (45) | 0.05 (40) | 0.31 / 0.31 | 0.09 (77) |
| 6 | 0.05 (43) | 0.12 (42) | 0.31 / 0.28 | 0.07 (74) |
| 7 | 0.17 (42) | 0.16 (44) | 0.47 / 0.75 | 0.14 (72) |

**No per-fact benefit of label-free alignment**: the aligned half led at visits 3-5 by 0.05-0.08 and the halves are level by visit 7
(0.17 / 0.16) - within the noise of ~40 facts. Both halves rise with replay as in cg27a_replay (0.11 at visit 7). Plain 'A r' completion
tracks own-sentence completion (0.44 / 0.47 aligned; 0.69 / 0.75 pure). Gutenberg's private-parts reliance here 0.21 / 0.22 / 0.19 at
visits 5-7 vs 0.26 / 0.31 / 0.30 in cg27a_replay - the same replay, so the erosion seen there is not simply replay's doing (run-to-run
variation or the alignment rows; not separable on one seed).

DIAGNOSTICS: (1) **the closed-form edit's lift on unedited conclusions is shared names**: control sharing a subject or answer with an
edited conclusion 0.00 -> 0.60 (5) and 0.57 -> 0.86 (7) when the edit lands; with no shared name 0.18 -> 0.18 (11), 0.00 -> 0.00 (2) - no
generalisation to unrelated conclusions. (2) **What the trunk answers for written rule-world facts** (test wording, edits in place, family
size 6, chance ~0.17): taught 16 of 17 right at visit 7 (2 of 4 at visit 3); held back 3 of 6 right (0 of 2 at visit 3, 0 of 3 at 4), the
wrong ones a consistent other name ('is overseen by -> Edgalzo' for two held-back subjects whose truth is Orgalqui), never the middle hop.
Taught 0.94 vs held back 0.50 - a gap again, but n 6.

### Source replay on the GPU: facts only READ are now stored (cg27a_replay, seed 4, interim visits 1-6; 12 November)

cg27a_align + inf_distill_replay 4 (4 stored source sentences of pool AND control facts per question-row step; the control is never
asked; 399 replay steps per visit, the replay loss 1.30 -> 0.35-0.40 within each visit).

| visit | control: completes its OWN source sentence (align run) | taught: own sentence | control, test-wording question (ledger) | taught, ledger | cosine own / ANOTHER fact's declaration (control) |
|---|---|---|---|---|---|
| 2 | 0.15 (13) (0.08) | 0.55 | 0.00 (43) | 0.19 | 0.963 / 0.881 |
| 3 | 0.13 (31) (**0.00**) | 0.61 | 0.04 (75) (0.01) | 0.43 (0.38) | 0.932 / 0.789 |
| 4 | **0.30 (30)** | 0.72 | 0.09 (81) (0.02) | 0.46 (0.42) | 0.904 / 0.654 |
| 5 | 0.21 (29) | 0.81 | 0.04 (77) | 0.58 | 0.887 / 0.594 |
| 6 | **0.40 (30)** | 0.82 | 0.03 (74) | 0.60 | 0.871 / 0.529 |
| 7 | 0.38 (29) | 0.82 | **0.11 (72)** (after-visit probe 0.11 (36)) | 0.62 | 0.872 / 0.488 |
| 8 | **0.59 (29)** | 0.97 | **0.14 (71)** (after-visit probe **0.23 (39)**) | **0.72** | 0.876 / 0.424 |
| 9 | 0.42 (24) | 0.91 | **0.18 (68)** (after-visit probe 0.15 (27)) | 0.70 | 0.883 / 0.496 |
| **10** | **0.69 (26)** | 0.96 | **0.21 (67)** (after-visit probe **0.31 (36)**) | 0.72 | 0.904 / 0.417 |

**Final (pass 1, visit 10): never-asked facts answered from the weights at 0.21 (ledger) / 0.31 (probe), against 0.01-0.03 without replay; their own sentences completed 0.69.** Cost: pass-1 mean forgetting +0.026, final BPC 1.767 (earlier pass-1 runs on other seeds: +0.006 / 1.725-1.734; no seed-4 pass-1 summary to compare), Gutenberg's private-parts reliance 0.38 at visit 10 (0.20-0.26 in earlier runs) - a real cost of replay, the target of spaced replay (cg27a_spaced).

By visit 8 the facts the trunk only READ and was never asked are answered in the test wordings at 0.14 (ledger, 71) / 0.23 (after-visit probe, 39)
against 0.01-0.03 in every run without replay - storage first (own sentence 0.00 -> 0.59), question recall following a visit or two behind.
Taught facts 0.72 (the best ledger figure so far). Gutenberg's private-parts reliance 0.31 -> 0.30 -> 0.27 -> **0.36** (visits 6-9: rising again). One seed.

**Replay stores facts the model only read** (own-sentence completion 0.00 without replay at visit 3 -> 0.30-0.40 by visits 4-6) and
taught facts too (0.47 -> 0.82). Taught recall in the test wordings is a little higher than without replay (ledger 0.43 / 0.46 vs 0.38 /
0.42 at visits 3 / 4; 0.58 / 0.60 at 5 / 6). But **the stored control facts are not yet answerable as questions**: control in the test
wordings 0.03-0.09 (after-visit probe 0.07 / 0.14 / 0.03 / 0.10), barely above cg27a_align (0.01-0.02). The states are separating by fact
(own vs another fact's declaration 0.96 / 0.88 -> 0.87 / 0.53; alignment loss 0.03 -> 0.07: alignment now has work to do). No cost seen
on the GPU: post-train BPC per visit 2.031 / 2.078 / 2.028 / 1.667 vs cg27a_chunk 2.036 / 2.071 / 2.024 / 1.669 (local short runs: 3.021
/ -0.127 vs 3.006 / -0.141 - a consistent +0.015 locally). Step 4: 23 facts written by visit 6, precision 1.00 (20 never planted).

### View alignment on the GPU (cg27a_align, seed 4, interim visits 1-4; 12 November) - the bottleneck is STORAGE, not routing

cg27a_rwclean + view alignment (0.5 x (1 - cosine), declaration a stop-gradient target, taught facts only). Against cg27a_chunk (seed 4):
taught facts, ledger "taught at least one visit" 0.38 / 0.42 at visits 3 / 4 (chunk 0.38 / 0.41); never-taught control facts 0.01 / 0.02
(chunk 0.01 / 0.01) - **no effect of alignment on recall**. The new readout says why:

| visit | cosine, test-wording question vs declaration (taught / control) | paraphrased relation (taught / control) | **trunk completes the fact's OWN source sentence (taught / control)** |
|---|---|---|---|
| 2 | 0.963 / 0.969 | 0.891 / 0.891 | **0.47 (32) / 0.08 (13)** |
| 3 | 0.951 / 0.961 | 0.863 / 0.869 | **0.48 (31) / 0.00 (31)** |

The question's state already sits at the declaration's (alignment loss 0.03-0.06 from the start; taught = control), so alignment has little
to do; but the facts the trunk only READ are not in its weights even in their own wording (0.00 of 31). Reading a fact a few times does not
store it; question rows do. Toy (tools/toy_align2.py, test A, no alignment): a never-taught fact read 2 / 8 / 32 / 128 times is completed
0.22 / 0.71 / 0.96 / 0.96 and answered in an unseen question wording 0.06 / 0.18 / 0.43 / 0.50 - storage needs ~8-30 readings, and stored
facts still answer only ~half in question form, the gap alignment closes in the toy. Next: SOURCE REPLAY (inf_distill_replay: stored source
sentences of pool AND control facts replayed as plain text, control never asked), then alignment on top. The mismatched-pair cosine (question
vs another fact's declaration) is added to the readout to tell a fact-specific 0.96 from a shared direction.

**Toy alignment tests (tools/toy_align2.py, seed 0, 6,000 steps; never-taught facts in UNSEEN question wordings):**

| test | result |
|---|---|
| A READINGS (never-taught facts read 2 / 8 / 32 / 128 times): alignment off | stored (completes 'A r') 0.22 / 0.71 / 0.96 / 0.96; question 0.06 / 0.18 / 0.43 / 0.50 |
| ... alignment on | stored 0.04 / 0.51 / 0.93 / 0.96; question 0.02 / 0.30 / **0.78 / 0.76** - alignment pays once a fact is stored (~32 readings) |
| B LABEL-FREE (aligned but never asked / read-only control) | none 0.33 / 0.33; taught facts aligned 0.79 / 0.76; **+ the never-asked facts aligned (no labels) 0.92 / 0.87** |
| C OBJECTIVE, 1,500 facts | none 0.40; **cosine 0.59**; InfoNCE 0.53; predictor head + cosine 0.48 (as LLM-JEPA: cosine best) |
| D WHERE | **answer position 0.87**; subject position 0.53; both 0.88; middle layer 0.89 |
| E [PRED] TOKENS (LLM-JEPA's tied predictor) | **k 0 (as built) 0.85**; k 1 0.50; k 2 0.47; k 3 0.61; no alignment 0.52 |
| F WEIGHT | **0.5 (as built) 0.85**; 2 0.83; 8 0.79 |

Every variant points back to the build: cosine, stop-gradient declaration, answer position, no predictor, weight 0.5. The open variable
is STORAGE (A) - hence source replay (cg27a_replay). Local safety, short runs: alignment with a stop-gradient target 3.006 / -0.141, both
sides trained 3.006 / -0.140, vs 3.006 / -0.141 without (local alignment loss 0.007: the states coincide at the local context length).

Rule world with the sleep layer held off (inf_rw_nosleep, the clean step-4 control): 0 rule-world conclusions leak (control conclusions 0.00
vs 0.44 in cg27a_chunk at visit 4); written facts 11 at visit 3, precision 1.00, 8 never planted; from the trunk, edits subtracted: taught
1.00 (4) vs held back 0.50 (2); never planted 1.00 (2) vs 1.00 (1) - n far too small to separate.

### Reader tests 2 and the toy transformer (tools/reader_tests2.py, tools/toy_compose_rows.py; 12 November)

READER (offline, Re-DocRED gold; gold rate of records between benchmark entities / gold relations covered):

| reader | gold rate | covered |
|---|---|---|
| baseline: adjacent + pronoun-centre records | 0.34 | 338 |
| **M2 subject-anchored records** (every sentence about the title: title, frame, each of its first 3 fillers) | **0.46** | 223 |
| baseline + subject-anchored | 0.38 | **373** |
| M3 lists (3+ fillers joined by one frame) and neighbour dates dropped | 0.39 | 293 |
| **M2 + M3** | **0.41** | 342 |

The anchored records' gold frames are relational ('was born in', 'was elected as a', 'was married to', 'was the flag bearer for').
M1 INFORMATIVE FRAMES (selectional preference strength over the co-clustered types): the top frames are units, dates and name fragments
('metres (', 'km h (', 'Summer', 'Film'; 'was born in' 1.44 vs 'and' 1.11) - the types are too weak; rules among them: 995 candidates, 879
with no chains, 0 rules (best 'May' o ', and' -> 'October'). The anchored reader's frames are the better input for rule mining.

TOY TRANSFORMER (2 layers, 128 wide, from scratch on synthetic facts; seed 0, 6,000 steps):
- TWO-HOP CURSE: held-out chains (never asked) answered **0.00** whether their premises were taught SEPARATELY or also CO-LOCATED in one
  training sequence - though every first hop (1.00) and every trained two-hop question (1.00) is right. Co-location in training is not
  enough at this scale; composition belongs at question time (chained retrieval) or outside the weights (the relational module).
- WORDING DIVERSITY and VIEW ALIGNMENT (recall in UNSEEN question wordings; facts declared 'A r B .', 70% also taught as questions):

| training wordings | alignment | taught facts | facts only read (never taught) |
|---|---|---|---|
| 2 | off / on | 0.20 / 0.20 | 0.19 / 0.19 |
| 7 | off / **on** | 0.93 / **0.95** | 0.64 / **0.88** |
| 20 | off / on | **0.96** / 0.96 | 0.55 / 0.87 |

With 2 wordings, teaching adds nothing in a new wording; with 7+ the taught facts carry over (0.93-0.96). Aligning the question's state
with the declaration's state after 'A r' (after LLM-JEPA) then lets the model answer questions about facts it only READ at 0.88 (0.64
without). For Platypus: many more question wordings, and an alignment term between a fact sentence and its question.

### Matching the Blanks on the fast channel (tools/mtb_probe.py, 12 November)

A frame encoder (BiGRU over '[A] frame words [B]') trained with no labels to bring together frames that link the SAME filler pair in
different sentences (74749 positive pairs from local Wikipedia + the benchmark), tested on 370 benchmark gold-pair records:
nearest other record of the same gold relation type - untrained 0.40, frame words (tf-idf) 0.44, **MTB code after 4 epochs 0.44, MTB +
frame words 0.47** (0.39 / 0.43 / 0.42 / 0.44 by epoch). Level with the connector words, a small gain combined: at 10 MB there are too few
repeated pairs for the encoder to learn more than the words already say (the published method used hundreds of millions of sentences).

### Rule constraints and entity identity (tools/toy_rules_constrain.py, tools/entity_identity.py; 12 November)

RULE CONSTRAINTS (after RiJEPA, 2026), toy world of tools/toy_rules.py, seed 0: once child o child -> grandchild and supervised o
supervised -> overseen are discovered, the operators are refitted with an energy term pulling the composite operator toward the product.
The composite operator applied DIRECTLY to new families and charts (never seen with a composite fact): **0.00 -> 0.81 / 0.79 (weight
0.3), 0.85 / 0.85 (1.0), 0.90 / 0.88 (3.0)**; composition itself 0.94 -> 0.85-0.90. A discovered rule, fed back as a constraint, becomes
part of the representation and transfers - the lookup no longer has to be composed at question time.

ENTITY IDENTITY (real text, local Wikipedia + the benchmark, adjacent and centre records): name variants merged within a document by head
match ('Belyi' -> 'Vladimirovich Belyi', 11,526 merges) and across the corpus (one-word fillers to their dominant multi-word name, 5,307:
'Latifah' -> 'Queen Latifah'): fillers 27,488 -> 22,653, records per filler 12.1 -> 14.2, gold relations covered **338 -> 358**, gold
rate 0.34 -> 0.33. Rule diagnosis unchanged: 1,476 of 1,902 candidates with no observed chain, the 401 testable at a median of 0.00, 0
rules - and the best testable are glue ('January' o 'and' -> 'March', ':' o 'and' -> 'and the'). The miner takes the 60 MOST FREQUENT
frames, which are function words, punctuation and month names; the relational frames ('was born in', 'is a town in') are rarer and cut.
Frames must be chosen by how relational they are, not by frequency.

### Hand-checked precision of the fast channel (tools/precision_sample.py, seed 0, 100 benchmark records; 12 November)

Each record judged by reading its sentence: does it state a real relation between the two fillers as written? **24 of 100 (0.24)** -
close to the gold-based 0.26-0.30, so the incomplete annotation was not hiding much. **Document-centre records: 12 of 22 (0.55)**
('Sibiu - the capital of - Sibiu County', 'Mera Ki Rishta - distributed by - Eros International', 'Stoddard Hurd Martin - was a son of -
Stoddard Martin'); **adjacent records: 12 of 78 (0.15)**. The adjacent errors fall in three classes: DATES AND NUMBERS as fillers paired
with whatever follows ('1979 - in - Bratislava', '2011 - the town had a population of - 73'), LISTS read as relations ('Kenya , Zambia',
'Changzhou , Fuzhou', 'Michael Sarrazin , Susannah'), and NAME FRAGMENTS ('Lake - National - Park', 'Port - of - Quanzhou'). Real
adjacent relations: 'Westwood , Los Angeles', 'Leipzig , Germany', 'Wong Nai Siong - was born in - Fuzhou', 'Ubu Roi - is often cited as
a forerunner of - Dada', 'Bee Gees : Barry'. A two-item apposition is a relation, a three-plus list is not; a date belongs to its event,
not to its neighbour.

### Why no real-text rule passes (tools/rule_diag.py, 12 November)

The relational module on the fast channel's real-text records (local Wikipedia + the benchmark; adjacent and centre records only, so the
reader's own list structure is gone; untyped frames): 16,577 facts, 3,827 fillers, 24 frames, 2,009 candidates beat the null. Of them
**1,491 fail the chain-existence guard** (fewer than half of the subjects with a composite fact have an observed two-hop chain at all),
50 have fewer than 3 testable subjects, and the **467 testable ones predict their composite facts at a median of 0.00** (none reaches
0.5). Real-text facts at this scale do not form chains: each filler sits in few records and the two hops are rarely both read (entity
identity is also fragmented - 'Paris' vs 'Paris , France'). The bottleneck for real-text inference is the density and identity of the
fact graph, before typing. (AMIE's partial-completeness confidence, test C, turned out to equal the current check: only subjects with a
recorded composite fact are tested.)

### DCT-style consistency on writes (tools/relmod_hard.py CONSIST=1, seed 1, 10% wrong parents; 12 November)

After Deductive Closure Training (Akyürek et al. 2024: keep the implications that are jointly consistent): the module finds the INVERSE
relation in the data ('is a child of' <-> 'is the parent of', 80%+ agreement) and keeps a write only if one of its chains has no premise
the inverse contradicts. Round 1: writes 738 -> 705, precision 0.96 -> **0.97**; round 2 (inferred facts as premises, where noise
compounds): writes 130 -> 94, precision **0.79 -> 0.84**, recall unchanged (great-grandchildren 58 of 72 new, 80 of 84 training). The
first precision lever that works; it needs an inverse (or other constraining) relation in the data.

### Overnight reader tests T1-T6 (tools/reader_tests.py, 12 November, offline against Re-DocRED gold) and the clean step-4 control (T7)

| test | result | verdict |
|---|---|---|
| T1 REDUNDANCY: gold rate of benchmark-entity filler pairs (records from 630 Wikipedia articles + the benchmark) by independent sightings | sentences 1 / 2 / 3-4 / 5+: **0.12 / 0.09 / 0.04 / 0.00** (n 4,351 / 383 / 138 / 67); frames and documents alike | **Refuted here**: pairs seen more often are LESS often gold - repeated pairs are generic co-occurrences (common places, dates, name fragments), and the gold covers only the benchmark documents' relations, so the 'urns' boost does not show at this scale |
| T2a TYPES: fillers co-clustered by frame sides (PPMI, SVD 50, k-means), against gold entity types | purity 0.53 / 0.57 / 0.61 at k 6 / 12 / 24 vs majority 0.52; NMI 0.07 / 0.17 / 0.18 vs shuffled 0.03-0.08; one clean TIME cluster, LOC and PER mixed | Weak: 10 MB of text gives too few frame contexts per filler |
| T2b TYPED FRAMES: nearest gold-pair record shares the gold relation type | majority 0.23, frame words 0.44, types only 0.12, **frame words + types 0.47** | Small gain from types |
| T5 FUNCTIONAL FILTER (subjects seen twice in a frame) | **0 single-valued frames** found - no effect | Sparse: few subjects repeat in a frame |
| T3 TYPED DOCUMENT CENTRE (latest filler of the title's type) | 438 gold relations vs 444 for the title | No gain over the title |
| T6 RULES ON TYPED FRAMES ('t3 , t10' ...), adjacent + centre records | 9,871 facts: 1,031 beat the null, **0 rules**; pairs sighted 2+ times: 1,708 beat the null, 0 rules | No real-text rule passes the functional check (diagnosis below) |
| T7 CLEAN STEP-4 CONTROL, local short run (`inf_rw_nosleep 1`) | 3.006 / -0.141 vs 3.008 / -0.139 without it; no errors | Safe; preset `cg27a_rwclean_pass1` in the notebook |

What is learned: at 10 MB the fast channel's real-text records are too sparse and too generic for the statistics that work in large
corpora (redundancy, co-clustered types, functional relations); none of them lifts precision here. The gains so far come from structure,
not counts: frames (9x), reference by the document's title (+37%).

### Fast channel: reference by salience, simplified frames, rules on real text (tools/chunk_reader2.py; GPU cg27a_chunk; 11 November)

Offline, against the 4,726 Re-DocRED gold relations: REFERENCE BY SALIENCE (a sentence opening with frame material - 'He was born in',
'The company was founded by' - gets the document's centre as its subject) raises the gold relations covered from **325 to 444** and the
gold-pair rate from 0.26 to 0.30 (centre = the document's first filler; the previous sentence's subject: 426). SIMPLIFIED FRAMES (inner
fillers as a slot, 'was born on [F] in') leave coverage unchanged and raise the records whose frame repeats 5+ times from 0.18 to 0.22.
RULES ON REAL TEXT (240,991 records from 630 Wikipedia articles + the benchmark; 17,721 facts in the 60 commonest frames): 3,941
candidates beat the null; 5 rules pass the functional check, all of them the reader's own list / apposition structure - ',' o ',' ->
', [F] ,' (0.56 of 385), ', [F] ,' o ',' -> ', [F] , [F] ,' (0.62 of 202) ... - true by construction (a record between fillers two
apart IS the chain of the two adjacent ones); their 6,011 proposals are gold 0.02 on benchmark entities (records themselves: 0.26-0.30)
and none would be written (every rule predicts < 0.9). The guards hold - nothing false is written - but untyped frames (',' joins
places, dates, people, list items) cannot carry semantic rules: frames need TYPED slots (filler types by co-clustering) and the slot
records must not be mined as relations.

GPU, cg27a_chunk (seed 4), the fast channel as a readout: gold relations covered at sleeps 1 / 2 / 3: **54 / 144 / 204** against 4 / 11 / 16
for the current reader (13x); the relational module on the real-text records: 438-968 candidates beat the null, 0 rules. Local short run
with the reader on: identical to the run without it (3.008 / -0.139). Step-4 readout at sleep 3 with the new split: taught 0.25 (1 of 4)
vs held back 0.00 (0 of 2), unchanged with the edits subtracted - but 3 of the 4 taught and both held-back facts were ALSO concluded by
the sleep layer as 'A is connected to C' (and replayed into the trunk); not also a sleep conclusion: 0.00 (0 of 1). The relwrite run's
0.75 is not reproduced; consolidation of self-inferred facts is NOT yet shown - the rule world's chains must be held out of the sleep
layer's own joining for a clean test.

### Step 4 on the GPU: written conclusions consolidate (cg27a_relwrite, seed 4; interim, sleeps 1-3)

The fixed relational module on the reader's facts: sleep 2 - paraphrases merged (resides close to = lives near, was taught by = trained
under, works with = is a colleague of; all true), supervised o supervised -> overseen (6 of 6), 0 false, 6 WRITTEN (precision 1.00, 3
never planted); sleep 3 - four true paraphrase merges (+ partners with = is a partner of, answers to = reports to), both rules (child o
child -> grandchild 11 of 11; overseen 2 sleeps running), 0 false, 5 more written (11 in all, precision 1.00, 8 never planted; 8 taught,
3 held back). **FROM THE TRUNK (store and memory off), written at sleep 2 and taught for one visit: 0.75 (3 of 4) vs held back 0.00 (0 of
2); composites NEVER PLANTED anywhere: taught 1.00 (2 of 2) vs held back 0.00 (0 of 1).** Ordinary facts taught one visit: 0.33,
control 0.00. Tiny n. **Not supported as n grew** (cg27a_replay sleep 6: taught 0.69 (13) vs held back 0.60 (5), edits in place; never planted 0.64 (11) vs 0.75 (4)): the trunk answers held-back rule-world composites about as often as taught ones - teaching the module's writes is not shown to be the cause; why held-back composites are known (guessing hub names in small families?) is open.

### A fast sequence channel as the reader (tools/chunk_reader.py, 11 November, local)

Frames are the short COMMON word sequences, fillers what varies between them - counted on local Wikipedia plus the benchmark text
(68,123 sentences, 1.86 M tokens), no labels, no word lists. Frame material: the K most frequent word types and punctuation, and (variant
B) words that are rarely capitalised when not first in a sentence (a statistic of the text); filler runs between them, joined across one
frame word when the three words are a strong collocation ('Gulf of Mexico'); a record per pair of fillers with the frame between them
(punctuation included: 'Stockholm , Sweden'). Scored against the Re-DocRED gold relations like the current reader:

| reader | records | gold pairs covered (of 4,726) | precision on benchmark-entity records |
|---|---|---|---|
| current (two rarest words of a clause) | 1,252 | 36 (0.8%) | 0.23 |
| A: frequent words are frame material, K 1000 | 8,763 | 195 (4.1%) | 0.27 |
| B: + rarely capitalised words, K 300, adjacent fillers | 4,345 | 194 (4.1%) | 0.28 |
| C: Harris slot entropy instead (best threshold) | 11,558 | 150 (3.2%) | 0.25 |
| **B, fillers up to 3 apart, frames up to 12 tokens** | 8,613 | **325 (6.9%)** | 0.26 |

Ceiling with both entities named in one sentence: 555 (11.7%) - the best variant reaches 59% of it, 9x the current reader. Multi-word
names are read whole ('Fredrik Laurentz Ekman', 'Margaret Thatcher' - 'and her husband' - 'Denis Thatcher'); the commonest frames of
gold records are punctuation (',' and '(' - appositions, 'Stockholm , Sweden'), 'in', 'was born in'. Precision stays near 0.26 (the
gold annotation is incomplete, so this is a floor). Still missing: pronouns and cross-sentence reading (88% of the gold pairs).

### Relational module on the GPU (cg27a_relmod, seed 4, first version: nearest-only candidates, whole-matrix distance, no guards)

On the facts the store's reader recorded (every relation, none marked composable): sleep 1 (70 facts, 10 relations) no rule - too
few composites; sleep 2 (168 facts, 16 relations) **supervised o supervised -> overseen** (0.73 vs null 0.81; predicts 6 of 6), 0
false, 6 proposals all right, **3 of 4 held-out composites** (never planted) proposed; sleep 3 (300 facts, 21 relations) **child o child
-> grandchild** (0.71 vs 0.88; 11 of 11), 5 proposals all right, 5 of 8 held-out composites - but overseen was missed (nearest-only
candidates) and one FALSE rule passed on 2 of 4 facts ('was taught by' o 'reports to' -> 'answers to'): the check now needs a Wilson
95% lower bound >= 0.5 (2 of 4 gives 0.15; 6 of 6 0.61; 11 of 11 0.74) - in the hard world it also drops the 50% partial rules, writes
unchanged (746 and 92, precision 1.00).

### Composition failure modes, fixed and tested (tools/relmod_hard.py, 11 November; the relational module _rm_mine)

The failure analysis predicted where operator composition breaks: whole-matrix distances squeeze the contrast; paraphrase splits halve
the evidence; a matrix cannot say "except self"; inverse and missing hops snap to the most similar name; written partial-rule facts
confirm their own rule; noise compounds per inferred hop. A HARD RULE WORLD plants each one: 4-generation families (great-grandchild =
three hops), the inverse "is the parent of" (one-to-many), siblings (child o parent-of, minus self), supervision said as "is supervised
by" / "reports to" / both, a 70% partial rule (mentored), two arbitrary relations; 30% of the groups NEW (one-hop facts only); half of
every composite held out elsewhere; 840 people. Rules graded by their TRUE confidence on the full world.

| seed 0, clean | rules TRUE / PARTIAL / FALSE | facts that would be written, precision | held-out recall: grandchild / great-grandchild / sibling / overseen |
|---|---|---|---|
| as first built | 8 / 5 / 0 (the paraphrase split gives 4 copies of one rule) | 274, 1.00 | 232/232 / 64/156 / **0/273** / 177/177 |
| fixed, round 1 | 6 / 3 / 0 (paraphrases merged) | **746, 1.00** | 232/232 / 64/156 / **273/273** / 177/177 |
| fixed, round 2 (round 1's writes as hops, never evidence) | 6 / 3 / 0 | 92 more, **1.00** | 232/232 / **156/156** / 273/273 / 177/177 |
| round 2 WITHOUT provenance | 7 / 2 / **2** ("mentored" merged into "overseen") | 92, 1.00 | mentored 0/174 |
| seed 1, 10% wrong parents: fixed round 1 / round 2 | 6 / 2 / 0 both | 729 at **0.96** / 97 at **0.79** | great-grandchild 132/156 |

Fixes kept: paraphrase merge (objects agree on 80%+ of shared subjects); rule distance on the subjects' codes (half the null-beaters of
the whole-matrix distance in the stress check); except-self at each snap; every relation beating the null a candidate (a strict and a
partial rule can share a product); the functional check counts only chains whose middle is a known subject AND needs an observed chain
ending on someone else (rejects parent-of o child = self and grandchild o grandchild, both FALSE rules otherwise; a distance-to-identity
test was tried and removed: it also killed the true sibling rule, child o parent-of being half identity); written facts carry
PROVENANCE (hops, never evidence); WRITES only from a rule verified at 0.9+ through STORED premises - the latent chain finds and verifies
rules but never writes alone (57 of 206 latent-only round-2 writes were wrong: it snaps through a missing hop). Not fixed: noise compounds
through inferred premises (0.96 -> 0.79 at the second round); corroboration by a second rule did not help (0.77, recall down) because
both routes share the corrupted fact - limit inference depth where sources are noisy (props carry a depth flag). Partial rules (true
confidence 0.5-0.7) are found and labelled but not written.

Repair replay brings forgetting back to the replay-only level (-0.156). The stochastic theta run's lower cost is not reproduced by fixed
phases and coincides with a 10-dream-step final visit (its allocation per visit was 5 / 0 / 1 / 3 / 10 against 5-6 normally): not
separable from where the random phases fell in 24-step visits; a full-run trio with bounded phases is queued. Spindle-style writing
fails; the beta brake changes nothing measurable.

**Theta, full runs (11 November, seed 0, mlp_out distillation, bounded stochastic phases):** no gate (TH0) vs stochastic theta (THS):
pass 1 mean forgetting -0.045 / -0.036, BPC 2.753 / 2.758; pass 2 forgetting +0.014 / +0.008, final BPC 2.428 / 2.438; lost between
visits at the revisits (Gutenberg, Wikipedia, ArXiv, PubMed, FreeLaw) -0.136 / -0.134, -0.046 / -0.031, +0.010 / -0.010, -0.015 / -0.017,
-0.010 / +0.009 (means -0.039 / -0.037). **No effect** - every difference inside seed noise; the short-run gain was the dream allocation.
(Replay-only reference without distillation, FR: -0.005 / 2.415.) Fixed phases (THF): pass 1 -0.047 / 2.751, pass 2 +0.011 / 2.432, lost between visits at the revisits -0.145, -0.043, +0.002, -0.019, -0.017 (mean -0.044) - the trio is level within noise. Rhythm gating is not carried forward.

### Null-space probe (11 November, local full runs, seed 0; readout only)

Inputs to each block's mlp_out (H = 512) on stored sentences of the old slots, against a question-row gradient on taught facts (sleeps 2-3,
BR0/BR1): 90 / 99 / 99.9% of the old-domain energy sits in 18-30 / 94-157 / 256-329 of 512 directions, and **0.95 / 0.99 / 1.00 of the
fact update lies inside them** (an isotropic update would put 0.04-0.05 / 0.22-0.27 / 0.54-0.61 there). The facts are written in the very
directions the old domains use most - which is why the question rows cost old-domain BPC and why repair replay, pulling those directions
back, erased the facts. A hard null-space projection (AlphaEdit) would keep 0.05 / 0.01 / 0.00 of the fact update: **not viable at this
width**; any separation has to come from elsewhere (separate weights for facts, or a key that is not the old text's main directions).

### Fact units (neurogenesis), first local test (11 November, short runs, seed 0)

New units beside each block's MLP, one per new taught fact (up to 64 per sleep), keyed on the fact's question context; the question rows
train only them. First version (raw cosine, tau 0.5): the units fired on 0.36-0.48 of ordinary text - the MLP inputs share one dominant
direction. Second (inputs centred, per-unit thresholds at the 0.99 quantile of cosine on ordinary text; FU): at visit 10, 22 units, any
unit firing on 0.08 of old-domain and 0.05 of visited text positions against 0.18 / 0.11 of taught questions (taught / test wording) - about
2x selective; question-row answer loss flat (1.77 -> 1.76); **taught recall 0.00** (DWM 0.11); forgetting -0.151, final BPC 2.980
(DWM -0.139 / 2.991) - the same as soft targets alone (PD, -0.152 / 2.977, taught 0.00): no fact learning, so no cost. Not a trade yet;
the units do not reach enough answer positions of their own fact.

### Prompt distillation (soft targets from a teacher that reads the source sentence; 11 November 2026, local full runs, mlp_out writes)

| Run | Taught at the end | Forgetting | Final BPC | Teacher vs student, answer bytes right |
|---|---|---|---|---|
| none (DWM) | 0.11 | -0.140 | 2.991 | - |
| soft targets + 0.5 CE (PDC) | 0.07 (n 27) | -0.140 | 2.990 | 0.73 vs 0.73, 0.72 vs 0.72, 0.82 vs 0.84 (visits 8-10) |
| soft targets only (PD) | **0.00** (n 27) | -0.152 | 2.977 | 0.79 vs 0.79, 0.72 vs 0.73, 0.79 vs 0.79, 0.50 vs 0.51, 0.59 vs 0.59 (visits 6-10, 570-690 bytes) |

Soft targets alone teach nothing (answer loss flat at 1.96-1.97; the slightly lower cost is that of no question learning). The teacher is no better than the student: reading the fact's source sentence first does not help the local trunk answer, so the soft
targets carry nothing the hard answer does not. No harm, no gain. The GPU readout (inf_distill_pd_probe, SOURCE IN CONTEXT) asks the same
question at the GPU window, for taught and never-taught facts.

## Write side (cg27a), configuration results

| Result | Numbers | Conditions |
|---|---|---|
| Held-out wording two-hop from the weights comes from closed-form edits fitted on three wordings, not from replay wordings | 0.36-0.39 at visit 10 (three edit wordings) vs 0.22 with one edit wording + replay wordings = 0.22 eval-fix (no write side), visit for visit | GPU seed 1; one run per arm; multiple choice among 20. **Seed 2** (self-contrast build): 0.30 at visit 10, 0.33 at the end (seed 1 same build 0.31 / 0.40); visits 6-9 within 0.06 of seed 1 |
| **Self-contrast in the closed-form edit (`inf_cf_selfc`, on by default)** | Edited conclusions, fitted wording, open-ended from the weights: 0.88 / 0.80 / 0.77 at visits 4-6 vs 0.30 / 0.40 / 0.54 without (same seed); held-out 0.06 / 0.24 / 0.32 vs 0.00 / 0.24 / 0.22; all edits kept. End of run: fitted open 0.67 vs 0.54, held-out open 0.25 vs 0.19; 130-probe open two-hop 0.54 vs 0.48, held-out multiple choice 0.40 vs 0.41 (unchanged); BPC 1.731 vs 1.724, forgetting -0.002 vs -0.012; real forgetting (per domain, mean of 10) +0.012 vs +0.005, largest USPTO +0.109 vs +0.064 and FreeLaw +0.022 vs -0.004: a possible cost, to check on seed 2 | **Real build** (with competitor contrast), GPU seed 1, one run per arm, slot given, edited conclusions only. Shadow (no competitor contrast), 7 sleeps: fitted open 0.17 -> 0.59, held-out MC 0.51 -> 0.65. Local: no effect, no harm (2 seeds) |
| Greedy decoding loses names the model ranks well | Held-out misses: true name more probable than the greedy output in 0.88; rank among 200 probe names median 6 | GPU end of run; constrained decoding is reported beside greedy, never as the headline |
| Temporal gradient under store lesion | Two-hop on the system's conclusions, store off: concluded before visit 8 0.91 (n 47), from visit 8 0.50 (n 20); end of run, split at visit 7: 0.88 (n 56) / 0.51 (n 69), and with self-contrast 0.87 (n 53) / 0.59 (n 70); seed 3 (10 Nov) 0.92 (n 52) / 0.51 (n 69), by exposure older edited 0.88 (n 25), older not edited 0.83 (n 12), recent edited 0.50 (n 14), recent not edited 0.35 (n 17); seed 2 (self-contrast) 0.91 (n 53) / 0.53 (n 68), and its store lesion split by exposure: older edited 0.96 (n 23), older not edited 0.78 (n 18), recent edited 0.67 (n 9), recent not edited 0.50 (n 26); sleeps in the edit 0 / 1-2 / 3+: 0.61 (n 44) / 0.67 (n 9) / 0.96 (n 23) - so the age effect holds among conclusions never edited (0.78 vs 0.50), and re-solved edits add to it; replay count is not yet split; store lesion split at visit 8, with self-contrast 0.86 (n 42) / 0.71 (n 24) (store on 1.00 / 1.00), so self-contrast narrows the gap by holding recent conclusions better; build without closed-form edits (1 Nov): 0.02 / 0.04 | **Two GPU runs of one seed (1)**, near-identical builds, so not independent; age confounded with exposure (sleeps, replays, re-solves since concluding); needs the exposure control and a second run before it is a claim. Local mixed (n 5-9) |
| Weights hold the system's conclusions, not the source facts | Never-edited conclusions 0.21 open-ended (fitted wording); source one-hop open-ended from the weights ~0.00 | GPU seed 1 |

## Slow features (north-star rungs 0-2)

| Result | Numbers | Conditions |
|---|---|---|
| Rung 0: closed-form SFA against the trained JEPA on retrieval | **Local: ties on held-out wording** 0.15 vs 0.12, 0.15 vs 0.10, 0.20 vs 0.12 (seeds 0, 2, 1); written wording JEPA ahead. **GPU (seed 3, 10 Nov): does NOT hold** - retrieval of a planted chain's second fact from its first (150 chains, 550 candidates): trained JEPA 0.61/0.75, held-out wording 0.61/0.76; SFA 0.05/0.11, held-out 0.03/0.09; CCA 0.03/0.05; trunk cosine 0.06/0.16; rare names 0.96-0.97. Judge and relations: all near equal (bits saved +0.066 to +0.078; Re-DocRED top pick holds a gold relation 0.27-0.30 vs random sentence 0.23) | The JEPA is trained in sleep on the system's linked pairs (it learns names); SFA on adjacent sentences does not. At GPU scale the slow code is not a retrieval key |
| Rung 1: online slow head (**amended 9 Nov, 17:05: rung 1 is now slow features re-solved each sleep in closed form, as the latent judge already does; online rules parked**) | Not passed after five versions. v3 reaches closed-form slowness at 16 bytes (0.811 vs 0.814) but not its retrieval; v4 (sentence pairs) ties SFA top-1 in the written wording on two seeds (0.07, 0.15), not held-out (0.05 vs 0.12-0.15). Three alternative rules, seed 0 (top-1/top-5, written | held-out; SFA 0.07/0.35 | 0.12/0.35): SFA from running statistics 0.05/0.17 | 0.05/0.20; IncSFA-style two-stage (GHA whitening + minor components) 0.10/0.20 | 0.03/0.12 (its first version diverged to NaN, and NaN scores ranked first gave a false 1.00: both fixed); LPL 0.00/0.10 | 0.00/0.03 (codes collapsed, units correlated ~0.5); Koopman (SFA read off a next-step predictor, Klus et al. 2018): delta rule 0.03/0.12 | 0.03/0.12, recursive least squares 0.03/0.15 | 0.07/0.20. None passes held-out. LIKE FOR LIKE: closed-form SFA on the slow head's own pooled features 0.10/0.35 | 0.17/0.38 (reference codes 0.10/0.35 | 0.12/0.35) - the features are not the problem; the online rules do not reach the closed-form solution, which is what amended rung 1 uses | Local, stop-gradient (no BPC cost) |
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
- **Latent judge on harder controls (local seed 0, 8 sleeps)**: known-linked pairs vs random 0.94-0.98, but with the shared name replaced by 'the person' 0.41-0.73 (mostly ~0.6): mostly a name matcher. The truth score on planted links was withdrawn (this judge never sees planted pairs). As a first pass (skip half the proposals) it saved 2-3 s per sleep and lost up to half the accepts in one sleep.
- **Route 2 (sleep replays recorded facts in varied relation-agnostic forms, `inf_fact_forms`)**: no change in the subject-to-answer path at 25% of replay rows (0.23/0.26/0.27 bits vs plain 0.22/0.24/0.31 vs baseline 0.22/0.24/0.27) or at 75% (0.22, one sleep, then stopped); no harm. Caveat: the forms keep the relation phrase, so the bytes before the answer barely vary.
- **ROME-style subject edit**: negative (see rung 3).
- **WISE-style self-firing name memory**: fires 43-100 times per 1000 ordinary bytes, first answer byte 0.00-0.03 (two seeds).
- **Rung 2b is negative locally**: first-byte surprise is a capital-letter detector (AUC 0.90 overall, 0.48-0.50 within capitalised words); entity codes are no more stable across occurrences (0.43-0.56); relations on Re-DocRED at chance (substitution 0.45, compression 0.50, latent judge 0.50-0.57). The reader remains the ceiling on all natural-text results; the name-channel key only helps where a name has been found.
- **Case factoring** (`case_factor`, c/C share a letter table): BPC -0.008 and -0.002 (two seeds, noise), entity test unchanged.
- **Stale keys**: prototypes recorded at a domain's first visit route at chance by the end (0.20-0.26). An anchor term on the slow head (`slow_anchor` 1) did not hold its reference codes (cosine to the previous visit -0.04 to 0.30), interim, local seed 0.
- **Six wordings** in the edit add nothing over three (GPU, five sleeps).
- **System-written question forms: GPU confirms and they hurt** (seed 3, eight sleeps): held-out multiple choice 0.22-0.39 vs 0.44-0.78 for the configured edit, answer bytes 0.60-0.65 vs 0.72-0.79.
- **ROME-style fit on the GPU** (seed 3, eight sleeps): the best change at the subject buys 0.00-0.11 bits per answer byte (3.53 -> 3.52 ... 2.90 -> 2.79) - even less than locally.
- **System-written question forms** in the edit ('Notes on A: ...', 'Q: A is connected to whom? A: ', 'It is recorded that ...', 'A is connected to ... ') add nothing either (shadow, local, two seeds, six sleeps): held-out multiple choice 0.09-0.27 vs 0.12-0.28 without, answer bytes 0.56-0.65 vs 0.58-0.69. More forms of any kind do not move held-out recall: the edit's key (the state at the question end) carries the wording. GPU check rides on the seed-3 readout run.

## cg27a_kmf on GPU (15 Nov 2026, A100, seed 4; log up to visit 6 sleep)
- MEMORY FEEDBACK FROM READING (new readout): text read after the write, naming a written subject. AUC right vs wrong of bits saved 0.62 / 0.47 / 0.49 (visits 3-5, n 56 / 87 / 157); confirmations minus contradictions 0.54 / 0.52 / 0.53; Spearman with write-time rank -0.10 / +0.07 / -0.03. **Not a usable judge** (matches the local negative). On the second name the memory COSTS bits for right and wrong inferences alike (visit 5: right -4944, wrong -1594 bits over 658 spans): an untyped "A ~ C" answer is the wrong prediction in real sentences even when the pair is right.
- Wikipedia cost of the memory grows with fill: domain 1 +0.0015 / +0.0072 / +0.0144 bits per byte at visits 3 / 4 / 5 (gate open 2.1 / 5.0 / 12.3 per 1000 bytes); gate 0.95 roughly halves it (+0.0085 at visit 5) but it still doubles each visit. Other domains +0.0004 to +0.006.
- Judge (docs signal) AUC 0.637; KEPT right 0.76 vs FLAGGED 0.65; the trunk receives 122 right + 38 wrong (oracle 122 + 0). Retracted and flagged inferences never reach the trunk (recall 0.00).
- Byte-edit retraction degrades as the memory fills: WRONG rank 0.92 -> 0.03 (visit 3), 0.53 -> 0.03 (visit 4), 0.95 -> 0.25 (visit 5; 302 bytes above the margin left). Right inferences and facts unharmed.
- Relation typing on the GPU trunk: MTB key NMI 0.547; typed rules fail (TYPED right 0.00-0.21 vs untyped 0.74). Learning speed as a judge: AUC 0.46-0.52 (none).

## Pretraining stage, equal compute (15 Nov 2026, tools/pretrain_lm.py, d 192, 3 blocks, 6000 steps)
- A6: Wikipedia only, 6000 steps: held-out Wikipedia 1.991 bits per byte.
- B6: TinyStories 3000 steps (1.87 -> 1.47 training) then Wikipedia 3000 steps: held-out Wikipedia 2.008, held-out TinyStories 2.498.
- Fluency on Wikipedia is level at equal compute (+0.017, about seed noise); B6 reaches A6's Wikipedia loss in fewer Wikipedia steps (2.0 bits at 2000 steps vs 3000), but TinyStories itself is largely forgotten after the switch (1.47 -> 2.50). Whether B6 judges and types relations better is the separate comparison (rel_mtb, weights_judge, jepa_mem) still to run.

## Learned selector from the judge's own verdicts (15 Nov 2026, tools/learned_selector.py, distant_50k, GPU write path, 3 rounds)
Candidates and write-time features from premises._rcode_step at visits 2 / 4 / 6; self-label = relational records of the pair within two visits (label-free); logistic selector trained on earlier rounds' self-labels picks the next 200; distant labels score only. Rounds 2 / 3 (200 written each; SE about 0.03):
| selector | round 2 | round 3 |
|---|---|---|
| RULE (code + AA, asymmetry filter; GPU) | 0.730 | 0.675 |
| learned, all candidates | 0.430 | 0.455 |
| learned, all candidates, co-mention features dropped | 0.540 | 0.570 |
| learned rerank inside the rule's top 1000 | 0.650 | 0.715 |
| learned rerank inside the rule's top 1000, co-mention dropped | 0.750 | 0.720 |
| rule + learned rank sum | 0.665 | 0.735 |
| rule, then the judge (kept only) | 0.789 | 0.709 |
- **Negative (Goodhart).** The selector predicts its own label very well (AUC 0.95) by choosing pairs ALREADY in a relational sentence (weight +3.7 to +4.1), which are re-mentioned later but are right only 0.43-0.46 of the time. The judge's verdict works as a FILTER after the rule (+0.03 to +0.06) but not as a training target over the whole pool. The best variant (rerank within the rule's shortlist, no co-mention) is +0.03 on average, within noise. Learned features that carry: resource allocation, both degrees, object type TIME negative.

## Pretraining comparison, A6 vs B6 (15 Nov 2026; equal compute, d 192; the GPU path's 200 writes, 0.74 right)
| | A6 (Wikipedia only) | B6 (TinyStories, then Wikipedia) |
|---|---|---|
| truth probe, read vs name-swapped spans (held-out acc) | 0.485 | 0.485 |
| truth probe on the inferences (AUC) | 0.419 | 0.404 |
| the trunk's likelihood of the inferences (AUC) | 0.515 | 0.537 |
| interference with what was read (AUC) | 0.498 | 0.511 |
| held-out gain (AUC) | 0.491 | 0.521 |
| relation types, MTB key type-masked (NMI) | 0.455 | 0.451 |
| jepa_mem two-hop decoded (rank) | 0.07 | 0.22 |
- **Negative: a basic-English stage does not give a trunk that judges inferences or types relations better at this size.** No trunk signal separates right from wrong inferences (all 0.40-0.54). (jepa_mem on these trunks is far below the 6k / 18k trunks' 0.73-0.97: few slots written, 60-64 of 4,096 - these trunks' states may not suit the frame-trained head; not investigated.) sparse_mem stopped to free the CPU for builds.

## GPU cg27a_kmr: the relational-records judge (seed 4, 15 Nov 2026, visits 1-4 and the visit-5 sleep)
- **The relational judge holds at scale: AUC 0.688 (signal docs 0.639 on the same writes; offline 0.697).** FLAGGED 40: wrong 0.45 vs unflagged 0.21 (kmf, docs: 0.35 vs 0.24). KEPT right 0.79 vs flagged 0.55 (kmf 0.76 vs 0.65); the trunk receives 126 right + 34 wrong (kmf 122 + 38; oracle 126 + 0). List-only pairs 25, right 0.44. Retracted / flagged never reach the trunk (recall 0.00).
- **THE GATE RULE removes the Wikipedia cost (readout):** with the untyped inferences' slots shut on text, Wikipedia +0.0035 -> +0.0002 (visit 3) and +0.0054 -> +0.0004 (visit 4) bits per byte; other domains +0.0000-0.0004. Questions still read the slots. Now an option (inf_kma_nori), touches no training.
- Memory feedback from reading again no judge: AUC 0.417 / 0.628 (bits saved), 0.488 / 0.541 (confirm - contradict); right inferences cost bits on the second name (-2,588 / -4,093) as in kmf.
- Byte edit (copy): WRONG 0.42 -> 0.05, RIGHT and facts unchanged, undo exact. Trunk bits level with kmf (2.0269 / 2.0265 / 2.0082 / 1.9207 vs 2.0296 / 2.0292 / 2.0075 / 1.9169).

## Routes to typed inference from what the text confirms (16 Nov 2026, tools/typed_routes.py; GPU write path, distant_50k, 200 writes a round)
A typed claim 'X <wording> Y' is right when the wording's commonest Wikidata relation (over labelled relational records) is one of the pair's relations in that direction (evaluation only). Exact relation / with the 'is in' relations (P131, P17, P30, P706, P276, P361) scored as one (TR_GROUP 1). Writes right as pairs 0.69 (round 1), 0.73, 0.675.
| route | typed | typed claim right (exact / 'is in' as one) | among pairs that are right |
|---|---|---|---|
| 1 COARSE at write time: class pair, head = the more specific name (lower store degree), the store's commonest relational wording for that class pair | 200 of 200 (LOC-LOC 'in' 152) | 0.325 / **0.610** | 0.47 / **0.88** |
| 2 FROM THE CONFIRMING relational sentence (after the write) | 115 (0.57) | 0.617 / 0.635 | 0.78 / 0.80 |
| 3 COMPOSITION learned from earlier rounds' confirmations (premise wordings -> confirming wording), at write time | 54 / 72 (0.27 / 0.36) | 0.556 / **0.59-0.63** | - |
| class-pair table learned the same way (no specificity) | all | 0.16 / 0.27-0.28 | - |
- **Positive, two ways.** (1) Coarse typing at write time is nearly free at the 'is in' level: 0.61 typed-right against 0.69 right as pairs - 88% of right pairs get relation group and direction right, the direction from SPECIFICITY (without it the same class-pair idea scores 0.27). On the 115 confirmed writes route 1 (0.71) beats the confirming sentence's own wording (0.64: raw wordings are messy - ') was', 'in cascade range of'). The exact relation (state vs country) is not recoverable this way (0.33). (2) Typed COMPOSITION learned from the system's own confirmations works where rule mining failed: it types a third of the writes at write time, right 0.56-0.63, and **the pairs it types are right 0.83 against 0.73 / 0.675 for all writes - composition support is also a precision signal**. Its table is noisy (723 cells, TV-network and 'in united states on' wordings).
- Build implication: write inferences TYPED ('head is in tail' for place pairs) instead of 'A is connected to C', and prefer pairs with composition support. Both label-free.

## Keyed units with CAUSAL firings (16 Nov 2026, tools/keyed_units.py RU_CAUSAL 1; Re-DocRED, 600 scored documents, memory from the 2,453 before)
The 14 Nov firings were decided from the whole token list (a NAME fired only at the end of the maximal run, a DATE / NUMBER exactly where a year or number starts), so a firing told the memory what came next. Causal: every capitalised word fires with the run so far, DATE / NUMBER fire at every word starting with a digit (DATE: or a capital) and are scored from its SECOND byte; stored continuations only from real dates / numbers. The stand-in trunk was retrained for this run and is weaker (2.532 bits per byte on these documents vs 2.246), so absolute savings are larger than on 14 Nov and not comparable; compare within the run.
| bits saved per 1,000 bytes | fixed mixture | learned gate (last 400 documents) |
|---|---|---|
| NAME alone | 61.4 | 67.3 (2.67% of the trunk's bits) |
| DATE on top of NAME | +2.54 | +3.81 |
| NUMBER on top of NAME | +1.21 | +2.31 |
| DATE + NUMBER on top of NAME | - | +3.60 |
| FACT / PLACE on top of NAME | -1.86 / -0.56 | +1.10 / -0.10 |
- **DATE and NUMBER survive the causal check** (+5% on top of NAME with the gate, as on 14 Nov: +1.29 / +0.30 on NAME's 30). NAME alone is the bulk. Expected on a stronger trunk: smaller in absolute terms (the 18k / 6k comparison: savings scale with the trunk's bits).

## Build B smoke: the sparse memory layer in the package (16 Nov 2026, tools/smf_smoke.py; d 64, 3 blocks, 64 x 64 slots, top 8; 400 + 400 steps)
Visit 1 (text, the layer learning with the trunk): held-out wiki 2.315 bits per byte, layer OFF 2.338 (+0.023: the trunk uses it), 2,660 of 4,096 slots used by text. Visit 2, text training continues + 8 question rows a step (60 facts x 7 wordings, the test wording never used):
| question rows written to | test wording rank (of 10) | held-out wiki | forgetting vs no questions | question loss |
|---|---|---|---|---|
| nothing (control) | 0.12 | 2.181 | - | - |
| the layer only (smem, lr 0.01, 256 slots a step) | 0.15 | 2.198 | **+0.018** | 3.76 -> 3.55 |
| every block's mlp_out (the GPU builds so far) | 0.10 | 2.267 | **+0.086** | 3.00 -> 3.79 |
- The path works end to end (1,277 slots written; the trunk untouched by the questions) and forgets 5x less than mlp_out at this scale, but neither learns these facts in 400 steps on a d-64 trunk (rank ~ chance 0.10-0.15). A grid of the write lr and slots (0.03-0.3, 256-1,024) is running.

## GPU cg27a_kmr, visits 5-6 (seed 4, 15 Nov 2026)
- **The gate rule holds as the memory fills:** Wikipedia +0.0133 (v5, gate open 10.2 per 1000) and +0.0203 (v6, 13.4) as built -> **+0.0010 / +0.0013** with the untyped inferences' slots shut on text (open 1.6 / 2.5): 92-94% of the cost removed, still creeping.
- **The judge did NOT replicate on the second batch** (written at visit 4, judged at visit 6): AUC 0.519 (relational records 0.524, since the write 0.550; documents 0.492), FLAGGED wrong 0.35 vs unflagged 0.34, 39 with no confirming record (first batch 19), batch wrong 0.34 (first 0.26), list-only pairs right 0.63. By visit 6 the trunk has received 231 right + 89 wrong (28% wrong; oracle 231 + 0). Offline the same pattern (tools/learned_selector.py candidates, distant_50k): the judge's AUC is 0.64-0.70 on the rule's top 200 but 0.54-0.60 on ranks 200-400 - and on the GPU the second batch IS the next tier (the first batch's pairs are excluded). The judge works for the best-supported pairs and weakens on rarer ones.
- Retraction: retracted inferences never reach the trunk (0.00, rank 0.06-0.11), but in the MEMORY the retracted ones are still answered at visit 6 (rank 0.65 with the memory, 26 retracted); byte edit on a copy WRONG 0.68 -> 0.20 (363 bytes above the margin) - the byte edit weakens as the memory fills. Trunk bits 1.8618 / 1.9007 (v5 / v6).

## Trunk feedback on the type: the trunk picks the wording (16 Nov 2026, tools/typed_routes.py TR_STAGE=T; A6 stand-in trunk, distant_50k, 200 writes)
Route 1 proposes the class pair's 8 commonest relational wordings; the trunk picks the most probable between the two names. Typed claim right: route 1's top 0.325 / 0.610 (exact / 'is in' as one), **the trunk's choice 0.335 / 0.580**, the best of the 8 (ceiling) 0.615 / 0.635; agreement with the later confirming wording: table 0.087, trunk 0.026. **Negative: the trunk's likelihood does not choose the relation** (as it does not judge truth); there is room (exact ceiling 0.615 vs 0.325) for a chooser with a better signal.
- Grid (d 64, 4,096 slots): write lr 0.03-0.3 and 256-1,024 slots a step - recall 0.12-0.20 (chance), forgetting +0.047 to +0.101, the question loss RISES. **Not address drift**: 0.90-0.96 of the slots the questions' answers read at the end were written by them. **With the trunk frozen (as in Lin et al.) the layer learns** (question loss 3.58 -> 1.31 / 2.99 -> 0.79; rank 0.28 / 0.45) **but forgetting is catastrophic** (+1.07 / +2.38: at 4,096 slots, 65% used by text, the questions' slots are text's slots). So: with a training trunk the layer's writes are undone downstream (the blocks above keep changing how they read it); with a frozen one they collide with text. Next: 128 x 128 slots, d 96, the layer beside the middle vs the LAST block (less downstream to drift).
- 128 x 128 slots, d 96 (5,232 / 4,632 slots used by text), write lr 0.01 / 0.03, 1,024 slots a step:
| layer beside | trunk in visit 2 | rank (control 0.07-0.10) | forgetting | question loss |
|---|---|---|---|---|
| middle block | training | 0.20 / 0.20 | **+0.0065** / +0.042 | 3.69 -> 3.60 / 3.35 -> 3.44 |
| middle block | frozen | ~0.4 / 0.52 | +1.27 / +2.18 | -> ~1.2 / 2.93 -> 0.77 |
| last block | training | 0.18 / 0.13 | +0.017 / +0.048 | 3.74 -> 3.84 / 3.49 -> 4.04 |
| last block | frozen | 0.35 / 0.48 | +1.12 / +2.16 | 3.57 -> 1.23 / 3.11 -> 0.82 |
**Negative as built: the training trunk undoes the writes wherever the layer sits; with the trunk frozen the layer learns the frames (test wording rank 0.35-0.52, open 0.00) but drowns text** (only ~800-900 slots written; the written rows grow far past the text-trained ones - Adam at these rates moves a row by up to lr per step). The bigger memory fixes the collision only while nothing is learned. A row-norm cap (written rows <= the 99th percentile norm of text-trained rows) is being tested.
- Row-norm cap (written rows <= the 99th percentile norm of text-trained rows), middle block, 128 x 128: frozen trunk forgetting +0.10 (from +1.3) but no learning (rank 0.12-0.15, loss 3.49 -> 2.73); training trunk +0.002 and no learning (rank 0.08). **Build B parked: in this package's continually trained trunk, sparse memory finetuning either does not stick (training trunk) or drowns text (frozen); capped, it learns nothing.** Lin et al.'s result rests on a frozen, large model with ~1M slots. Facts stay in the keyed memory (0.97 with it on GPU); consolidation into the trunk's own weights remains open.

## Does a written inference improve the model? Typed writes and usefulness as the selector (16 Nov 2026, tools/useful_writes.py; distant_50k, the GPU path's 200 writes at visit 2, right 0.690; A6 stand-in trunk)
The writes stored in a keyed continuation memory (key: the head name; continuation: the claim's rest) mixed with the trunk by the learned gate (inference.kunits, prequential); scored on 16,350 later relational spans whose first name is a written name (397 state a written pair), bits saved against the trunk alone (2.422 bits per byte):
| form | all later spans | spans stating a written pair | spans of a written name with another partner |
|---|---|---|---|
| F1 'A is connected to C.' (as the GPU) | **-0.42** | **-1.90** | -0.48 |
| F2 typed, route 1 ('head <class-pair wording> tail', head = the more specific name) | +0.06 | **+10.41** | -0.13 |
(bits saved per 1,000 bytes; negative = the memory costs bits)
USEFULNESS AS THE SELECTOR (label-free): keep a write only if it saved bits on the first half of the later text; score on the second half:
| form | writes with later text | kept | right: kept / dropped | second half, all writes -> kept only |
|---|---|---|---|---|
| F1 connected to | 181 | 24 | **0.833 / 0.675** | -0.86 -> +0.22 |
| F2 typed | 171 | 37 | 0.730 / 0.709 | +0.09 -> +0.34 |
- **Positive.** (1) The GPU's finding reproduces offline ('connected to' costs bits even where the pair is stated) and **typing turns it into a saving where the pair is stated (+10.4 per 1,000 bytes vs -1.9)**, with a quarter of the cost elsewhere. (2) **Usefulness - bits saved on later text, no labels - is a selector: it keeps writes that help on held-out text (+0.22 / +0.34 vs -0.86 / +0.09) and, for the untyped form, writes that are more often right (0.83 vs 0.68)**: a prediction-gain signal that, unlike re-mention, the system cannot satisfy without predicting the text better.

## Improving typed inference (16 Nov 2026, tools/typed_plus.py; round 2 = visit 4 on distant_50k, batch = the rule's ranks 0-200 (right 0.730), NEXT TIER = ranks 200-400 (right 0.545, as the GPU's second batch); composition table from round 1's 286 confirmations; A6 stand-in trunk; usefulness from the first half of the later text, scored on the second)
| typing of the batch | typed claim right exact / 'is in' as one | bits saved per 1,000 bytes (second half) |
|---|---|---|
| F1 'connected to' | 0 / 0 | -0.56 |
| F2 class pair (route 1, = cg27a_kty) | 0.365 / 0.615 | -0.12 |
| **F3 composition wording where the table types it (18%), else class pair** | **0.400 / 0.575** | **+0.69** |
| F4 usefulness chooses among 4 class-pair wordings + the composition one | 0.215 / 0.305 | +0.40 |
| precision | all | kept by usefulness | composition-supported | either | neither |
|---|---|---|---|---|---|
| batch (0-200) | 0.730 | 0.795 (39) | 0.833 (54) | 0.797 (79) | 0.686 |
| **next tier (200-400)** | 0.545 | **0.735 (34)** | 0.667 (18) | 0.708 (48) | 0.493 |
- **Combined typing (F3) is the best form**: more exact relations (0.40) and the most bits saved (+0.69 vs -0.12 for route 1 alone). **Usefulness should keep or drop, not choose the wording**: chosen by bits it picks wordings that predict text ('state of', "'s") but name the wrong relation (0.215 exact). **Usefulness keeps precision on the next tier where the GPU judge failed (0.735 vs 0.545 for all; the relational judge 0.52 on GPU)** - the label-free filter for later batches. Fix: list wordings in quotes ('" and "', 9 writes) now excluded from the class-pair table in the package.

## Are the 'wrong' inferences wrong? (16 Nov 2026, tools/label_check.py; the GPU path's 200 writes at visit 2, distant_50k; read by hand)
62 of 200 are wrong by the distant (Wikidata) labels; 24 of those are stated later in a relational record (the judge keeps them). Read by hand: **about 8 are plainly right** (Orlando ~ Florida, Memphis ~ Tennessee, Seattle ~ Washington, Houston ~ Texas, Eritrea ~ Ethiopia, Namibia ~ South Africa, Egypt ~ Sudan, Asia ~ Europe: containment or borders missing from the documents' labels), **about 6 plausible** (Collins Crime Club ~ UK, Dodd ~ US, Nazi ~ World War II, Nordic ~ Norwegian, Bavaria ~ Bavarian, United Kingdom ~ United States), **about 10 not a relation**: co-members of a list the filter misses ('Microsoft Windows , OS X and Android'; consoles; 'African - American'), a number ('10 ~ UK Singles Chart'). So the true precision of the writes is about 0.73-0.76, not 0.69 (8-14 of the 62 'wrong' are right), and typed claims called wrong include right ones ('Seattle in Washington', 'Houston in Texas'). Fixable: one comma + 'and' ('X , Y and Z') is a list too - the judge's list filter misses it.

## Replications (16 Nov 2026, night)
- typed_plus on ROUND 3 (visit 6; batch right 0.675, next tier 0.685): F3 combined typing again best (exact 0.365 vs 0.335 for the class pair; bits +0.47 vs -0.16; 'connected to' -0.75); usefulness chooses wordings badly again (0.225 exact). Precision: batch all 0.675, kept by usefulness 0.771, composition-supported 0.833, either 0.793, neither 0.584; next tier all 0.685, usefulness 0.781, composition 0.789, either 0.778. Round 2 with the B6 trunk: usefulness keeps 0.816 (batch) / 0.765 (next tier). **The usefulness and composition filters replicate across rounds and trunks (+0.07 to +0.16 over all writes).**
- THE JUDGE'S LIST FILTER (tools/judge_list.py): also calling 'X , Y and Z' (one comma + and/or) a list removes 27-58 of ~550 relational records a round and **raises the judge's AUC in every round and tier by +0.02-0.03** (batch 0.659 / 0.657 / 0.617 -> 0.682 / 0.688 / 0.642; next tier 0.544 / 0.596 / 0.539 -> 0.568 / 0.614 / 0.558) and kept precision by +0.03-0.05 (batch 0.791 / 0.810 / 0.743 -> 0.818 / 0.865 / 0.772). A small, consistent judge fix; the next tier stays weak (AUC 0.56-0.61), where usefulness holds.
- DENSER COMPOSITION TABLE by merging wordings that state the same name pairs (tools/typed_plus.py TP_CLUSTER 1, Jaccard 0.1 / 0.05): **negative** - 6,477 wordings with 3+ pairs fall into 6,466 / 6,462 clusters (the same pair is rarely stated in two wordings; the few merges are sensible: 'summer olympics in / olympic games in / summer olympics held in', 'state of / representative from'); the table's coverage stays 72 of 400 and every typing and precision figure is unchanged. Denser composition needs a different similarity (the trunk's relation key, or the pair's class) - not pair overlap.

## THE FULL LOOP, offline over three rounds (16 Nov 2026, tools/loop_sim.py; distant_50k, written pairs persist across rounds as on the GPU, so rounds 2-3 write the next pairs - batch right 0.690 / 0.570 / 0.595, next tier 0.530 / 0.515 / 0.535; typing = composition wording (table from kept, confirmed writes) else class pair; A6 trunk)
| policy | kept r1 / r2 / r3 (right) | cumulative memory: right | bits saved by the memory on the round's held-out text r1 / r2 / r3 |
|---|---|---|---|
| JUDGE (batch, relational record later; the GPU now) | 115 (0.791) / 85 (0.718) / 77 (0.636) | 0.726 (277) | +0.04 / -0.03 / +1.01 |
| USEFUL (all 400, saved bits on the first half) | 65 (0.723) / 58 (0.690) / 53 (0.660) | 0.693 (176) | **+0.31 / +0.20 / +1.29** |
| JUDGE + USEFUL (batch by judge, next tier by usefulness) | 144 (0.771) / 110 (0.709) / 97 (0.629) | 0.712 (351) | +0.13 / -0.04 / +1.10 |
| ALL BATCH (no filter) | 200 (0.690) / 200 (0.570) / 200 (0.595) | 0.618 (600) | +0.01 / -0.12 / +0.88 |
- **Usefulness builds the memory that helps the model most in every round** (by the author's criterion), with precision between the judge's and none (0.66-0.72); the judge is the more precise filter in round 1 (0.79) but loses that edge by round 3 (0.636 vs 0.660), as on the GPU. Unfiltered writing makes the memory cost bits in round 2 (-0.12). The composition table types a growing share of kept writes (0.10-0.18 -> 0.21-0.30). Next: a policy keeping what EITHER filter passes, and usefulness thresholds.

## RETURN-TO-QUERY: composition through the key-to-key memory (16 Nov 2026, tools/jepa_mem.py JM_RQ 1; 18k stand-in trunk, two seeds, two-hop on chains never taught, rank / open)
| | K-JEPA s0 | K-JEPA s1 | K-RAW s0 | K-RAW s1 |
|---|---|---|---|---|
| LATENT (the read's value as the next name key) | 0.67 / 0.27 | 0.67 / 0.42 | 0.42 / 0.20 | 0.50 / 0.20 |
| **SOFT DECODE (attention over the known names' keys; no training)** | **0.85 / 0.65** | **0.87 / 0.70** | **0.88 / 0.67** | **0.75 / 0.55** |
| learned map G (trained on single hops only) | 0.50 / 0.23 | 0.62 / 0.32 | 0.45 / 0.15 | 0.35 / 0.18 |
| DECODED (hard: nearest known name) | 0.85 / 0.65 | 0.88 / 0.73 | 0.85 / 0.68 | 0.77 / 0.57 |
- **Positive: a soft decode closes the latent gap completely** (0.67 -> 0.85-0.87 rank), with no training and no discrete step - the next query is a weighted mix of known names' keys, so composition stays differentiable. A map learned from single hops does WORSE than none (it fits the hops it saw). For the package: composition through the memory needs a key-valued read (the GPU memory returns byte logits), then soft decode between hops.
- POLICIES, A6 and B6 trunks (r1 / r2 / r3; kept (right), memory saves): USEFUL (gain > 0) 65 (0.72) / 58 (0.69) / 53 (0.66), +0.31 / +0.18 / +1.31 (B6: +0.30 / +0.17 / +1.24); **USEFUL>2 55 (0.71) / 50 (0.72) / 45 (0.69), +0.35 / +0.22 / +1.25 (B6: +0.32 / +0.17 / +1.29)**; USEFUL>5 48 (0.69) / 36 (0.72) / 37 (0.70), +0.36 / +0.24 / +1.21; EITHER (judge or useful, all 400) 215 (0.70) / 174 (0.64) / 165 (0.64), +0.02 / -0.15 / +0.88; EITHER-BATCH 129 (0.78) / 97 (0.69) / 93 (0.63), +0.04 / -0.02 / +1.05; JUDGE 115 (0.79) / 85 (0.72) / 77 (0.64), +0.04 / -0.05 / +1.03. **Keeping what the judge passes adds wrong and useless writes; usefulness above a small threshold (2 bits per 1,000 bytes of the name's later text) is the best keep rule on both trunks: the most bits saved, precision 0.69-0.72 steady across rounds where the judge falls 0.79 -> 0.64.**
- BOTH (the judge AND usefulness), A6: kept 39 (0.769) / 38 (0.711) / 28 (0.714), memory right 0.769 / 0.740 / 0.733, saves +0.29 / +0.21 / **+1.61**; against USEFUL>2 55 (0.709) / 50 (0.720) / 45 (0.689), memory 0.709 / 0.714 / 0.707, +0.35 / +0.23 / +1.43. **The intersection is the precise variant (memory 0.73-0.77 right) at about the same usefulness**: the judge adds value as a second check, not as an alternative.
- WITH THE WIDER LIST FILTER (LS_LIST2 1; the judge's and the spans' list test also catch 'X , Y and Z'): JUDGE 110 (0.818) / 77 (0.779) / 74 (0.662), memory 0.762, saves +0.17 / -0.02 / +1.11; **BOTH 40 (0.800) / 37 (0.865) / 27 (0.741), memory 0.800 / 0.831 / 0.808 right, saves +0.36 / +0.18 / +1.37**; USEFUL>2 62 (0.726) / 60 (0.683) / 46 (0.674), +0.42 / +0.26 / +1.56. The list fix lifts every filter; **judge (wider lists) AND usefulness keeps a memory 0.80-0.83 right** that still saves bits every round. Package option inf_rret_list2 (off by default).

## CORRECTION (16 Nov 2026, 00:00): the composition-typing results were inflated by repeated pairs
tools/typed_routes.py (route 3) and tools/typed_plus.py ran each write round INDEPENDENTLY (the ledger reset), so later rounds re-wrote earlier pairs: 97 of round 2's 200 writes and 136 of round 3's were written before, and **65 of round 2's writes had been CONFIRMED in round 1** - the composition table then 'typed' them with their own confirmed wording. So the route-3 figures (types 27-36%, right 0.56-0.63), F3's edge (exact 0.40 vs 0.365, bits +0.69 vs -0.12) and **'composition-supported pairs right 0.83' are mostly re-reading confirmations, not inference.** Through the package's own path (premises._rcode_step with inf_rti_comp 1, the ledger persisting as on the GPU, tools: inline check), the table (244 / 427 cells) types only **3 of 200 and 6 of 200** new inferences at visits 4 / 6. Typed composition does not yet generalise to new pairs. Unaffected: route 1 / route 2 and tools/useful_writes.py (round 1 only), tools/loop_sim.py (persistent ledger: the usefulness, judge and BOTH results stand; its 'composition-typed' column counts non-'in' wordings, not the table). The package keeps inf_rti_comp as an option (off); it is not a build.
- DOES IT GENERALISE WITH COARSER KEYS? (tools/comp_general.py; persistent ledger, tables from earlier rounds' confirmed writes, applied to the next round's NEW writes; rounds 2 / 3, right as pairs 0.585 / 0.630): exact premise wordings type 3 / 6 (right 0.00 / 0.17 exact); premise last words 121 / 139 (exact 0.207 / 0.194, 'is in' 0.298 / 0.388); the three names' classes 172 / 191 (0.163 / 0.141, 0.273 / 0.267); the two ends' classes 173 / 192 (0.162 / 0.161, 0.260 / 0.323); back-off 181 / 196 (0.193 / 0.163, 0.276 / 0.306); **route 1, the store's class-pair wording, 200 / 200 (0.235 / 0.245 exact, 0.420 / 0.490 'is in') - better than every composition key. Negative: composition learned from confirmations does not type new pairs better than the store-wide class pair; route 1 (cg27a_kty) stays the typing.** (On new pairs typing is weaker than round 1's 0.61 because later rounds' pairs are worse.)
- NAME-SPECIFIC typing (the tail name's commonest relational wording with names of the head's class, else route 1), new pairs rounds 2 / 3: exact 0.230 / 0.260, 'is in' 0.335 / 0.410, against route 1's 0.235 / 0.245 and 0.420 / 0.490. **Negative: the store-wide class-pair wording (route 1, as in cg27a_kty) remains the best typing on new pairs.**
- THE GPU'S SIGNAL (usefulness measured only on the SECOND NAME, as the memory-feedback readout does; LS_LIST2 1): BOTH-NAME keeps 13 / 9 / 6 a round, right 0.923 / 0.889 / 1.000, memory 0.909-0.929 right, saves +0.15 / -0.11 / +1.03; USEFUL-NAME 15 / 14 / 8 (0.80 / 0.86 / 0.88); against BOTH on the whole span 40 / 37 / 27 (0.80 / 0.87 / 0.74), memory 0.80-0.83, saves +0.36 / +0.19 / +1.57. **The second-name signal is very precise but keeps a third as much and saves fewer bits.** Package: inf_kmc_useful_span (whole span after the subject: wording + name); cg27a_kus uses it, its [USEFULNESS] readout reports both measures.
- FOUR ROUNDS (LS_ROUNDS 1,3,5,7; wider lists), A6 / B6: rounds 1-3 replicate the three-round run on both trunks (B6: BOTH memory 0.769 / 0.806 / 0.784). **Round 4 (visit 8) degrades every rule**: the writes themselves are worse (batch 0.535, next tier 0.450); BOTH keeps 12 / 16 at 0.500, USEFUL>2 33 / 34 at **0.364 / 0.382**, JUDGE 68 / 67 at 0.618 / 0.612. The cumulative memory stays best under BOTH (0.776 / 0.743 right, saving +0.77 / +0.70) against JUDGE 0.733 / 0.732 (+0.15 / +0.17) and USEFUL>2 0.642 / 0.635. **As the good candidates run out, usefulness alone stops tracking truth; judge AND useful degrades least.** Implication: the write budget should shrink as candidate quality falls (e.g. stop a round when few writes pass), rather than writing 200 a round regardless.
- ADAPTIVE BUDGET (BOTH-ADAPT: blocks of 100 by rank, stop at the first block where under LS_STOP pass judge AND usefulness; four rounds, A6, wider lists): stop 0.1 keeps 23 / 20 / 17 / 2, memory 0.826 / 0.837 / 0.817 / 0.806 right, saves +0.33 / +0.17 / +1.48 / +0.60; stop 0.2 keeps 14 / 14 / 10 / 2, memory 0.929 / 0.929 / 0.895 / 0.875, saves +0.19 / +0.08 / +1.18 / +0.35; against BOTH on all 400: 40 / 37 / 27 / 11, memory 0.800 / 0.831 / 0.808 / 0.783, +0.36 / +0.19 / +1.59 / +0.93. **The budget is a precision dial: stopping early shuts round 4 down (2 kept) and lifts the memory to 0.81-0.93 right, at the cost of bits saved (about a third less at 0.1).** A clean trade-off, no free lunch; the round-4 collapse of 'keep everything that passes' is avoided.
- ADAPTIVE BUDGET on the B6 trunk (stop 0.1): keeps 23 / 18 / 9 / 4, memory 0.826 / 0.829 / 0.820 / 0.796 right, saves +0.25 / +0.15 / +1.32 / +0.46; against BOTH on all 400: 39 / 33 / 25 / 15, memory 0.769 / 0.806 / 0.784 / 0.750, +0.28 / +0.14 / +1.39 / +0.84. **Replicates: +0.02-0.06 on the memory's precision, about the same bits in rounds 1-3, less in the thin round 4.**
- LEARNED SELECTION FROM USEFULNESS (SEL-BOTH: a logistic selector on write-time features - score, rank, premises, degrees, classes - trained on earlier rounds' usefulness of all 400 candidates, picks the 200 to write; then judge AND useful; against the rule's top 200, RULE-BOTH), four rounds, A6 / B6: the selector's 200 are LESS often right (rounds 2-4: 0.565 / 0.485 / 0.480 vs the rule's 0.570 / 0.595 / 0.540; B6 0.565 / 0.525 / 0.465), overlap with the rule's 72-122; kept precision mixed (round 2 0.909 / 0.938 vs 0.850 / 0.833; round 3 0.667 / 0.688 vs 0.765 / 0.688; round 4 0.571 / 0.500 vs 0.333 / 0.333) and **the memory saves far fewer bits (round 3 +0.54 / +0.35 vs +1.50 / +1.35)**. **Negative: usefulness is a good FILTER but a poor training target for write-time selection** - from these features the selector cannot predict which pairs later text will make useful; the rule's ranking is better.
- RETYPE ON CONFIRMATION (BOTH-RETYPE: kept, confirmed inferences rewritten in the wording of the later relational sentence that states them), four rounds, A6 / B6: the memory's typed claims right EXACT 0.522-0.625 (vs 0.301-0.333 for the class-pair wording), 'is in' 0.584-0.727 (vs 0.575-0.653); bits saved +1.25 / +2.30 / +2.72 / +1.70 (A6) and +1.25 / +2.05 / +2.53 / +1.95 (B6) vs +0.14-1.59. **Caution: in this run the confirming sentence could come from the held-out half the memory is then scored on (a leak for the current round's entries; also for every policy's judge). Rerun with the judge and retyping restricted to the first half (LS_HALF 1) before trusting the bits; the exact-relation gain (scored by labels) does not depend on it.**
- HELD-OUT RERUN (LS_HALF 1: the judge and retyping see only the first half of each round's window; the memory scored on the second), four rounds, A6 / B6:
| policy | memory right (r1-r4) | typed claims right exact / 'is in' | bits saved r1 / r2 / r3 / r4 (A6; B6) |
|---|---|---|---|
| BOTH | 0.829 / 0.839 / 0.810 / 0.783 (B6 0.767-0.828) | 0.30-0.33 / 0.60-0.67 | +0.26 / +0.11 / +1.19 / +0.63; +0.20 / +0.05 / +1.12 / +0.59 |
| **BOTH-RETYPE** | same entries | **0.54-0.64 / 0.61-0.73** | **+1.05 / +1.72 / +1.63 / +1.17; +0.94 / +1.52 / +1.62 / +1.29** |
| BOTH-RETRACT (drop entries that now cost bits) | 0.829 / 0.848 / 0.792 / **0.656** (B6 0.667) | 0.22-0.33 | +0.26 / +0.13 / +1.19 / +0.44 |
| USEFUL>2 | 0.726 / 0.702 / 0.695 / 0.646 | 0.25-0.28 | +0.42 / +0.26 / +1.64 / +1.15 |
- **RETYPE ON CONFIRMATION HOLDS ON HELD-OUT TEXT: exact relations double (0.30-0.33 -> 0.54-0.64) and the memory saves far more bits in rounds 1-2 (+1.05 / +1.72 vs +0.26 / +0.11) on both trunks** - the confirming sentence's wording is the right type for the kept inference, and it predicts later text. (The judge's own held-out precision: BOTH memory 0.78-0.84 right.) **RETRACTION BY USEFULNESS over time is negative**: one round's text is too noisy a test - it drops 15-24 entries a round, mostly right ones, and by round 4 the memory is 0.656 / 0.667 right against 0.783 / 0.767.
- Package: inf_rti_retype (judge time: kept + relationally confirmed inferences take the record's wording and order; pool entries follow at the next write). Through the package path on distant_50k: 110 of the 200 visit-2 writes retyped at visit 4 (right as pairs 0.818), typed claims exact 0.627 against 0.233 for the rest. Some wordings are long or noisy (', located southeast of downtown', ': jpmorgan chase tower ('). Preset cg27a_krt (= cg27a_kus + retype) prepared.
- RETYPE WORDING CLEANUP (LS_RT_SHORT 1: retype only with <= 4 words and no ':' '(' etc.), A6, held out: exact 0.446-0.486 (vs 0.554-0.613 retyping with every wording), bits +1.10 / +1.75 / +1.84 / +1.11 (vs +1.05 / +1.77 / +1.81 / +1.33). **Negative: the long, odd wordings are often the exact relation ('located southeast of downtown'); keep every confirming wording.**
- RETYPED WORDINGS AS FUTURE TYPES (BOTH-RETYPE-T: per class pair, the wordings of earlier rounds' retyped kept inferences type the NEW writes; 3+ needed, else route 1), A6 / B6: new writes typed exact 0.203 / 0.198 / 0.185 vs 0.200 / 0.198 / 0.182 (B6 0.212 / 0.210 / 0.193 vs 0.200 / 0.198 / 0.182) in rounds 2-4. **Neutral**: the confirmed wordings per class pair are dominated by the same 'in'; fine types arrive by confirmation (retyping), not at write time.
- TRUE ADAPTIVE WRITES (LS_DEFER 1: candidates past the adaptive stop are never written - they leave the ledger and can be proposed again with more evidence), A6, stop 0.1, four rounds: later rounds' writes are better (batch right 0.585 / 0.625 / 0.605 vs 0.570 / 0.595 / 0.535 in rounds 2-4) and the memory more precise (0.850 / 0.897 / 0.848 / 0.820 vs 0.850 / 0.853 / 0.816 / 0.804) at about the same bits (+0.23 / -0.02 / +1.21 / +0.41 vs +0.23 / +0.04 / +1.19 / +0.48). **Modest positive: deferring unwritten candidates keeps later rounds' pool healthier** (round 4 writes 0.605 vs 0.535).
- EVIDENCE WINDOW (LS_UWIN: share of the two visits after a write that the judge and usefulness see; the memory scored on the rest), A6, four rounds, BOTH / BOTH-RETYPE:
| window | kept r1-r4 | memory right | retyped memory exact | retyped memory saves r1-r4 |
|---|---|---|---|---|
| 0.25 | 19 / 12 / 9 / 7 | 0.842 / 0.839 / 0.800 / 0.745 | 0.57-0.68 | +0.93 / +1.38 / +1.86 / +1.66 |
| 0.5 | 35 / 27 / 22 / 8 | 0.829 / 0.839 / 0.810 / 0.783 | 0.55-0.61 | +1.05 / +1.77 / +1.81 / +1.35 |
| 0.75 | 46 / 29 / 27 / 16 | 0.783 / 0.800 / 0.775 / 0.763 | 0.54-0.57 | +1.24 / +2.34 / +2.53 / +1.25 |
- More evidence keeps more (46 vs 19 in round 1) at slightly lower precision (0.76-0.80 vs 0.75-0.84) and saves the most bits in rounds 2-3; less evidence keeps fewer and loses the thin round 4. **No sharp optimum: half the window (the default, about one visit on the GPU) is a sound middle; the GPU's two visits correspond to the generous end.** Retyping helps at every window.
- JUDGE WITH WORDING SPECIFICITY (tools/judge_list.py; confirmations weighted by the wording's rarity, label-free), rounds at visits 2 / 4 / 6, AUC batch / next tier: wider filter (count) 0.682 / 0.568, 0.688 / 0.614, 0.642 / 0.558; IDF-weighted count 0.672 / 0.565, 0.694 / 0.603, 0.639 / 0.540; most specific wording 0.651 / 0.558, 0.695 / 0.593, 0.613 / 0.523. **Negative: rare wordings are not stronger evidence; the plain count of relational confirmations (with the wider list filter) stays the best judge.**

## Consolidation of kept typed inferences into a trunk (16 Nov 2026, tools/consolidate.py; A6 stand-in trunk, the GPU path's 200 typed writes at visit 2 (right 0.690), six frames, 8 fact + 24 wiki rows a step, 400 steps; test wording never taught, rank among 10, chance ~0.10; base rank 0.115, wiki 2.0268)
| rule | lr 1e-4 | lr 3e-4 | lr 1e-3 |
|---|---|---|---|
| FULL | rank 0.105, +0.071 bits | 0.155, +0.091 | 0.255, +0.173 |
| KL (wiki rows anchored to the old model) | 0.140, +0.064 | 0.185, +0.069 | **0.260, +0.116** |
| MLPOUT | 0.090, +0.065 | 0.090, +0.079 | 0.130, +0.088 |
| LORA (rank 8) | 0.090, +0.052 | 0.090, +0.053 | 0.085, +0.053 |
- Weak at this scale: the best recall in an unseen wording is rank 0.26 (chance 0.10), at +0.12 bits with the KL anchor (the same recall costs +0.17 without it). MLP-out (the GPU's path) and a rank-8 adapter barely learn at these rates. Every rule also adds +0.05 bits with almost no learning - a drift control (wiki rows only) and higher adapter rates are running.
- DRIFT CONTROL AND ADAPTER RATES: wiki rows only (NONE) add +0.014 (lr 1e-4) and **+0.139 (lr 1e-3)** bits by themselves - most of the high-rate 'forgetting' is the fine-tuning text's drift, not the facts (net of it: FULL 1e-3 +0.034 for rank 0.255; FULL 1e-4 +0.057). A rank-8 adapter learns nothing at any rate (rank 0.095 / 0.100 / 0.115 at 3e-3 / 1e-2 / 3e-2; +0.059 / +0.091 / +0.172). **The limit is recall in an unseen wording (0.26 at best), not forgetting**: the trunk learns the taught frames but transfers little to a new one (open-ended 0.13). Physics of LMs 3.1 (Allen-Zhu & Li): extraction needs varied wordings during learning ('knowledge augmentation') -> next: 6 vs 24 frames.
- KNOWLEDGE AUGMENTATION (CS_FRAMES 24: the same 200 inferences taught in 24 varied frames instead of 6; 600 steps; test wording never taught): **KL lr 1e-3: rank 0.230 -> 0.710, open-ended 0.080 -> 0.520** (right 0.732, wrong 0.661), +0.118 bits (6 frames +0.099); KL 3e-4: 0.225 -> 0.365 (open 0.270), +0.054; FULL 1e-3: 0.265 -> 0.335 (+0.148); FULL 3e-4: 0.215 -> 0.225. **Positive and large: varied teaching wordings are what lets a trunk recall consolidated inferences in a new wording, and the KL anchor makes the most of them** (as Allen-Zhu & Li's knowledge augmentation). The package teaches with 7 frames (inf_distill_forms 7, up to 21 available) through mlp_out only -> next: MLPOUT with 6 vs 24 frames, and 12 frames.
- FRAMES ON THE GPU'S PATH: MLPOUT (only the blocks' MLP output matrices take the question gradient, as the GPU builds) gains nothing from frames - 6 frames rank 0.135 / 0.190 (lr 1e-3 / 3e-3), 24 frames 0.140 / 0.180 (open 0.125 / 0.165), +0.09 / +0.13-0.15 bits. KL with 12 frames: rank 0.560, open 0.415, +0.112 (24 frames 0.710 / 0.520). **The varied-wording gain needs every weight (with the KL anchor on text); the GPU's mlp_out path (chosen 11 Nov for its low cost: recall 0.11-0.16 vs 0.26-0.38 with all parameters) cannot use it.** A KL-only drift control (wiki rows anchored, no facts) is running to net out the cost.
- KL DRIFT CONTROL (KLNONE: wiki rows anchored to the old model, no facts): +0.007 (lr 3e-4) / +0.040 (lr 1e-3) bits. **Net of drift, KL + 24 frames costs about +0.05 (lr 3e-4: rank 0.365, open 0.27) to +0.08 bits (lr 1e-3: rank 0.71, open 0.52)** - the first consolidation route on this stand-in with real recall in an unseen wording at a moderate cost. Candidate build (one change: the question rows' consolidation path): inf_distill_wake_params '' (every weight) + inf_distill_forms 21 + a KL anchor on text; needs a package-level check of its forgetting first.
- PACKAGE RUN ON THE LOCAL MINI-PILE (tools/local_run.py: the GPU preset through python -m platypus, five local domains, d 64): cg27a_kty runs end to end (20 steps a task: no failures); [KEYED UNITS] at visit 5: trunk 3.588, + NAME -0.0330, + NAME + DATE + NUMBER -0.0353 bits per byte (DATE + NUMBER -0.0024 on top). A 250-steps-a-task run is under way.
- LOCAL PACKAGE RUN, 250 STEPS A TASK - SETUP BUG FOUND: the first full-length local run of cg27a_kty had inference OFF. tools/local_run.py passed the preset file as LAB_OVR, which REPLACES the build's whole default set (DEFAULT_LAB_OVR: inf_on, n_passes, the inference options), so only the trunk, dreams and keyed units ran. Fixed (merged onto DEFAULT_LAB_OVR, as scripts/merge_overrides.py does; the GPU path was never affected). That run is still a usable trunk-only reference: forgetting -0.016, final BPC 2.098 (5 domains, d 64, 250 steps); keyed units -0.003 to -0.005 bits per byte at visits 1, 2, 5 and +0.002 / -0.001 at visits 3, 4 (small at this width). The run with inference on is going now.
- LOCAL PACKAGE RUN WITH INFERENCE ON (cg27a_kty, 250 steps a task, distant_50k; stopped in visit 3 for cost): every inference readout runs, no failures. Visit 2 in the package: **[TYPED WRITES] 200 of 200 written inferences typed** ('in' 171, 'as' 8, 'of' 5, 'to' 4, 'during' 4; e.g. 'Nashville in Tennessee', 'Bergen in Norway'; 'Seattle in Washington' and 'Houston in Texas' are labelled WRONG by the distant labels though true - the undercount seen offline); compressed fact table code AUC 0.633 vs Adamic-Adar 0.597 (top 5% 0.309 vs 0.232; GPU-like); integrated keyed memory taught facts 0.17 / 0.33 open / rank vs trunk 0.00 / 0.32, two-hop first hop 0.50 vs 0.12; 200 inferences awaiting judgement (the judge needs the later visits). COST at this scale: a visit's sleep 1855-2709 s (JEPA index readout training 1165-1239 s, memory writes 552-633 s), so a 5-visit run takes ~2.7 h; package A/B runs now use the JEPA index at its local size (128 x 3, 100 steps), distant_10k and 150 steps a task.
- PACKAGE A/B, RUN A (cg27a_kty reference; 150 steps a task, distant_10k, JEPA index 128 x 3; ~13-19 min a visit), interim to visit 4: the smaller store writes worse pairs (right 0.33 at visit 2 vs 0.69 on the GPU with 50k+ documents), so the judge and usefulness see a hard batch. **Relational judge at visit 4: AUC 0.635, flagged 40 wrong 0.85 (unflagged 0.62), kept right 0.38 vs flagged 0.15; 14 retracted, wrong 0.93.** **[USEFULNESS] (readout; the kus keep rule is off here): useful right 0.692 (13) vs not useful 0.325 (40); useful AND judged-kept right 0.750 (12) vs kept 0.38 - the kus rule's precision gain shows inside the package (small n).** Memory feedback AUC 0.594 (mean bits saved). Typed inferences in the memory still COST bits where read at this scale (right -224, wrong -1395 over 190 spans) - the gate rule's case. Keyed units -0.012 bits per byte at visit 4. A JUNK WORDING appears in the class-pair table: ') was' (30 of 200 at visit 2, 11 at visit 4) - a span that starts at a closing parenthesis.
- PACKAGE A/B, RUN A FINAL (cg27a_kty, 150 steps a task, distant_10k, JEPA index 128 x 3): **HEADLINE forgetting -0.036, final BPC 2.603** (weights only; store on -0.034 / 2.605). Trunk recall of taught facts in test wordings (the reference for run B): ever taught 0.04 open (106), 4+ visits 0.00; written inferences taught right 0.00 / wrong 0.01 from the trunk; integrated keyed memory taught facts 0.68 / 0.83 vs trunk 0.02 / 0.47 rank, paraphrased 0.03 / 0.49 vs trunk 0.00 / 0.46. USEFULNESS at visit 5 fell back: useful right 0.500 (26) vs not useful 0.452 (84); useful AND kept 0.462 (13) (visit 4: 0.750 (12)) - at this store size the usefulness gain is NOT stable from one visit to the next; the GPU-scale offline result (0.78-0.84) stands, but the package evidence is mixed.
- PACKAGE A/B, RUN B FINAL - **NEGATIVE: the consolidation path (every weight, 21 wordings, soft anchor 0.5) FORGETS.** Same settings as run A plus LR_OVR {"inf_distill_wake_params": "", "inf_distill_forms": 21, "inf_distill_wake_akl": 0.5}. HEADLINE forgetting **+0.094 vs A -0.036**, final BPC **2.731 vs 2.603**; Wikipedia forgetting +0.346 (A -0.018), ArXiv / PubMed +0.023 / +0.024 (A -0.039 / -0.017). Held-out Wikipedia bits through the run: 2.616, 2.626, 2.646, then 3.012 at visit 5 (A: 2.604, 2.578, 2.560, 2.591) - a slow drift and a jump during the last domain. It does what it was for - taught facts recalled from the trunk in test wordings 0.16 / 0.18 / 0.19 / 0.21 at visits 2-5 vs A 0.06 / 0.01 / 0.03 / 0.04 - but written WRONG inferences are recalled too (0.07 taught vs A 0.01: false memories). The offline consolidation test anchored on the very text it scored (wiki); in the package the soft anchor covers stored sentences of old slots only, so the rest of Wikipedia drifts. Not a build. Next: B2 = the same + the gradient projection (inf_distill_wake_proj 1: drop the question gradient's component that raises next-byte loss).
- PACKAGE A/B, RUN B2 (run B + inf_distill_wake_proj 1, the gradient projection) - **NEGATIVE, a little less bad than B.** HEADLINE WEIGHTS ONLY (fast store off): FORGETTING +0.080 | FINAL BPC 2.724 || with the fast store on: FORGETTING +0.082 | FINAL BPC 2.726. Per domain: Wikipedia forgetting +0.286 (B +0.346, A -0.018), ArXiv +0.035, PubMed +0.010. Held-out Wikipedia bits 2.617 / 2.628 / 2.647 / 2.927 at visits 2-5 (B 2.616 / 2.626 / 2.646 / 3.012; A ... / 2.591). Trunk recall of taught facts 0.25 (B 0.21, A 0.04), but false memories rise with it: written WRONG inferences recalled 0.10 taught (B 0.07, A 0.01) vs right 0.06. Conclusion: in the package, every-weight question rows drift the trunk on text the anchor does not cover, and the projection only trims it. The consolidation path stays offline; the GPU keeps MLP-out question rows. A fix would need the anchor on held-out-like text of every old domain (dream / replay rows), not stored sentences - queued as a later idea, not a build.
- kus IN THE PACKAGE, interim to visit 4 (same settings as run A; keep rule on): same trunk path as A (post-train BPC 2.941 / 2.591 / 2.655 / 2.562, identical). Judge AUC 0.625, flagged 40 wrong 0.80, 13 retracted (wrong 0.92). **Usefulness feedback covers only 36 of 400 written inferences at this store size (distant_10k), so the keep rule (judge AND useful) passes just 3 to the trunk (right 0.667)** - precise but nearly empty. Among the 400: useful right 0.50 (4), not useful 0.31 (32), no feedback 0.41 (364). RISK FOR THE GPU kus BUILD: where reading feedback is sparse, kus teaches very few inferences; check [USEFULNESS] coverage in the kty GPU log first. A softer variant - judge AND NOT HARMFUL (drop only the measured not-useful) - queued offline.
- kus IN THE PACKAGE, FINAL (same settings as run A; keep rule on, whole-span usefulness): **HEADLINE forgetting -0.040 vs A -0.036, final BPC 2.599 vs 2.603 - no harm (within noise; Wikipedia -0.044 vs -0.018).** At visit 5 feedback covered 129 of 400 written inferences (36 at visit 4): **useful right 0.656 (32) vs not useful 0.423 (97) vs no feedback 0.365; useful AND judged-kept right 0.667 (15) vs all judged-kept 0.36** - the keep rule's precision gain holds in the package at visit 5 (run A's readout gave 0.462 (13) at visit 5 with the second-name measure, 0.750 (12) at visit 4). Written WRONG inferences recalled from the trunk 0.00 (A 0.01). The coverage risk stands: the trunk gets 15 of 160 judged-kept inferences when feedback is sparse. kus is safe to release; JUDGE AND NOT HARMFUL (queue 25) is the variant to test for coverage.
- TYPED WORDING CLEANUP (inf_rti_clean / LS_CLEAN; loop_sim BOTH and BOTH-RETYPE, LS_LIST2 1, A6, distant_10k and 50k) - **NEUTRAL TO NEGATIVE, not adopted.** Write-time exact type accuracy unchanged (50k: 0.253 / 0.203 / 0.198 vs 0.253 / 0.200 / 0.198; 10k identical); memory precision a little lower (50k BOTH 0.771 / 0.803 / 0.787 vs 0.829 / 0.839 / 0.810) and bits saved lower in round 3 (BOTH +0.22 vs +1.32, BOTH-RETYPE +0.86 vs +1.81). ') was' is mostly a real wording (a parenthesis closing after the head: 'X (1950-2010) was ...'), not junk. Option stays off.
- JUDGE AND NOT HARMFUL (offline, loop_sim JUDGE-ALL / BOTH / JUDGE-NOHARM, LS_LIST2 1, A6): offline, nearly every judged pair has a later span of its head (no feedback 1-5 of 81-125), so the rule differs from BOTH by a few pairs: 50k kept 37 / 28 / 27 (right 0.811 / 0.857 / 0.704) vs BOTH 35 / 27 / 22 (0.829 / 0.852 / 0.727) vs the judge alone 125 / 94 / 81 (0.760 / 0.713 / 0.679); memory 0.793 vs 0.810 at round 3, bits +1.12 vs +1.13; 10k similar (memory 0.609 vs 0.556 at round 3). **Keeps BOTH's precision with slightly more kept; but the offline harness cannot reproduce the package's sparse feedback (there, feedback needs the gate open on the second name), so the deciding test is in the package:** keep-rule mode inf_kmc_useful 2 added (only kept inferences MEASURED not useful stay memory-only) with a 'NOT HARMFUL AND judged-kept' readout; 8 package tests pass; local package run started (log lr_kus2.log).
- kus WITH JUDGE AND NOT HARMFUL (inf_kmc_useful 2) IN THE PACKAGE, interim to visit 4: feedback covers 58 of 400; useful right 0.769 (13), not useful 0.289 (45), no feedback 0.404 (342). **NOT HARMFUL AND judged-kept right 0.382 (123) vs useful AND judged-kept 0.833 (12)** (all judged-kept about 0.36): with sparse feedback the soft rule passes nearly everything the judge keeps and gains almost no precision; the strict rule is precise but passes about a tenth. The trade-off is set by feedback coverage, so the GPU kty log's [USEFULNESS] coverage decides which kus setting to release.
- kus WITH JUDGE AND NOT HARMFUL (inf_kmc_useful 2) IN THE PACKAGE, FINAL - **NEGATIVE: the softer rule passes nearly everything the judge keeps and lets false memories into the trunk.** HEADLINE forgetting -0.035, final BPC 2.602 (no harm, as A / kus). Visit 5: feedback on 145 of 400; useful right 0.744 (39), not useful 0.387 (106), no feedback 0.357 (255); **NOT HARMFUL AND judged-kept right 0.371 (116) vs useful AND judged-kept 0.769 (13)**. Written WRONG inferences recalled from the trunk 0.06 taught (kus strict 0.00, A 0.01). The strict rule (the released kus, inf_kmc_useful 1) stays; coverage is the open question for the GPU log.

## Learned unit types (16 Nov 2026, tools/learned_units.py; Re-DocRED, 6k stand-in trunk, memory from 2,753 documents, 300 scored; causal firings; feedback gate)
Can the hand-written keyed units (NAME / DATE / NUMBER) be replaced by GENERIC keys with LEARNED types? Generic = a firing at every word end, key = the last 1 or 2 words; typed = the same with the nearest of 16 codes of the trunk's state (k-means on early documents, then frozen) in the key, the gate per (type, key length).

| bits saved per 1,000 bytes (feedback gate) | all 300 | after warm-up |
|---|---|---|
| HAND (NAME + DATE + NUMBER, the package's units) | 32.5 | 35.6 |
| **GENERIC (every word end, last 1-2 words, no hand rule)** | **82.0** | **87.7** |
| TYPED (generic + learned type in the key) | 41.1 | 53.7 |
| HAND + TYPED | 40.1 | 53.2 |
| HAND + GENERIC | 82.2 | 88.7 |

- **POSITIVE, and larger than expected: generic word keys save 2.5 times the bits of the hand units, and the hand units add almost nothing on top (+0.1 / +0.9).** The hand detectors are not needed for the bits; a generic keyed continuation memory over every word end subsumes them (a byte-level cache in the spirit of cache LMs / kNN-LM / infini-gram, here with a learned gate).
- **Putting the learned type IN the key hurts** (41 vs 82): 16 types split each key's continuations 16 ways, so fewer are found. Types should select or weight keys, not partition them.
- **The learned types rediscover the hand categories unsupervised:** type 7 is 81% NAME firings; type 0 is 63% DATE + 30% NUMBER (one 'date-or-number' type); the rest are mostly other words. The gate trusts type 7 (+0.43) and type 1 (+0.49, 35% NAME) most; types 2, 3, 4, 14 cost bits (-0.4 to -1.4k) and get the lowest trust.
- Caveats: a small trunk (6k steps, 2.24 bits per byte on this text) leaves much for a cache to catch; a stronger trunk will gain less, and the hand units' advantage is cost (far fewer firings). Next: generic + typed as SEPARATE units (types weighting, not partitioning), longer keys, and a stronger trunk (queue 27b).

## Learned subject keys under key drift (16 Nov 2026, tools/learned_keys.py; Re-DocRED, 300 scored documents; NAME firings, causal; feedback gate)
The NAME unit's exact-string key (the package's subject hash) vs a LEARNED key: the nearest of 1,024 codes (k-means) of the trunk's state at the name. DRIFT: the memory is written with the 6k trunk and read after that trunk was trained 1,500 more steps on arXiv text (a continual-learning drift; mean state change 15.0 against a state norm of 17.7 - large). Every arm is scored with the drifted trunk's predictions (3.81 bits per byte here - it forgot much of its wiki text, so absolute savings are inflated; the RANKING is the result).

| bits saved per 1,000 bytes | all 300 | after warm-up |
|---|---|---|
| HASH (exact name; today) | 158.1 | 169.5 |
| REKEY (learned key, every stored key recomputed with the new trunk at sleep) | **181.6** | **191.7** |
| STALE (stored with the old trunk, no fix) | 153.6 | 163.3 |
| QDC (query corrected by the mean drift) | 161.8 | 171.9 |
| FROZEN (a frozen key encoder: old trunk's states both sides) | **180.7** | **191.7** |
| **HASH + REKEY / HASH + FROZEN** (exact key and learned key as two units) | **210.5 / 210.1** | **224.0 / 224.2** |

- **POSITIVE: learned keys beat the exact hash (+15%), and the hash plus a learned key beats either (+33% over the hash alone).** The learned key reaches stored text the exact name cannot: ~18k ALIAS positions (the exact name had nothing stored) saving ~13.9k bits.
- **Drift matters and a FROZEN key encoder solves it as well as re-keying:** stale keys lose 28 bits per 1,000 bytes against re-keyed ones (153.6 vs 181.6, below the hash); query drift compensation recovers only about a quarter (161.8); a frozen encoder matches re-keying exactly (180.7 / 191.7) with no re-keying pass - the Discrete Key-Value Bottleneck recipe holds here.
- The learned codes are COARSE: 1,024 codes over ~109k name firings, about 44 distinct names per code, 1% of codes hold one name. So the learned key works as a 'kind of name in this context' key, not an identity key - it complements the hash rather than replacing it. Next (28b): more codes (8,192) for identity-level keys, a milder drift where the trunk stays good, and the frozen-encoder hybrid in the package's keyed units as a build candidate.

## GPU cg27a_kty (typed writes), interim to visit 3 (log of 16 Nov 18:17 run; vs cg27a_kmr, 15 Nov)
- **Typed writes on the GPU: 200 of 200 typed** ('in' 180, 'as' 10, 'of' 4, "'s" 3; e.g. 'Bergen County in New Jersey', 'Tehran in Iran'; 'Houston in Texas' / 'Miami in Florida' labelled wrong though true - the label undercount). No ') was' wordings at this scale.
- **The memory's cost on written inferences fell sharply (the point of typing): bits saved on the second name, visit 3: RIGHT inferences -584 vs kmr -2,588 (4.4 times less), WRONG -649 vs -389; all inferences -1,233 vs -2,977 (59% less), with the gate open on more spans (494 vs 334).** Label-free feedback separates right from wrong better: AUC 0.527 vs 0.417 (mean bits saved).
- Integrated keyed memory, taught facts: 0.67 / 0.80 open / rank vs kmr 0.50 / 0.78 (trunk alone 0.22 / 0.65 vs 0.30 / 0.73). Fact table code AUC 0.625 vs Adamic-Adar 0.587 (unchanged by design). Keyed units -0.026 bits per byte (NAME), DATE + NUMBER a further -0.003.
- Post-train BPC so far: Gutenberg 2.035 (kmr 2.036), Wikipedia 2.079 (kmr 2.062, +0.017 - at the edge of seed noise; watch the final).
- **USEFULNESS coverage on the GPU: 74 of 200 written inferences have feedback at visit 3 (0.37; local 0.19).** Labelled right: useful 0.792 (24), not useful 0.740 (50), no feedback 0.730 (126) - the GPU batch is easy (about 0.73 right), so usefulness separates only weakly here; the strict kus rule will pass few inferences for a small precision gain. The judge (visit 4) and kus's own log decide.

## GPU cg27a_kus (usefulness keep rule), interim to visit 3-4 (16 Nov 18:26 run) - and a CORRECTION to the kty interim
- **CORRECTION (kty interim above):** kus is identical to kty up to the visit-4 judge (the keep rule only acts on judged inferences), yet at visit 3 its memory cost on written inferences was RIGHT -2,471 / WRONG -1,239 over 728 spans (kty -584 / -649 over 494; kmr -2,588 / -389 over 317). So the raw totals vary a lot between runs of the same configuration (GPU nondeterminism moves where the gate opens). Per span opened, typed writes still cost less than kmr's untyped ones (kty -2.6, kus -5.1, kmr -9.4 bits per span), but the '59% less' figure overstates it; the per-span reduction (about 2-3.5 times) is the fairer reading, from two runs.
- Post-train BPC: kus 2.036 / 2.063 / 2.037 (Gutenberg / Wikipedia / ArXiv) vs kty 2.035 / 2.079 vs kmr 2.036 / 2.062 / 2.019 - kty's Wikipedia +0.017 was seed noise (kus 2.063); ArXiv +0.018 vs kmr, watch.
- **USEFULNESS on the GPU, whole-span measure, visit 3: feedback on 100 of 200 (0.50); useful right 0.870 (23) vs not useful 0.701 (77) vs no feedback 0.740 (100)** - a clear separation (+0.13-0.17), against kty's second-name readout 0.792 vs 0.740. Feedback AUC (mean bits saved) 0.641. The keep rule acts from visit 4 (judge AND useful): expected about 20-25 inferences to the trunk at about 0.87 right, vs kmr's 160 kept at 0.79 (126 right, 34 wrong).

## GPU cg27a_kty to visit 4 (phase 5 running; vs cg27a_kmr)
- **USEFULNESS on the GPU (readout; the rule is off in kty), visit 4: useful right 0.909 (22) vs not useful 0.681 (91) vs no feedback 0.686 (287); whole span 0.938 (16). useful AND judged-kept right 0.938 (16), NOT HARMFUL AND judged-kept 0.847 (85), all judged-kept 0.79 (160: 126 right, 34 wrong).** The strict kus rule's precision holds at GPU scale (0.94 vs 0.79) with about 16 inferences reaching the trunk; unlike locally, the softer rule also gains here (0.85 with 85) - a middle point if the strict rule passes too few.
- Judge unchanged by typing (identical retraction readout: AUC 0.688, flagged wrong 0.45 vs 0.21; it judges pairs on relational records). **Typing sharpens the label-free confirmation signal: confirmations - contradictions AUC 0.707 vs kmr 0.541**; mean bits saved 0.645 vs 0.628; feedback on 110 of 147 (0.75 vs 0.58).
- Memory cost on written inferences at visit 4: -9,687 bits over 1,238 spans (-7.8 per span) vs kmr -5,357 over 519 (-10.3 per span): about 25% less per span, with the gate open on 2.4 times as many spans. Held-out Wikipedia bits with the memory +0.0030 (kmr +0.0054). Typed inferences still cost bits where read; the gate rule (inf_kma_nori) remains the fix for that.
- Integrated keyed memory taught facts 0.67 / 0.77 (kmr 0.77 / 0.92; trunk 0.35 / 0.60 vs 0.40 / 0.85 - this run's trunk recalls less); its taught-inference rows read '- (0)': the readout does not find typed entries (readout gap, to fix: key typed entries head -> tail as in kmfb).
- Post-train BPC kty / kmr: 2.035 / 2.036, 2.079 / 2.062, 2.026 / 2.019, 1.670 / 1.659 - each within seed noise (0.017) but all on the same side (+0.001 to +0.017); kus 2.036 / 2.063 / 2.037. The final forgetting decides.

## GPU cg27a_kty to visit 6 (phase 6 Github running; vs cg27a_kmr)
- **USEFULNESS is NOT stable on the GPU: visit 6 useful right 0.667 (21) vs not useful 0.670 (215) vs no feedback 0.684; useful AND judged-kept 0.625 (16) - below all judged-kept 0.72 (320: 231 right, 89 wrong); NOT HARMFUL AND kept 0.757 (148).** At visit 4 the same readout gave 0.938 (16). The second batch (next-tier pairs, where the judge also fails: AUC 0.519, identical to kmr) defeats usefulness too. So kus's strict rule helps on the first batch and hurts on the second; the softer rule gains a little on both GPU batches (0.85 vs 0.79, 0.76 vs 0.72) but failed locally. Mixed - the kus GPU log decides.
- **The typed confirmation signal works where the judge fails: confirmations - contradictions AUC 0.626 on the second batch (kmr untyped 0.540; the relational judge 0.519)**, after 0.707 vs 0.541 at visit 4. Typing turns later reading into a usable label-free check. Queued (31): a judge built on the typed confirmation-minus-contradiction score.
- Memory cost on held-out Wikipedia at visit 5: +0.0071 bits per byte (kmr +0.0133, about half); the gate rule (no inference slots on text) would make it +0.0004. Visit 6 gate open on 1,494 spans (kmr 927), inferences -17,697 bits (-12.4 per span; kmr -10.4) - per-span cost is NOT lower at visit 6.
- False memories from the trunk unchanged (kty teaches what kmr taught): taught WRONG inferences recalled 0.28 (kmr 0.30), RIGHT 0.42 (0.37).
- **Post-train BPC kty vs kmr: +0.001 / +0.017 / +0.007 / +0.011 / +0.010 on the five domains so far** - each within seed noise, but five of five on the same side (mean +0.009). A small real cost of typed writes is likely; the final forgetting decides.

## GPU cg27a_kus to visit 5 (phase 6 running) - THE KEEP RULE BLOCKS FALSE MEMORIES AT SCALE
| from the TRUNK, test wording, visit 5 | taught RIGHT inferences recalled | taught WRONG inferences recalled (false memories) | held back (control) right / wrong |
|---|---|---|---|
| kmr (judge keeps -> trunk) | 0.22 (366) | 0.14 (170) | 0.01 / 0.02 |
| kty (the same + typed writes) | 0.33 (183) | 0.27 (85) | 0.04 / 0.00 |
| **kus (judge AND useful -> trunk)** | **0.10 (182)** | **0.02 (85)** | 0.00 / 0.03 |
- **POSITIVE for the false-memory goal: with the usefulness keep rule, wrong inferences are recalled from the trunk at the control level (0.02 vs held-back 0.03), against 0.27 in kty and 0.14 in kmr.** Right ones are still recalled above control (0.10 vs 0.00); the ratio of right to wrong recalled is about 5 : 1 (kty 1.2 : 1, kmr 1.6 : 1). The cost is recall of right inferences from the trunk (0.10 vs 0.33) - they stay available through the memory.
- The rule's per-visit readout is unstable, as in kty: useful AND judged-kept right 0.882 (17) at visit 4, 0.429 (7) at visit 5 (usefulness re-measured on new text); NOT HARMFUL AND kept 0.877 (81) / 0.841 (69) against all kept 0.79. The trunk-level outcome above is what matters, and it is clearly better.
- No harm so far: post-train BPC 2.036 / 2.063 / 2.037 / 1.670 / 1.587 vs kty 2.035 / 2.079 / 2.026 / 1.670 / 1.588 (same; both about +0.009 above kmr). Memory cost on held-out Wikipedia, visit 5: +0.0058 (kty +0.0071, kmr +0.0133).
- **krt IN THE PACKAGE (local, run A's settings): PASSES.** HEADLINE forgetting -0.040, final BPC 2.599 (A -0.036 / 2.603, kus -0.040 / 2.599). 32 kept inferences retyped on confirmation at visit 4 (e.g. 'Sydney , most populous city in Australia', 'Billboard Hot 100 in US'; some long wordings: ', son of albert iii of'). False memories from the trunk at visit 5: taught wrong recalled 0.00 (147) vs held back 0.01 - as kus. Safe to run on the GPU.
- **GPU kus at visit 6 (the hard second batch): false memories stay blocked.** Taught WRONG inferences recalled from the trunk 0.05 (79) vs held back 0.00 - against kty 0.28 and kmr 0.30 at visit 6; RIGHT 0.08 (171) vs kty 0.42. Usefulness at visit 6: useful right 0.857 (21) vs not useful 0.633 (218); useful AND judged-kept 0.812 (16) vs all judged-kept 0.72. The keep rule holds across both batches at GPU scale.

## GPU cg27a_kgk (krt + gate rule + generic word units readout), interim to visit 2 (16 Nov 20:29 run)
- **GENERIC WORD UNITS at GPU scale: NOT better than the hand units on text they never stored.** Held-out bits per byte saved (visit 2): Wikipedia (read) words -0.0049 vs hand NAME + DATE + NUMBER -0.0122 (words + hand -0.0125); ArXiv (NEXT, unread) words -0.0173 vs hand -0.0179, words + hand -0.0291 (+0.011 on top of hand). Visit 1, ArXiv... / Wikipedia NEXT: words -0.0026 vs hand -0.0027.
- **The headline 'ALL DOMAINS -0.229 / -0.163 vs hand -0.025' is a DUPLICATION ARTEFACT from Gutenberg:** words save -0.456 / -0.468 bits per byte there (hand -0.047). The run's own overlap check finds 2,222 of 259,122 128-byte Gutenberg training chunks inside the validation text (Pile documents are disjoint, but PG-19 has duplicated books); a memory that copies 24-byte continuations after any 1-2 words reproduces duplicated passages almost free. The hand units on Gutenberg (-0.047 vs -0.014 on Wikipedia, every build) are inflated the same way, less. Readouts of copy-type memories should exclude or flag Gutenberg (and Github: 943 overlapping chunks).
- So the offline 2.5x (6k stand-in trunk, 2.24 bits per byte) was mostly the weak trunk, as cautioned: a strong trunk already predicts what a 1-2 word cache knows. Generic keys may still ADD on top of the hand units on some domains (ArXiv -0.011); the learned-keys line moves to KEYS FOR COMPOSITION (key_return.py, route C) rather than byte prediction.
- Gate rule: the memory's cost on held-out text at visit 2 is 0.0000 on every domain. Post-train BPC: Gutenberg 2.022 (kty 2.035, kmr 2.036), Wikipedia 2.058 (kty 2.079, kmr 2.062) - lower so far.
- **CHECK (queue 37): post-train BPC does NOT include the keyed memory** (it is the trunk's own forward, platypus/runner.py `bpc_m`; the keyed memory and the gate rule enter only the readouts: compress.py prequential bits, kmem.py feedback and recall). So kgk's lower post-train BPC (Gutenberg 2.022 / Wikipedia 2.058 vs kty 2.035 / 2.079, kus 2.036 / 2.063) is not the gate rule - kty, kus and kgk are the same configuration up to visit 3 (the judge acts at visit 4), so these differences are **run-to-run GPU variation of about 0.015-0.02 per domain**. kty's +0.009 mean against kmr is within it: no evidence of a cost from typed writes.
- **cg27a_kgk IN THE PACKAGE (local, run A's settings): PASSES.** HEADLINE forgetting -0.042, final BPC 2.597 (A -0.036 / 2.603, krt -0.040 / 2.599); 32 retyped at visit 4; every readout prints, no failures. Generic word units locally beat the hand units (visit 4: -0.071 vs -0.012 bits per byte) - the small-trunk effect the GPU run does not reproduce on unseen text.

## A judge from typed confirmations (16 Nov 2026, tools/typed_judge.py; offline, GPU write path, distant_50k, writes at visits 2 / 4 / 6, judged on the next two visits' records)
Signals per typed inference over later relational records whose FIRST name is its head: CC-ANY (confirm - contradict) / total, any wording; CC-WORD only the inference's own wording; CC-CLASS only records whose second name has the tail's class; JUDGE+CC = judge rank + CC-CLASS rank. AUC right vs wrong over all 200 of a tier (uncovered = 0) and COVERAGE (share with any record).

| AUC (coverage) | batch, visit 2 / 4 / 6 | next tier, visit 2 / 4 / 6 |
|---|---|---|
| JUDGE (relational records of the pair) | **0.682** / **0.688** / **0.642** (0.51-0.55) | 0.568 / 0.614 / 0.558 (0.33-0.46) |
| CC-ANY | 0.635 / 0.562 / 0.622 (0.84-0.93) | 0.584 / 0.587 / **0.629** (0.81-0.88) |
| CC-WORD (own wording) | 0.429 / 0.476 / 0.503 | 0.445 / 0.415 / 0.486 |
| CC-CLASS | 0.561 / 0.588 / 0.567 (0.76-0.83) | 0.575 / 0.563 / 0.612 (0.67-0.79) |
| **JUDGE + CC** | 0.656 / 0.678 / 0.635 (0.79-0.85) | **0.597 / 0.622 / 0.610** (0.69-0.80) |

- **MODEST POSITIVE.** On the first batch the judge stays best (the typed signal adds nothing); **on the harder next tier judge + typed confirmation beats the judge in all three rounds (+0.03 / +0.01 / +0.05) and covers about twice as many inferences (0.69-0.80 vs 0.33-0.46).** Smaller than the GPU readout suggested (0.626 vs 0.519 there; the GPU's signal is read through the memory's gate, this one from records).
- **'Contradictions' in the inference's OWN wording are misleading (AUC below 0.5):** most typed wordings are 'in', and a head is 'in' several true places at once (a city in its county, state and country), so 'X in Z' with another Z is usually NOT a contradiction. Containment relations need a hierarchy-aware contradiction test (Z not containing / contained by the tail) before same-wording contradictions can count.
- Use: the combined score as the judge for the next tier, and its coverage for the kus keep rule (queue 35: judge AND (useful OR typed-confirmed)).

## Key-returning memory, three routes, FULL RUN (16 Nov 2026, tools/key_return.py; all 3,053 Re-DocRED documents, 85,932 facts written, 4,000 chains A -> B -> C with A -> C never stated; 27,794 name keys from the 6k stand-in trunk; relation key = the gold relation)
| two-hop, top-1 / top-5 | all 4,000 | ALIASED B (186, 5%) | plain B (3,814) |
|---|---|---|---|
| DECODE (today's path, spelling assumed perfect) | 0.358 / 0.666 | 0.059 / 0.075 | 0.373 / 0.695 |
| A POINTER (a distribution over answer names) | **0.395 / 0.745** | 0.081 / 0.145 | **0.410 / 0.774** |
| B LEARNED KEY SPACE (similarity addressing, untrained) | 0.301 / 0.479 | 0.204 / 0.333 | 0.306 / 0.486 |
| **C HYBRID (A, B where the exact address is empty)** | **0.395 / 0.748** | **0.210 / 0.312** | 0.404 / 0.769 |
- **POSITIVE, smaller than the 60-document trial (0.505 / 0.557 / 0.549 / 0.575): returning a DISTRIBUTION of keys instead of decoding one name gains +0.04 top-1 and +0.08 top-5 (A and C vs DECODE), and the hybrid triples aliased chains (0.21 vs 0.06-0.08) at almost no cost on plain ones (0.404 vs 0.410).** C ties A overall because only 5% of these chains write B two ways (gold evidence sentences); in the package's store, built from noisy distant records, the alias share is likely higher, which is where C's fallback pays.
- B alone (similarity addressing over the whole store, untrained keys) is worst on plain chains (0.306): look-alike names collide. The learned half earns its place only as the fallback - as with the hash + learned name key.
- So the design for the package: POINTER first (answer-name distribution per slot, top-k carried to hop 2), the learned key as fallback for empty exact addresses (frozen encoder). Absolute two-hop accuracy here is with ideal relation keys; the package's learned wording key will cost some.

## GPU cg27a_kgk to visit 6 (krt + gate rule + word units readout; vs kus / kty)
- **False memories stay blocked:** taught WRONG inferences recalled from the trunk at visit 5: 0.01 (85) vs held back 0.00 (kus 0.02, kty 0.27); RIGHT 0.06 (kus 0.10, kty 0.33).
- **The gate rule works on the GPU:** the memory's cost on held-out Wikipedia at visit 5 is +0.0004 bits per byte (kus +0.0058, kty +0.0071: -93%).
- **Retyping is active at scale: 135 kept inferences retyped at visit 4, 108 at visit 6** (local runs: 32). But the confirming wordings are often long spans rather than relation words ('Iran , but especially from Tehran', 'Ohio , located 61 miles southwest of Cleveland'), and some are lists ('Houston and galveston , Texas' - the judge's list filter inf_rret_list2 is off). Exact relation accuracy cannot be read from this log (readout gap, queue 30).
- **INTERACTION FOUND: the gate rule starves the usefulness keep rule.** Usefulness is measured through the memory with the gate rule applied (kmfb_readout passes no_rti = inf_kma_nori), so inference slots rarely open on text: the gate opened on 177 spans at visit 4 (kty 1,238), feedback on 48 of 400 inferences (kus 118), and **only 5 inferences reached the trunk** (all right, 1.000; kus 16-17). Fix: measure usefulness with the inference slots readable (a counterfactual 'would it have helped' read) while prediction keeps the gate rule (queue 40).
- Judge unchanged (visit 4 AUC 0.688, visit 6 0.520). Post-train BPC 2.022 / 2.058 / 2.024 / 1.664 / 1.579 - lowest of the four builds but within run variation.
- LEARNED SUBJECT KEYS, 28b (LK_K 8,192 codes, drift = 300 arXiv steps; tools/learned_keys.py): the drift is barely milder (mean state change 14.0 vs norm 17.2; the trunk at 3.31 bits per byte on this text - it forgets wiki fast). Finer codes hold 5.8 names per code (24% hold one). Bits saved per 1,000 bytes (all | after warm-up): HASH 116.8 | 124.9, REKEY 121.8 | 129.0, STALE 86.7 | 96.4, QDC 97.8 | 107.7, **FROZEN 125.6 | 135.3, HASH + FROZEN 144.7 | 155.0 (+24% over the hash)**. The ranking holds: a frozen key encoder is the best learned key (at least as good as re-keying), stale keys lose 26% (now well below the hash), query drift compensation recovers a third. Finer codes give the learned key alone a smaller edge over the hash (+7%, vs +15% at 1,024 codes): coarse codes share more across aliases. For route C's fallback: a frozen encoder, codes on the coarse side.
- READOUT FIX (queue 30) verified (tools/kmi_smoke.py, cg27a_kgk): the integrated keyed memory readout now reports typed inferences (taught RIGHT 40 / WRONG 20, held back 19) instead of '- (0)', and they no longer sit in the 'taught FACTS' rows. NOTE for earlier typed builds (kty, kus, kgk on the GPU): their 'taught FACTS' recall included typed inferences.
- **CONTAINMENT-AWARE CONTRADICTIONS (queue 38; typed_judge.py, '-H': 'X in Z' is not a contradiction when the store already links Z and the tail): the best judge is JUDGE + CC-ANY-H** (judge rank + confirm-minus-contradict over any wording, containment-aware): batch 0.678 / 0.671 / 0.677 vs the judge 0.682 / 0.688 / 0.642 (about equal on average, 0.675 vs 0.671), **next tier 0.608 / 0.632 / 0.603 vs 0.568 / 0.614 / 0.558 (+0.04 / +0.02 / +0.05), covering 0.79-0.92 of inferences vs the judge's 0.33-0.55.** Own-wording contradictions stay below chance even containment-aware (CC-WORD-H 0.40-0.52): the store rarely states the containment link, so drop that variant. Candidate model change: the relational judge's score replaced by judge + typed confirmation (containment-aware) - mainly for the harder tier and for coverage (queue 42).
- **RETYPING WITH THE WIDER LIST FILTER (queue 41; loop_sim BOTH-RETYPE, A6, distant_50k, three rounds): POSITIVE - the list filter belongs with retyping.** LS_LIST2 1 vs 0: memory right 0.829 / 0.839 / 0.810 vs 0.771 / 0.742 / 0.739; exact relation types 0.600 / 0.613 / 0.595 vs 0.543 / 0.545 / 0.534; bits saved +1.05 / +1.77 / +1.81 vs +0.76 / +1.43 / +1.55. The GPU kgk retyped list wordings ('Houston and galveston , Texas') because inf_rret_list2 is off there. Candidate for the next judge build: inf_rret_list2 1 with the typed confirmation judge (both change the judge's evidence).
- **WIDER KEEP RULE (queue 35; loop_sim, LS_LIST2 1, A6, distant_50k): judge AND (useful OR typed-confirmed) keeps 17-50% more at the same precision** - kept 41 / 32 / 33 (right 0.854 / 0.812 / 0.758) vs judge AND useful 35 / 27 / 22 (0.829 / 0.852 / 0.727), memory 106 at 0.811 vs 84 at 0.810 by round 3, bits the same (+1.12 vs +1.13); the judge alone keeps 81-125 at 0.68-0.76. Typed confirmation alone with the judge keeps few (8-15) at 0.75-0.90. Offline, usefulness covers nearly every judged pair, so the typed signal adds only a few; on the GPU, where usefulness covered 16 of 160 (kus) and 5 under the gate rule (kgk), it should add more - a cheap coverage fix to pair with inf_kmc_useful_open.
- GPU cg27a_kgk to visit 7 (8 domains in this run; no headline yet): **false memories from the trunk stay at the control level through visits 5-7: taught WRONG recalled 0.01 / 0.00 / 0.01 vs held back 0.00** (kmr 0.14 / 0.30 at visits 5 / 6); taught RIGHT 0.06 / 0.06 / 0.07 vs held back 0.01-0.02. Useful AND judged-kept right 1.000 (10) at visit 7 - precise, but only 10 reach the trunk (the gate-rule starvation, fix queued). The gate rule keeps the memory's cost on held-out Wikipedia at +0.0007 / +0.0012 (visits 6 / 7). Generic word units (ALL DOMAINS, incl. duplicated Gutenberg) -0.069 to -0.080 vs hand -0.009 to -0.013.
- JUDGE WINDOW (queue 13; loop_sim LS_UWIN = share of the two visits after a write the judge sees, LS_LIST2 1): the judge's precision barely depends on the window (judge alone, round 1: 0.756 / 0.760 / 0.750 at 0.25 / 0.5 / 0.75) while coverage grows with it (kept 78 / 125 / 152); judge AND useful memory by round 3: 40 at 0.800 / 84 at 0.810 / 102 at 0.775. No sharp optimum - more reading judges more pairs at about the same precision; the GPU's two-visit window is a reasonable setting. (Bits are not comparable across windows: the scoring text is what remains after the judge's share.)
- **PACKAGE CHECK OF THE TYPED CONFIRMATION JUDGE (inf_rret_sig 'rel+cc'; the GPU write path offline, distant_50k, writes of 200 at visits 2 and 4, judged two visits later): NOT BETTER in the package's own sequence.** First batch AUC 0.657 vs 'rel' 0.654 (flagged 40 wrong 0.40 vs 0.47); second batch 0.650 vs 0.659 (flagged wrong 0.50 vs 0.60). The typed signal alone 0.623 / 0.582 (covers 0.93 / 0.86). The offline gain (+0.02-0.05) was on the next tier of ONE 400-write round; the package writes 200 a round, and its second batch is not the hard case here (the relational count holds at 0.659 offline, while the GPU's fell to 0.52 at visit 6 on its larger store). **Decision: no judge change yet; the [TYPED CONFIRMATION (evaluation)] readout rides along in the next build to show whether the typed signal holds where the GPU judge collapses.**
- **DUPLICATION AUDIT (queue 33) - CORRECTION to every GPU keyed-units claim.** Recomputed from the GPU logs over the domains already read, without Gutenberg (domain 0) and Github (domain 5) whose training text overlaps the validation text: hand keyed units (NAME + DATE + NUMBER) save **-0.0035 to -0.0051 bits per byte**, not the -0.011 to -0.015 the all-domain mean shows (kty visits 4 / 5: -0.0146 / -0.0117 -> -0.0051 / -0.0040; kus -0.0140 / -0.0112 -> -0.0044 / -0.0035; kgk visits 6 / 7: -0.0142 / -0.0132 -> -0.0039 / -0.0046). Gutenberg alone contributes -0.040 to -0.043. **About two thirds of the reported keyed-unit gain was duplicated text.** Generic word units without the duplicated domains: -0.0047 / -0.0048 (kgk) - about the same as the hand units, not 6-10 times more. Earlier summaries quoting 'keyed units -0.013 to -0.026 on the GPU' should read about -0.004.

## Consolidation anchor on replay text from every domain (16 Nov 2026, queue 24; tools/consolidate.py CS_ANCHOR; A6 stand-in trunk, 200 typed inferences in 24 frames, 600 steps; forgetting = held-out bits added on each of five domains)
| KL anchor (every weight) | recall rank / open | wiki | prose | arxiv | pubmed | law | mean |
|---|---|---|---|---|---|---|---|
| anchor rows wiki only, lr 3e-4 | 0.380 / 0.285 | +0.056 | -0.035 | +0.100 | +0.044 | +0.007 | +0.034 |
| anchor rows from all five, lr 3e-4 | 0.345 / 0.250 | +0.075 | -0.030 | -0.042 | +0.013 | -0.031 | **-0.003** |
| anchor rows wiki only, lr 1e-3 | 0.690 / 0.515 | +0.123 | +0.007 | +0.145 | +0.120 | +0.062 | +0.091 |
| anchor rows from all five, lr 1e-3 | 0.510 / 0.415 | +0.120 | -0.030 | -0.178 | +0.020 | -0.055 | **-0.025** |
- **POSITIVE (partly): anchoring on replay text from every domain removes the drift on the domains a one-domain anchor never sees** (mean forgetting +0.034 / +0.091 -> -0.003 / -0.025), the failure mode of the package's consolidation path (its anchor covers stored sentences only; GPU-scale package run B: Wikipedia +0.35). Costs: recall somewhat lower at the high rate (rank 0.51 vs 0.69), and forgetting on the facts' own domain (wiki) is unchanged (+0.06-0.12): the facts compete with wiki text directly. Some 'gains' on unanchored domains (arxiv -0.18) are unexplained by a KL anchor alone - treat the sign with caution.
- FULL (cross-entropy) with rows from every domain is not a forgetting test: it trains on those domains (arxiv -4.1 bits) - excluded.
- Next: in the package, the consolidation path's soft anchor on DREAM / REPLAY rows of every visited domain instead of stored sentences (queue 44), against run A; pass = forgetting within noise of A with trunk recall above A's.
- **USEFULNESS STABILITY (queue 34; loop_sim LS_STAB, rounds at visits 2-6, every written inference re-measured each round on that round's first-half text, as the GPU re-measures at every visit): the CUMULATIVE measure is the steadier and, later on, the more precise keep rule.** Latest-measure decisions flip for 37 / 52 / 69 / 83 judged inferences per round (about 20%), and its precision swings 0.83 -> 0.78 -> 0.83 -> 0.77 -> 0.71; usefulness summed over every round since the write passes more (54 / 68 / 84 / 97 vs 45 / 63 / 78 / 90) at 0.72 / 0.79 / **0.81 / 0.78** (latest 0.78 / 0.83 / 0.77 / 0.71). This matches the GPU flips (0.94 -> 0.63; 0.88 -> 0.43 -> 0.81). Candidate package option: inf_kmc_useful_cum 1 (the keep rule on the running sum of a written inference's usefulness) with inf_kmc_useful_open (queue 45).
- WRITE CHOICE BY SURPRISE (queue 29; loop_sim LS_SURP: the trunk's bits per byte on a candidate's premise sentences at write time, Titans-style): **NOT a reliable signal.** AUC right vs wrong 0.476 / 0.513 / 0.595 over the three rounds (the sign flips); most vs least surprising half right 0.615 / 0.615, 0.561 / 0.520, 0.643 / 0.490; within the code's batch high half 0.670 / 0.600 / 0.673 vs low 0.723 / 0.551 / 0.520. Only the third round shows a gain - not enough to replace or reweight the fact-table code's ranking. Learned write choice stays open; surprise on the premises is not it.
- **ADAPTIVE WRITE BUDGET IN THE PACKAGE (queue 5; inf_rti_adapt, tools/adapt_check.py, four write rounds, judged two visits later): a QUANTITY dial, not a precision dial - NEUTRAL, not for a build.** Off (200 a round): 800 written right 0.627, judged-kept 480 at 0.673. Threshold 0.3 (share of a 100-block with a confirming relational record): writes grow to 300 / 400 / 400, 1,300 at 0.575, kept 720 at 0.637. Threshold 0.5: 200 / 300 / 100 / 100, 700 at 0.627, kept 480 at 0.667. Threshold 0.6: 200 / 200 / 100 / 100, 600 at 0.643, kept 400 at 0.672. Precision of the kept set barely moves (0.667-0.673 except the expanding setting). The offline dial (up to 0.93) stopped on judge AND usefulness per block; confirmation alone is too loose a stop signal. Option stays off.

## Key-returning memory IN THE PACKAGE (17 Nov 2026, inf_kmc_keyread readout; tools/kmi_smoke.py, cg27a_kgk, d 64 smoke model)
| visit | the model decodes B, then asks (today) | ROUTE A pointers: first hop / C rank among 10 / C top-1 of all answers | ROUTE C (+ trunk-code fallback) |
|---|---|---|---|
| 2 | first hop exact 0.43, C rank 0.50 (trunk alone 0.00 / 0.03) | 0.93 / **1.00** / 0.75 | 0.93 / 1.00 / 0.72 (147 fallbacks) |
| 3 | 0.00, 0.20 (trunk alone 0.00 / 0.10) | 0.98 / **1.00** / 0.85 | 0.98 / 1.00 / 0.82 (138 fallbacks) |
- **POSITIVE: returning answer names from the memory's own slots composes two-hop chains far better than decoding** - C ranked first among 10 in every chain (1.00 vs 0.50 / 0.20), the exact C the top answer of all names 0.75-0.85, over 60 chains each visit of which 25-27 have an inference hop (composition through written inferences). The pointer table (answer names per slot, 1,332-1,441 slots) is built from the same addresses the memory writes with, so on chains of entries IN the memory it is close to an exact index - the test that matters is the GPU: noisy store, collisions in shared slots, far more entries.
- The trunk-code fallback fires often (names that are answers but not written subjects) and costs a little here (top-1 0.72 / 0.82 vs 0.75 / 0.85): on exact chains it only replaces names that need no replacing. Keep it as a readout arm; it should earn its place only where aliases exist.
- Ready to ride along in the next GPU build (readout only).
- **cg27a_kck IN THE PACKAGE (local, run A's settings): PASSES** - HEADLINE forgetting -0.039, final BPC 2.600 (A -0.036 / 2.603). 27 retyped at visit 4. **KEY-RETURNING MEMORY at local scale (inference hops in most chains): route A first hop 0.64 / 0.58 / 0.55 / 0.55, C rank among 10 0.73 / 0.67 / 0.52 / 0.57, C top-1 of all answers 0.55 / 0.42 / 0.40 / 0.45 (visits 2-5; 11-60 chains, chance 0.10)** - well above chance but below the smoke model's 1.00, and the decoding path found too few chains to compare (1-3). Route C's fallback: top-1 0.27 / 0.25 / 0.38 / **0.50** - worse early, better than pointers alone at visit 5 (0.50 vs 0.45). [TYPED CONFIRMATION (evaluation)] AUC 0.617 vs the relational count 0.652. One readout error (COUNTER-WRITE: a long retyped prompt past the row's width) - fixed (skip such rows); readout only, caught.
- **CONSOLIDATION PATH WITH A STRONGER, EVERY-DOMAIN ANCHOR (queue 44; B3 = run B + inf_distill_wake_akl 2.0, aklall 1 (every old slot each step), aklrows 8, aklcache 96): PARTLY FIXED, NOT YET A PASS.** HEADLINE forgetting **+0.031** (B +0.094, B2 +0.080, A -0.036), final BPC 2.676 (B 2.731, A 2.603); trunk recall of taught facts in test wordings **0.22 / 0.23** at visits 4 / 5 (B 0.19 / 0.21, A 0.03 / 0.04) - two thirds of the forgetting removed with recall kept. ArXiv / PubMed / Gutenberg now improve (-0.008 / -0.015 / -0.032, as A); **the remaining damage is Wikipedia alone: +0.186** (B +0.346, A -0.018) - held-out Wikipedia bits 2.610 / 2.610 / 2.613 then 2.822 at visit 5 (B 3.012): steady until the last domain, then a jump while the facts (Wikipedia-like sentences) are taught during FreeLaw. Matches the offline result: an all-domain anchor fixes the other domains, not the facts' own domain. Next (queue 46): the anchor weighted toward the facts' own domain (more Wikipedia rows) and / or a lower question rate in the last visits; pass = forgetting within noise of A.
- **CONSOLIDATION, ANCHOR WEIGHTED TO THE FACTS' OWN DOMAIN (queue 46; B4 = B3 + inf_distill_wake_aklslot 1 (Wikipedia), aklfw 4.0): MOSTLY FIXED.** HEADLINE forgetting **+0.002** (B3 +0.031, B +0.094, A -0.036), final BPC 2.643 (A 2.603, B3 2.676); **Wikipedia +0.048** (B3 +0.186, B +0.346); other domains as A (Gutenberg -0.030, ArXiv +0.001, PubMed -0.005). Held-out Wikipedia bits 2.610 / 2.607 / 2.608 / **2.668** at visits 2-5 (B3 2.822 at visit 5, A 2.591). **Trunk recall of taught facts 0.22 / 0.27 at visits 4 / 5** (A 0.03 / 0.04, B3 0.22 / 0.23). Six times A's recall for about +0.04 forgetting against A - not yet within seed noise of A (about 0.02), but the closest yet. Next: aklfw 8 and / or the question rate lower in the last visit (queue 47).
- CONSOLIDATION, WIKIPEDIA WEIGHT x8 (B5): HEADLINE forgetting +0.007 (x4 +0.002), final BPC 2.649; Wikipedia +0.039 (x4 +0.048) but ArXiv / PubMed +0.013 / +0.016 (x4 +0.001 / -0.005) - the extra weight moves the anchor away from the other domains; recall 0.21 / **0.29** at visits 4 / 5. **A plateau: x4 and x8 are equal within noise (+0.002 / +0.007); the remaining gap to A (-0.036) is about 0.04, now spread over domains rather than one jump.** Held-out Wikipedia bits 2.610 / 2.601 / 2.598 / 2.663. The next lever is not the anchor's weight but how hard the questions push in the last visit (queue 47: question rate or frames reduced after visit 3), or a longer anchor cache.
- **CONSOLIDATION INTO THE TRUNK WITHOUT FORGETTING: PASSES (B6 = anchor on every old domain, Wikipedia x4, + the question rows halved each later visit: inf_distill_wake_decay 0.5 from visit 2).** HEADLINE forgetting **-0.032** (A -0.036: within seed noise), final BPC **2.611** (A 2.603), Wikipedia **-0.015** (A -0.018), every domain improves (Gutenberg -0.042, ArXiv -0.030, PubMed -0.033); held-out Wikipedia bits 2.610 / 2.601 / 2.590 / 2.600 - no last-visit jump. **Trunk recall of taught facts in unseen test wordings 0.19 / 0.22 at visits 4 / 5 vs A 0.03 / 0.04 - about five times, at no measurable forgetting.** The path: every weight (inf_distill_wake_params ''), 21 frames (inf_distill_forms 21), a soft KL anchor to the model at the last sleep on stored real sentences of EVERY old domain each step (akl 2.0, aklall 1, 8 rows, 96 cached), the facts' own domain weighted x4 (aklslot 1, aklfw 4.0), question rows x0.5 per later visit. Build candidate (one model change) after cg27a_kck.
- **cg27a_kcs IN THE PACKAGE (local, run A's settings): PASSES.** HEADLINE forgetting **-0.036** (A -0.036), final BPC 2.608 (A 2.603); every domain improves (Gutenberg -0.043, Wikipedia -0.033, ArXiv -0.033, PubMed -0.031); trunk recall of taught facts in unseen wordings 0.19 / **0.22** (A 0.03 / 0.04); written WRONG inferences recalled from the trunk 0.00 (false memories stay out). Key-returning two-hop at visit 5: route A C rank 0.50, top-1 0.37 (58 of 60 chains through an inference hop). Ready for the GPU after cg27a_kck.

## GPU cg27a_kcs (kck + consolidation into the trunk), interim to visit 3 (17 Nov 17:12 run; vs cg27a_kgk)
- **CONSOLIDATION WORKS AT SCALE: taught facts recalled from the TRUNK in unseen test wordings 0.77 (198) at visit 3, vs kgk 0.35 (198)** (taught 1 visit 0.75, 2-3 visits 0.81; entered at this sleep, not yet taught, 0.02). Wrong inferences: the pool's few conclusions recalled 0.00.
- **KEY-RETURNING MEMORY ON THE GPU: route A first hop 0.92 / 1.00, C ranked first among 10 1.00 / 1.00, C top-1 of all answers 0.46 / 0.92 (visits 2 / 3; 13 / 39 chains)**; route C 0.92 / 1.00, 1.00 / 1.00, 0.54 / 0.92. The decoding path at visit 3: first hop exact 0.40, C rank 0.87 (kgk 0.37 / 0.83). Few chains go through an inference hop yet (3).
- **WATCH: post-train BPC higher than kgk on the current domain** - Gutenberg 2.025 / 2.022, Wikipedia 2.076 / 2.058 (+0.018), **ArXiv 2.063 / 2.024 (+0.039, beyond the ~0.02 run variation)**; held-out Wikipedia with the memory at visits 2 / 3 2.047 / 2.058 vs 2.029 / 2.044 (+0.015-0.018). The every-weight question rows (and the anchor rows) may take capacity from learning the current domain; the final forgetting and the later domains decide. Visit timing as kgk (phase 4 at 42.5 vs 40.6 min).

## cg27a_kcs on the GPU, interim to visit 4, and the cg27a_kcp smoke test (17 Nov 2026, evening)

**Recall (trunk, test wordings).** Compared with kgk at visit 4:
- Facts taught for at least one visit: 0.77 (kgk 0.42).
- Known facts answered right: 0.69 (kgk 0.38).
- Confidence when right: 0.93 (kgk 0.54).
- Confidence separates right answers from never-read questions with an AUC of 0.90.

**False memories.**
- Gated answers on never-read questions: given a stored name 0.11 (kgk 0.25), abstained 0.89. Known facts right when answered: 1.00.
- Ungated stored-name rate on fresh names: 0.81. This is in line with earlier builds (0.72–0.86).

**Kept inferences.**
- Kept right 0.81 (kgk 0.79); flagged right 0.45 (kgk 0.55), a wider gap between kept and flagged.
- Useful AND judged-kept right 0.95 (21).
- 131 kept inferences retyped on confirmation.

**Key-returning memory, visit 4.**
- Route A: first hop 0.98, final answer ranked first among 10 in 0.98 of chains, top-1 0.93.
- Route C: 0.98 / 0.98 / 0.90.
- 60 chains, 27 of them with an inference hop.

**Cost (learning, not consolidation).** Held-out BPC at visit 4, kcs vs kgk:

| Domain | kcs | kgk | Change |
|---|---|---|---|
| Gutenberg | 2.013 | 2.006 | +0.007 |
| Wikipedia | 2.064 | 2.049 | +0.015 |
| ArXiv | 2.060 | 2.028 | +0.032 |
| PubMed | 1.695 | 1.664 | +0.031 |

The ArXiv and PubMed changes exceed GPU run variation (about 0.015–0.02). The cost is concentrated on the newest domains: the anchor slows learning. Next test: a lighter consolidation dose. (Revised at visit 5, below: most of this was run variation.)

**cg27a_kcp smoke test (synthetic names).**
- `[COMPOSED WRITES]` runs at visits 2 and 3 with no errors.
- 1,129–1,269 unstated two-hop conclusions found; the top 100 written per sleep.
- The labels read 0.00 because synthetic names have no distant labels. This is not a recall result.


## cg27a_kcs on the GPU, final (visit 5)

**Held-out BPC at visit 5.**

| Domain | kcs | kgk | kty |
|---|---|---|---|
| Gutenberg | 2.006 | 2.030 | 2.063 |
| Wikipedia | 2.050 | 2.040 | 2.070 |
| ArXiv | 2.058 | 2.025 | 2.024 |
| PubMed | 1.702 | 1.686 | 1.699 |
| FreeLaw | 1.592 | 1.579 | 1.588 |

- Mean change: kcs vs kgk +0.010; kcs vs kty -0.007. kgk and kty themselves differ by up to 0.03 on a domain.
- The visit-4 "learning cost" was mostly run variation. Only ArXiv is consistently higher (+0.033), and ArXiv is not forgotten after it is learned (2.063 post-train -> 2.058).

**Facts.**
- Trunk recall of facts taught for at least one visit: 0.80 (kgk 0.48, kty 0.46).
- Known facts right: 0.70 (kgk 0.52); confidence when right 0.93 (kgk 0.66).
- AUC, known-right vs never-read: 0.93 (kgk 0.84).

**Gated answers.**
- Right when answered: 0.94 (kgk 0.71).
- Never-read given a stored name: 0.11 (kgk 0.31, kty 0.16); abstained 0.88.

**Real-text inferences from the trunk, taught vs held back.**
- kcs: right 0.14 vs 0.03, wrong 0.01 vs 0.00.
- kgk: right 0.06; kty: right 0.33 but wrong 0.27.
- Written at visit 2: right 0.24 vs 0.04. kcs recalls more right inferences without the false memories that kty had.

**Key-returning memory, visit 5.**
- Route A: first hop 0.95, rank 0.98, top-1 0.95.
- Route C: 0.95 / 0.98 / 0.93.

**REGRESSION: taught sleep-layer conclusions (n 22) recalled from the trunk 0.05 (kgk 0.36, kty 0.45).**
- Hypothesis: `inf_distill_wake_decay` is GLOBAL. Every question row is weighted 0.5^(visit - 1), about x0.125 at visit 5, while the anchor (akl 2.0) is not decayed.
- Conclusions enter the pool late and have no source sentences, so the anchor holds them out. Facts survive through source replay and early teaching.
- This cannot be tested locally: only 2–4 conclusions are taught there.
- Next: a local harm check of decay off (1.0), then a GPU run.

## Real-text composition rules mined from typed frames: NEGATIVE (17 Nov 2026, evening; tools/rt_rules.py, distant_10k)

**Setup.** The relation of a record is its typed frame: (type A, the words between, type B). Rules r1 o r2 -> r3 are mined on the first 70% of documents, against a shuffled null, with a Wilson lower bound on confidence. Composed pairs that were never stated in training are then checked two ways: stated in a later document, or carrying a distant label.

| Run | Rules kept | Composed pairs | Stated later | Distant label |
|---|---|---|---|---|
| Surface frames, chains within one document allowed | 49 | 578 | 0.038 | 0.021 |
| All chains, no rule (baseline) | n/a | sample of 50,000 | 0.005 | 0.009 |
| Random same-type pairs (control) | n/a | 576 | 0.000 | 0.000 |
| Hops from different documents, evidence from a third | 1 | n/a | n/a | n/a |
| Cross-document, frames ending in a word | 0 | n/a | n/a | n/a |
| Cross-document, word frames, last word only | 0 | n/a | n/a | n/a |
| Same, threshold lowered to 0.15 | 1 (TIME frames) | 10 | n/a | n/a |

- In the first run, the rules are mostly list and punctuation artefacts within one document.
- Across documents, an A-C pair from a chain is almost never stated elsewhere (0.005). There is too little evidence to learn or verify cross-document composition in this corpus.
- The distant labels exist only for pairs named in the same document, so they cannot score cross-document conclusions anyway.

**Conclusion.** Real-text composition across documents cannot be learned or checked from this corpus with surface frames.
- Within-document "never stated together" inference is what the typed writes already do: 0.6-0.8 right.
- Rule-world composition works through the relational module, with verified rules and no false ones on the GPU.
- Measuring the next step needs a corpus whose composite facts are checkable (Wikidata relation paths over the names read), not later text.

## GPU cg27a_kcb (kcs + the conclusion boost), visit 4: NEGATIVE for the conclusion fix (17 Nov 2026, 20:13)

**The boost worked mechanically.** Conclusion rows were sampled x2 at visit 3 and x4 at visit 4; their share of question samples was 0.132 and 0.219 (uniform 0.055 and 0.049).

**Recall did not move.**
- Taught conclusions recalled from the trunk at visit 4: 0.06 (34). kcs: 0.09 (22).
- Never-taught CONTROL conclusions: 0.25 (8).
- The amount of practice is not the cause. The global decay hypothesis is wrong.

**Everything else as kcs.** Facts taught for at least one visit recalled 0.79. Relational-module composites taught 1.00 (8) vs held back 0.33 (3).

**Next suspects.**
- The 21 question wordings: conclusions have no source sentences to carry them. kgk used 7 wordings and recalled 0.33–0.36.
- Every-weight training.

GPU arm cg27a_kcw = kcb + inf_distill_forms_conc 7.

**Local kcs with half the anchor weight (akl 1.0).** Timed out before HEADLINE. Visit-4 held-out BPC 2.913 / 2.577 / 2.658 / 2.571 vs kcs 2.904 / 2.569 / 2.643 / 2.571, and the same recall. No benefit; closed.

## Correction: chain discovery is NOT a large gap (17 Nov 2026, 20:35)

"Probe chains found 15 of 203" counts every planted chain, but a chain can only be found once the domains of both its facts have been read. Consecutive facts are in different domains, with 10 domains in the GPU runs.

Expected number of discoverable two-hop chains after v visits: 203 x v/10 x (v-1)/9.

| After visit | Discoverable | Found (kcs GPU) | Share found |
|---|---|---|---|
| 3 | ~14 | 15 | ~100% |
| 4 | ~27 | 23 | ~85% |
| 5 | ~45 | 35 | ~78% |
| 6 | ~68 | 57 | ~84% |

Sleep finds about 80% or more of the chains it can reach. The remaining inference gaps are composition by the trunk itself and real-text composition, not discovery.

**Conclusion recall (kcb): not a wording mismatch.**
- Conclusions are tested in the held-out paraphrase "A is tied to" / "The one tied to A is" (taught as "is connected to" in every form).
- This is the same for 7 and 21 wordings, so the drop from about 0.36 (kgk, kty, kus) to about 0.06 (kcs, kcb) comes from the consolidation package itself: every weight, every-domain anchor, 21 forms, decay.
- kcw tests the 21 forms.

## COMPOSITION ON REAL TEXT WITH RELATION KEYS: POSITIVE (17 Nov 2026, 20:30-20:50; tools/latent_compose.py, Re-DocRED)

### E1. Rules over relation keys (gold relation ids = the ideal relation key)

- Rules mined on 3,053 training documents: 200 kept (support >= 10, Wilson LB >= 0.5).
- On 1,000 unseen documents, the rules composed 17,661 conclusions. **Precision 0.796** (labelled with that relation in the document).
- Of 17,572 labelled facts reachable by a two-step chain, **0.44 are composed by a rule**.

| Rule | Training support | Held-out precision |
|---|---|---|
| located-in o located-in | 6,577 / 6,577 | 1.00 |
| located-in o country | 4,862 / 4,889 | 0.99 |
| birthplace o located-in -> citizenship | 504 / 770 | 0.68 |
| performer o label -> label | 470 / 727 | 0.73 |

The relation names were printed from a local table (rel_info.json was not downloaded), so some names may be mislabelled. The ids and the precision are exact.

**Conclusion.** Tonight's real-text negative (tools/rt_rules.py) came from the surface-frame relation keys, not from composition itself.

### E2. Latent operators on learned name keys (the user's fuzzy-composition hypothesis)

Name key = the stand-in trunk's state over the name's bytes (tg_lm192_18000). Relation maps are ridge fits on the training documents (92 relations). For 13,413 composites in unseen documents, C is ranked among the document's entities (chance 0.055, about 18 entities).

| Route | C ranked first | Top quarter by margin |
|---|---|---|
| ONE learned map W_r3 key(A) | 0.249 | 0.461 |
| **Pure latent composition W_r2 W_r1 key(A)** | **0.248** | 0.478 |
| **Memory hop 1 (B read) + latent hop 2 W_r2 key(B)** | **0.336** | **0.674** |
| NULL (random r2) | 0.063 | 0.076 |

- Composing two learned maps is as good as a direct map for the composite relation.
- It is 4.5x chance, and the null shows it depends on the right relations.
- Hop 1 through the memory raises it to 0.34, and to 0.67 when only the confident quarter answers.

**Conclusion.** Fuzzy composition in a learned key space works on real text, beside the exact rules of E1 (0.80 precision). Next steps:
1. Rules over LEARNED relation keys (the JEPA wording key) instead of gold ids.
2. A write path in the package: rule-composed conclusions, with latent composition as the fallback where a premise is missing, judged, kept and taught.

## GPU cg27a_kfc3 (integration run: kcs + conclusion boost + rule-verified composing + cumulative keep rule + rule world x4), to visit 4 (17 Nov 2026, 20:54)

The rule world x4 is active: 34-116 rule-world facts per domain (kcb: 15-33).

**Relational module.**
- Both true rules are found at sleep 1: child o child -> grandchild (z +44) and supervised o supervised -> overseen. 0 false rules.
- Composites written: 15 / 21 / 43 / 58 by visits 1-4. All are correct by the rule world's truth, and none was planted.

**The trunk, by test wording.**

| Visit | Taught | Held back |
|---|---|---|
| 2 | 1.00 (9) | **0.00** (6) |
| 3 | 1.00 (11) | **0.00** (4) |
| 4 | 0.74 (19) | **0.00** (6) |

- Held-back answers at visit 4 are 'other' names, 6 of 6: not the middle hop, not the family.
- The trunk does not compose on its own. kcs's held back 0.60-0.75 at visit 5 (n 4-5) was small-sample noise.

**OPERATOR ALGEBRA, frozen trunk codes.**

| Visit | Held-out composites | Composed operator | Shuffled null | Direct composite map | Chance |
|---|---|---|---|---|---|
| 3 | 45 | **1.00** for both rules | 0.00 | 0.04-0.05 | 0.006 |
| 4 | 60 | **1.00** | 0.00-0.03 | 0.03-0.04 | 0.004 |

At 4-8x the earlier sample, composition in the trunk's code space is exact while the trunk's forward pass gives 0.00.

**Composing writes through verified rules (old code: stand-in names still used).**
- Right 0.139 (36), 0.070 (100), 0.122 (90) at visits 2-4.
- The relational module already wrote the correct composites. The remaining chains mostly go through stand-in names (986-1,967 per visit). The exact-chain fix (commit 8c9bb4c) postdates this run.

**Other readouts.** Conclusions taught for at least one visit 0.12 (16) at visit 4 (kcs 0.09). Facts 0.76-0.80. No errors.

**Conclusion.** Composition belongs in key space, and the trunk is taught its results. Together with the offline real-text result (tools/latent_compose.py: rules over relation keys 0.80 precision, latent maps 4.5x chance), this is the design for real-text inference: compose in key space, write, judge, keep, teach.

## README updated (17 Nov 2026, 21:05)

README.md has a new section, "Recent results (15-17 November)". Ten new table rows were added, and open questions 7, 17 and 18 were updated or added. The "Note from author" is unchanged.

## Rules over LABEL-FREE relation keys: the reader, not the key, is the blocker (17 Nov 2026, 21:20; tools/frame_compose.py, Re-DocRED)

**Setup.** Premises are facts readable in one sentence (both names within 12 tokens). Held-out precision = the composed pair is gold-related in the document. The base rate for random pairs in the same documents is 0.132.

| Key \ reader | Rules | Composed | Precision | Recall of chain-reachable gold pairs |
|---|---|---|---|---|
| GOLD relation id, ideal reader (gold facts) | 16 | 826 | **0.527** | 0.368 |
| FINE frame (types + 1-3 words between), ideal reader | 29 | 152 | 0.382 | 0.049 |
| COARSE frame (types + last word), ideal reader | 30 | 260 | 0.423 | 0.093 |
| TYPES only (type pair + order), ideal reader | 7 | 1,052 | **0.452** | **0.402** |
| FINE / COARSE / TYPES, REAL reader (every mention pair within 12 tokens) | 2,226 / 2,546 / 365 | 5,668 / 7,872 / 26,006 | **0.150 / 0.155 / 0.149** | 0.11 / 0.16 / 0.51 |
| Same, REAL reader within 5 tokens | 305 / 442 / 153 | n/a | 0.22 / 0.22 / 0.20 | n/a |

**Simulated readers (ideal premises plus non-gold records to a premise precision).** COARSE / TYPES rules:

| Premise precision | Composed precision | Recall (TYPES) |
|---|---|---|
| 0.8 | 0.32 / 0.41 | 0.38 |
| 0.6 | 0.31 / 0.33 | n/a |
| 0.4 | 0.30 / 0.28 | n/a |

**Conclusions.**
- With an ideal reader, label-free keys keep about 85% of the gold key's precision (0.45 vs 0.53), and the type pair alone matches its recall. A better relation key is NOT the main blocker.
- Deciding which co-occurrences are facts (the reader) is. Over every co-occurrence, composition is barely above base (0.15 vs 0.13).
- Precision degrades gracefully with premise quality. Composing only over premises the package already trusts, such as judged-kept inferences (right about 0.8), should give conclusions about 3x base (~0.4).
- That is below what the fact-table code writes today (0.6-0.7 right, distant labels; a different metric). A real-text composition write path is NOT yet worth a GPU build.

**Next.** Premise quality: compose only over judged-kept and typed-confirmed facts. In the rule world, composition through verified rules is already exact.
- **Label-free premise filters (same tool, FC_READERS).** None helps. The pair co-occurring in 2+ sentences gives 48 / 10 composed pairs (precision 0.29 / 0.00). A short gap ending in a preposition gives 62 / 96 (0.00). Both together give 0 rules. They are too sparse within one document. A premise filter needs the package's own judged and kept facts, or the label-free extractor (tools/consistency_teacher.py, top 5% 0.54).

## READER DIAGNOSIS (17 Nov 2026, 21:20-21:45; tools/reader_diag.py on distant_10k and distant_50k; tools/consistency_teacher.py CT_COMPOSE)

**R1. Where gold is lost (Re-DocRED, 1,000 held-out docs, 21,390 gold related pairs).**

| Reader's candidates | Gold visible | Precision |
|---|---|---|
| Mention pairs within 12 tokens | 0.359 | 0.258 |
| Same sentence | 0.464 | 0.226 |
| + pronoun-led sentences resolved (the title, or the previous same-type name) | **0.578** | 0.223 |

Simple cross-sentence resolution adds 60% coverage at the same precision.

**R2. Cross-document redundancy (distant_50k, 605,667 pairs).** Label rate by the number of distinct documents with the same coarse frame:

| Documents | 1 | 2 | 3-4 | 5+ |
|---|---|---|---|---|
| Label rate | 0.124 | 0.263 | 0.311 | 0.401 |

Strong but narrow: under 3% of pairs recur.

**R3. Self-trained reader on redundancy pseudo-labels: NEGATIVE.**
- AUC 0.55 on held-out distant records, 0.50 on Re-DocRED. The pseudo-positives mark popular names, not relational wording.
- The same reader trained on DISTANT LABELS (outside knowledge, the upper bound): AUC 0.81 / 0.79, Re-DocRED top 10% 0.69-0.70 related.
- **R4.** Pair redundancy from the whole distant corpus as the Re-DocRED reader: gold rate 0.26 / 0.35 / 0.36 / 0.30 for 0 / 1 / 2 / 3+ documents. Weak.

**Composition over each reader's premises** (rules over coarse keys; composed pair gold-related; base 0.132, every co-occurrence 0.155):

| Reader | Premise precision | Composed precision |
|---|---|---|
| Self-trained, top 10-25% | n/a | 0.22-0.25 |
| Redundancy, top 25% | 0.30 | 0.27 |
| Distant-label reader, top 10% | 0.68-0.70 | **0.36-0.42** |
| Distant-label reader, top 25% | 0.54-0.56 | 0.25-0.28 |
| **13 Nov label-free extractor** (consistency teachers + self-training, AUC 0.63, top 10% 0.45) | 0.43 / 0.37 (top 10 / 25%) | 0.07 (30) / 0.18 (122) |

The extractor's composed figures (top 10 / 25%) are too few to count.

**Conclusion.** The reader's signal is in the wording features (a labelled reader reaches AUC 0.79), but no label-free teacher tried so far gives it (0.50-0.63). Composition over real text needs premises right about 0.7. Today only a reader trained on outside (distant) labels gets there, and composes at 0.36-0.42 precision.

**Decision for the author.** Use distant supervision (Wikidata labels, an outside source, labelled as such), or keep the system label-free and find a better internal teacher, e.g. the package's typed confirmations of written inferences, which separated right from wrong at AUC 0.63-0.71 on the GPU.

**New build ready: cg27a_kcr.** kcb + the POWERED COMPOSITION TEST (rule world x4, half the relational module's written composites held back, as cg27a_step4p) + the two-hop probe readout. The local safety run is queued after the consolidation paraphrase test.

## GPU cg27a_kcw (kcb + conclusions in 7 wordings), visit 4; label-free AGREEMENT teacher for the reader (17 Nov 2026, 21:35)

**kcw, visit 4.**
- Taught conclusions recalled from the trunk: 0.09 (22). kcs 0.09, kcb 0.06, kfc3 0.12. **Fewer wordings is not the fix at visit 4.** Visit 5 decides whether kfc3's 0.36 at visit 5 was a late effect of the boost (x8).
- Facts 0.81-0.82. Gated: right when answered 1.00; never-read given a stored name 0.21.
- Relational-module composites held back recalled 2 of 3 (taught 0.88). The late held-back recall seen in kcs and kfc3, on tiny n; kcr powers it.

**Reader, label-free AGREEMENT teacher (tools/reader_diag.py RD_TEACH agree, distant_50k).** Positives: a pair in 2+ documents with the same relational last word. Negatives: pairs joined only by list glue, or seen once.
- AUC 0.51 (distant held-out), 0.54 (Re-DocRED); top 10% 0.36.
- Composition over its top 10%: 0.35 (34 conclusions), top 25% 0.16. **Negative.** No label-free teacher yet approaches the labelled reader (AUC 0.79).

The author chose to stay label-free (no distant supervision for the reader).

## Label-free reader teachers and COMPOSE-THEN-CONFIRM (17 Nov 2026, 21:40-21:50)

**TYPED-CONFIRMATION teacher for the reader (tools/reader_diag.py RD_TEACH typed, distant_50k).**
- Teacher: a record is confirmed by a later document with the same head, last word and tail (11,235 records), or only contradicted (63,836).
- Pseudo-positive label rate 0.445 vs negative 0.148.
- The reader reaches AUC 0.62 on both corpora: the best label-free reader, level with the 13 Nov extractor (0.63) and far below the labelled reader (0.79).
- Composition over its top 25 / 50%: 0.20 / 0.22.

**COMPOSE, THEN CONFIRM (tools/compose_confirm.py; label-free, distant labels for evaluation only).** Compose from every co-occurrence, then keep conclusions that later reading confirms.

| distant_50k (10k in brackets) | Precision | n |
|---|---|---|
| Base: random within-document pair never recorded together | 0.070 (0.066) | 308,149 |
| Two-step chain, no rule | 0.082 (0.075) | 462,383 |
| Chain, CONFIRMED by a later document | 0.265 (0.275) | 25,834 |
| Composed by a rule (TYPES / COARSE keys) | 0.083 / 0.098 | n/a |
| Composed + confirmed, any wording (TYPES / COARSE) | 0.261 / 0.281 | n/a |
| **Composed + confirmed, relational wording (TYPES / COARSE)** | **0.311 / 0.337 (0.344 / 0.296)** | 13,752 / 5,543 |

**Conclusions.**
- Confirmation by later reading does the work: x3.8-4.8 over base, matching the GPU's predict-then-confirm (cg27a_rtc: 0.49 vs 0.21).
- Rules over label-free keys add little on top: 0.31-0.34 vs 0.27 for chains confirmed without a rule.
- The confirmed pairs are later STATED, so this is anticipation (as rtc), not inference of something never read.
- **For label-free real-text inference, the judge (later reading) matters more than the composer.** Composition rules earn their place only once relation keys are clean, as in the rule world.

## Local: two-hop probe questions through the key-returning memory (17 Nov 2026; inf_kmc_probe2 readout, kcs preset, local run to visit 5)

| Visit | Trunk from the weights | Through the memory, C first | First hop right | Both facts in the memory |
|---|---|---|---|---|
| 2 | 0.02 | 0.00 (22 chains) | 0.23 | 0.00 |
| 3 | 0.02 | 0.07 (46) | 0.24 | 0.02 |
| 4 | 0.04 | 0.02 (89) | 0.15 | 0.01 |
| 5 | 0.02 | n/a | n/a | n/a |

- Locally, the probe facts are almost never in the memory: the memory holds the question pool, which is small here. The readout measures coverage, not composition. On the 1 covered chain, C was first (1.00).
- The existing KEY-RETURNING readout (chains of entries in the memory) composes above chance locally: top-1 0.33-0.55, rank 0.52-0.75 (chance 0.10).
- The probe readout rides in cg27a_kcr for the GPU, where the pool is up to 2,000 facts.

## LABEL-FREE RELATION OPERATORS ON REAL TEXT: POSITIVE (17 Nov 2026, 22:25; tools/latent_compose.py LC_LABELFREE 1, Re-DocRED, tg_lm192_18000 name keys)

**Setup.** The relation key is what a reader sees: the coarse frame (type A, last word between, type B, order) of every mention pair within 12 tokens in one sentence. No gold facts or relations are used. 524 ridge maps (keys with 50+ training records).

**Test.** Held-out gold composites whose chain A-B-C is readable as two records. C is ranked among the document's entities.

| Route (n 2,367) | Label-free keys: C first | Confident quarter | Gold relation keys (E2, n 13,413) | Gold, confident quarter |
|---|---|---|---|---|
| Pure latent composition W_k2 W_k1 key(A) | **0.155** | 0.279 | 0.254 | 0.481 |
| Memory hop 1 + latent hop 2, W_k2 key(B) | **0.216** | **0.442** | 0.339 | 0.677 |
| Null (random second map) | 0.045 | 0.042 | 0.066 | 0.078 |
| Chance | 0.057 | n/a | 0.055 | n/a |

**Conclusions.**
- The system CAN learn relation operators from its own reading of real text, without labels, and compose them. Pure composition is 2.7x chance, and 3.8x with the first hop read from memory. The null is at chance.
- That is about 60-65% of the gold-key result, and 0.44 for the confident quarter.
- This is the label-free route to real-text composition that the rule mining lacked: learned maps degrade gracefully where symbolic rules over noisy keys did not.
- Caveats:
  - Ranking among a document's entities (about 18).
  - Name keys from a stand-in trunk.
  - Chains where both hops are readable in one sentence.
- Next: compose with these maps, CONFIRM by later reading (tools/compose_confirm.py: confirmation x4), and teach.

## Conclusion paraphrase transfer under consolidation, offline (17 Nov 2026, 21:45-22:40; tools/consolidate.py CS_CONC 1, A6 stand-in trunk, 600 steps, anchor mix)

**Setup.** Untyped conclusions 'A is connected to C', tested in the paraphrase 'A is tied to' (never taught), as the package tests conclusions.

| Rule, lr | Frames 6: rank / open | Frames 21: rank / open |
|---|---|---|
| Every weight + KL anchor, 3e-4 | 0.128 / 0.020 | 0.107 / 0.030 |
| Every weight + KL anchor, 1e-3 | **0.150 / 0.055** | 0.122 / 0.037 |
| MLP output only, 3e-4 | 0.102 / 0.007 | 0.115 / 0.015 |
| MLP output only, 1e-3 | 0.110 / 0.025 | 0.110 / 0.013 |

Base 0.113 / 0.000; chance 0.10.

**Conclusions.**
- Neither every-weight training nor 21 frames reduces paraphrase transfer here. Every weight with the anchor is if anything best (6 frames, lr 1e-3).
- But the stand-in learns untyped conclusions barely in any wording (taught wording 0.10-0.15), so the test is weak.
- The output-weights-only arm is not supported as a fix. kcw (7 wordings) was negative on the GPU at visit 4.
- The open lead for the GPU conclusion regression remains the late recovery in kfc3 (0.36 at visit 5). kcw's visit 5 decides it.
- The MLPOUT per-domain changes come from the stand-in learning the anchor's domains, not from forgetting.

## LEARNED KEYS FOR COMPOSITION (label-free, jointly with the maps): POSITIVE (17 Nov 2026, 23:35; tools/latent_compose.py LC_JOINT 1)

**Setup.** Name embeddings start from the trunk codes and are trained jointly with the 524 label-free coarse-frame maps on the training documents' co-occurrence records only. Objective: in-batch contrastive (tails), with an anchor to the initial codes; 3,000 steps; 140,238 records; no labels. Test: the same held-out composites as E3 (n 2,367; chance 0.057).

| Route | Fixed trunk codes (E3) | **Learned keys (E4)** | Gold relation types (E2) |
|---|---|---|---|
| Pure latent composition | 0.153 | **0.192** (+25%) | 0.254 |
| Memory hop 1 + map | 0.217 | **0.278** (+28%) | 0.339 |
| Memory hop 1 + map, confident quarter | 0.441 | **0.651** | 0.677 |
| Null (random second map) | 0.045 | 0.062 | 0.066 |

**Conclusions.**
- Keys learned for composition, without labels, beat the trunk's fixed codes by a quarter.
- In the confident quarter they reach the gold-relation result (0.65 vs 0.68).
- **This supports the author's hypothesis: learned keys make (fuzzy) composition work on real text.**
- Caveats:
  - Ranking among a document's entities.
  - A stand-in trunk.
  - Chains readable in one sentence per hop.
  - Names that occur only in held-out documents keep their initial codes; there is no leakage of held-out relations.

## Overnight run log (17-18 Nov 2026)

- 22:40-23:30 kcr local safety run: killed with its waiter shell at the 2-hour background limit, in phase 2. Phase 1 took 53 min with another job alongside and the 4x rule world. To re-run alone after compose-then-confirm, with a longer local timeout (LR_TIMEOUT).
- 23:35 container restart: compose-then-confirm (tools/latent_confirm.py) restarted. Queue after it, one at a time: kcr local safety run, then the long synthetic composition run (tools/compose_synth.py CS_STEPS 20000, CS_NTRAIN 2000, CS_NHELD 200).

## COMPOSE WITH LABEL-FREE MAPS, THEN CONFIRM (17 Nov 2026, 23:40; tools/latent_confirm.py, distant_10k, 5,000 training docs, 502 maps, fixed trunk name keys)

| Conclusions | Precision (distant labels) | n |
|---|---|---|
| Base: random within-document pair never recorded together | 0.066 | 61,320 |
| Every chain's C (no map) | 0.078 | 228,358 |
| Map's choice (memory hop 1 + map) | 0.073 | 129,535 |
| Map's choice, confident quarter | 0.101 | 32,385 |
| Map's choice + confirmed later, any wording | 0.261 | 3,374 |
| Map's choice + confirmed later, relational wording | 0.311 | 1,609 |
| **Map's choice, confident quarter + confirmed (relational)** | **0.397** | 526 |

**Conclusions.**
- On the noisy distant corpus the maps alone add little over chains. Their confident quarter is 0.10 vs 0.078.
- Margin and later-reading confirmation STACK: 0.40, 6x base. That beats rules + confirmation (0.31-0.34) at a smaller n.
- **Below the about 0.5 target, so not yet a GPU build.**
- Next, cheap: the same with the JOINT LEARNED KEYS (+25-28% on Re-DocRED), and two independent confirmations.
- 00:12 second container restart: the kcr local run was lost in visit 3. It ran alone at normal speed: phase 3 at 23.7 min. Its partial log (scratchpad lr_kcr_part1.log) shows no errors through visit 2: the conclusion boost ran, the memory two-hop readout ran, and the probe readout covered both facts of 9% of probe chains (C first on those, 2 of 2). Relational writes were 'none yet' at visit 2 locally. Restarted at 00:13.

## cg27a_kcr local safety run: PASSES (18 Nov 2026, 01:24; kcb + rule world x4 + half the written composites held back + probe readout)

| Run | HEADLINE forgetting | Final BPC |
|---|---|---|
| kcr | **-0.041** | **2.596** |
| kcs (reference) | -0.036 | 2.608 |

- Within local noise, and slightly better. No errors.
- Facts taught at least one visit, from the trunk: 0.13 / 0.17 / 0.19 at visits 3-5. Gated never-read 0.17 at visit 4.
- Locally the relational module writes nothing ('none yet' at visits 1-5: the rules need the GPU's larger sample). Conclusions n 2. The composition test and conclusion recall can only be read on the GPU.
- Probe readout: memory coverage of both facts 0.01-0.09; C first on the covered chains 1.00 (2).
- **Ready for the GPU:** colab/platypus_gpu_kcr.ipynb (branch cg27a-distill).

## Long synthetic composition test: NEGATIVE (18 Nov 2026, 01:25-02:15; tools/compose_synth.py, 3 x 192, 20,000 steps, 2,000 taught people, 200 held out)

| Step | Person's city | Citizenship TAUGHT (2,000) | Citizenship NEVER TAUGHT (200; chance 0.10) |
|---|---|---|---|
| 5,000 | 0.20 | 0.15 | 0.12 |
| 10,000 | 0.86 | 0.15 | 0.10 |
| 12,000 | 1.00 | 0.32 | 0.11 |
| 15,000 | 1.00 | 0.68 | 0.10 |
| 20,000 | 1.00 | **0.97** | **0.10** |

- The trunk knows both hops perfectly, and has MEMORISED all 2,000 taught composites (0.97). It composes none of the 200 held-out ones (chance).
- After pretraining, new facts taught briefly: CITIZENSHIP 0.09-0.15 in all four arms (old city, new random, new typed, typed only), with both hops at 0.83-1.00.
- This is memorisation instead of composition, as in the 6,000-step runs and the literature. Composition reported by Wang et al. needs training far past memorisation.
- The GPU's late hint (held back 2/6, kfc3) is therefore more likely noise. kcr's powered test (about 30 held back) settles it.
- The working route stays compose in key space, then teach.

## Compose with LEARNED keys, then confirm ONCE or TWICE (18 Nov 2026, 02:15-02:37; tools/latent_confirm.py LF_JOINT 1, distant_10k)

Name keys learned jointly with 502 label-free maps (161,610 training records). Precision by distant labels:

| Conclusions | Fixed keys (23:40) | **Learned keys** | n (learned) |
|---|---|---|---|
| Base: random pair never recorded together | 0.066 | 0.066 | 61,320 |
| Every chain's C | 0.078 | 0.078 | 228,358 |
| Map's choice | 0.073 | 0.076 | 129,301 |
| Map's choice, confident quarter | 0.101 | 0.100 | 32,327 |
| + confirmed later, relational wording | 0.311 | 0.341 | 1,552 |
| Confident quarter + confirmed (relational) | 0.397 | 0.353 | 589 |
| **+ confirmed TWICE (2+ later documents, relational)** | n/a | **0.533** | 304 |
| **Confident quarter + confirmed twice** | n/a | **0.565** | 131 |

**Conclusions.**
- Two independent confirmations by later reading take label-free composed conclusions **above 0.5 precision (8x base)**, the threshold for a real-text composition write path. Learned keys add a little to single confirmation (0.34 vs 0.31).
- Caveats:
  - Small n (131-304 of 129,301 composed). The write path would be selective, about 0.2% of candidates.
  - Confirmed pairs are later stated (anticipation, as cg27a_rtc), so the value is in writing and teaching them BEFORE they are read.
  - Distant labels undercount.
- **Candidate build:** compose with label-free maps, hold as predictions, write when confirmed twice, then judge, keep and teach. Needs a package implementation, a local test, then the GPU.
