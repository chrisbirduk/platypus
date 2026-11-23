# Results log (north-star rungs and the cg27a write side)

Results go here as they come, with the conditions that limit them; negatives stay in. Local = the five-domain harness (35 probes; runs
without an explicit seed are seed 0; local recall differences under about 0.15 are seed noise). GPU = Colab, 130 probes, pass 1 (10 visits).
"Shadow" = a readout that fits a variant on one sleep's conclusions alone, applies it to the pre-edit weights, measures and removes it.

**Latest summary:** "Summary, 20-22 November" at the end of this file (earlier: "Summary, 18-19 November"). The log below is in date order.

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

## Replication at 50,000 documents (18 Nov 2026, 03:00; fixed keys: joint keys for 351k names exceeded 8 GB)

1,462 label-free maps. Precision by distant labels:

| Conclusions | 50k, fixed keys | n | 10k, learned keys |
|---|---|---|---|
| Base | 0.070 | 308,149 | 0.066 |
| Every chain's C | 0.085 | 1,170,051 | 0.078 |
| Map's choice, confident quarter | 0.116 | 191,995 | 0.100 |
| + confirmed once (relational) | 0.290 | 28,331 | 0.341 |
| Confident quarter + confirmed once | 0.391 | 8,487 | 0.353 |
| Confirmed twice | 0.398 | 11,858 | 0.533 |
| **Confident quarter + confirmed twice** | **0.499** | **3,951** | 0.565 |

**Conclusions.**
- At 40x the sample, margin + two confirmations hold at about 0.50 (7x base). The 10k figures were optimistic at small n, or helped by the learned keys.
- Each filter adds:

| Filter added | Precision |
|---|---|
| Margin alone | 0.12 |
| Two confirmations alone | 0.40 |
| Both | 0.50 |

- **A label-free write path at about 0.5 precision is feasible, at about 0.5% of composed candidates (3,951 of 767,950).**

## CORRECTION and RELIABILITY LEARNED FROM CONFIRMATION (18 Nov 2026, 07:00; tools/latent_confirm.py, distant_50k, strict time order)

**Correction.** The overnight 0.50 (map's choice + confident + confirmed twice) is the precision of conclusions that later reading STATES TWICE. Such a pair has been read, so a write path that waits for two confirmations teaches nothing new. Before confirmation, the map's confident quarter alone is 0.116 (base 0.070).

**Confirmation as a TEACHER (label-free).**
- Maps are trained on docs 0-25k.
- Each (k1, k2) key pair's confirmation rate is learned on predictions in docs 25k-37.5k, confirmed only by docs before 37.5k (12,664 key pairs with 5+ predictions).
- Predictions in docs 37.5k-50k are kept by their key pair's past rate, with NO confirmation of their own, and judged by distant labels.

| Test predictions (382,539) | Precision | n |
|---|---|---|
| Map's choice, all | 0.080 | 382,539 |
| Confident quarter | 0.116 | 95,635 |
| Key pair confirmed >= 0.10 | 0.095 | 15,616 |
| Key pair confirmed >= 0.10, confident quarter | 0.183 | 3,640 |
| Key pair confirmed >= 0.20 | 0.120 | 4,711 |
| Key pair confirmed >= 0.20, confident quarter | 0.258 | 1,215 |
| Key pair confirmed >= 0.30 | 0.168 | 1,297 |
| **Key pair confirmed >= 0.30, confident quarter** | **0.356** | **427** |

**Conclusions.**
- Learned reliability + margin reach 0.26-0.36 BEFORE confirmation (4-5x base): genuine anticipation, label-free.
- But that is below what the package's fact-table code already writes (right 0.6-0.7 on its own candidates). As a separate write path, kcl would add lower-precision conclusions.
- Better use: as a SECOND, independent vote (typed by a relation map) on the code writer's candidates, or to type them. Test offline before any build.

## AGREEMENT TEST: learned maps as a second vote on the package's writer: NEGATIVE (18 Nov 2026, 08:00; tools/agree_test.py, distant_10k)

**Setup.** The writer is premises._rcode_step: code + Adamic-Adar, pairs never stated together. The vote is pure latent composition of the two bridge records' maps (A -> B -> C, both directions). Rank among 50 random same-type names (+ the true one). Within the writer's top N:

| Visit | Writer's top N | All | Map rank 1 | Map top 3 | Map rank > 10 | No usable map |
|---|---|---|---|---|---|---|
| 4 | 100 | 0.630 | 0.646 (65) | 0.650 (80) | 0.500 (6) | n/a |
| 4 | 400 | 0.438 | 0.410 (234) | 0.415 | 0.621 (29) | 0.273 (22) |
| 4 | 3000 | 0.144 | 0.163 (1,372) | 0.153 | 0.227 (181) | 0.083 (663) |
| 7 | 100 | 0.680 | 0.721 (68) | 0.735 (83) | 0.750 (4) | 0.000 (2) |
| 7 | 400 | 0.505 | 0.490 (261) | 0.502 | 0.609 (23) | 0.231 (13) |
| 7 | 1000 | 0.359 | 0.341 (628) | 0.347 | 0.508 (61) | 0.233 (43) |

**Conclusions.**
- The maps' vote adds nothing to the writer. Agreeing candidates are as right as the writer's own ranking, and the few disagreeing ones are, if anything, more often right.
- Only "no usable map" (no bridge with fitted maps) marks worse candidates. Chain existence is already in the writer's Adamic-Adar term.
- **kcl, the label-free composition write path, is POSTPONED.** On real text, the existing compressed-fact-table writer already captures what label-free composition offers:
  - before confirmation, 0.26-0.36 at best vs the writer's 0.6-0.7;
  - as a vote, no gain.
- Label-free learned maps remain useful for TYPING and as a judge signal, not as a writer.

## GPU cg27a_kcr (powered composition test) to visit 6, and cg27a_kcr2 (held back by answer) to visit 4 (18 Nov 2026, 09:04)

**Relational-module composites, never planted, recalled by the trunk (test wording).** Chance among 168 names is about 0.006.

| Visit | kcr: taught | kcr: held back | kcr2: taught | kcr2: held back BY ANSWER |
|---|---|---|---|---|
| 2 | 1.00 (7) | 0.12 (8) | 1.00 (9) | 0.00 (6) |
| 3 | 1.00 (8) | 0.23 (13) | 1.00 (10) | 0.18 (11) |
| 4 | 0.89 (18) | 0.32 (25) | 0.84 (19) | 0.17 (24) |
| 5 | 0.86 (28) | 0.30 (30) | n/a | n/a |
| 6 | 0.91 (32) | 0.21 (34) | n/a | n/a |

- Holding back by answer (no TAUGHT sibling shares the answer) lowers held-back recall from 0.30 to 0.17, but it stays far above chance.
- However, half of each family's grandchildren composites are PLANTED in the text (by name hash), so association with a planted sibling remains possible.
- The definitive test is cg27a_kcr3: half the families with NO composite planted at all (inf_rule_world_pure 0.5), and a new readout for them.

**kcr, other readouts.**
- Taught conclusions recalled: 0.00 (4) / 0.06 (18) / **0.19 (16)** / **0.29 (24)** at visits 3-6. The late recovery is confirmed, but control conclusions also reach 0.21 (14) at visit 6, so it may not be the teaching.
- Facts 0.75-0.83.
- Gated: right when answered 0.94-1.00; never-read given a stored name 0.14 / 0.02 / 0.05 / **0.00** at visits 3-6.
- **TWO-HOP PROBES THROUGH THE MEMORY** (chance 0.05; the trunk's own two-hop 0.02-0.06):

| Visit | Chains | C first | Chains with both facts in memory: C first |
|---|---|---|---|
| 3 | 10 | 0.80 | 1.00 (n 5) |
| 4 | 16 | 0.81 | 1.00 (n 10) |
| 5 | 29 | 0.83 | 1.00 (n 19) |
| 6 | 50 | 0.62 | 1.00 (n 27) |

- Held-out BPC at visit 5 vs kcs: 1.959 / 2.083 / 2.063 / 1.716 / 1.599 vs 2.006 / 2.050 / 2.058 / 1.702 / 1.592. Mean -0.002, within run variation.
- Fix: [INTEGRATED KEYED MEMORY] crashed at visit 6 on non-latin names (Github). The encode now uses 'replace'.

## GPU cg27a_kcr2 final (to visit 7; held back by answer) (18 Nov 2026, 10:02)

**Relational-module composites, never planted** (chance about 0.006):

| Visit | 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| Taught | 0.84 (19) | 0.96 (23) | 0.92 (25) | 0.80 (40) |
| Held back | 0.17 (24) | 0.14 (35) | 0.10 (41) | 0.09 (55) |

- Held-back answers that are RIGHT: 4 / 5 / 4 / 5. The count stays flat while n grows, so this is a small fixed set (association with planted siblings or hubs, or a few genuine), not composition that generalises.
- kcr3 (pure families) settles it.

**Conclusions.**

| Visit | 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| Taught | 0.11 | 0.14 | 0.36 | 0.50 |
| CONTROL (never taught) | 0.00 | 0.30 | 0.21 | 0.80 (10) |

- The late rise happens WITHOUT teaching: conclusions reach the trunk through another channel (closed-form edits / conclusion replay). The question rows add nothing for conclusions.
- The 'regression' under consolidation is a timing effect of that channel, not the teaching.

**Other readouts.**
- Facts 0.81 / 0.81 / 0.77 / 0.67 at visits 4-7. A decline at visit 7.
- Gated never-read given a stored name 0.02-0.06.
- Two-hop probes through the memory: 0.69 / 0.72 / 0.60 / 0.60 (16-63 chains); 0.97-1.00 where both facts are in memory (coverage 0.54-0.66).

## GPU cg27a_kcr3 (pure test families) to visit 3: the trunk does NOT compose (18 Nov 2026, 11:03)

- The pure families (no composite planted anywhere in the family; 164 leaves) are active. Rule-world facts per domain dropped (32-75 vs 34-116).

| Visit | Pure families: taught | Pure families: held back | All written composites: taught | All: held back |
|---|---|---|---|---|
| 2 | n/a | 0.00 (8) | 1.00 (3) | 0.00 (8) |
| 3 | 1.00 (8) | **0.00 (12)** | 1.00 (16) | 0.12 (17) |

- The 2 held-back hits at visit 3 are both in families with PLANTED sibling composites.
- **Conclusion: at this scale the trunk does not compose untaught facts.** kcr's 0.21-0.32 and kcr2's 0.09-0.17 held back were association with planted or taught siblings, not composition. This agrees with the long synthetic run (2,000 taught, 200 held out at chance) and the two-hop literature.
- The working route is the one already in place: compose in key space (the relational module, exact rules), write, and teach. The trunk then recalls taught composites at 0.80-1.00.
- Two-hop probes through the memory at visit 3: 0.70 (10 chains), 1.00 where both facts are in memory.

## kcr3 final readouts; memory COVERAGE is the experimental control; cg27a_kcf (decay floor) (18 Nov 2026, 12:10)

**kcr3, pure families.**

| Visit | 4 | 5 | 6 |
|---|---|---|---|
| Taught | 0.95 (20) | 0.96 (24) | 0.96 (24) |
| Held back | **0.00 (20)** | **0.00 (28)** | **0.00 (33)** |

- 0 of 81 across visits 2-6: the trunk does not compose.
- All families, held back: 0.06 / 0.06 / 0.10.
- Gated never-read 0.01-0.08; facts 0.78-0.87.

**Memory coverage is mostly the control, not a limit.**
- The pool is every fact the store holds (780 at visit 5, budget 2000). 1 in 5 is held back as the consolidation control (by key hash), and held-back facts are not written to the memory.
- A chain has both facts in the memory with probability about 0.8 x 0.8 = 0.64. That matches the readout's 0.52-0.66.
- Every chain whose two facts ARE in the memory is answered right (0.97-1.00). Without the experimental hold-out (deployment), two-hop answering through the memory is about that rate. No coverage build is needed.

**cg27a_kcf (decay floor).**
- kcr2 facts decline 0.81 -> 0.77 -> 0.67 at visits 5-7, as the global question decay reaches 1/16-1/64.
- kcf = kcr + inf_distill_wake_decay_min 0.125. Visits 1-5 are identical to kcr by construction. The 5-domain local harness cannot see it; tests pass.

**cg27a_kcf GPU result (18 Nov 2026, A100, seed 4, 7 visits; compared with cg27a_kcr2, which has no floor).**

| Visit 7, from the trunk | kcr2 | kcf |
|---|---|---|
| Facts taught at least one visit (n 246) | 0.67 | **0.72** |
| ... taught 1 visit (n 42) | 0.33 | **0.48** |
| ... taught 2-3 visits (n 68) | 0.60 | **0.75** |
| ... taught 4+ visits (n 136) | 0.80 | 0.79 |
| Gated: known facts right overall / never-read given a stored name | 0.31 / 0.02 | **0.50** / 0.03 |
| Confidence AUC, known-right vs never-read | 0.93 | 0.96 |
| Conclusions taught (n 30) | 0.50 | 0.23 |
| Fixed set of conclusions, every memory off (n 178-192) | 0.34 | 0.20 |
| Held-out bits per byte, read domains | 1.8796 | 1.8849 |

- Visits 5-6 are the same in both (facts 0.81-0.82, then 0.77).
- The floor halves the visit-7 fact decline (0.77 -> 0.72 instead of 0.67). The gain is in recently taught facts. Facts taught for 4+ visits were not declining.
- Gated answering improves (0.31 -> 0.50 right overall). Never-read stays at 0.03.
- Conclusions are lower (fixed set 0.34 -> 0.20). Their share of replay samples is unchanged (0.64 vs 0.61), so the floor does not crowd them out. The fixed set depends on the closed-form edits (0.01-0.02 without them; 4 kept vs 5, norm 57.7 vs 63.4). This may be seed variation in the edits, but it is not settled with one seed.
- Bits per byte are within noise (+0.005; noise about 0.017).

## Looped (weight-shared) trunk vs a deeper trunk: NO composition (18 Nov 2026, 12:10-13:50; tools/compose_synth.py CS_LOOP, 2,000 taught, 200 held out)

| Step | Looped (3 shared blocks x 2 passes): taught / NEVER taught | Deeper (6 blocks): taught / NEVER taught |
|---|---|---|
| 6,000 | 0.12 / 0.09 | 0.24 / 0.14 |
| 10,000 | 0.79 / 0.14 | 0.91 / 0.13 |
| 12,000 | 0.95 / 0.10 | 1.00 / 0.12 |
| 15,000 | 1.00 / 0.12 | 1.00 / 0.14 |
| 16,000 | 1.00 / 0.12 | 1.00 / 0.14 |

Deeper (6 blocks), taught briefly after pretraining (citizenship never taught for these people; 100 / 300 steps): old city 0.20 / 0.16; new random city 0.05 / 0.09; new typed city 0.09 / 0.10; typed only 0.09 / 0.08. Both hops are learned (city 0.91-1.00, city's country 0.75-1.00) but composition stays at chance.

Chance for the never-taught people is 0.10.

**Conclusions.**
- Both trunks know both hops (1.00) and memorise all 2,000 taught composites (1.00). Neither composes the 200 held-out people's (0.12-0.14, chance 0.10, n 200).
- A second pass through shared weights does not overcome the two-hop curse at this scale and training length.
- Composition in the weights (route B) is negative here. Route A, two reads through the integrated memory layer, is the remaining route to composition inside the model.

## Route A: composition through two reads of the integrated memory (18 Nov 2026, local package run, cg27a_kcr3 + inf_kmc_rulecomp 2)

A composite-relation question ('A is a grandchild of') is answered at prediction time by two reads of the integrated keyed memory. Read 1 (A, r1) gives B, read 2 (B, r2) gives C, and C is ranked among the rule world's 672 names (chance 0.001). The rule r3 = r1 o r2 is the TRUE rule (ORACLE), because the relational module learns none at this local scale. The pointer table, which maps memory slots back to names, covers every rule-world premise read (mode 2). Mode 1, which covers only question-pool facts, held both premises for only 3 of 269 composites and is not informative.

| Visit | Pure families | All composites never planted | First hop right |
|---|---|---|---|
| 2 | 0.66 (88) | 0.66 (111) | 0.86 |
| 3 | 0.77 (112) | 0.78 (157) | 0.89 |
| 4 | 0.52 (168) | 0.50 (239) | 0.82 |
| 5 | 0.47 (208) | 0.45 (294) | 0.77 |

- On the GPU (cg27a_kcr3), the trunk alone answered 0 of 81 pure-family held-back composites.
- The model's own memory composes facts never stated together, at about 470x chance by visit 5.
- Accuracy falls as the memory fills (pointer slots 910 -> 1,553 at visit 4). Both hops lose accuracy: the first falls 0.89 -> 0.77, and the product is below first hop squared by visit 4-5 (0.47 vs 0.59). This is likely interference in the small local memory; the GPU memory has 65,536 slots.
- Headline (readout only, no model change): forgetting -0.041, final BPC 2.623.
- Claims: this is composition at prediction time through the memory, NOT consolidation into the trunk. The rules are oracle here. The learned-rule test is the GPU build cg27a_kra.

### cg27a_kra GPU (18 Nov 2026, A100, seed 4; log to visit 4): route A with LEARNED rules

The relational module learned 2 rules from reading at every sleep, with 0 false: 'is a child of' o 'is a child of' -> 'is a grandchild of' (z +9 to +16) and 'is supervised by' o 'is supervised by' -> 'is overseen by' (z +8 to +41).

| Visit | Route A, pure families | Route A, all never planted | First hop right | Trunk alone, pure-family held back |
|---|---|---|---|---|
| 2 | **0.85** (20) | 0.76 (34) | 0.94 | 0.00 (8) |
| 3 | **0.88** (40) | 0.77 (69) | 0.90 | 0.00 (12) |
| 4 | 0.65 (52) | 0.65 (92) | 0.82 | 0.00 (20) |
| 5 | 0.40 (60) | 0.42 (106) | 0.75 | 0.07 (28) |
| 6 | 0.57 (88) | 0.53 (148) | 0.85 | 0.00 (33) |

(Run stopped at visit 7 by the user. Visit 6: taught facts from the trunk 0.74; gated never-read given a stored name 0.04.)

- End to end and label-free: the model learns the rules from text it read, then its own memory composes facts never stated, where its trunk answers 0. Chance is 0.001.
- The same visit-4 decline as locally. Claims: prediction time through the memory, NOT consolidation into the trunk.

Local diagnostics (kra3, visit 2): second hop from the true B 0.91; composite 0.70 when the first hop is right, 0.40 when wrong. CARRYING ONLY THE TOP B: 0.80 vs 0.66 carrying the top 5. Mixing in the lower-ranked candidates costs 0.14.
Local diagnostics (kra3) at visits 3-4: top B only 0.77 / 0.73 vs top 5 0.71 / 0.57; second hop from the true B 0.88 / 0.87; first hop 0.85 / 0.83. With the top B only, the composite reaches about first hop x second hop (0.83 x 0.87 = 0.72). The remaining decline is the first hop. No domain-age effect: 0.53-0.63 across domains at visit 4. On the GPU, the first hop falls further (0.94 -> 0.75 by visit 5), so top-1 alone will not remove the whole GPU decline.
kra3 visit 5: top B only 0.63 vs top 5 0.45; first hop 0.79, second hop from the true B 0.79 (both falling now). BY DOMAIN at visit 5: the newest domain (d4) 0.75 composite / 0.89 second hop vs the older four 0.31-0.41 / 0.74-0.78. So by visit 5 older domains ARE hit after a new domain is written: interference from later writes, in both hops. Headline (readout only): forgetting -0.041, final BPC 2.622.

### kraw (local, 18 Nov 2026 evening): top B only, slot-normalised decoding, route A writes (ORACLE rules locally)

| Visit | Route A, pure families, top B only (raw table) | Slot-normalised table: first hop / composite | kra3 top 5 (raw) | Never-read AUC | Route A write precision |
|---|---|---|---|---|---|
| 2 | 0.82 | 0.97 / 0.95 | 0.66 | 0.80 | 0.33 (3) |
| 3 | 0.84 | 0.99 / 0.99 | 0.70 | 0.75 | 0.50 (4) |
| 4 | 0.77 | 0.95 / 0.92 | 0.58 | 0.58 | 0.33 (6) |
| 5 | 0.65 | 0.94 / 0.88 | 0.47 | n/a | 0.10 (10) |

- **The decline is mostly decoding through crowded pointer slots.** With the slot-normalised table, route A stays 0.88-0.99 through visit 5 (all composites never planted, top B only), against 0.45-0.71 for the original readout. The first hop stays 0.94-0.99.
- **The false-memory check fails.** For families never read, the memory gives a name 0.97 of the time, and the composite score separates right answers from never-read ones at AUC 0.80 falling to 0.58. Route A needs an abstention rule before its answers are trusted.
- **The writes are mostly WRONG locally:** precision 0.10-0.50 at tiny n (relational module write precision 0.09-0.14 overall). They decode through the POOL's facts only, about 90 locally, with the raw table at score >= 0.5. Trunk recall of them: TAUGHT 0.25-0.67 vs HELD BACK 0.17-0.50 (n 2-8).
- **kra2 (GPU)** uses the slot-normalised table for writes and a much larger pool, but its write precision must be checked at visits 2-3 before it is trusted.
- Headline: forgetting -0.041, final BPC 2.623, unchanged.

### cg27a_kra2 GPU (18 Nov 2026, seed 4; route A writes with learned rules, slot-normalised pool table, top B only), visit 2

- **Route A writes:** 20 composites written for 46 subjects. The route A line said precision 0.77 (n 13), but that count left out subjects with no true composite. The relational module's line at visit 3 gives the TRUE precision for all 20: **0.50**. The 7 left out are middle names whose parent is a root; they have no grandparent, yet the memory returned a name. The readout is now fixed (2bb6b61).
- **Slot-normalised readout:** first hop 0.97 (raw 0.85), composite 0.97 (raw 0.76). Pure families, raw table: 0.90 (20).
- **Never-read subjects:** the memory gives a name 0.89; AUC 0.67. Abstention is still missing.

kra2 visit 3:
- **Route A writes:** 27 new. The old readout gave 1.00 (n 18); counting the 9 subjects with no true composite as wrong gives about 0.67.
- **Slot-normalised table:** first hop 0.97, composite 0.96 (raw 0.80 / 0.71). Pure families, raw table: 0.80 (40).
- **Never-read subjects:** given a name 0.92, AUC 0.73.
- **Trunk recall of written composites:** first measurable at visit 4.
- **The main write error is the same as the abstention failure:** route A answers for subjects that have no composite. kra3's gate requires the first read's top name to be a written subject of the memory. A root is not a subject of a parent fact, so the gate should block these writes.

kra2 visit 4: THE LOOP CLOSES MECHANICALLY, BUT WITH WRONG WRITES.

From the trunk, test wording, composites written at an earlier sleep:

| | Taught | Held back |
|---|---|---|
| Recall of the written answer | **0.82** (11) | 0.00 (7) |
| ... never planted | 0.71 (7) | 0.00 (4) |
| ... with the closed-form edits subtracted | 0.82 / 0.71 | 0.00 / 0.00 |

- The composites route A writes ARE consolidated into the trunk's weights, not into the closed-form edits.
- **But many written answers are wrong.** Precision over all 47 writes is 0.53. Of the 11 taught and recalled, the trunk's answer matches the rule world's truth for 3 (others: middle hop 1, hub 4, other 3). The trunk learned false composites written by route A: a measured false-memory cost.
- Visit 4's batch: precision 0.92 (n 13) by the OLD readout, which leaves out subjects with no composite.
- Route A, slot-normalised: 0.93 / 0.92. Pure families, raw table: 0.79. Never-read subjects given a name 0.96, AUC 0.78.
- The gate (cg27a_kra3) is required before route A writes are trusted.

### CORRECTION (18 Nov 2026, 20:00): what route A measures

Route A's reads use only the memory's ADDRESSING: the slots that the model's keys select (_kma_ptr_reads returns slot indices and read weights). They never use the memory's learned values. The pointer table that turns slots into names is built from the same prompts' recorded answers, so route A is a lookup table keyed by the model's addresses, holding recorded facts:
- **Mode 2:** every rule-world premise in the visited domains, a HARNESS ASSIST.
- **Mode 1 and the writes:** the question pool's facts, what the reader recorded.

Composition through it chains two recorded facts by address. That is the same function as the relational module's symbolic chaining over STORED hops, which already writes composites at precision 1.00. Route A's writes were less precise (0.50-0.67 before the gate).

Hence:
- 'The model's own memory composes facts never stated' (route A 0.85-0.97) is NOT supported as worded. It is retrieval and chaining of recorded facts through the model's addressing. The slot-normalised gain fixes address collisions.
- **Still supported:**
  - the trunk does not compose untaught composites (0 of 81, synthetic negatives);
  - the relational module learns the true rules from read facts, 2 found, 0 false;
  - written composites are consolidated into the trunk's weights (relational module writes: taught 0.80-0.96 vs held back 0.00-0.09; kra2's route A writes: taught 0.82 vs 0.00).
- The earlier KEY-RETURNING MEMORY two-hop readouts (0.97-1.00) are the same address lookup over pool facts.
- **The honest in-model test is ROUTE D** (inf_kmc_rulecomp_decode): the trunk decodes B for 'A r1', then C for 'B r2', with store and memory OFF, compared with the composite asked directly. Local run started.

### The correction tested, and routes D and E (local, krc, visit 2; ORACLE rules)

| Same composites (n 111) | Right |
|---|---|
| Route A, top B only | 0.80 |
| The same reads, table answers SHUFFLED among the facts | **0.01** |
| A PLAIN DICTIONARY over the same recorded facts, no model | **1.00** |

Route A's answers come entirely from the table: confirmed.

| Asking the composite | First step B | Two-step C | Asked directly | Pure families, two-step |
|---|---|---|---|---|
| Route D: trunk decodes, store and memory off (n 60) | **0.00** | 0.00 | 0.00 | 0.00 (45) |
| Route E: trunk + integrated memory's learned values (n 30) | 0.10 | 0.03 | 0.00 | 0.05 (22) |

- Route D writes: 0 written; 14 of 17 refused because B was not a recorded name.
- Locally, neither the trunk nor the memory's learned values can produce the FIRST hop of a rule-world composite: the premise 'A is a child of B', read in text but not taught. The bottleneck is knowing read premises (trunk recall of read-only facts on the GPU: 0.19-0.30), not the composition step.
- The GPU (kra4) gives the learned-rule, larger-trunk figures.

### Real-text location chains (tools/loc_chain.py, distant 10k; labels for evaluation only)

Within one document (the labels annotate a document's own pairs), pairs never recorded together:

| | Precision | n |
|---|---|---|
| Random pair | 0.064 | 116,624 |
| Any two-record chain | 0.075 | 182,658 |
| **Containment chains** (A in/, B; B in/, C; B and C places) | **0.167** | 2,313 |

- 2.2x the chain baseline, but far below the existing writer (0.6-0.7 at the top). Containment alone is weak; it needs the reader's typing or confirmation on top.
- Across documents, the precision is 0.013-0.025, but the labels annotate only same-document pairs, so that figure is not meaningful.

### cg27a_kra4 GPU, visit 2 (18 Nov 2026, seed 4): ROUTE D - THE TRUNK COMPOSES BY DECODING TWO STEPS (learned rules)

Store and memory OFF, nothing looked up. The relational module's LEARNED rule splits the composite question; the trunk decodes B for 'A r1', then C for 'B r2':

| Composites never planted | First step B | Two-step C | Asked directly |
|---|---|---|---|
| All (n 34) | 0.44 | **0.44** | 0.09 |
| Pure families (n 20) | — | **0.60** | **0.00** |

- **The first in-model composition result:** where the trunk alone answers 0 of 20 pure-family composites directly, it answers 0.60 when the learned rule splits the question into two steps.
- **The second step is perfect given B** (C right 0.44 = B right 0.44). The limit is the first step: premises the trunk knows (0.44 on the GPU, 0.00 on the small local trunk).
- This matches the literature: composition works when the bridge entity is written out (Balesni et al., v4). Here the decomposition comes from a rule the system learned itself.
- **Route D writes:** 24 written, precision **0.50** (subjects with no composite count as wrong). Refused: B unknown 4, B unstable 8, same name 10. The gate does not block middle names whose parent is a root: the decoder still outputs a known name for the root's 'parent'.
- One seed, small n.

kra4 visit 3:
- **Route D:** first step B 0.42; two-step C 0.35 vs asked directly 0.07 (n 69); pure families 0.38 vs 0.00 (n 40). Composition by two decoding steps holds, lower than visit 2 (0.60, n 20).
- **Route D writes:** precision 0.27 (n 37); 56 written in all. Refused: B unknown 15, B unstable 23, C unknown 1, same name 12.
- **The writes must not be trusted without abstention.** cg27a_kra5 adds it (B must have a recorded fact for the second relation).

### LEARNED DECOMPOSITION in the synthetic world (18 Nov 2026, evening; tools/compose_synth.py CS_COT 1, 3 blocks x 192, 2,000 training people, 200 held out, seed 0)

The training people's citizenship is written as its chain, 'A is a citizen of B > C.'. Held-out people never have citizenship in any form. Asked 'A is a citizen of ', their greedy continuation is parsed.

| Step | Taught: B / C | **NEVER TAUGHT: B / C** |
|---|---|---|
| 4,000 | 0.06 / 0.11 | 0.08 / 0.12 |
| 5,000 | 0.50 / 0.52 | 0.24 / 0.25 |
| 6,000 | 0.82 / 0.83 | 0.66 / 0.68 |
| 7,000 | 0.99 / 0.99 | 0.81 / 0.81 |
| 8,000 | 0.96 / 0.96 | **0.92 / 0.92** |
| 10,000 | 1.00 / 1.00 | 0.99 / 0.99 |
| 12,000 | 1.00 / 1.00 | **1.00 / 1.00** |

- The same world with the implication taught DIRECTLY ('A is a citizen of C') never composed for held-out people: 0.12-0.14 at every depth (3, 6, looped 3x2; chance 0.10).
- **Trained to write the bridge, the model composes for people it was never told about: 0.92 (n 200).** It learned the decomposition itself. No rule table and no lookup; every step is its own decoding.
- This matches Balesni et al. (step-by-step reasoning composes where latent composition fails) and gives the in-package route: the model learns from its consolidated composites to write the chain.
- Caveats: synthetic, one seed. Every person's city fact is known (person's city 1.00). The second step has B in context.

### cg27a_kra5 GPU (18 Nov 2026, seed 4; route D abstention + source replay 8), visits 2-3

| | kra4 v2 | **kra5 v2** | kra4 v3 | **kra5 v3** |
|---|---|---|---|---|
| Route D first step B | 0.44 | **0.65** | 0.42 | 0.41 |
| Route D two-step C, all (asked directly) | 0.44 (0.09) | **0.50** (0.12) | 0.35 (0.07) | 0.39 (0.07) |
| Route D, PURE FAMILIES (asked directly) | 0.60 (0.00) | 0.60 (0.00) | 0.38 (0.00) | 0.40 (0.00) |
| **Route D write precision** (n) | 0.50 (24) | **1.00 (12)** | 0.27 (37) | **0.89 (9)** |

- **The abstention works.** "B has no recorded r2 fact" refused 23 and 39 candidates; write precision 0.50 / 0.27 -> 1.00 / 0.89. It writes far fewer: 12 + 4.
- **Premises:**
  - Reading wording: first step 0.74 / 0.46, composite 0.68 / 0.48, pure families 0.70 / 0.50. Better than the test wording (0.65 / 0.41).
  - Composites whose premises are TAUGHT pool facts: 0.67 (15) / 0.25 (40). Not better.
  - The first step falls from visit 2 to visit 3 as domains are added: premise forgetting, not just never-learned.
- **Route E** (trunk + integrated memory's learned values): first step 0.63 / 0.50; composite 0.47 / 0.40 vs direct 0.10 / 0.03; pure families 0.58 / 0.38. About the same as route D. The memory's values do not add much beyond the trunk.
- **Route A controls on the GPU:** shuffled table 0.00 / 0.01; plain dictionary 1.00 / 1.00. The correction is confirmed at scale.

### cg27a_kra6 GPU (18 Nov 2026, seed 4): ROUTE J, the JEPA-style relational memory, visit 2

One linear operator per relation on the trunk's name codes, fitted on the 364 facts the reader recorded (15 relations). The learned rule's two operators are applied in latent space (the bridge is never decoded); C is the nearest of 572 recorded names (chance 0.002).

| Visit 2 (n 34; pure families n 20) | Route J | Route D (same run) | Route E | Asked directly |
|---|---|---|---|---|
| First hop B | **0.76** | 0.56 | 0.53 | — |
| Composite C | 0.47 | 0.41 | 0.37 | 0.09 |
| **Pure families** | **0.65** | 0.40 | 0.37 | 0.00 |
| Null (operators fitted on shuffled facts) | **0.00** | | | |

- **Margin gate >= 0.05:** answers 0.79, right when answering 0.59.
- **Never-read families:** 0.52 pass the gate; AUC right vs never-read **0.89**. The separation is good, but the threshold is too low.
- **Route J writes:** 17, precision **1.00** (17). Refused: margin 10, same name 3, B has no recorded r2 fact 16.
- **The best composition so far that is the model's own.** The facts are in learned operators fitted on reading (null 0.00), the composition is latent, and the rules are learned.
- The second step loses the most: first hop 0.76, composite 0.47. The predicted B code is applied directly, not snapped to the nearest name. Snapping is the obvious next test.

**BELIEF REVISION (CS_REV, same run).** 100 held-out people move to an old city in ANOTHER country. Only the new 'lives in' fact is taught briefly, with replay; the country is never mentioned.

| Steps | Chain gives the NEW city / NEW country | Still the OLD city / OLD country |
|---|---|---|
| 100 | 0.77 / 0.78 | 0.13 / 0.13 |
| 300 | **0.96 / 0.95** | 0.04 / 0.04 |

- Learned decomposition updates its conclusions for free: the composite is recomputed from the current premise, not stored.
- Composites consolidated as facts go stale when a premise changes (RippleEdits). Writing the chain is how the model can avoid that.

### Self-supervised pronoun linker (18 Nov 2026, night; tools/pron_link.py, distant 10k, stand-in trunk 192 x 3, 128-byte window)

Held out (a later mention replaced by a pronoun; 1,435 training and 285 held-out examples):

| Linker | Accuracy |
|---|---|
| Learned linker (best epoch) | 0.18-0.23 |
| First mention (title) | **0.22** |
| Nearest earlier mention | 0.05 |
| Nearest earlier mention of the matching type | 0.15 |

Real pronouns in 1,000 documents:
- New records through linked pronouns: precision 0.091 (n 6,324), against 0.138 for the reader's name-pair records (n 25,755).
- 574 right pairs added (+16%), at lower precision.

**NEGATIVE as built.** The learned linker does not beat 'link to the title'. Likely causes:
- The stand-in trunk's states see only 128 bytes, and a pronoun's referent is often further back.
- It uses one state per mention.
- 1,435 examples.

Next: the title heuristic as the baseline to beat; the GPU trunk's states; context-wide features (the mention's sentence and the pronoun's sentence).

### Bootstrapped reader, Snowball-style without labels (18 Nov 2026, night; tools/boot_reader.py, distant 10k; held-out last 20% of documents)

- **Seeds:** 23 generic connective templates (', born in', 'is located in', 'was founded in' ...). 588 records, labelled 0.367 against a base rate of 0.151.
- **Negatives:** list glue (',', 'and' ...).
- **Model:** a logistic classifier over the text between the names, the type pair and the distance, with its confident records added each round.

| Ranker | Held-out AUC | Precision, top 1% / 5% / 20% |
|---|---|---|
| Distance only | 0.560 | 0.079 / 0.161 / 0.184 |
| Last word is relational | 0.534 | 0.156 / 0.162 / 0.183 |
| Bootstrapped, rounds 0-3 | **0.49 -> 0.49** | 0.18 -> 0.14 / 0.17 -> 0.15 / 0.17 -> 0.16 |
| Same, list glue only between names of the same type | 0.49 -> 0.47 | 0.15 -> 0.14 / ... |

**NEGATIVE.** The bootstrapped reader is no better than chance on held-out documents and drifts with each round. In round 3, the 2,000 records it was surest were WRONG are labelled 0.58.
- **The assumption fails:** pairs joined by commas ('Paris , France') are often related (glue negatives labelled 0.15-0.17, the base rate).
- **The seeds are too narrow** (588 records) to generalise from wording features.
- **Surface wording does not carry the relation in this data.** That agrees with the earlier frame-based results (AUC 0.62 at best).
- The next reader attempt should use what the model knows (trunk states, cross-document consistency), not surface wording.

### cg27a_kra6 visits 3-4: THE CLOSED LOOP WITH THE MODEL'S OWN LATENT MEMORY

| Route J | Visit 2 | Visit 3 | Visit 4 |
|---|---|---|---|
| Recorded facts fitted / relations / names decoded among | 364 / 15 / 572 | 514 / 20 / 745 | 758 / 22 / 1,059 |
| First hop B | 0.76 | 0.81 | 0.77 |
| Composite C (all never planted) | 0.47 | **0.65** | **0.61** |
| PURE FAMILIES | 0.65 | **0.78** | **0.67** |
| Null (shuffled facts) | 0.00 | 0.01 | 0.05 |
| Margin gate: answers / right when answering | 0.79 / 0.59 | 0.81 / 0.77 | 0.77 / 0.79 |
| Never-read families passing the gate / AUC | 0.52 / 0.89 | 0.47 / 0.92 | 0.37 / **0.95** |
| Writes this sleep (total) / precision | 17 (17) / 1.00 | 29 (46) / 1.00 (39) | 13 (59) / **1.00** (37) |

**Consolidation of route J's writes** (visit 4, from the trunk, test wording, memory off):

| | Taught | Held back |
|---|---|---|
| All written composites | **1.00** (7) | 0.00 (8) |
| Never planted | **1.00** (7) | 0.00 (8) |
| With the closed-form edits subtracted | 1.00 (7) | 0.00 (8) |
| Pure families | 1.00 (5) | 0.00 (7) |
| What the trunk answers | right 7 of 7 | 'other' 8 of 8 |

- **THE LOOP CLOSES, CORRECTLY, WITH THE MODEL'S OWN MECHANISMS:**
  1. it reads;
  2. its reader records facts;
  3. its relational module learns the true rules;
  4. its latent relational memory (operators fitted on the records; null 0.00-0.05) composes facts never stated;
  5. the gated writes are all right (59 written, precision 1.00 where truth is known);
  6. the trunk then knows the taught ones in an unseen wording with the memory off (1.00 vs 0.00 held back), in its weights.
- Route D in the same run: pure families 0.35 / 0.52 (asked directly 0.00 / 0.10).
- One seed; small n for the consolidation (7 / 8).

kra6 visit 5:

**Consolidation of route J's writes** (from the trunk, test wording, memory off):

| | Taught | Held back |
|---|---|---|
| All | **1.00** (22) | 0.05 (22) |
| Never planted | 1.00 (22) | 0.05 (21) |
| With the closed-form edits subtracted | 1.00 (22) | 0.05 (22) |
| Pure families | **1.00** (15) | 0.00 (14) |
| What the trunk answers | right 22 of 22 | right 1, middle hop 4, other 17 of 22 |

- **Route J:** fitted on 888 recorded facts, 25 relations, 1,236 names. First hop 0.78; composite 0.59; pure families 0.65; null 0.01. Gate: answers 0.73, right 0.81. Never-read families passing 0.42, AUC 0.93.
- **Writes:** 68 in all, precision **1.00** (42).
- **Route D:** pure families 0.64 vs asked directly 0.27. The direct answers now include composites route J wrote and the trunk was taught.
- The consolidation result holds at a larger n: 22 taught, all right, vs 0.05 held back.

### Synthetic world, overnight (CS_IMPL 0: pretrained with NO citizenship at all; 2,000 + 200 people, 3 x 192, 12,000 steps)

**1. LATE CHAIN TEACHING WORKS (the package's situation).** After pretraining, 300 people are taught 'A is a citizen of B > C.' briefly (8 rows + 24 replayed per batch). The other people are never taught.

| Steps | Taught C | **Never taught: B / C** |
|---|---|---|
| 300 | 0.97 | 0.81 / 0.81 |
| 750 | 0.98 | 0.83 / 0.84 |
| 1,500 | 1.00 | **0.88 / 0.86** |
| 3,000 | 1.00 | 0.84 / 0.84 |

- A trained trunk learns the decomposition from a few hundred chains in a few hundred steps, and generalises it to untaught people at 0.81-0.88. Direct teaching of the same 300 is in the log for comparison.
- **DIRECT teaching of the same 300 people** ('A is a citizen of C.'): taught 0.37 / 0.65 / 0.95 / 0.99 at 300 / 750 / 1,500 / 3,000 steps, but NEVER TAUGHT **0.10 at every step** (rank among 10, chance 0.10).
- The same teaching budget in chain form generalises (0.81-0.88); in direct form it does not generalise at all.
- Chain-form teaching of consolidated composites (kra7) is supported.

**2. ROUTE J AT SCALE FAILS HERE: capacity.**

| Held-out people (n 200) | Right |
|---|---|
| First hop, city among 300 | **0.04** |
| Composite, country among 20 (chance 0.05) | 0.17 |
| Composite among all 2,520 names | 0.14 |
| With B snapped to the nearest city | 0.10 |
| Null | 0.07 |

- One linear operator per relation on 192-d name codes cannot hold 2,200 arbitrary person-city facts. A linear map memorises at most about D facts.
- On the GPU rule world it holds about 35 facts per relation (758 facts over 22 relations), within capacity.
- **Route J must become a higher-capacity memory before it scales:** a keyed or nonlinear associative memory, the JEPA predictor as an MLP, or random-feature expansion.
- Snapping does not help when the first hop is wrong.


**3. ROUTE J WITH AN MLP PREDICTOR (InfoNCE, hidden 1024, 3,000 steps; CS_OPJ_MLP 1) FIXES THE CAPACITY LIMIT. Two seeds.**

| Held-out people (n 200), citizenship never taught | Linear, seed 0 | Linear, seed 1 | **MLP, seed 0** | **MLP, seed 1** |
|---|---|---|---|---|
| First hop, city among 300 | 0.04 | 0.08 | **1.00** | **1.00** |
| Composite, country among 20 (chance 0.05) | 0.17 | 0.18 | 0.42 | 0.46 |
| With B snapped to the nearest city | 0.10 | 0.13 | **1.00** | **1.00** |
| Null (lives-in predictor fitted on shuffled facts) | 0.07 | n/a | 0.10 | 0.06 |

- The MLP predictor on the trunk's own name codes holds all 2,200 person-city facts, where the linear operator held almost none.
- Composed latently without snapping, the second hop reaches 0.42-0.46. Snapping B to the nearest recorded city first makes it exact (1.00). The null stays near chance.
- What this claims and what it does not: the predictor is fitted on recorded facts, and snapping is a cleanup to a recorded name, so the composite is two memorised hops chained. A plain dictionary would also score 1.00 here. The result is that the model's own latent memory can now hold facts at this scale and compose them without ever seeing a composite. It is not evidence that the trunk composes.
- Next: kra7 uses the MLP predictor with snapping for route J. The abstention gate still has to be checked at this capacity.

**4. LATE CHAIN TEACHING, SEED 1 (replication).** Never taught B / C: 0.88 / 0.86 at 300 steps, 0.86 / 0.85 at 750, 0.85 / 0.82 at 1,500, 0.89 / 0.88 at 3,000 (taught C 0.99-1.00). It replicates seed 0 (0.81-0.88).

**5. DIRECT TEACHING, SEED 1 (replication).** Taught 0.32 / 0.88 / 0.99 / 1.00 at 300 / 750 / 1,500 / 3,000 steps; never taught 0.14 / 0.10 / 0.12 / 0.14 (chance 0.10). It replicates seed 0: direct teaching does not generalise, chain teaching does (0.82-0.89 on this seed).

**6. ROUTE J ABSTENTION AT CAPACITY (CS_OPJ_ABST 1; MLP predictor refitted without the facts of 100 held-out people).** Three groups: RECORDED held-out people; people KNOWN to the trunk from pretraining but whose city fact was never recorded; 200 NOVEL names never read anywhere. The threshold keeps 95% of recorded subjects; the table gives the share of the other groups that would still pass (lower is better).

| Gate on the first hop | Hidden 1024, seed 0 | Hidden 1024, seed 1 | Hidden 256, seed 0 |
|---|---|---|---|
| **Top-1 minus top-2 margin**, known-not-recorded / novel | **0.00 / 0.01** (AUC 1.000 / 0.999) | **0.00 / 0.00** (AUC 1.000 / 1.000) | 0.08 / 0.07 (AUC 0.978 / 0.980) |
| Max cosine | 0.04 / 0.03 (AUC 0.994 / 0.993) | 0.00 / 0.01 (AUC 0.999 / 0.997) | 0.47 / 0.49 (AUC 0.887 / 0.864) |
| Margin over the shuffled-fact null | 0.85 / 0.84 (AUC 0.49 / 0.50) | 0.83 / 0.79 (AUC 0.59 / 0.56) | 0.94 / 0.94 |
| First hop right: recorded / known-not-recorded | 1.00 / 0.00 | 1.00 / 0.02 | 1.00 / 0.00 |
| Composite, latent / snapped (section 3) | 0.42 / 1.00 | 0.46 / 1.00 | 0.23 / 0.99 |

- **The top-1 minus top-2 margin is the abstention gate for kra7.** With hidden 1024 it rejects 99-100% of subjects whose fact was never recorded, including novel names, while keeping 95% of recorded ones. The linear route J let 37-52% of never-read families through on the GPU.
- The gate costs nothing here: the predictor does not infer a city from the trunk's knowledge of a person (0.00-0.02 right for known-but-unrecorded people), so everything it rejects would have been wrong.
- The null-margin gate does not work and should not be used.
- Hidden 256 still memorises the facts (first hop 1.00) but separates worse (margin gate lets 7-8% through) and composes worse latently (0.23). Hidden 1024 is the setting for kra7.
- Caveat: synthetic world, 2,200 facts, one relation pair. The GPU rule world has 22 relations and fewer facts per relation, so it is within this capacity.

### kra7 draft, local smoke (19 Nov, 01:50-03:10; route J MLP 1024 + snapping + auto gate; oracle rules, 800 predictor steps)

- **Inconclusive, not yet a pass.** First hop B 0.06-0.07, composite 0.01-0.04, null 0.00-0.01 over visits 2-5 (local trunk, 191-356 recorded facts). Writes: 2, both right. Forgetting -0.040, final BPC 2.623 (no harm; the readout trains nothing in the trunk).
- The automatic gate came out at 0.000: one-to-many relations (several recorded answers per subject) tie at the top, so their margins are 0 and the 5th percentile is 0. Fixed: the gate is now calibrated on single-answer subjects only.
- There is no local linear route J run to compare against (kra6 was only run on the GPU, first hop 0.76). An offline check shows the predictor memorises facts (0.95 on random codes, 800 steps), so the low first hop is either unrecorded first-hop facts (the MLP does not generalise to them, as in the synthetic world) or weak local trunk codes. The rerun (krj7b) adds a same-run linear comparison and splits the first hop by whether the fact was recorded.

**Rerun with diagnostics (krj7b, 03:35-04:50): the MLP and the linear operator tie; the local reader is the limit.**

| Visit | Recorded facts memorised (MLP) | First hop, fact RECORDED: MLP / linear | First hop, fact NOT recorded: MLP | All targets: MLP / linear |
|---|---|---|---|---|
| 2 | 0.80 (156) | 1.00 / 1.00 (n 7) | 0.00 (104) | 0.06 / 0.06 |
| 3 | 0.82 (173) | 1.00 / 1.00 (n 10) | 0.00 (147) | 0.06 / 0.06 |
| 4 | 0.83 (289) | 1.00 / 1.00 (n 17) | 0.00 (222) | 0.07 / 0.07 |
| 5 | 0.84 (308) | 1.00 / 1.00 (n 20) | 0.00 (274) | 0.07 / 0.07 |

- Locally only 6-7% of the composite targets have their first-hop fact recorded; both predictors get all of those right and none of the rest. On the GPU the reader recorded far more (linear first hop 0.76 in kra6).
- Neither predictor invents a first hop that was never recorded (0.00), consistent with the synthetic abstention result.
- The gate was still 0.000: the 16-20% of recorded facts the predictor does not reproduce tie at the top. Fixed: the gate is calibrated only on recorded facts the predictor reproduces (the recorded answer is known, so no label is used).
- Forgetting -0.040, final BPC 2.623 (identical to the first smoke).

**kra7 local safety run (kra7l, 05:00-06:20; chain form + route J MLP + snapping + automatic gate; oracle rules locally): PASS.**
- Forgetting -0.040, final BPC 2.623 (krj7b 2.623): no harm.
- **Automatic gate now works:** 0.058-0.062 (was 0.000). Never-read families passing fell from 1.00 to **0.22** (visit 4; AUC 0.77). Answers 0.13-0.24 of targets locally (few first hops are recorded here).
- Chain form exercised: 2 written composites entered the pool in chain form ('A <relation> B > C'), both held back by the control hash; held-back chain recall 0.00 (n 1-2). Too few locally to judge chain teaching; the GPU decides.
- Bug found and fixed: the route J diagnostics dict was named like route A's (dg), so route A's readout failed with KeyError in krj7b and kra7l (readout only; training unaffected). Renamed.

### Self-consistency reader on real text (19 Nov, 06:20-07:30; tools/self_reader.py, distant 10k, held-out last 20% of documents, label-free; distant labels for evaluation only)

**1. Cross-document agreement** (all 40,563 held-out records; base rate 0.151):

| Score | AUC | Precision top 5% / 20% |
|---|---|---|
| Distance only | 0.560 | 0.161 / 0.184 |
| Other documents recording the pair | 0.578 | **0.370** / 0.240 |
| Agreement + distance | **0.611** | 0.379 / 0.242 |

- Only 11.5% of held-out records appear in any other document, so agreement sharpens the top records but cannot lift the reader as a whole.

**2. Route J on real records** (free name codes + an MLP predictor per link frame, InfoNCE, trained on 55,827 unlabelled training records; reproduces 0.79 of them). Scored on the 13,566 held-out records whose two names both occur in training (base rate 0.179):

| Score (same records) | AUC | Precision top 5% / 20% |
|---|---|---|
| Distance only | 0.590 | 0.180 / 0.224 |
| Cross-document agreement | **0.678** | **0.487** / 0.355 |
| Route J: fit with everything else read | 0.606 | 0.426 / 0.304 |
| Route J under the record's own link frame | 0.611 | 0.413 / 0.286 (n 4,956) |
| Route J + agreement + distance (ranks summed) | 0.646 | 0.370 / 0.304 |

- Route J's fit is a real but weak signal (top 5% at 2.4x the base rate); agreement is stronger where both apply. Neither is near 0.85.

**3. Inference and abstention on real text: NEGATIVE.** Route J's top never-recorded object per subject and frame (111,700 proposals) is almost never a labelled pair: precision 0.0003-0.003 in every margin bin. The automatic gate (0.126) passes 12.5% of proposals at precision 0.0014.
- On real text, route J does not infer new facts, and its margin gate would NOT stop false writes there. Route J writes must stay restricted to rule-based composition with recorded premises (as in kra6/kra7: a learned rule, both hops recorded), never free link prediction on real records.

**Conclusion for the reader:** label-free signals from what else was read (agreement, route J fit) add a little at the top (precision 0.4-0.5 in the top 5%) but leave AUC at 0.61-0.68. Reaching about 0.85 needs a stronger signal: the trunk's own reading of the sentence (substitution reader, queued) or a GPU-scale trunk.

### Chain teaching: premises only read, false memories, forgetting (19 Nov, 07:30-07:50; compose_synth CS_COTLATE_X 1, seed 0, pretrained checkpoint)

Late chain teaching itself replicates (never taught B / C 0.85 / 0.83 at 300 steps, 0.87 / 0.85 at 3,000; direct teaching never taught 0.10 at every step).

**1. PREMISES ONLY READ - the premise bottleneck is real.** 300 new people whose 'lives in' sentence is only read in the text stream during teaching (never taught, never asked), 75 per exposure level:

| Read | Premise known (rank among 10) | Chain gives B / C |
|---|---|---|
| 1x | 0.25 | 0.00 / 0.08 |
| 2x | 0.40 | 0.00 / 0.04 |
| 4x | 0.45 | 0.01 / 0.11 |
| 8x | 0.81 | 0.05 / 0.08 |

- Chain decoding needs the premise RECALLED, not just recognised: at 8 reads the trunk picks the right city among 10 (0.81) but cannot produce it (chain B 0.05). The pretrained people (premise read 4x per pass over 12,000 steps) chain at 0.85.
- On the GPU this is route D's first-step limit (premise recall 0.41-0.76). Chain teaching will pay off only for premises the trunk can generate: source replay / premise questions must come first.

**2. FALSE MEMORY - A REAL RISK OF CHAIN TEACHING.** 200 NOVEL names never read anywhere, asked 'X is a citizen of':
- Before chain teaching: 0.00 give a well-formed chain (the model rambles).
- **After: 0.93 give a well-formed chain to a known city and country** (e.g. 'Beljo Selvenlin is a citizen of Morvenfi-qi > Petoryo') - a made-up premise and conclusion.
- The model's own confidence (mean byte log-prob of its chain) separates known from novel people at AUC 0.83 after teaching (0.56 before): a usable but imperfect abstention signal.
- For kra7: the dose is far smaller (route J writes, tens of composites, against 300 people x 3,000 steps here), and the GPU false-memory readouts will show any effect. A fix is queued: abstention rows for never-read names during chain teaching.

**3. FORGETTING:** bits per byte on fixed pretraining windows 0.472 -> 0.493 (+0.021) after 3,000 steps of chain teaching with replay.

### Bootstrapped reader seeded by cross-document agreement (BS_SEED agree, BS_NEG sametype, distant 10k): NEGATIVE
- Seeds 2,302 records (0.39 labelled, 2.6x base). The classifier over the words between the names learns nothing that transfers: held-out AUC 0.47-0.47 over 4 rounds (distance 0.56).
- Agreement is a property of the PAIR, not of the wording, so it does not teach a wording classifier. Agreement stays useful as a score, not as a teacher.

### Substitution reader (19 Nov, 07:50-08:08; tools/sub_reader.py, stand-in trunk 192 x 3 (tg_lm192_6000), 90,629 records from 3,000 real documents, 2 swaps): NEGATIVE locally

| Score (base rate 0.138) | AUC | Precision top 5% / 20% |
|---|---|---|
| Distance only | 0.555 | 0.162 / 0.174 |
| Cross-document agreement | 0.572 | 0.352 / 0.207 |
| **Substitution gain** (trunk surprise on Y when X is swapped for a same-type name), per byte / summed | 0.510 / 0.511 | 0.171 / 0.180 |
| Original surprise of Y (control) | 0.567 | 0.235 / 0.177 |
| Gain + distance + agreement (ranks) | 0.577 | 0.226 / 0.192 |
| Distance + agreement (no trunk) | 0.586 | 0.294 / 0.203 |

- The stand-in trunk's prediction of the second name barely depends on the first (gain at chance): a 192 x 3 byte model trained 6,000 steps does not know these real-world facts, so swapping a name costs it nothing.
- This does not test the idea at scale: the GPU trunk's own reading is the real test (a readout for kra8: substitution gain on the benchmark records, scored against the gold relations).
- Reader status after today's local tests: every label-free signal tried locally sits at AUC 0.51-0.68 (agreement best, where names are known). The local stand-in cannot carry the reader further; the remaining route is the GPU trunk.

### Abstention rows during chain teaching, and seed-1 replication (19 Nov, 08:08-08:31; CS_COTLATE_X 1, CHAIN arm only)

| | Seed 0, no abstention rows | **Seed 0, + abstention rows** (300 never-read names taught 'X is a citizen of ?.') | Seed 1, no abstention rows |
|---|---|---|---|
| Never-taught KNOWN people, chain C (300 / 3,000 steps) | 0.83 / 0.85 | **0.57 / 0.65** | 0.83 / 0.86 |
| NOVEL names given a made-up chain | 0.93 | **0.46** | 0.95 |
| NOVEL names answering '?' | 0.00 | 0.51 | 0.00 |
| KNOWN held-out people answering '?' | 0.00 | **0.23** | 0.00 |
| Confidence separates known from novel (AUC) | 0.83 | 0.73 | 0.86 |
| Premise read 8x: chain B / C | 0.05 / 0.08 | 0.01 / 0.04 | 0.12 / 0.17 |
| Forgetting, bits per byte | 0.472 -> 0.493 | 0.472 -> 0.497 | 0.478 -> 0.500 |

- **Seed 1 replicates both findings:** chain teaching reaches never-taught people (0.83-0.89) and confabulates chains for novel names (0.95); premises read up to 8 times do not chain (B 0.01-0.12).
- **Abstention rows are only a partial fix, with a real cost:** made-up chains halve (0.93 -> 0.46), but a quarter of known people now abstain and correct chains fall from 0.85 to 0.65. The trunk's sense of who it knows is too weak (AUC 0.73-0.86) to abstain cleanly by training alone.
- **Recommendation:** keep chain teaching for consolidation, but do not rely on the trunk to abstain. Answer composite questions through the gated memory (the package's integrated memory answers only where its address was written; route J's margin gate rejected 99-100% of unrecorded subjects in the synthetic world), and use the trunk's own chain confidence as a second check (AUC 0.83-0.86).

### Belief revision of consolidated composites (19 Nov, 11:16-11:37; CS_COTLATE_REV 1, seed 0): chain teaching does NOT update by itself

After late teaching (3,000 steps), 100 taught and 100 never-taught people move to a city in a different country; only 'A lives in B2.' is taught (600 steps; the new city is then known, rank 1.00). Asked 'A is a citizen of':

| | Chain form: NEW / STALE country | Direct form: NEW / STALE country |
|---|---|---|
| Taught people | 0.19 / **0.65** | 0.03 / **0.61** |
| Never-taught people | 0.14 / **0.52** | 0.08 / 0.00 (never knew it) |

- Chain-taught composites go stale like direct ones (stale 0.65 vs 0.61): the trunk's chain still runs through the OLD city, even though the new city is known (rank 1.00). It updates a little (new 0.14-0.19 vs 0.03-0.08). The pretraining-time chain result (CS_REV, 17 Nov) does not carry over to late teaching.
- So consolidated composites need explicit retraction and re-derivation when a premise changes (the package's written-composite retraction pops route J / A / D writes when a premise is retracted); the trunk will not revise them on its own. This needs a GPU readout: a premise change after consolidation, then the composite's answer.

**Seed 1 (11:38-11:59) changes the reading: chain teaching updates PARTLY and unreliably; direct teaching never does.**

| Taught people: NEW / STALE country | Chain form | Direct form |
|---|---|---|
| Seed 0 | 0.19 / 0.65 | 0.03 / 0.61 |
| Seed 1 | **0.53 / 0.28** | 0.02 / 0.54 |
| Never-taught people, seed 0 / seed 1 (chain) | 0.14 / 0.52 -> 0.50 / 0.19 | direct 0.08 / 0.00, 0.10 / 0.04 |

- Direct-taught composites never update (new country 0.02-0.03 on both seeds). Chain-taught composites update for 19% (seed 0) to 53% (seed 1) of taught people: better than direct, but seed-dependent and never complete.
- Design conclusion unchanged: consolidated composites must be retracted and re-derived when a premise changes; chain teaching makes the trunk's own update partial, not reliable.

### THREE HOPS (19 Nov, 12:00-12:18; CS_COTLATE_ARMS CHAIN3,DIRECT3, seed 0): CHAIN TEACHING EXTENDS TO THREE HOPS

Continents are NEW facts ('C is part of K.', 20 countries -> 5 continents) taught as rows during late teaching; 300 people taught 'A is a resident of B > C > K.' (chain) or 'A is a resident of K.' (direct). Never-taught people (n 200) asked 'A is a resident of' (chance 0.20):

| Steps | Chain: never taught B / C / K | Direct: never taught K | Taught (chain / direct) |
|---|---|---|---|
| 300 | 0.77 / 0.75 / **0.78** | 0.26 | 0.96 / 0.31 |
| 750 | 0.78 / 0.72 / **0.76** | 0.18 | 1.00 / 0.32 |
| 1,500 | 0.86 / 0.87 / **0.90** | 0.27 | 1.00 / 0.76 |
| 3,000 | 0.85 / 0.84 / **0.90** | 0.24 | 0.98 / 0.96 |

- Three-hop chain teaching transfers to never-taught people at 0.90, composing two pretrained premises with a fact learned only during teaching. Direct teaching learns the taught people (0.96) and transfers nothing (0.18-0.27, chance 0.20).
- The same limits as two hops should apply (premises must be recallable; confabulation on novel names; revision partial) - not measured here.
- **Seed 1 replicates (12:19-12:28):** chain three hops, never taught K 0.81 / 0.86 / 0.84 / **0.92** at 300 / 750 / 1,500 / 3,000 steps (B 0.85-0.89, C 0.76-0.90); taught 0.94-1.00.

### kra8 local safety run (19 Nov, 12:45-14:10; cg27a_kra8, oracle writes, MLP steps 800): PASS
- HEADLINE weights only: forgetting -0.039, final BPC 2.622 (kra7l -0.040 / 2.623). With the fast store: -0.038 / 2.624. No errors, no "unavailable".
- Source replay 16 (was 8): the trunk's open recall of taught facts 0.18 (kra7l 0.20), rank among 10 0.85 (0.78); within local noise. Whether more replay raises recall of read facts is a GPU question.
- SUBSTITUTION READER (new readout, local 192-wide trunk): gain AUC 0.532, B's own surprise 0.521, distance 0.566, agreement 0.500, combined 0.553 on 2867 benchmark records (gold-related 0.191). As expected from the stand-in test (0.51): the small trunk knows too few facts. The GPU trunk is the real test.
- CHAIN FORM, MADE-UP CHAINS (new readout): runs; locally only 1-2 chain composites are written (oracle writes), so n is too small to judge (novel subjects given a well-formed chain 0.00, AUC 0.75-1.00 on n 1-2).
- Route J gate: never-read questions given a stored name 0.34 / 0.20 (kra7l 0.37 / 0.22). Recall claim vs consolidation claim: nothing here is a consolidation claim; all are readouts.

### Answering composites through the gated memory (19 Nov, 15:50-16:12; CS_COTLATE_X 1, CS_COTLATE_GATE 1, CHAIN arm, seeds 0 and 1): THE EXACT-NAME GATE REMOVES MADE-UP CHAINS AT NO COST TO RECORDED PEOPLE
After chain teaching (300 people, 3000 steps), the trunk's chain answer is given only when the memory's gate opens for the subject. Store: the 300 taught people plus 100 never-taught people whose premise ('lives in') was recorded; 100 more never-taught people are known to the trunk from pretraining but never recorded. Soft gate: cosine of the question's name code to the nearest stored name code, threshold at the 5th percentile of stored people (label-free).

| Group (n 100 each) | Trunk always answers | Exact-name gate | Soft gate |
|---|---|---|---|
| Taught: right country | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.95 |
| Never taught, recorded: right country | 0.85 / 0.87 | 0.85 / 0.87 (answers 1.00) | 0.78 / 0.82 |
| Never taught, NOT recorded: right country | 0.86 / 0.86 | 0.00 / 0.00 (abstains) | 0.05 / 0.41 |
| Novel names: made-up chain | 0.92 / 0.97 | 0.00 / 0.00 | 0.10 / 0.46 |
| Near-miss of a recorded name (one syllable changed): made-up chain | 0.95 / 0.92 | 0.00 / 0.00 | 0.64 / 0.71 |

(seed 0 / seed 1)
- The exact-name (hash) gate, which the package's keyed memory already uses, removes made-up chains for novel and near-miss names (0.92-0.97 to 0.00) and keeps every right answer for people the memory recorded. That is far better than the '?' abstention rows (made-up 0.46, known people 0.85 to 0.65).
- Its cost is coverage: people the trunk knows but the memory never recorded lose their answers (0.86 to 0.00). Answer yield therefore equals the share of people the reader records.
- The soft (code-similarity) gate is worse on both counts. It leaks 64-71% made-up chains for near-miss names and is unstable across seeds. Keep exact keys.
- Caveat: the exact key trivially rejects never-read strings. On real text the open risks are aliases (the same person under another surface form: misses) and different people sharing a name (false opens). Neither is tested here.

### Retract and re-derive on premise change (19 Nov, 16:25-16:48; CS_COTLATE_REV 1, CS_COTLATE_RETRACT 1, CHAIN arm, seeds 0 and 1): THE CHAIN'S NAMED BRIDGE MAKES STALE COMPOSITES DETECTABLE AND FIXABLE, LABEL-FREE
After chain teaching, 100 taught and 100 never-taught people move to a city in another country; only the new premise 'A lives in B2.' is taught (600 steps). Check: the chain's bridge B is compared with the model's current greedy 'A lives in'. If they differ, the composite is flagged stale, retracted and re-derived as the model's 'B2 lies in'. Unmoved people are the false-flag control.

| Group (n 100; seed 0 / seed 1) | Flagged | Right country as answered | After retract and re-derive |
|---|---|---|---|
| Moved, taught | 0.76 / 0.39 | 0.19 / 0.53 | 0.93 / 0.85 |
| Moved, never taught | 0.82 / 0.37 | 0.14 / 0.50 | 0.93 / 0.82 |
| Unmoved, taught (control) | 0.21 / 0.19 | 0.92 / 0.79 | 0.83 / 0.74 |
| Unmoved, never taught (control) | 0.38 / 0.30 | 0.66 / 0.60 | 0.81 / 0.73 |

- Moved people: right country 0.14-0.53 to 0.82-0.93 after retraction. This closes the revision gap found this morning (chain composites alone updated only 0.19-0.53).
- On seed 1 only 37-39% of movers were flagged, yet they still reach 0.82-0.85. The unflagged movers had already updated their chain, so the check only needs to catch the stale ones.
- False flags on unmoved people are 0.19-0.38. Re-deriving costs taught unmoved people 0.05-0.09 and gains never-taught unmoved people 0.13-0.15 (their single-hop premises are better than their chains after the move training).
- Note: the move training itself degrades unmoved taught chains (1.00 before, 0.79-0.92 after): interference from teaching new premises.
- Recall claim, not consolidation: the re-derived answer is produced at question time by two single-hop queries; it is not written back.

### Writing back the re-derived composites (19 Nov, 17:00-17:30; CS_COTLATE_WRITEBACK 600, seeds 0 and 1): CONSOLIDATION WORKS ON ONE SEED ONLY
The flagged people's re-derived chains ('A is a citizen of B2 > C2.') are taught for 600 steps (6 rows a batch, plus 2 replayed rows of the original chains of taught people 200-300), then the composite is asked directly with no check.

| Group (seed 0 / seed 1) | As answered before | Re-derived at question time | Asked directly after write-back |
|---|---|---|---|
| Moved, taught | 0.19 / 0.53 | 0.93 / 0.85 | 0.78 / 0.43 |
| Moved, never taught | 0.14 / 0.50 | 0.93 / 0.82 | 0.84 / 0.49 |
| Unmoved taught, not replayed | 0.92 / 0.79 | - | 0.73 / 0.77 |
| Unmoved never taught | 0.66 / 0.60 | - | 0.67 / 0.73 |
| Unmoved taught, replayed | - | - | 1.00 / 0.99 |

- Seed 0 consolidates (moved 0.14-0.19 to 0.78-0.84 when asked directly). Seed 1 does not (0.50-0.53 to 0.43-0.49): only 37-39% were flagged there, and the write-back did not hold the movers it did not flag.
- Unreplayed composites lose 0.02-0.19, while replayed ones are kept at 0.99-1.00. Write-back needs replay of the other composites.
- Verdict: the question-time check (retract and re-derive) is the reliable mechanism (0.82-0.93 on both seeds); consolidating it is not yet reliable. Next: write back every checked composite (flagged re-derived, unflagged as currently given), so unflagged movers and unmoved people are replayed too.

### cg27a_kra7 GPU (19 Nov 2026, seed 4, 95 GB GPU; log to visit 9, run still going): ROUTE J WITH THE MLP PREDICTOR AND THE AUTOMATIC GATE HOLDS UP AT SCALE
Route J: MLP predictor (1024), B snapped to the nearest recorded name, label-free gate at the 5th percentile of reproduced single-answer margins. Composites never planted; C chosen among all recorded names (chance 0.002-0.000).

| Visit | Facts fitted | First hop B | Composite C | Pure families | Null | Gate answers / right when answering | Never-read pass / AUC |
|---|---|---|---|---|---|---|---|
| 2 (n 34) | 364 | 0.74 | 0.47 | 0.60 | 0.09 | 0.47 / 0.94 | 0.01 / 0.99 |
| 3 (n 69) | 507 | 0.81 | 0.64 | 0.75 | 0.01 | 0.57 / 1.00 | 0.04 / 0.98 |
| 5 (n 106) | 863 | 0.78 | 0.58 | 0.63 | 0.04 | 0.59 / 0.97 | 0.04 / 0.99 |
| 7 (n 180) | 1243 | 0.77 | 0.61 | 0.62 | 0.00 | 0.58 / 0.99 | 0.02 / 0.99 |
| 9 (n 244) | 1588 | 0.79 | 0.63 | 0.66 | 0.00 | 0.60 / 0.99 | 0.03 / 0.99 |

- Against kra6 (linear, visit 2): composite 0.47 -> 0.58-0.64 at visits 3-9 and stable as domains are added; never-read families passing the gate 0.52 -> 0.00-0.04. When it answers it is right 0.94-1.00, on 47-62% of composites.
- Same-run linear operator at visit 9: composite 0.57 (MLP 0.63). Recorded facts memorised 0.97-0.98.
- The remaining limit is the reader's coverage: first hop is 1.00 where the first-hop fact was recorded and 0.00 where it was not (51 of 244 at visit 9).
- Writes: 141 by visit 9, precision 1.00 (130 composites never planted), 133 in chain form.
- The trunk's recall of TAUGHT written composites (test wording): 1.00 / 0.77 / 0.92 / 0.92 / 0.64 / 0.38 at visits 4-9 (n 7-47). It decays as writes accumulate: consolidation of composites does not hold over many visits. HELD BACK composites stay 0.00: no transfer to untaught composites on the GPU, unlike the synthetic world (0.81-0.89). That is also no false recall of the right answer; what the trunk says instead (made up or not) is what kra8's MADE-UP CHAINS readout will show.
- Trunk probe recall at visit 9 (weights, rank among candidates): one-hop 0.74, two-hop 0.54, open two-hop 0.30, three-hop 0.08. No HEADLINE yet (log ends mid-visit 10).
- Recall vs consolidation: route J answers are a recall claim (the module answers); the 0.38 taught-composite figure is the consolidation claim.

### Writing back EVERY checked composite (19 Nov, 17:15-17:45; CS_COTLATE_WB_ALL 1, seeds 1 and 0): CONSOLIDATION OF REVISIONS WORKS ON BOTH SEEDS
As the write-back above, but every checked person's chain is taught: flagged ones re-derived, unflagged ones as currently given (375 / 397 chains).

| Group (seed 0 / seed 1), asked directly with no check | Before | Flagged only written back | EVERY checked chain written back |
|---|---|---|---|
| Moved, taught | 0.19 / 0.53 | 0.78 / 0.43 | **0.90 / 0.76** |
| Moved, never taught | 0.14 / 0.50 | 0.84 / 0.49 | **0.83 / 0.82** |
| Unmoved, taught (checked, so written back too) | 0.92 / 0.79 | 0.73 / 0.77 | 0.82 / 0.79 |
| Unmoved, never taught (checked, written back) | 0.66 / 0.60 | 0.67 / 0.73 | 0.82 / 0.76 |
| Unmoved taught 200-300, replayed originals | - | 1.00 / 0.99 | 0.99 / 0.97 |

- Seed 1's failure is fixed (moved 0.43-0.49 to 0.76-0.82); seed 0 is as good or better. The revision is now consolidated, not only recomputed at question time: a consolidation claim.
- Rewriting every checked composite also lifts never-taught unmoved people (0.60-0.66 to 0.76-0.82). Taught unmoved people are kept within 0.00-0.10 of before.
- The label "not replayed" in the log is wrong for this variant: those people were checked, so their chains were written back.
- Design: after a premise changes, re-check and rewrite every composite that depends on the changed name (via the stored bridge), not only those flagged.

### High-precision reader and route 2, usefulness over time (19 Nov, 17:50-18:10; tools/prec_reader.py, distant 10k and 50k, label-free, distant labels for evaluation)
Held-out records = the last 20% of documents. Two label sets: ANY (the pair is labelled in some document, which favours pairs seen in many documents) and DOC (labelled in the record's own document: no such bias; PR_Y doc). Precision at the top 1 / 2 / 5 / 10%:

| Signal | 10k, any labels (base 0.151) | 50k, any labels (base 0.159) | **50k, own-document labels (base 0.141)** |
|---|---|---|---|
| Distance | 0.09 / 0.13 / 0.17 / 0.19 | 0.11 / 0.14 / 0.17 / 0.18 | 0.06 / 0.10 / 0.15 / 0.16 |
| Agreement (other documents) | 0.48 / 0.44 / 0.38 / 0.33 | 0.55 / 0.51 / 0.43 / 0.37 | 0.40 / 0.37 / 0.30 / 0.26 |
| Pair specificity (PMI-like) | 0.48 / 0.50 / 0.43 / 0.35 | 0.36 / 0.43 / 0.45 / 0.42 | 0.35 / 0.40 / 0.38 / 0.34 |
| **ROUTE 2: usefulness over time** | 0.47 / 0.53 / 0.48 / 0.36 | 0.42 / 0.42 / 0.49 / 0.47 | **0.38 / 0.39 / 0.44 / 0.38** |
| Route 2, online (earlier documents only) | 0.47 / 0.52 / 0.48 / 0.35 | 0.41 / 0.42 / 0.49 / 0.46 | 0.38 / 0.39 / 0.44 / 0.38 |

- Route 2 (if this pair were stored, how often would it predict the name when A or B turns up again, minus the misses, weighted by the name's surprise) is the best label-free top-slice signal: 0.44 at the top 5% with unbiased labels, 3.1x the base rate. AUC 0.74 among records whose pair is seen elsewhere (21% of records at 50k). The online version (earlier documents only) is as good, so it can run while reading.
- Reading 5x more raises agreement coverage (11.5% to 20.7% of records) and agreement's AUC (0.58 to 0.62, any labels).
- Route 3 (conjunctions declared in advance) does not beat route 2 alone: best 0.39 (PMI >= 0.2 and agreement >= 1, 2% kept); the rest 0.23-0.27 at 11-14% kept. Rank sums with distance or repeats lower the top-slice precision.
- AUC over all records stays 0.58-0.62: the reader is still weak. A 0.44-precise top slice is not clean enough to write facts from directly; it could be one stage, with route J's fit or the GPU substitution signal as a second.
- Caveat: distant labels are incomplete, so true precision is likely higher than measured; the comparison between signals is what is reliable.

### kra8 GPU, visit 2 (19 Nov, log at 17:40; 95 GB GPU, seed 4): SUBSTITUTION READER AT CHANCE SO FAR
- SUBSTITUTION READER (benchmark records 2867, gold-related 0.191): gain AUC **0.519** (top 10% 0.22), B's own surprise 0.514, distance 0.566, agreement 0.505, combined 0.550. The GPU trunk at visit 2 knows little (one-hop recall about 0.16 at visit 2 in kra7); the readout repeats at visit 4.
- Route J at visit 2 as in kra7: composite 0.44, pure families 0.60, null 0.06, gate answers 0.38 right 1.00, never-read pass 0.07.

### Self-trained wording reader (19 Nov, 18:00-18:20; tools/wording_reader.py, label-free; own-document labels for evaluation)
Route 2 (usefulness over time) TEACHES a wording model: on training documents, records of pairs seen in other documents get pseudo-labels (top 20% by usefulness = 1, bottom half = 0; true related share 0.42-0.46 vs 0.09-0.10). A logistic regression on the words between the names (unigrams, bigrams, last word, distance bucket, name shapes) is applied to held-out records, including pairs never seen elsewhere, where co-occurrence says nothing.

| 50k documents (10k in brackets) | All held-out records (base 0.141) | Pairs NEVER seen elsewhere (79%; base 0.116) | Pairs seen elsewhere (base 0.234) |
|---|---|---|---|
| Distance | AUC 0.562 (0.563) | 0.548 | 0.566 |
| Usefulness (route 2) | 0.596 (0.562) | - | 0.743 |
| **Wording reader, self-trained (label-free)** | **0.646** (0.634) | **0.641**, top 1% 0.45 | 0.748 (0.759) |
| **Wording + usefulness (ranks)** | **0.672** (0.650), top 1/2/5% 0.47/0.47/0.47 | - | - |
| Reference: same features, TRUE labels (ceiling, not label-free) | 0.801 (0.791) | 0.795 | 0.822 |

- The best label-free reader so far on ALL records: AUC 0.672 (previous best 0.61-0.65), and the first signal that works on pairs never seen in another document (0.641, top 1% 0.45 against a base of 0.12).
- It learns sensible wordings: trusted 'capital', 'county', 'district', 'province', 'municipality', 'son', 'brother', 'his', "'s", 'dynasty'; distrusted 'during', 'countries', 'politician', 'ambassador', continent and country names in the gap (lists).
- It improves with more reading (0.634 at 10k to 0.646 at 50k; combined 0.650 to 0.672).
- The ceiling with true labels on the same features is 0.80: about half the gap to supervision is closed by self-training. This also sizes route 6 (a labelled seed).
- Caveat: pairs seen elsewhere can share boilerplate wording between training and held-out documents; the never-seen subset has no such overlap.

### Does the reader improve as it reads more? (19 Nov, 18:05-18:10; tools/grow_reader.py, distant 50k in blocks of 10k, own-document labels)
After each block, the reader is rebuilt from everything read so far and scored on the NEXT, unread block (online). Co-training: the wording reader's most confident records on pairs seen once join the pseudo-labels (2% positive, 4% negative), refitted twice.

| Read so far | Pairs seen elsewhere | Usefulness: AUC / top 5% | Wording reader: AUC / top 5% | Wording, pairs seen once | Co-training round 2: AUC / top 5% |
|---|---|---|---|---|---|
| 10k | 0.150 | 0.568 / 0.43 | 0.652 / 0.37 | 0.636 | 0.654 / 0.28 |
| 20k | 0.172 | 0.580 / 0.43 | 0.646 / 0.36 | 0.634 | 0.643 / 0.31 |
| 30k | 0.190 | 0.588 / 0.42 | 0.646 / 0.34 | 0.631 | 0.643 / 0.29 |
| 40k | 0.204 | 0.596 / 0.44 | 0.654 / 0.35 | 0.642 | 0.647 / 0.30 |

- Usefulness improves steadily with reading (AUC 0.568 to 0.596; coverage 15% to 20%): more of each new block's pairs have been seen before. This part of the reader does learn as it reads, slowly.
- The wording reader is flat (0.646-0.654): bag-of-words features saturate early. Its ceiling with TRUE labels is 0.80, so the limit is the feature set, not the amount of pseudo-labelled data.
- Co-training does NOT help: its added positives are only 0.19-0.30 related, and top-slice precision drifts down (0.35-0.37 to 0.28-0.31). The same failure as the agreement-seeded bootstrap (0.47).
- For the reader to keep improving with reading, the wording representation itself must improve with reading: the trunk's own states (which do learn) rather than fixed word counts. That is a GPU readout: the wording reader on the trunk's hidden states between the names.

### kra9 GPU, visit 2 (19 Nov, started 17:52 by the user before its local safety run; seed 4): THE WORDING READER BEATS DISTANCE ON THE BENCHMARK'S GOLD LABELS
- WORDING READER, self-trained from usefulness (10,204 / 25,504 pseudo-labels; true related 0.44 / 0.10): **benchmark records (gold, base 0.205): AUC 0.634** (top 5% 0.40) vs distance 0.557, usefulness 0.508; pairs seen once 0.629. Held-out distant records (own-document labels): 0.654, with usefulness 0.658, as offline (0.646 / 0.672).
- The first label-free reader signal that clearly beats distance on the benchmark's own gold relations (substitution in kra8: 0.519; trunk surprise 0.53).
- Rule-world facts planted per domain include the moved leaves' new premises (inf_rule_world_move 0.15). No errors to visit 2.

### Two stages: usefulness, then route J's fit (19 Nov, 18:00-19:05; tools/self_reader.py SR_USE 1, distant 10k; ANY labels, held-out records whose two names are known from training, n 13,566, base 0.179)
| Reader | AUC | Precision top 5% | top 20% |
|---|---|---|---|
| Usefulness alone | 0.694 | 0.506 | 0.439 |
| Usefulness + route J fit (ranks) | 0.674 | 0.578 | 0.376 |
| TWO STAGES: top 10% by usefulness, then route J fit (n 1,357) | 0.632 (within the slice) | 0.552 | **0.638** |
| Two stages from the top 20% | 0.583 | 0.519 | 0.518 |
- Route J's fit adds precision on the top slice: about 2% of these records (271) at 0.64, against usefulness's top 5% at 0.51. A second check helps, but stays well short of clean writes.
- Labels here are ANY labels (favour pairs seen in many documents), so the absolute precision is optimistic; the comparison between rows is what holds.

### Parser test (19 Nov, 19:05-19:15; tools/parse_reader.py, spaCy en_core_web_sm, distant 10k, own-document labels, base 0.133): PATHS ADD A LITTLE TO WORDS
| Reader (self-trained from usefulness unless stated) | All held-out: AUC / top 1% / top 5% | Pairs seen once: AUC / top 1% |
|---|---|---|
| Distance | 0.559 / 0.07 / 0.16 | 0.546 / 0.08 |
| Rule: not a coordination path | 0.510 | 0.521 |
| Rule: shorter dependency path | 0.569 / 0.21 / 0.19 | 0.567 / 0.17 |
| Words between the names | 0.630 / 0.26 / 0.30 | 0.618 / 0.26 |
| Dependency path only | 0.619 / **0.49** / 0.29 | 0.618 / 0.28 |
| **Words + path** | **0.644** / 0.45 / 0.30 | **0.637** / 0.30 |
| Reference: words + path on TRUE labels | 0.768 / 0.77 / 0.61 | 0.761 / 0.72 |
- The parser adds +0.014-0.019 AUC to words, and the very top slice gains most (top 1% 0.26 to 0.45). Coordination paths are not less related (0.147 vs 0.126): lists are not the main confusion.
- The 30k run ran out of memory after parsing (174k sentences held as parsed documents); the 10k result stands.
- By the user's decision, the parser reader goes into kra10 as a readout (inf_premises_parse 4000; the notebook installs spaCy and en_core_web_sm).

### cg27a_kra8 GPU to visit 5 (19 Nov, log at 18:49): THE TRUNK RARELY MAKES UP CHAINS ON THE GPU; SUBSTITUTION STAYS AT CHANCE
- MADE-UP CHAINS (trunk alone, store and memory off): NOVEL subjects (never read) given a well-formed chain **0.08 (13) at visit 4, 0.15 (39) at visit 5**, against 0.93-0.95 in the synthetic world. Held-back composites: well-formed 0.00 / 0.11, right 0.00. Taught: right 0.80 / 0.81.
- Its confidence separates known from novel subjects: AUC 0.60 / **0.82**. The label-free confidence gate (keeps 95% of taught) lets through **0.08 / 0.03** of novel made-up chains, but none of the held-back composites it answers are right (0 of 6, 0 of 7).
- SUBSTITUTION READER at visit 4 (benchmark, 5764 records): gain AUC **0.515**, distance 0.573. Still chance at visit 4 (0.519 at visit 2): this route is closed. The wording reader (kra9 visit 2: 0.634) replaces it.
- Trunk recall of taught written composites: 0.60 (5) / 0.86 (21) at visits 4 / 5 (kra7: 1.00 (7) / 0.77 (22)). Too few to tell whether source replay 16 slows the decay; visits 7-9 will.
- Route J at visit 5 as in kra7: composite 0.58, pure families 0.63, null 0.02; gate answers 0.54, right 1.00; never-read pass 0.07.

### cg27a_kra9 GPU to visit 3 (19 Nov, log at 18:51): RE-DERIVE RUNS CLEAN; THE FIRST MOVE READ WAS NOT RECORDED BY THE READER
- ROUTE J WRITES 16 / 39 by visits 2 / 3, precision 0.94 / 0.97 (kra7 1.00; moved leaves count against the old truth until their move is read). Route J at visit 3: composite 0.58, pure 0.65, null 0.01, gate right 1.00, never-read pass 0.03.
- ROUTE J RE-DERIVE (model change): KEPT 16 (right 1.00) / 39 (right 0.97); none rewritten or retracted. The re-derivation does not churn stable writes.
- PREMISE CHANGE: control (move not yet read; n 7 / 17): reader recorded the OLD premise 1.00, route J and the written composite give the old answer 0.71 / 0.76 (correct before the move). MOVE READ (n 1, visit 3): the reader recorded the OLD premise only (new 0.00), so route J and the written composite stay stale. One case; visits 4-6 will show whether the reader takes up new premises at all. The name memory's material does keep the latest object per (subject, relation), so a miss here would point at the store not holding the new sentences, or at the fact pool keeping entries from earlier sleeps.

### cg27a_kra9 GPU visit 4 (19 Nov, log at 19:01): THE WORDING READER IMPROVES AS IT READS; PREMISE CHANGE NOT YET TAKEN UP
- WORDING READER on the benchmark's gold (6378 records, base 0.201): **AUC 0.653, top 5% 0.52** (visit 2: 0.634 / 0.40); pairs seen once 0.652 / 0.50. Distance 0.559, usefulness 0.516. Held-out distant records: 0.645, with usefulness 0.662 / 0.45.
- PREMISE CHANGE: MOVE READ n 3: reader recorded the NEW premise 0.33, OLD 0.33; route J composite NEW 0.00, OLD 0.33; written composite stale 0.33. Control (move not read, n 17): old premise recorded 0.94, route J old answer 0.82. Still too few moves read; the reader takes up new premises only partly so far, and route J has not switched any.
- ROUTE J RE-DERIVE: KEPT 58 (right 0.98), none rewritten or retracted. Writes 58, precision 0.97.
- MADE-UP CHAINS: novel subjects given a chain **0.00** (14); confidence separates known from novel AUC **0.86**; confidence gate passes 0.00 of novel. Taught right 0.88; held back right 0.00.

### cg27a_kra9 GPU visit 5 (19 Nov, log at 19:25): THE READER TAKES UP MOST NEW PREMISES BUT ROUTE J DOES NOT YET FOLLOW; MADE-UP CHAINS RISE
- PREMISE CHANGE, MOVE READ (n 11): reader recorded the NEW premise **0.64**, OLD 0.18 (visit 4: 0.33 / 0.33). Route J composite NEW 0.09, OLD 0.18, passes gate 0.27; written composite NEW 0.09, stale 0.18. Control (not read, n 12): old premise 1.00, old composite 0.83.
- ROUTE J RE-DERIVE: KEPT 62 (right 0.97) | **REWRITTEN 1 (old wrong -> new right)** | **RETRACTED 4 (3 of them wrong)**. The first stale composites are being corrected or withdrawn; write precision 0.95.
- Route J at visit 5: composite 0.52 (kra7 0.58), pure 0.52, null 0.02; gate answers 0.54, right 0.93 (kra7 0.97); never-read pass 0.03.
- MADE-UP CHAINS **rise**: novel subjects given a well-formed chain **0.46** (37; visit 4 0.00, kra8 visit 5 0.15); held back well-formed 0.36, right 0.00; confidence AUC 0.77; the confidence gate lets through **0.30** of novel made-up chains. Trunk recall of taught written composites 1.00 (15). As in the synthetic world, chain teaching starts to make the trunk produce the chain format for any subject. Composite answers must come through the gated memory (exact-name keys: novel subjects have no written entry), not from the trunk.

### Confirmation-trained reader (19 Nov, 19:32-19:50; tools/confirm_reader.py, distant 50k in 10 blocks of 5k, own-document labels; the user's design)
The reader writes its top 5% of each new block as TENTATIVE; a tentative pair is CONFIRMED when a later block has it in another document, with a different record text and a relational wording (the reader's top 30% of that block); unconfirmed pairs FADE after 2 blocks. The reader is scored on each block BEFORE it learns from it.

| Arm (how the reader is refitted each block) | AUC on unread blocks 1 -> 9 | Confirmed: n, right | Faded: right |
|---|---|---|---|
| STATIC (never refitted) | 0.625 -> 0.627 (flat) | - | - |
| USEFUL (usefulness pseudo-labels, everything read) | 0.625 -> 0.648 | - | - |
| CONFIRM (confirmed = positive, FADED = negative) | 0.625 -> **0.459 (collapses)** | 1,769, 0.38 | 0.24 |
| RECUR (plain recurrence confirms; faded negative) | 0.625 -> 0.486 (collapses) | 3,184, 0.39 | 0.24 |
| CONFIRM_PU (confirmed vs a random sample of what was read) | 0.625 -> 0.610 | 1,061, **0.60** | 0.33 |
| USEFUL + confirmed (PU) | 0.625 -> 0.644 | 842, **0.70** | 0.35 |

Tentative top 5% right 0.32-0.36 (base 0.14).
- AS A FILTER, CONFIRMATION WORKS: confirmed pairs are right 0.60-0.72, about twice the tentative 0.33-0.36 and 4-5 times the base rate. Relational confirmation beats plain recurrence (0.72 vs 0.54 at block 2, the cleanest round). This supports the design for what gets consolidated and answered.
- FADING IS NOT EVIDENCE OF FALSEHOOD: faded pairs are right 0.32-0.35, the same as tentative ones, because most pairs never recur (79% seen once). Treating faded as negatives teaches the reader that its own top picks are wrong, and it collapses below chance (0.46). Unconfirmed must stay unlabelled.
- THE READER DOES NOT IMPROVE FROM CONFIRMATIONS with word-count features: positive-unlabelled refitting is flat or slightly down (0.61), usefulness plus confirmations equals usefulness alone (0.644 vs 0.648). The yield is too small (about 100-160 confirmations a block) and the features saturate (the ceiling with true labels is 0.80). Improvement by learning has to come from a representation that learns: the trunk's states (kra10's state reader curve).
- Coverage cost: of about 47,000 tentative pairs, about 1,000 are confirmed; about 11,000 true pairs fade unconfirmed.

### cg27a_kra9 GPU visit 6 (19 Nov, log at 19:57)
- WORDING READER on the benchmark's gold (7775 records, base 0.203): **AUC 0.659, top 5% 0.51** (visits 2 / 4 / 6: 0.634 / 0.653 / 0.659); pairs seen once 0.659 / 0.50. Held-out distant 0.656. It keeps rising as it reads.
- MADE-UP CHAINS back down: novel subjects given a chain **0.08** (50; visit 5: 0.46, visit 4: 0.00); confidence AUC **0.87**; the confidence gate passes **0.00**. Visit 5 was a spike, not a trend so far.
- PREMISE CHANGE, MOVE READ (n 18): reader recorded NEW 0.50, OLD 0.39; route J composite NEW 0.11, OLD 0.33; written NEW 0.11, stale 0.28. The old premise still wins in the reader for about 40% of read moves (the kra11 fix targets this). Control (n 13): old 1.00, old composite 0.85.
- RE-DERIVE: KEPT 87 (right 0.94), RETRACTED 2 (1 right). Writes 89, precision 0.94.
- Route J at visit 6: composite 0.53, pure 0.53, null 0.00; gate answers 0.53, right when answering **0.92** (kra7 visit 6: 0.60 / 0.96). Slightly below kra7; the moved leaves (scored against their new truth) and recency fitting are the likely causes.

### cg27a_kra10 GPU visit 2 (19 Nov, restarted 19:35 with the fix): THE TRUNK'S OWN STATES BEAT WORD COUNTS ON THE BENCHMARK
- STATE READER (the trunk's last-block states, 6000 + 6000 usefulness pseudo-labels): benchmark gold (3147 records, base 0.205) **AUC 0.646** (top 5% 0.36) vs word counts 0.634 (0.40); **states + words 0.671** (top 5% 0.43), the best label-free benchmark score so far. Held-out distant: states 0.617, words 0.647, both 0.655. The curve over visits starts at 0.646 (visit 2).
- PARSER READER on the GPU benchmark: path 0.545, words 0.634, **words + path 0.612** - the parser does not help here (local distant test: +0.014). The record texts are short fragments (60 characters at most), which parse poorly.
- WORDING READER identical to kra9 (same seed): 0.634 on the benchmark.

### Unit check of the premise-update fix (19 Nov, 20:05; tools/test_update_first.py): the fix does what it should
A store history for one subject and relation; the current object chosen by the old rule (latest sighting) and the fix (inf_nm_update_first: latest FIRST sighting among objects seen 2+ times):
| Case | Old rule | Fix |
|---|---|---|
| Update, then replay re-sights the old parent | OLD parent (the kra9 failure) | **NEW parent** |
| Noise: the new parent seen once | old | old (needs 2 sightings) |
| No change | old | old |
| Update, no replay | new | new |

### End-to-end real text, offline (19 Nov, 20:05-20:08; tools/e2e_reader.py, distant 50k; the steps of kra11's readout)
Read every record -> the reader keeps its top slice -> compose A-C through a shared name (different documents; hubs skipped) -> score. Premises: own-document labels (base 0.14). A-C: distant labels on checkable pairs.

| Reader policy | Kept 25%: premises / A-C right | Kept 10% | **Kept 2%** (A-C checkable; 2+ bridges right, n) |
|---|---|---|---|
| Random | 0.14 / 0.093 | 0.14 / 0.104 | 0.14 / 0.138 (4,093; 0.33, 80) |
| Nearer | 0.17 / 0.090 | 0.16 / 0.089 | 0.10 / 0.048 |
| Usefulness | 0.21 / 0.191 | 0.37 / **0.230** (26,895) | 0.37 / 0.166 |
| Wording reader | 0.24 / 0.170 | 0.30 / 0.205 | 0.26 / 0.241 (1,472; 0.47, 32) |
| **Wording + usefulness** | 0.25 / 0.158 | 0.34 / 0.169 | **0.53 / 0.302** (3,831; **0.49**, 167) |
- End to end, the combined reader's top 2% gives premises right 0.53 and composed A-C pairs right 0.30 (2.2x random at the same slice), 0.49 with two or more bridges. Usefulness alone at 10% gives the most right composites (about 6,200 of 26,895).
- Composition roughly halves precision from the premises (0.53 -> 0.30): a second weak link compounds the first. Two or more bridges restore it (0.49), at a small yield.
- The distant labels favour pairs seen in many documents, which flatters usefulness; the comparison within a column is the reliable part.

### Composing from confirmed premises (19 Nov, 20:35; tools/confirm_compose.py, distant 50k, blocks of 5k)
The wording reader's top 5% of each block are tentative premises (43,784; right 0.326); 1,207 are confirmed by a later relational record in another document (right 0.592). A-C composed through a shared name:
| Premises used | A-C composed | Checkable, right | 2+ bridges, right |
|---|---|---|---|
| Every tentative premise | 46,745 | 4,478, **0.177** | 87, 0.356 |
| **Confirmed premises only** | 262 | 181, **0.392** | 5, 0.80 |
| One confirmed + one tentative | 4,102 | 1,433, 0.232 | 19, 0.74 |
- Composing only from confirmed premises more than doubles the precision of the inferences (0.18 -> 0.39), at a much smaller yield (262). It supports kra12's order (confirm, then build on), and suggests the next step: compose from confirmed facts only (kra12 composes from the whole store graph, then confirms the composite).
- Note: locally the premise-change readout cannot test kra11's update fix - at local scale the reader records almost none of the moved leaves' facts, old or new (control: old 0.00-0.07). The GPU run is the test.

### Naming relations on real text (19 Nov, 20:52; tools/relname_reader.py, distant 50k; gold Wikidata relations for evaluation only)
The wording reader's top 10% (104,080 records, related 0.298); 32,352 of them have a gold relation (187 types; commonest P569 date of birth 22%, P577 publication date, P131 located in, P17 country).
| Grouping (label-free) | Groups | Purity | NMI | Records in usable groups (5+, majority 60%+) |
|---|---|---|---|---|
| One group (baseline) | 1 | 0.224 | 0.000 | 0.000 |
| Last word between the names | 399 | 0.466 | 0.473 | 0.131 |
| **Words between (backed off to the last word)** | 1,028 | **0.561** | **0.509** | **0.285** |
- Grouping kept pairs by the words between already names the relation for about 29% of them (groups at least 60% one relation); purity 0.56 against 0.22 for one group. A usable start for typed real-text facts ('A was born in C'), better than expected.
- The k-means rows ran out of memory (dense centres on 2^20 features) and were killed; the kra11 safety run was not affected. To rerun with smaller features (RN_BITS 16) once the safety runs are done.

### Several teachers combined without labels (19 Nov, 21:05-21:20; tools/teachers_reader.py, distant 10k in blocks of 2k; own-document labels, base 0.139): COMBINING WEAK TEACHERS DOES NOT BEAT USEFULNESS ALONE
Teachers voting +1 / -1 / abstain from what was read so far: USEFUL (usefulness over time), CONFIRM (another document states the pair in a relational wording: 1-8 words, not a list; recurs only in lists: -1), WORDING (functional wording +1; list -1), FAR (more than 8 words: -1). Student: logistic regression on the words between, scored on the next unread block.
| Arm | AUC on unread blocks 1 -> 4 |
|---|---|
| **USEFUL ONLY** | **0.612 -> 0.634** (rises) |
| Label model (Dawid-Skene, abstains not modelled) | 0.484 -> 0.494 |
| Majority vote | 0.557 -> 0.554 |
| Hand-written union of rules | 0.555 -> 0.569 |
| Useful-voted records relabelled by the label model | 0.539 -> 0.571 |
- The other teachers are weak or misleading as built here. True precision of their votes: CONFIRM +1 0.25, WORDING +1 0.19 (base 0.14); CONFIRM -1 (recurs only in lists) is right only 0.67-0.78 against a base of 0.86, so it points the wrong way. Usefulness: +1 0.45-0.55, -1 0.86-0.90.
- The label model estimated usefulness's accuracy almost exactly (0.46 / 0.90 against 0.45 / 0.90) but misjudged the correlated teachers (all depend on the pair recurring) and turned them against the student. Modelling abstention as evidence made it worse: it learned 'recurs' as the related class (fixed by not modelling abstains).
- Conclusion: combining teachers only helps when each teacher is individually strong and they are not all built on recurrence. The strong signals we have are the reader-scored relational confirmation (0.60-0.72, sparse) and the relational memory's fit (top slice 0.51 -> 0.64): a teacher set built from those, with a learning representation (trunk states), is the version worth testing. The cheap proxies used here are not.

### Strong teachers jointly, in a feedback loop (19 Nov, 21:25-21:30; tools/strong_teachers.py, distant 10k, blocks of 2k; own-document labels, base 0.139): NEGATIVE - THE STRONG SIGNALS ARE FILTERS, NOT TEACHERS
Teachers: USEFUL (usefulness over time); RCONF (reader-scored confirmation: another document states the pair in a wording the CURRENT reader puts in its top 30%; the newly trained reader scores the next round, a feedback loop); FIT (the names share neighbours in other documents, Adamic-Adar; top 10% +1, none -1). Student on the words between, scored on the next unread block.
| Arm | AUC on unread blocks 1 -> 4 | Positives' true precision |
|---|---|---|
| **USEFUL ONLY** | **0.612 -> 0.637** | 0.44-0.55 |
| USEFUL + RCONF | 0.612 -> 0.629 | 0.33-0.35 |
| ALL THREE | 0.533 -> 0.510 | 0.36-0.37 |
| AGREEMENT ONLY (two teachers agree) | 0.504 -> 0.469 | 0.32-0.35 |
- As teachers over ALL records, RCONF and FIT are weak: +1 precision 0.22-0.23 and 0.22-0.34 (base 0.14). Their strength measured earlier (0.60-0.72 for confirmation, 0.64 for the fit check) was CONDITIONAL on the reader's own top picks: they are good second-stage FILTERS of what the reader already chose, not teachers of the reader. FIT's 'no shared neighbour' -1 covers about 80% of records at the base rate (0.87) and swamps the negatives.
- The feedback loop (a better reader scoring the next confirmations) did not lift the reader: USEFUL + RCONF stays below usefulness alone at every round.
- With today's tests together: usefulness is the one teacher that improves the reader with reading (0.612 -> 0.637 offline, 0.634 -> 0.659 on the GPU). Confirmation and fit belong in the pipeline as filters for what is kept and consolidated (kra12). Further gains in the reader have to come from its representation (trunk states: kra11's curve; a pretrained base: the Colab test).

### What the reader misses (19 Nov, 21:25; tools/miss_reader.py; wording reader self-trained from usefulness, its top 10%; labels to analyse errors only)
1. STRUCTURAL: of 294,904 labelled pairs in 20,000 documents, only **0.337 ever become a record** (two names within 12 words of one sentence); 0.577 are stated only ACROSS sentences, 0.086 in one sentence but further apart. By relation: country P17 0.28, citizenship P27 0.18, country of origin P495 0.14, birthplace P19 0.27 become records; date of birth P569 0.85, publication date P577 0.63. The reader never sees two thirds of the relations at all.
2. RECALL of the top 10% (precision 0.256, recall 0.331 of related records):
   - by words between: **1 word 0.70**, 2-3 words 0.19, 4-8 words 0.11, 9-12 words 0.04. It finds relations stated by adjacency or a comma and misses those stated by a phrase.
   - by form: list-like 0.63 (commas: 'Paris, France'), verb or preposition 0.19, parenthesis 0.12.
   - by relation: located in P131 0.58, publication date P577 0.67, date of birth P569 0.54, but **date of death P570 0.01**, citizenship P27 0.07, birthplace P19 0.11, country P17 0.28.
   - by type pair: LOC-LOC 0.40, PER-TIME 0.36, but **PER-LOC 0.09**, ORG-LOC 0.14.
   - recurrence: pairs seen once 0.30, seen in another document 0.40.
3. Its errors: the top slice is dominated by ',' and '(' records (unrelated 70,655 + 18,640, related 18,026 + 12,087), mostly LOC-LOC and PER-PER lists. The MISSED related records are stated by short relational phrases: 'in' 5,719, '( born' 4,182, 'and' 2,453, 'of' 1,221, 'in the', 'state of', 'by', 'president', 'from'.
- PATTERN: the reader has learned FORM (adjacent names joined by a comma or parenthesis: 'Paris, France', 'Smith (1950') and not MEANING. It misses relations stated with a preposition or verb ('born in', 'president of', 'died'), especially person-place relations, and it cannot see cross-sentence relations at all. The teacher explains it: usefulness rewards pairs that recur, and comma-adjacent geography and dates recur most.

### cg27a_kra11 GPU to visit 4 (19 Nov, log at 21:25): ROUTE J READING EVERY RECORDED FACT JUMPS TO 0.93; THE STATE READER DOES NOT RISE
- **ROUTE J at visit 4 with inf_kmc_rulej_allrec (the held-back control fifth of recorded facts included): first hop B 0.95, COMPOSITE C 0.93, pure families 0.90, null 0.04; gate answers 0.95, right when answering 0.97** (kra9 at visit 4: 0.78 / 0.61 / 0.67; answers 0.62). The missing first hops were facts the reader had recorded but the control held out of route J. Never-read families passing the gate 0.08 (kra9 0.04). Writes 90, precision 0.97; re-derive kept 88 (0.99), retracted 2 (both wrong).
- STATE READER (benchmark gold): **0.592 at visit 2, 0.585 at visit 4** - not rising (word counts 0.634 -> 0.653). kra10's 0.646 at visit 2 was not repeated. The last-block states learn the text, not relations.
- END-TO-END (combined reader, top 2%): premises 0.54 / 0.52, composed A-C right **0.349 / 0.316** (2,291 / 2,456 checkable; random about 0.13), as offline (0.30).
- PREMISE CHANGE at visit 4 (n 3): reader recorded the NEW premise 0.67 (kra9 at visit 4: 0.33); no composite updated yet. Small n.
- MADE-UP CHAINS: novel 0.16, the confidence gate passes 0.10.
- The kra11 LOCAL safety run was killed by the memory limit at visit 5 (an analysis I ran alongside it; my mistake) after running every new part through visit 5 without errors; the GPU run is the check of its final figures.

### cg27a_kra12 GPU to visit 4 (19 Nov, started 20:23, log at 21:28): CONFIRMATION SEPARATES RIGHT FROM WRONG REAL-TEXT INFERENCES
- CONFIRM AND CONSOLIDATE at visit 4 (the 600 tentative inferences written at visit 2, judged on the documents read since): **CONFIRMED 351, right 0.73** | still tentative 249, right 0.49 | faded 0 (fading starts at visit 6). Plain recurrence of the pair (evaluation): 497 of 600.
- Only the confirmed enter the question pool: 238 taught, 113 held back (kra9 wrote 200 a readout at once, right about 0.71; kra12 consolidates 351 at 0.73 from 600 candidates).
- Route J with every recorded fact (as kra11): composite 0.97 / 0.94 at visits 2 / 3, gate answers 0.82 / 0.86, right when answering 1.00.

### Overnight tests started (19 Nov, 21:40; small-sample checks; full runs pending)
- WIDER RECORDS (tools/wide_reader.py, 3,000 documents): labelled pairs covered by some record: near (current) 0.310 -> + far in the sentence 0.387, + next sentence 0.426, + pronoun-led 0.333, all four **0.511**. The self-trained reader on the new kinds: far AUC 0.581, next sentence 0.572, pronoun 0.510 (near 0.569); the top 10% of all records is 0.69 near records. Coverage rises by two thirds; the reader does little with the new kinds yet.
- TEACHING MEANING (tools/teach_reader.py, 10,000 documents, blocks of 2,000): balancing the pseudo-labels across wording forms raises phrase recall at the last block (0.118 vs 0.060) but lowers AUC (0.604 vs 0.634); the knowledge-taught reader (every record of a confirmed pair positive) stays at or below usefulness alone (0.629) with fewer than 120 confirmed pairs. Small samples; the 50,000-document runs are pending.

## Summary, 18-19 November (for the merge to main)

**What now works (GPU, seed 4, one run each, templated rule world unless stated):**
- Composition by a learned relational memory beside the trunk (route J): held-out composites 0.93-0.97 when it reads every recorded fact, gate-answered 0.82-0.95 at 0.97-1.00 precision, never-read families let through 0.03-0.08, shuffled-fact null at most 0.09. The trunk itself still answers held-back composites at 0.00.
- Abstention and made-up answers: the memory's gates refuse questions about names never read; after chain-form teaching the trunk gives made-up chains to never-read names 0.00-0.16 at most visits (one spike 0.46), where a small synthetic trunk did so 0.93-0.95.
- Belief revision (synthetic world, two seeds): check the chain's middle name against the current premise, re-derive, write every checked chain back: people who moved 0.14-0.53 -> 0.76-0.93 right. On the GPU a store defect (replay made the old parent look newest) was found and fixed; composites updating after a move are not yet shown there.
- Real-text inferences confirmed before they are taught (`kra12`): confirmed 351 of 600 at the next readout, right 0.73 against 0.49 for the unconfirmed.

**What does not work yet:**
- Reading real text: the best label-free reader (taught by usefulness over time) is AUC 0.63-0.67 on gold relations; it learns comma and bracket forms more than phrases, and two thirds of labelled pairs never fall within its 12-word window. Substitution by the trunk, readers on the trunk's own states, a dependency parser, co-training, confirmation as a teacher, and several teachers combined (weak or strong) all failed to improve it.
- End-to-end real-text inference: about 0.30-0.35 right from the reader's top 2% (random 0.13); 0.39 when composed from confirmed premises only (offline, small yield).
- No final summary (forgetting, final BPC) from the 18-19 November GPU builds yet; one seed throughout.

### cg27a_kra12 GPU visit 4, remaining readouts (19 Nov, log at 21:46)
- Route J at visit 4: first hop 0.95, composite **0.92**, pure 0.90, null 0.00; gate answers 0.88, right **1.00**; never-read families passing 0.05. Re-derive kept 85 (right 0.99), retracted 4 (2 right).
- MADE-UP CHAINS: novel subjects given a chain **0.50** (28), the confidence gate passes 0.29 (kra11 visit 4: 0.16 / 0.10; kra9: 0.00-0.46 by visit). The trunk's chain answers for never-read names come and go between visits and builds; composite answers must come from the gated memory, not the trunk.
- State reader 0.607 -> 0.601 (visits 2 -> 4): flat, as in kra11. Premise change as kra11 (same seed): new premise recorded 0.67 of 3 moves read, no composite updated yet.

### Overnight reader tests, full runs (19 Nov, 21:45-22:00)
**Wider records** (tools/wide_reader.py, 20,000 documents, 1.46 M records): labelled pairs covered by some record: near (current) 0.316 -> + far in the sentence 0.395, + next sentence 0.434, + pronoun-led 0.339, **all four 0.520**. The self-trained reader (usefulness pseudo-labels over all kinds) on held-out documents: near AUC 0.622 (top 10% 0.24, base 0.131), far 0.597 (0.17, base 0.092), next sentence 0.580 (0.13, base 0.078), pronoun 0.569 (0.19, base 0.145). The new kinds roughly double the related pairs within reach at lower precision: worth reading, with confirmation as the filter.

**Teaching meaning** (tools/teach_reader.py, 50,000 documents, blocks of 5,000; AUC on each unread block, blocks 1 -> 9):
| Arm | AUC | Top 10% at block 9 | Phrase recall (2-12 words, not a list) | Person-place recall |
|---|---|---|---|---|
| USEFUL (now) | 0.625 -> 0.648 (best 0.653) | 0.301 | 0.165-0.183 | 0.098-0.126 |
| BALANCED across wording forms | 0.593 -> 0.634 | 0.296 | **0.214** | 0.103-0.114 |
| **KNOWLEDGE** (every record of a confirmed pair positive) | 0.625 -> **0.654 (best 0.658)** | **0.309** | 0.166-0.174 | 0.105-0.127 |
| KNOWLEDGE + BALANCED | 0.593 -> 0.616 | 0.221 | 0.04-0.05 | 0.04-0.06 |
- Knowledge-taught: a small, steady gain over usefulness alone (+0.004-0.006 AUC at every block from 5 on, top 10% 0.309 vs 0.301), from 952 confirmed pairs (right 0.67). The first loop in which what the system confirmed teaches its reader and the reader gets better; small so far.
- Balanced teacher: phrase recall up (0.214 vs 0.165-0.183) at an AUC cost (-0.015). A trade, not a gain.
- Combining the two drifts (its confirmations fall to 0.35 right) and is worse than either.

### Does the reader feed inference? (19 Nov, 22:05; tools/rcode_reader.py, distant 20k; the package's compressed-fact-table method at 128 dimensions; distant labels, candidates base 0.133)
Codes from the name x name graph of ALL records (as the package) or of the reader's top slice; candidates = pairs named together in held-out documents and never in the training graph.
| Graph built from | Records | Candidates coded | AUC | Precision top 1% / 5% / 10% |
|---|---|---|---|---|
| All records (now) | 374,587 | 0.29 | 0.551 | 0.168 / 0.157 / 0.140 |
| Reader top 50% | 187,294 | 0.27 | 0.557 | **0.243** / 0.168 / 0.131 |
| Reader top 25% | 94,055 | 0.22 | 0.546 | 0.222 / 0.144 / 0.120 |
| Reader top 10% | 57,346 | 0.18 | 0.550 | 0.231 / 0.135 / 0.115 |
- A graph from the reader's top half sharpens the very top of the inferences (top 1% 0.24 vs 0.17) and keeps 93% of the coverage; narrower slices lose coverage without further gain. A small, cheap model change for kra13 (`inf_rcode_reader 0.5`).

### Where the reader fails: teacher or features? (19 Nov, 22:00-22:05; tools/reader_ceiling.py, distant 30k, held-out documents; labels used only to diagnose)
1. Same features (words between), self-trained vs TRUE labels: comma / bracket / list 0.641 vs 0.768 (gap 0.126); short phrase 0.595 vs 0.726 (0.131); longer phrase 0.603 vs 0.700 (0.097); all 0.638 vs 0.741. **The teacher costs about 0.10-0.13 AUC in every form**, not only on phrases; the features themselves are weaker on phrases (0.70-0.73 even with true labels).
2. The teacher (usefulness) by form: +1 precision 0.450 on comma-type records (base 0.174: 2.6x), **0.385 on phrases (base 0.104-0.136: 2.8-3.7x)**; -1 precision 0.86-0.93; it votes on only 9-13% of records. It is informative for phrases, but sparse and noisy.
3. Learning curve on phrase records with true labels: 1k 0.599, 5k 0.638, 20k 0.671, 100k 0.703. The words-between features need many, clean labels.
4. Features with true labels: words 0.741; **+ the names' types 0.796; + 3 words of context each side 0.772; + both 0.816** (phrases 0.778-0.803); + the names themselves 0.790. Types are the largest single gain. But SELF-TRAINED with types + context drops to 0.616 (from 0.638): richer features let the reader learn the teacher's bias (type pairs that recur, such as place-place) faster.
- Summary: the reader loses about 0.10 to its teacher's noise everywhere and is capped near 0.74 by its features; types and context would lift the cap to about 0.82, but only with a teacher whose errors they cannot learn.

### Readers per horizon, combined (19 Nov, 22:10; tools/horizon_reader.py, distant 20k, 1.46 M records; the user's idea)
Horizons: H0 adjacent / comma / list (base 0.175), H1 short phrase (0.135), H2 longer phrase (0.104), H3 far in the sentence (0.093), H4 next sentence (0.077), H5 pronoun-led (0.142). Held-out documents:
| Arm | AUC all | Top 10% precision (related found) | Within-horizon AUC H0 / H1 / H2 / H3 / H4 / H5 |
|---|---|---|---|
| One reader (horizon as a feature) | **0.616** | **0.202** (5,946) | 0.592 / 0.600 / 0.609 / 0.596 / 0.580 / 0.573 |
| Per-horizon readers, combined by probability | 0.588 | 0.161 (4,735) | **0.619** / 0.603 / 0.611 / 0.598 / 0.578 / 0.511 |
| Per-horizon readers, equal shares | 0.595 | 0.180 (5,296) | as above |
- A specialist reader per horizon is slightly better inside the adjacent / comma horizon (0.619 vs 0.592) and equal elsewhere; the pronoun specialist is worse (too few pseudo-labels).
- Combining them without labels loses: the per-horizon probabilities (Platt scaling on each horizon's own pseudo-labels) are not comparable across horizons - every horizon's pseudo-labels are 20% positive by construction, whatever its true rate - so the next-sentence horizon (base 0.077) floods the top. Calibrating horizons against each other needs a label-free estimate of each horizon's true rate; the natural one is the pair: evidence for the SAME pair from several horizons and documents (which confirmation already uses).
- PAIR LEVEL (238,477 held-out pairs, related 0.091): combining a pair's records by noisy-or: one reader AUC **0.650**, top 10% **0.209** (its best single record 0.641 / 0.186; mean of records 0.622); per-horizon readers 0.627 / 0.193. Pairs seen in 2+ horizons are related 0.155 (all 0.091). Evidence for a pair accumulating across its records and horizons helps a little; separate readers per horizon still do not.

### A cleaner teacher, then richer features (19 Nov, 22:20-22:30; tools/clean_teacher.py, distant 40k in 8 blocks; own-document labels): NEGATIVE
Learned types (label-free: k-means on the words around each name's mentions, 24 clusters over 348,710 names): purity 0.416 against the dataset's types (majority 0.300).
| Teacher | Words between: AUC blocks 1 -> 7 | + learned types + context |
|---|---|---|
| USEFUL | 0.614 -> 0.629 (best 0.637) | 0.562 -> 0.578 |
| KNOWLEDGE (usefulness + every record of a confirmed pair) | 0.614 -> 0.634 (best **0.641**) | 0.562 -> 0.574 |
| CONFIRMED only (every record of a confirmed pair; usefulness -1 as negatives) | 0.614 -> 0.581 | 0.562 -> 0.535 |
- The confirmed pairs are NOT a cleaner teacher here: confirmed by the reader itself, they are right only 0.34-0.49 (records of them as positives 0.35-0.52), no better than usefulness's +1 votes (0.41-0.42). Confirmation's precision depends on the reader doing the confirming; earlier 0.60-0.72 came from a stricter set-up (top 5% picks, different-text, own-document labels on the original record). A weak reader cannot confirm its way to clean labels.
- Richer features hurt every teacher (about -0.06): learned types are too impure (0.42) and context words add more to learn from the teacher's bias than from the relation.
- KNOWLEDGE stays a small, steady gain on words (+0.004-0.006), as at 50k.
- Conclusion for the reader: within the label-free set-up tried, the teacher's noise is the binding limit and no available signal is clean enough to replace it; feature enrichment only pays with clean labels. The remaining levers are coverage (wider records), pooling evidence per pair, and filtering after the reader (confirmation, the relational memory's fit) - not a better reader from the same signals.

### Two reader fixes from the literature (19 Nov, 22:40-22:45; tools/research_reader.py, distant 30k, held-out documents, own-document labels)
**B. ReVerb's relation-phrase constraints** (verb, verb + preposition, or verb + words + preposition, by spaCy part-of-speech tags; and at least 10 distinct name pairs): 5.5% of phrase records pass the syntactic constraint, 1.3% both; those are related **0.166 vs 0.122** for other phrase records (1.36x). Commonest: 'was born in' 702, 'died in' 256, 'moved to' 152, 'was founded in' 109, 'returned to', 'died on', 'was released in', 'was established in', 'is located in', 'graduated from', 'led by'.
| Arm | Record AUC | Top 10% | **Phrase recall** | Phrase-record AUC | Pair AUC (noisy-or) |
|---|---|---|---|---|---|
| USEFUL (now) | 0.648 | 0.291 | 0.111 | 0.619 | 0.654 |
| **MULTI-INSTANCE** ('at least one' mention per positive pair; hard EM, 2 rounds) | **0.654** | 0.279 | 0.149 | 0.608 | **0.664** |
| **REVERB-TAUGHT** (passing phrase records added as positives) | 0.648 | 0.286 | **0.181** | 0.613 | 0.656 |
| Both | 0.653 | 0.272 | 0.163 | 0.599 | **0.665** |
- Multi-instance training lifts the pair-level score by 0.010 and phrase recall by a third, at a small cost in top-10% precision; its chosen mentions are less often labelled (0.374 vs 0.442) but more often phrases.
- ReVerb's constraints lift phrase recall by 63% (0.111 -> 0.181) at no AUC cost, where balancing the teacher cost 0.015. The phrases they find are the relations the reader missed ('was born in', 'died in', 'founded in'). Their own precision is modest because the distant labels cover few of these relations' pairs.
- Both are small, label-free gains in the direction the error analysis asked for; candidates for kra13's reader (multi-instance for pairs, ReVerb phrases as extra positives).

### How many labels does the reader need? (19 Nov, 22:50-23:00; tools/seed_reader.py, distant 30k; the seed's labels are the distant set's own-document labels, themselves noisy and incomplete)
| Seed size | Seed only | Seed + label-free (weight 3 / 15) | + context (weight 15) | + context + dataset types (reference, weight 3) |
|---|---|---|---|---|
| 0 | - | 0.632 | 0.627 | 0.602 |
| 500 | 0.630 | 0.634 | 0.633 | 0.612 |
| 2,000 | **0.662** | 0.649 / 0.657 | 0.663 | 0.640 |
| 5,000 | **0.680** | 0.657 / 0.666 | **0.683** (phrase recall 0.141) | 0.670 |
| 20,000 | **0.708** | 0.685 / 0.693 | **0.717** (top 10% 0.412, phrase recall 0.190) | 0.735 |
- 2,000 labelled records already beat every label-free reader tried (0.662 vs 0.632-0.665); 5,000 give 0.68; 20,000 give 0.71-0.72 (0.735 with types).
- Mixing in the label-free teaching does not help once the seed is 2,000 or more (it dilutes clean labels with noise), except through context features, which the seed can now use (5,000: 0.683; 20,000: 0.717 with phrase recall 0.190).
- These seeds use the distant set's own labels, which are noisy and incomplete; a clean, human-checked seed (the benchmark's gold) should do better per label. The gain per label is steady, not dramatic: about +0.03 per 4x labels.

### cg27a_kra13 local safety run (20 Nov, 23:10-00:14; five-domain harness, oracle writes, MLP 800 steps, parse off): PASSES; WIDER RECORDS AND THE READER'S GRAPH LIFT THE COMPRESSED FACT TABLE
kra13 = kra12 + wider records (FAR / NEXT / PRONOUN), the compressed fact table built from the reader's top half (`inf_rcode_reader 0.5`), the reader taught by confirmed inferences, ReVerb phrases and multi-instance selection, and the after-gate made-up readout; the states reader off.

| Readout (visit 4 unless stated) | kra11 local | kra13 local |
|---|---|---|
| HEADLINE, weights only: forgetting / final BPC | (kra8l) -0.039 / 2.622 | **-0.041 / 2.615** |
| Store records (4,000 distant documents) | 83,777 | 256,955 |
| Wording reader AUC, benchmark / held-out distant | 0.616 / 0.619 | **0.644** / 0.617 |
| Compressed fact table, pairs never stated together: CODE AUC, top 5% | 0.624, 0.304 | **0.706, 0.368** |
| Adamic-Adar on the same graph: AUC, top 5% | 0.597, 0.239 | 0.691, 0.337 |
| Confirm and consolidate: confirmed (right) vs still tentative (right) | - | 79 (0.49) vs 521 (0.23) |
| Through the exact-name gate: novel subjects answered / made up | - | 0.00 / 0.00 (held back answered from the memory 1.00) |

- Safe: forgetting and final BPC within noise of kra8l. No "unavailable" lines; ReVerb (spaCy) ran; READER TEACHING added 37 / 133 ReVerb phrase records at visits 2 / 4 with 2 multi-instance rounds (0 confirmed-inference records yet: the first confirmations come after that visit's reader is trained).
- The fact table gains most: inferred pairs never stated together are ranked at AUC 0.706 (from 0.624), top 5% precision 0.368 (from 0.304). Most of it is the three times larger store (wider records); the reader's top-half graph is the rest.
- Confirmation still separates right from wrong inferences (0.49 vs 0.23), as in kra12.
- Premise change and route J composites stay small at local scale (n 28-53 moves, 4 writes); these need the GPU run.
- Recall vs consolidation: the fact table and the gate are recall claims; confirmed inferences are what may be consolidated.

### Overnight 20 Nov (1): the synthetic world on a THIRD seed (seed 2; tools/compose_synth.py, same settings as seeds 0 and 1)
| | seed 0 | seed 1 | **seed 2** |
|---|---|---|---|
| Exact-name gate: made-up chains for NOVEL / NEAR-MISS names (trunk alone) | 0.00 / 0.00 (0.92 / 0.95) | 0.00 / 0.00 (0.97 / 0.92) | **0.00 / 0.00 (0.90 / 0.93)** |
| Exact-name gate: recorded held-out answered right (answers given) | 0.85 (1.00) | 0.87 (1.00) | **0.81 (1.00)** |
| Soft gate: made-up for novel / near-miss | 0.10 / 0.64 | 0.46 / 0.71 | 0.14 / 0.71 |
| Premise moved: stale composites, taught / never taught | 0.65 / 0.52 | 0.28 / 0.19 | 0.40 / 0.28 |
| Retract + re-derive, MOVED people: flagged; right country as answered -> after | 0.76-0.82; 0.14-0.19 -> 0.93 | 0.37-0.39; 0.50-0.53 -> 0.82-0.85 | **0.38-0.44; 0.15-0.20 -> 0.57-0.58** |
| Retract + re-derive, UNMOVED controls: as answered -> after | 0.92 / 0.66 -> 0.83 / 0.81 | 0.79 / 0.60 -> 0.74 / 0.73 | 0.53 / 0.36 -> 0.39 / 0.32 |
| Write-back of all re-derived chains (600 steps), asked directly: MOVED taught / never taught | 0.90 / 0.83 (397 chains) | 0.76 / 0.82 (375) | **0.57 / 0.62** (230) |
| Write-back: UNMOVED not replayed; replayed | 0.82 / 0.82; 0.99 | 0.79 / 0.76; 0.97 | 0.74 / 0.61; 0.99 |
| Forgetting (bits/byte on fixed pretraining windows) | - | - | 0.474 -> 0.493 |

- **The exact-name gate replicates exactly on all three seeds**: no made-up chains for names never recorded, and every recorded name answered. The soft gate still fails on near-miss names (0.64-0.71) on all three.
- **Retract and re-derive replicates in direction** (moved people 0.15-0.20 -> 0.57-0.58, about three times better) **but not in size**. The seed-2 trunk is a weaker chainer (unmoved people answered right only 0.53 / 0.36, against 0.79-0.92 on the other seeds), so re-derivation through it is weaker, and the re-derive step costs unmoved controls 0.04-0.14 on this seed (it did on seed 0 too for taught people).
- Write-back carries what re-derivation gets right into the weights (moved 0.57 / 0.62 asked directly, no check), with the replayed facts held at 0.99. Write-back quality is bounded by re-derivation quality.
- Claim to make: the gate is robust. Retract/re-derive/write-back works on every seed, with a size that depends on how well the trunk chains.

### Overnight 20 Nov (2): the SEEDED READER with the literature fixes, and seeded readers end to end (tools/seed_reader.py SD_MI / SD_REVERB; tools/e2e_reader.py E2_SEED)
Held-out documents of distant_50k (first 30,000 documents unless stated); AUC / top 10% precision:

| Seed labels | seed only (words) | **seed only + context** | seed + label-free + context | ... with multi-instance | ... with ReVerb | ... with both |
|---|---|---|---|---|---|---|
| 2,000 | 0.661 / 0.311 | **0.667 / 0.371** | 0.651 / 0.285 | 0.649 / 0.283 | 0.652 / 0.284 | 0.648 / 0.276 |
| 5,000 | 0.679 / 0.350 | **0.694 / 0.401** | 0.683 / - | 0.670 / 0.314 | 0.667 / 0.309 | 0.667 / 0.296 |
| 20,000 | 0.709 / 0.386 | **0.732 / 0.445** | 0.703 / 0.375 | 0.711 / 0.389 | 0.700 / 0.365 | 0.705 / 0.368 |

All 50,000 documents, seed only: 20,000 0.703 / 0.367; 50,000 0.722 / 0.401; 100,000 0.731 / 0.412 (with context 0.738 / 0.435; with the dataset's types, reference only, 0.776 / 0.463).

- **Once there is a seed, the label-free teaching only hurts, and the literature fixes do not rescue it.** Multi-instance selection keeps positives that are LESS often right (0.426 -> 0.363), and ReVerb phrase positives are right only 0.163 (base rate 0.131). Both are label-free signals as noisy as the teacher they were meant to clean.
- **The best reader is the seed alone with context words**: 0.667 / 0.694 / 0.732 at 2,000 / 5,000 / 20,000 labels, top 10% precision 0.37 / 0.40 / 0.45 (the label-free reader: 0.63-0.65, about 0.30). Gains flatten above about 50,000 labels (0.72-0.74); learned types would be the next step (+0.04 as an upper bound).

End to end (read -> the reader keeps its top share -> compose A-C through a shared name in another document -> checkable A-C pairs right by the distant labels), all 50,000 documents:

| Reader keeps its top 2% | premises right | A-C checkable | A-C right | with 2+ bridges right (n) |
|---|---|---|---|---|
| wording reader (label-free) | 0.26 | 1,472 | 0.241 | 0.469 (32) |
| wording + usefulness | 0.53 | 3,831 | 0.302 | 0.485 (167) |
| seed 2,000 only | 0.50 | 873 | 0.165 | 0.474 (19) |
| seed 5,000 only | 0.55 | 441 | 0.274 | 0.909 (11) |
| **seed 20,000 only** | **0.60** | 660 | **0.379** | 0.857 (14) |
| seed 20,000 + label-free | 0.53 | 1,708 | 0.287 | 0.607 (61) |

At the 10% cut every reader gives A-C right 0.18-0.21.
- A seeded reader more than doubles the precision of the premises it keeps (0.26 -> 0.60 at the 2% cut), but inferences right rise less (0.24 -> 0.38, small n). **Composition, not reading, now limits end-to-end inference**: a shared name in two documents is weak evidence that A and C are related. That is why confirmation before consolidation matters (kra12: confirmed 0.73 against 0.49).
- Next for the reader: a kra14 with a small labelled seed and context words (`inf_premises_seed`).

### cg27a_kra14 local safety run (20 Nov, 03:52-05:20; five-domain harness, oracle writes, MLP 800 steps, parse off; a container restart at 03:50 killed the first attempt in phase 4): PASSES; THE SEEDED READER DOUBLES THE PRECISION OF THE PREMISES IT KEEPS
kra14 = kra13 + the SEEDED READER (inf_premises_seed 20000 labelled records from documents already read, own-document labels; inf_premises_seed_ctx 1, 3 context words each side; label-free teaching, ReVerb and multi-instance off). Reading and composing only: the weights are untouched, so forgetting and BPC match kra13 exactly.

| Readout (visit 4 unless stated) | kra13 local | kra14 local |
|---|---|---|
| HEADLINE forgetting / final BPC (weights only) | -0.041 / 2.615 | -0.041 / 2.615 |
| with the fast store on | -0.039 / 2.617 | -0.039 / 2.617 |
| Wording reader AUC, benchmark (top 5%) | 0.644 (0.33) | **0.692 (0.58)** |
| Wording reader AUC, held-out distant (top 5%) | 0.617 (0.19) | **0.671 (0.39)** |
| INFERENCE A-C, reader top 10%: premises / A-C right (checkable) | 0.21 / 0.193 (9,346) | **0.33 / 0.223** (12,061) |
| INFERENCE A-C, reader top 2%: premises / A-C right (checkable) | 0.20 / 0.269 (695) | **0.57 / 0.486** (917) |
| Compressed fact table ALL: CODE AUC, top 5% | 0.706, 0.368 | 0.704, 0.382 |
| Compressed fact table (second slice): CODE AUC, top 5% | 0.714, 0.448 | 0.745, 0.455 |
| Confirm and consolidate: confirmed (right) / tentative (right) | 79 (0.49) / 521 (0.23) | 70 (0.49) / 530 (0.24) |

No Traceback, no "unavailable". Visit 2 (first attempt and rerun alike): seeded reader held-out AUC 0.671, top 2% premises 0.63, A-C right 0.454 (n 436).
RECALL vs CONSOLIDATION: these are reading and inference-proposal readouts (premises kept, A-C proposals checked against distant labels); the consolidation readout (confirmed right 0.49) did not move with the reader in this short run; why (where the tentative writes come from relative to the reader's ranking) is not yet checked.
Caveat: the seed is labelled data (20,000 records of own-document labels, themselves noisy), so kra14 is no longer label-free on the reading side; the trunk, store and consolidation remain label-free.

### Which A-C proposals to believe? (20 Nov, 05:25-05:30; tools/compose_rank.py, distant 50k, seeded reader 20k + context, held-out AUC 0.750): A CORRECTION TO THE END-TO-END A-C FIGURES
Proposals A-C through a shared name B from the reader's top share, ranked five ways. Two scorings:

| Ranking | all documents read: AUC, top 100 / 500 / 2000 (keep 10%, base 0.277) | NEW facts only (CR_HOLD 1): AUC, top 100 / 500 (keep 10%, base 0.040) |
|---|---|---|
| bridges (number of B) | 0.620, 0.95 / 0.66 / 0.50 | 0.657, 0.13 / 0.05 |
| best min premise | 0.615, 0.73 / 0.61 / 0.51 | 0.667, 0.15 / 0.05 |
| hub-weighted | 0.603, 0.96 / 0.64 / 0.39 | 0.593, 0.04 / 0.05 |
| sum x hub (rank sum) | 0.635, 0.85 / 0.63 / 0.47 | 0.658, 0.09 / 0.06 |

(Keep 25%: all-documents base 0.161, new-facts base 0.020, top 100 at most 0.15.)
NEW facts = composed only from the first 80% of documents, scored only on pairs NEVER named together there but named together in a held-out document, labelled by held-out labels - the way the package's compressed fact table is judged.
READING: the high A-C figures in the end-to-end readouts (e2e_reader, kra13/kra14 INFERENCE A-C: 0.27-0.49 at 2%) are mostly RE-DERIVING pairs that some read document states directly; they are not evidence of new facts. For genuinely new pairs, path composition is weak (top 100 right 0.13-0.15 against base 0.04), while the compressed fact table on its own held-out candidates reaches top 5% 0.38 at base 0.087 (kra14 visit 4). The two candidate sets differ, so this is not a head-to-head; but it says the fact table, not A-B-C chaining, is the route to new facts on real text, and the INFERENCE A-C readout must be read as recall of stated facts.

### Does the SEEDED reader feed the compressed fact table? (20 Nov, 05:35; tools/rcode_reader.py RC_SEED, distant 20k, 128 dims; 33,380 held-out candidate pairs never in the training graph, labelled 0.133): ONLY A LITTLE
| Graph built from | label-free reader: AUC, top 1% / 5% | seeded reader 20k + context: AUC, top 1% / 5% |
|---|---|---|
| ALL records | 0.551, 0.168 / 0.157 | (same) |
| reader top 50% | 0.557, 0.243 / 0.168 | 0.559, 0.261 / 0.161 |
| reader top 25% | 0.546, 0.222 / 0.144 | 0.567, 0.249 / 0.150 |
| reader top 10% | 0.550, 0.231 / 0.115 | 0.557, 0.189 / 0.116 |
| SOFT: all records, edge = reader probability | - | **0.573**, 0.171 / **0.179** |
A much better reader (held-out AUC 0.62 -> 0.75) moves the fact table by 0.01-0.02 AUC, matching kra14 in the run (CODE AUC 0.704 vs 0.706). The fact table's limit is not which records go in but the method and its COVERAGE: only 29% of candidates have both names in the graph. Next levers: coverage (names first seen in held-out documents) and the scoring of the code, not the reader.

### Goal 1, NEW FACTS: a SEEDED SELECTOR for the compressed fact table (20 Nov, 06:30-06:40; tools/rcode_select.py): POSITIVE, LARGE
Documents split by hash: TRAIN 80% (the name graph), VAL 10% (candidate pairs named together there, never in TRAIN, with labels: the selector's seed), TEST 10% (never in TRAIN or VAL; evaluation only). Write-time features: the code (plain and SOFT graph, edge = seeded reader probability), Adamic-Adar, resource allocation, common neighbours, degrees and their ratio, best two-step path. Scored among TEST candidates with both names in the graph.

| Ranking (TEST, among coded) | 20k documents (base 0.111): AUC, top 1% / 5% / 10% | 50k documents (base 0.101): AUC, top 1% / 5% / 10% |
|---|---|---|
| code (package) | 0.555, 0.204 / 0.174 / 0.169 | 0.540, 0.217 / 0.153 / 0.152 |
| code + RA (package rule) | 0.559, 0.280 / 0.213 / 0.182 | 0.543, 0.210 / 0.161 / 0.145 |
| soft code | 0.575, 0.204 / 0.191 / 0.198 | 0.562, 0.210 / 0.170 / 0.165 |
| best two-step path (A-B-C) | 0.518, 0.151 / 0.116 / 0.109 | 0.512, 0.120 / 0.084 / 0.088 |
| **SEEDED SELECTOR, logistic** | **0.698**, 0.441 / 0.344 / 0.268 | **0.701**, 0.388 / 0.305 / 0.245 |
| **SEEDED SELECTOR, trees** | **0.704, 0.613** / 0.400 / 0.311 | **0.721, 0.612** / 0.377 / 0.295 |

HOW MANY LABELS (20k documents, about k labelled coded VAL candidates): logistic k 300: AUC 0.672, top 1% 0.387; **k 1,000: 0.696, 0.452**; k 3,000: 0.698, 0.441 (trees need 3,000+: 0.680, 0.581). About 1,000 labelled candidate pairs are enough for the logistic selector.
- The selector more than doubles the precision of the best new facts (top 1%: 0.20-0.28 -> 0.44-0.61, base 0.10-0.11): the first real lever on GOAL 1 since the code itself. The logistic weights lean on Adamic-Adar (+) against common neighbours (-) and on name degrees: shared neighbours that are SPECIFIC, between names that are not hubs.
- COVERAGE is 0.28 (20k) -> 0.35 (50k): the rest have a name never read in TRAIN, which no method can infer from; it grows with reading, not with scoring.
- Caveat: the labels are distant labels (incomplete), the same seed assumption as the seeded reader; one split per size.
- Build candidate: inf_rcode_sel 'seeded' - the package's write step ranks candidates by a logistic selector trained on about 1,000 labelled candidate pairs from documents already read.

### Goal 2, CONSOLIDATION AT SCALE: REHEARSAL HOLDS what the trunk is taught (20 Nov, 06:40-07:15; tools/consolidate_scale.py, A6 stand-in trunk, 5 rounds x 200 of the package's typed real-text inferences (right 0.527), 600 steps a round, every weight, KL anchor to the original trunk on replay rows from all five domains, 24 frames, lr 1e-3; test wording never taught: rank among 10 (chance 0.10) / open-ended exact)
| After round | SEQ (each round alone): rounds 1..k rank | SEQ all taught, rank / open | REHEARSE (half the fact rows from earlier rounds): rounds 1..k rank | REHEARSE all taught, rank / open | never taught (SEQ / REHEARSE) | forgetting, mean of 5 domains (SEQ / REHEARSE) |
|---|---|---|---|---|---|---|
| 1 | 0.78 | 0.780 / 0.465 | 0.69 | 0.695 / 0.375 | 0.17 / 0.14 | -0.020 / -0.029 |
| 2 | 0.18, 0.90 | 0.542 / 0.372 | 0.83, 0.68 | 0.755 / 0.443 | 0.12 / 0.18 | -0.031 / -0.055 |
| 3 | 0.21, 0.30, 0.92 | 0.478 / 0.300 | 0.72, 0.66, 0.74 | 0.705 / 0.388 | 0.15 / 0.15 | -0.052 / -0.053 |
| 4 | 0.15, 0.20, 0.34, 0.98 | 0.420 / 0.259 | 0.71, 0.64, 0.65, 0.80 | 0.699 / 0.360 | 0.14 / 0.18 | -0.063 / -0.053 |
| 5 | 0.21, 0.18, 0.24, 0.30, 0.96 | **0.379 / 0.206** | 0.76, 0.58, 0.65, 0.67, 0.82 | **0.694 / 0.350** | 0.17 / 0.18 | -0.057 / **-0.083** |
- SEQ reproduces the GPU's decay (taught composites 1.00 -> 0.38 over visits 4-9): each round is learned (0.78-0.98) and lost after the next (0.15-0.34, near the never-taught 0.12-0.17). The loss is interference AMONG the inferences; general text is not harmed (mean forgetting negative; Wikipedia +0.12-0.17 in both arms).
- **REHEARSE holds it: after 1,000 inferences in 5 rounds, every round is still recalled 0.58-0.82 (all 0.694 rank, 0.350 open) vs 0.379 / 0.206 without rehearsal**, at no forgetting (-0.083). The cost: a round's own recall when fresh is lower (0.68-0.82 vs 0.90-0.98) - half the fact rows go to rehearsal.
- Never-taught inferences stay at chance in both arms (0.12-0.18): the trunk HOLDS what it is taught; it does not compose new ones (as before).
- Build candidate for the package: question-pool rows for consolidated inferences rehearsed across visits (half the conclusion rows from earlier visits' consolidated inferences), without the global question-row decay for them. One stand-in trunk, one seed.
- **REHEARSE_R (rehearsal + the KL anchor's teacher = the model at the start of each round, rolling): better recall, but FORGETTING ACCUMULATES - rejected.** All taught after rounds 1-5: 0.840 / 0.810 / 0.815 / 0.794 / **0.770** rank (open 0.432 at round 5), every round 0.69-0.87; but mean forgetting -0.018 -> -0.004 -> +0.013 -> +0.031 -> **+0.051** (Wikipedia +0.14 -> +0.37, PubMed +0.01 -> +0.14), and never-taught drifts up to 0.22. A rolling anchor lets drift compound round on round; the FIXED anchor to the original trunk (REHEARSE: 0.694, -0.083) is the one to build. The package's anchor (inf_distill_wake_akl on every old domain) is fixed in the same sense.
- RECHECK against kra14's actual write rule (code + Adamic-Adar rank sum among |log degree ratio| > 1), same candidates (20k / 50k documents): kra14 rule AUC 0.609 / 0.579, top 100 / 200 / 400 0.270 / 0.265 / 0.270 and 0.170 / 0.175 / 0.165; SELECTOR (trees) under the same filter 0.681 / 0.678, top 100 0.610 / 0.740, top 400 0.417 / 0.555; logistic under the filter top 100 0.490 / 0.470. The selector still wins. BUT this candidate set (record pairs, 128-dim codes from every record) is not the package's: in the package kra14's rule reports top 100 0.72 (labelled base 0.088, every name pair co-occurring in a held-out document, 512 dims, the graph from the reader's top half). The decisive comparison is on the package's own candidates (tools/rcode_select_pkg.py), queued.

### cg27a_kra14 GPU to visit 2 (20 Nov, started 06:46, log at 07:25; seed 4): THE SEEDED READER HOLDS AT SCALE; NO ERRORS
- SEEDED READER: 20,000 labelled records (2,062 related), context on, from 20,375 documents read (1.30 M records). **Wording reader AUC: benchmark 0.683 (top 5% 0.56), held-out distant 0.680 (top 5% 0.40)** - as local (0.692 / 0.671); kra11 GPU: state reader 0.592, word counts 0.634.
- INFERENCE A-C at visit 2 (reader's top share): 25% premises 0.21, A-C right 0.176 | 10% premises 0.32, 0.279 (2+ bridges 0.505, n 673) | **2% premises 0.55, A-C right 0.595 (n 822; 2+ bridges 0.944, n 18)**; random 0.106. kra11 GPU (combined reader, 2%): premises 0.54, A-C 0.349. NOTE (20 Nov correction): these A-C figures are mostly RE-DERIVED stated pairs, not new facts.
- COMPRESSED FACT TABLE (new facts, held-out candidates, labelled 0.093): CODE 512 AUC 0.689, top 5% 0.331 (with a bridge 0.721 / 0.403); SELECTION top 100 / 200 / 400: code 0.26 / 0.32 / 0.38, Adamic-Adar 0.77 / 0.64 / 0.53, **kra14's rule (code + AA among |log degree ratio| > 1) 0.73 / 0.76 / 0.755; WROTE 600, right 0.740**.
- Route J (learned rules on recorded facts) visit 2: first hop / composite 0.97 (linear operator), 33 composites written at precision 0.97.
- Prequential bits, domains read: 2.0125 (visit 1) -> 2.0371 (visit 2). No Traceback, no 'unavailable'. Forgetting and final BPC await the full run.
- For the selector decision: on the package's own candidates kra14's rule is right 0.73-0.76 in its top 100-400 - the bar a seeded selector must beat (tools/rcode_select_pkg.py, queued).
- **UNIFORM (the package's current RTI dose: fact rows drawn uniformly from every inference taught so far) HOLDS AS WELL: all taught after rounds 1-5 0.860 / 0.755 / 0.720 / 0.731 / 0.724 rank (open 0.441), every round 0.58-0.81, mean forgetting -0.075.** So the package's sampling already rehearses; the GPU fading is not a sampling problem. Suspected cause (code): the whole question loss, RTI dose rows included, is scaled by inf_distill_wake_decay 0.5 per visit (kra14: from visit 1, no floor), while the KL anchor is not - at visit 9 the inferences are pushed at about 0.004 of their first strength. Arms UNIFORM_D (the decay) and UNIFORM_DF (decay with a 0.25 floor) queued.

### Goal 1, DECISIVE: the seeded selector ON THE PACKAGE'S OWN CANDIDATES (20 Nov, 07:45; tools/rcode_select_pkg.py; premises._rcode_step as kra14 - 512 dims, graph from the seeded reader's top half, every name pair of a held-out document; 20k documents read; candidates split in half by pair hash, selector taught on half A, scored on half B, 37,909 candidates, labelled 0.089): NOT FOR A BUILD WITH A SMALL SEED
| Ranking (half B) | AUC | top 100 / 200 / 400 / 600 |
|---|---|---|
| **kra14 rule (code + AA among |log degree ratio| > 1)** | 0.631 | **0.750 / 0.675 / 0.615 / 0.542** |
| code | 0.609 | 0.470 / 0.460 / 0.370 / 0.357 |
| selector, 37,397 labelled candidates: logistic / trees | 0.728 / **0.793** | 0.640 / 0.645 / 0.542 / 0.502 and **0.830 / 0.780 / 0.720 / 0.682** |
| selector, 1,000 labelled candidates: logistic / trees | 0.718 / 0.686 | 0.620 / 0.545 / 0.472 / 0.438 and 0.460 / 0.475 / 0.465 / 0.460 |
- **CORRECTION to this morning's goal-1 result:** on the package's own candidates kra14's rule is far stronger than on tools/rcode_select.py's candidate set (top 100 0.75 vs 0.17-0.27 there), and **a selector taught on 1,000 labelled candidates does WORSE than kra14's rule in the top 100-600** (0.62-0.44 vs 0.75-0.54), though better on AUC (0.72 vs 0.63: it ranks the bulk better, not the head the package writes). The 'two to three times' claim held only against a weaker version of the rule.
- With about 37,000 labelled candidates, gradient-boosted trees beat kra14's rule by +0.08 to +0.14 in the top 100-600 (0.83 / 0.68 vs 0.75 / 0.54) - a real but modest gain that needs a large labelled set.
- DECISION: kra14's write rule stays; no selector in kra15. A middle seed (5,000-10,000 labelled candidates) is worth one more offline check.

### cg27a_kra14 GPU to visit 4 (log at 07:44): MEMORY AND GATES STRONG; TAUGHT FACTS CONSOLIDATE, CONCLUSIONS DO NOT
- Prequential bits, domains read: 2.0125 / 2.0371 / 2.0111 (visits 1-3). No Traceback, no 'unavailable'.
- Route J visit 3: first hop 0.97, **composite 0.94, pure families 0.93, null 0.03**; 64 composites written at precision 0.98; re-derive kept 64 (right 0.98).
- GATED ANSWERS visit 3: known facts right when answered **0.97**; never-read names given a stored name **0.06** (abstained 0.93).
- Keyed memory visit 3: taught facts 0.77 / 0.85 vs held back 0.02 / 0.21; persistent across readouts 0.69-0.78 open by visit written.
- FROM THE TRUNK at visit 4 (test wordings): **facts taught for at least one visit 0.88 (241; 2-3 visits 0.91) vs control 0.23** - facts consolidate. **Conclusions 0.00 (18), in the pool 0.03 (32); 2x2: taught 0.02 (62)** - conclusions still do not (the regression first seen in kgk/kcs).
- Probe recall from the weights: one-hop 0.36, open 0.28, two-hop 0.04 (chance 0.05).
- Real-text inferences: 600 written tentative at visit 2; none taught by visit 3 (confirmation at visit 4).
- Reading of goal 2: facts have source sentences replayed and enter the pool early; conclusions and composites have neither, and every question row is scaled by the visit decay (0.5 per visit, no floor) while the anchor is not - the decay arms queued on the stand-in test exactly this.
- **THE CAUSE OF THE FADING, CONFIRMED ON THE STAND-IN (UNIFORM_D: the package's sampling + its question-row decay, the fact loss x 0.5 per round, the anchor not decayed): all taught after rounds 1-5 0.800 / 0.720 / 0.497 / 0.432 / 0.316 rank (open 0.120), every round 0.22-0.49 at round 5 - vs UNIFORM without the decay 0.724 / 0.441.** The decay gives NO forgetting benefit here (mean -0.068 vs -0.075 without). A 0.25 floor (UNIFORM_DF) helps little: 0.421 / 0.197 (newest rounds learned poorly, 0.28-0.31).
- In the package the decay does protect forgetting for FACTS (B5 without it +0.007, B6 with it -0.032, local), so the fix is targeted: **cg27a_kra15 = kra14 + inf_rti_nodecay 1 - the confirmed real-text inferences' dose rows are exempt from the visit decay; facts keep it.** Local safety run next.
- MIDDLE SEEDS (same split): 5,000 labelled candidates - trees top 100 / 200 / 400 / 600 0.660 / 0.650 / 0.588 / 0.572 (AUC 0.751), logistic 0.620 / 0.635 / 0.560 / 0.487; **10,000 - trees 0.830 / 0.785 / 0.677 / 0.628 (AUC 0.772)**, with the asymmetry filter 0.850 / 0.765 / 0.662 / 0.598. kra14 rule 0.750 / 0.675 / 0.615 / 0.542. **The crossover is about 10,000 labelled candidate pairs: then trees beat kra14's rule by +0.08 to +0.11 across the 100-600 the package writes.** A candidate for a later build (labelled candidate pairs from documents already read, as the reader's seed); not kra15, which keeps one change.

### cg27a_kra15 local safety run (20 Nov)
- **HEADLINE identical to kra14l: forgetting -0.041 | final BPC 2.615 (fast store on -0.039 / 2.617).** No Traceback, no "unavailable".
- Every readout matches kra14l within probe noise (real-text recall 0.09-0.14 vs 0.11-0.12; the rest identical). Real-text inferences: 1,200 written, pool 41 taught / 29 held back at visit 4; from the trunk right 0.00 (15) vs 0.00 (19), rank 0.13 vs 0.16, the same as kra14l.
- **The change is NOT exercised locally**: the dose rows (861 rows of taught inferences) are ready only after visit 4's sleep, and no dose loss is reported at visit 5 (no "dose rows last visit" line), so the 5-visit local run ends before inf_rti_nodecay acts. The local run shows only that it does no harm and raises no error; whether the exemption stops the fading must be read on the GPU (visits 5-10: REAL-TEXT INFERENCE WRITES taught vs held back, and RANK RECALL with "dose rows last visit").
- Passed as a safety run; kra15 released to the GPU.

### Route to inference in the trunk: CHAIN-FORM TEACHING on the stand-in (20 Nov, 09:50-10:35; tools/chain_scale.py, A6 trunk)
1,911 of the package's typed code-written inferences have both premises on record (A-B, B-C via a bridge; right 0.444). 400 TAUGHT + 400 CONTROL; every premise read by the trunk (6 frames each) for both halves; 2,000 steps, 8 fact + 24 anchor rows, KL to the base trunk, lr 1e-3. Rank of the true tail among 10 (chance 0.10).

| arm (step 2000) | TAUGHT closed book | TAUGHT in context (premises in prompt) | CONTROL closed book | CONTROL in context | premises recalled taught / control | forgetting |
|---|---|---|---|---|---|---|
| base | 0.113 | 0.113 | 0.087 | 0.077 | 0.11 / 0.09 | - |
| PREMISES only | 0.135 | 0.125 | 0.077 | 0.072 | 0.34 / 0.33 | -0.053 |
| DIRECT 'h w t' | **0.375** (open 0.117) | 0.195 | 0.110 | 0.110 | 0.24 / 0.16 | -0.053 |
| CHAIN '<A-B>. <B-C>. So h w t' | 0.287 (open 0.030) | **0.475** (open 0.165) | 0.102 | **0.135** (open 0.033) | 0.67 / 0.21 | -0.030 |

- **No derivation from stored premises: CONTROL closed book stays at chance in every arm (0.08-0.11).** But the trunk holds few premises (control 0.21-0.33 recalled), so this does not yet separate "cannot combine" from "does not know the premises".
- CHAIN teaches the taught set's composition in context (0.475 vs DIRECT 0.195) and transfers weakly to never-taught chains given in the prompt: **0.135 (open 0.033) vs 0.110 (DIRECT) / 0.072 (PREMISES)** - about 2 standard errors above DIRECT on 400; a hint of a learned composition step, not yet a result.
- Follow-up running: PREMISES FIRST (3,000 premise-only steps, shared), then DIRECT vs CHAIN with 1,400 taught + 500 control, and control scored on the pairs whose BOTH premises the trunk recalls.
- **PREMISES FIRST follow-up (11:00-12:05; 6,460 inferences with both premises on record, 1,400 taught + 500 control, 3,000 premise-only steps shared, then 3,000 steps an arm):**

| step 3000 | TAUGHT closed book | TAUGHT in context | CONTROL closed book | **CONTROL in context** | premises recalled taught / control | forgetting |
|---|---|---|---|---|---|---|
| base | 0.094 | 0.097 | 0.094 | 0.092 | 0.10 / 0.11 | - |
| premises first (3,000) | 0.120 | 0.111 | 0.134 | 0.096 | 0.22 / 0.24 | -0.056 |
| DIRECT | 0.193 (open 0.038) | 0.117 | 0.142 | 0.104 (open 0.006) | 0.21 / 0.20 | -0.053 |
| CHAIN | 0.166 | **0.268** (open 0.084) | 0.114 | **0.158 (open 0.036)** | 0.42 / 0.25 | -0.043 |

  - **The in-context transfer replicates and grows: never-taught chains given in the prompt 0.158 with chain teaching vs 0.104 with direct teaching (500 each, about 3 standard errors; first run 0.135 vs 0.110).** The trunk learns a composition step it applies to chains it never saw - a first real-text sign of inference in the trunk, in context.
  - Closed book (premises stored, nothing in the prompt), no derivation: CONTROL 0.11-0.14 in every arm, and the premise-only arm already gives 0.134 (association through the shared bridge, not a learned step). The stand-in holds few premises (0.20-0.42), so pairs with BOTH premises recalled are too few (26-33) to judge.
  - Consequence: kra16 (inf_rti_chain) prepared; the small clean set (150 + 150, 4,000 premise steps) with DIRECT / CHAIN / BOTH running next.

### kra14 GPU to visit 8 (seed 4; log 74b7c91e, 20 Nov 11:40, visit 9 under way; no errors)
| visit | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| real-text inferences from the trunk, open: TAUGHT right vs HELD BACK right | 0.00 vs 0.00 | 0.00 vs 0.00 | **0.07 vs 0.02** | **0.10 vs 0.02** | **0.13 vs 0.03** |
| the same, rank among 10: TAUGHT right vs HELD BACK right | 0.15 vs 0.12 | 0.16 vs 0.10 | 0.26 vs 0.13 | 0.30 vs 0.20 | **0.45 vs 0.21** |
| taught WRONG inferences recalled open (false memories) vs held back | 0.00 vs 0.00 | 0.00 vs 0.00 | 0.08 vs 0.00 | 0.09 vs 0.00 | 0.08 vs 0.01 |
| facts taught >= 1 visit, from the trunk | 0.88 | 0.85 | 0.77 | 0.73 | **0.62** |
| conclusions taught >= 1 visit, from the trunk | 0.00 | 0.08 | 0.12 | 0.25 | **0.31** |
| route J composite (learned rules) | 0.93 (92) | 0.84 (106) | 0.87 (148) | 0.87 (180) | 0.87 (219) |
| fact table: right among the 600 written | 0.667 | - | 0.642 | - | 0.627 |
| confirmed so far (right) / faded (right) | - | - | 353 (0.75) / 172 (0.59) | - | 370 (0.72) / 205 (0.57) |
| gated answers: right when answered / never-read given a stored name | - | 1.00 / 0.00 | 0.88 / 0.00 | 1.00 / 0.01 | 0.86 / 0.02 |

- **GOAL 2 ON THE GPU: confirmed real-text inferences now CONSOLIDATE into the trunk - taught vs held back 0.13 vs 0.03 open, 0.45 vs 0.21 rank at visit 8, rising every visit since the dose began (visit 5).** The first real-text inferences the trunk recalls, under kra14's decay; kra15 (no decay on these rows) should lift them further.
- Costs: taught WRONG inferences are recalled too (0.08 vs 0.00-0.01): the confirmation filter passes right 0.72-0.75, so about a quarter of what is consolidated is wrong.
- **FACTS fade as the ledger grows: 0.88 -> 0.62 over visits 4-8 (ledger 659 -> 1,151)** while conclusions rise 0.00 -> 0.31 - the same shape as the stand-in's decay arm (UNIFORM_D). kra15 exempts only the inference dose rows; facts keep the decay. A candidate later change: a floor on the facts' decay or rehearsal of facts taught earlier (the stand-in: REHEARSE 0.694 vs SEQ 0.379), to test locally.
- The relational memory holds at 0.87 on 219 composites; the gate keeps never-read answers at 0.00-0.02.

### Tighter confirmation, offline (20 Nov 12:00; tools/confirm_min.py: the package's 2,400 code writes from 20,000 documents read, confirmed on the 30,000 read after, distant labels)
| rule (all 50k read) | confirmed (share of writes) | right | right writes kept |
|---|---|---|---|
| >= 1 relational record (kra14) | 1,056 (0.44) | 0.604 | 0.62 |
| >= 2 records | 648 (0.27) | 0.664 | 0.42 |
| >= 3 records | 439 (0.18) | 0.720 | 0.31 |
| >= 2 documents | 627 (0.26) | 0.670 | 0.41 |
| **1 record the seeded reader rates (top half)** | 865 (0.36) | **0.659** | **0.56** |
| 2 rated records | 467 (0.19) | 0.749 | 0.34 |
- The same order at 60% and 80% read. A rated record is the cheapest gain: +0.055 right for 10% of the right ones; counting records buys precision only by halving what is confirmed. None reaches 0.85 on distant labels (incomplete: "right" is a lower bound).

### cg27a_kra17 prepared (kra16 + two model changes for kra14's visit-8 problems)
- inf_distill_wake_decay_min 0.125 (a floor on the question rows' visit decay; GPU kcf: facts 0.72 vs 0.67 at visit 7, gated right overall 0.50 vs 0.31, bits within noise) - facts faded 0.88 -> 0.62 in kra14.
- inf_rti_conf_rated 1 (new option: a tentative inference is confirmed only by a later relational record the seeded reader rates in its top half; falls back to any record, with a log line, if the reader's scores do not match the records).
- Local safety runs: kra17 first (12:10), then kra16. The stand-in chain tests resume after them.

### cg27a_kra17 local safety run (20 Nov 15:40; tools/safety_run.sh, the kra15 settings: 150 steps, distant_10k)
| | kra15 (safety) | kra17 (safety) |
|---|---|---|
| HEADLINE weights only: forgetting / final BPC | -0.041 / 2.615 | -0.040 / 2.616 |
| with the fast store | -0.039 / 2.617 | -0.039 / 2.617 |
| confirmed at visit 4 (right) | 70 (0.49) | 54 (0.54) |
| new inferences keeping premises as text (visit 2 / 4) | - | 194 of 599 / 226 of 600 |
| chain dose rows (visit 4) | - | 160 for 420 inferences, beside 588 plain rows |
- PASS: forgetting and BPC within noise; no Traceback; the rated confirmation ran (no fallback line). It confirms fewer and more right, the direction tools/confirm_min.py found (small n here: one judging visit on 10k documents).
- The IN CONTEXT readout runs but n is 3-9 per cell at this size; it needs the GPU run. The chain dose rows are built at visit 4 and are first trained at visit 5, the last visit here, so their effect is not exercised locally.

### cg27a_kra17 GPU, visits 1-8 (20 Nov, log to the middle of visit 8; kra14 GPU beside it, same seed)
| readout (kra17 vs kra14) | visit 4 | visit 6 | visit 7 | visit 8 |
|---|---|---|---|---|
| facts ever taught, from the trunk | 0.83 vs 0.88 | 0.78 vs 0.77 | 0.72 vs 0.73 | **0.72 vs 0.62** |
| conclusions ever taught | 0.09 vs 0.00 | 0.36 vs 0.12 | 0.31 vs 0.25 | 0.34 vs 0.31 |
| gated answers: right overall / never-read given a stored name | 0.47 / 0.02 vs 0.55 / 0.03 | 0.41 / 0.02 vs 0.36 / 0.00 | 0.39 / 0.00 vs 0.42 / 0.01 | **0.50 / 0.02 vs 0.23 / 0.02** |
| confirmed this visit (right) | 326 (0.82) vs 365 (0.80) | 342 (0.77) vs 353 (0.75) | - | (not yet) |
| real-text inferences open, TAUGHT right / TAUGHT wrong / HELD BACK right | - | 0.06 / 0.01 / 0.01 vs 0.07 / 0.08 / 0.02 | **0.06 / 0.01 / 0.02 vs 0.10 / 0.09 / 0.02** | (not yet) |
| rank among 10, TAUGHT right / TAUGHT wrong / HELD BACK right | 0.16 / 0.09 / 0.11 vs 0.15 / 0.19 / 0.12 | 0.26 / 0.19 / 0.13 vs 0.26 / 0.17 / 0.13 | 0.32 / 0.12 / 0.18 vs 0.30 / 0.22 / 0.20 | (not yet) |
| IN CONTEXT (premises before the question), TAUGHT right / HELD BACK right | 0.21 / 0.20 | 0.29 / 0.17 | 0.32 / 0.20 | (not yet) |
| held-out bits per byte, domains read | - | 1.9222 vs 1.9078 | 1.8956 vs 1.8797 | (not yet) vs 1.8200 |
| route J learned composites: precision | - | - | 0.90 (176) vs 0.89 (170) | - |
- **FALSE MEMORIES DOWN: taught WRONG inferences recalled 0.01 vs 0.09 open (visit 7), 0.12 vs 0.22 rank;** taught right ones 0.06 vs 0.10 open, 0.32 vs 0.30 rank. Right over wrong goes from about 1x to 6x open.
- **The fact floor holds at visit 8: 0.72 vs 0.62 (n 250), flat over visits 7-8; below the 0.75 target.** Gated answers right overall 0.50 vs 0.23 at visit 8 (kra14 answered only 0.28 of KNOWN questions there).
- The rated confirmation moves less than offline: +0.02 right, about 5-10% fewer confirmed. It cannot explain the wrong-recall drop alone (101 taught wrong vs 119); the chain-form rows or the floor may. Not separable without kra16 on the GPU.
- IN CONTEXT shows no learned composition step: held back with the premises in front 0.17-0.20, the same as held back closed book (0.13-0.18). Taught in context = taught closed book. The chain form as taught does not yet make the trunk use premises it is shown.
- Cost: held-out bits +0.014 / +0.016 at visits 6 / 7 (seed noise on the harness about 0.017; consistent sign). Route J unchanged.

### cg27a_kra17 GPU, visits 8-10 (log to the middle of visit 10; kra14 stopped at visit 9)
| readout | visit 8 | visit 9 | visit 10 |
|---|---|---|---|
| facts ever taught (kra14 at visit 8: 0.62) | 0.72 | 0.63 | 0.56 |
| facts taught 4+ visits | 0.83 | 0.71 | 0.60 |
| conclusions ever taught | 0.34 | 0.18 | 0.21 |
| gated answers: right overall / never-read given a stored name (kra14 v8: 0.23 / 0.02) | 0.50 / 0.02 | 0.41 / 0.02 | 0.17 / 0.01 |
| name memory: facts / conclusions of 384 entries (fact candidates) | 228 / 156 (1,248) | 192 / 192 (1,333) | - |
| real-text inferences open, TAUGHT right / wrong vs HELD BACK right / wrong (kra14 v8: 0.13 / 0.08 vs 0.03 / 0.01) | 0.07 / 0.03 vs 0.03 / 0.03 | 0.06 / 0.04 vs 0.02 / 0.03 | (not yet) |
| rank among 10, TAUGHT right / wrong, HELD BACK right (kra14 v8: 0.45 / 0.19, 0.21) | 0.35 / 0.14, 0.21 | 0.35 / 0.13, 0.19 | (not yet) |
| IN CONTEXT, TAUGHT right / HELD BACK right | 0.37 / 0.27 | 0.31 / 0.24 | (not yet) |
| confirmed this visit (right) (kra14 v8: 370 (0.72)) | 354 (0.75) | - | - |
| held-out bits, domains read (kra14 v8: 1.8200) | 1.8372 | 1.7797 | - |
| route J precision (composites) | - | 0.89 (240) | - |
- **False memories stay at the held-back level: taught wrong 0.03-0.04 = held back wrong 0.03 (kra14: 0.08 vs 0.01).** The price: taught right 0.07 vs kra14 0.13 open, 0.35 vs 0.45 rank. Consolidation is weaker but no longer biased toward wrong inferences.
- **The floor delays the fact fade by about one visit; it does not stop it: 0.72 -> 0.63 -> 0.56, and facts taught 4+ visits 0.83 -> 0.60.** The pool grows (1,232 -> 1,287) on a fixed dose budget, so each fact's share keeps falling.
- **Gated answers fall 0.50 -> 0.17:** the name memory is capped at 384 entries and conclusions take a growing share (90 -> 192), leaving 192 facts for 1,333 candidates. A capacity limit, separate from the trunk's fade.
- IN CONTEXT, held back with premises 0.27 / 0.24 vs held back closed book 0.21 / 0.19 at visits 8 / 9: a +0.05 hint of using shown premises (SE about 0.035), not yet a result.
- Bits +0.017 vs kra14 at visit 8.

### cg27a_kra18 prepared (kra17 + two existing options for kra17's visit 8-10 limits)
- inf_nm_max 1024 (was 384): conclusions take up to half the name memory; by visit 9 they held 192 of 384 and facts 192 of 1,333 candidates; gated right overall 0.50 -> 0.17.
- inf_distill_wake_rows 8 (was 4): the question pool grows on a fixed budget; facts taught 4+ visits 0.83 -> 0.60 over visits 8-10 with the 0.125 floor.
- Local safety run after kra16's (about 17:25-18:45). The stand-in fact-fading test (uniform / floor / rehearse the fading first) follows; if it favours targeted rehearsal over more rows, that goes into kra19.

### cg27a_kra17 GPU, END OF RUN (visit 10 and the headline; run complete, then stopped by the user)
- **HEADLINE weights only: FORGETTING +0.005 | FINAL BPC 1.793** (fast store on +0.006 / 1.793). The lowest forgetting of the seed-4 GPU runs recorded here (cg27a_replay +0.026 / 1.767, cg27a_rtxw +0.037 / 1.779, cg27a_rtxt +0.077 / 1.807); final BPC 0.026 above cg27a_replay. Per domain: Gutenberg -0.055, Github -0.033, Wikipedia -0.022 (better than after their own visit); USPTO +0.067, StackExchange +0.053, DM Mathematics +0.028 the largest.
- End of run probe recall from the weights: one-hop 0.78, open one-hop 0.50 (n 134).
| visit 10 | kra17 |
|---|---|
| real-text inferences open: TAUGHT right / wrong vs HELD BACK right / wrong | 0.04 / 0.04 vs 0.01 / 0.02 |
| rank among 10: TAUGHT right / wrong vs HELD BACK right / wrong | 0.35 / 0.17 vs 0.19 / 0.18 |
| IN CONTEXT: TAUGHT right vs HELD BACK right | 0.31 vs 0.28 |
| confirmed this visit (right); all so far | 322 (0.71); 1,344 (0.76) |
| name memory | 192 facts / 192 conclusions of 384 (1,361 candidates) |
| route J precision | 0.89 (264 composites) |
| held-out bits, domains read | 1.7781 |
- Rank: taught wrong 0.17 = held back wrong 0.18 - no false-memory excess at the end; taught right 0.35 vs 0.19 held back.
- IN CONTEXT minus closed book, HELD BACK right: +0.06 / +0.05 / +0.09 at visits 8 / 9 / 10 (n 150 each; about 2 SE pooled). The first sign the trunk uses premises put in front of it for inferences it was never taught; the cells are not the same inferences (in context needs both premises on record), so this needs a matched readout before it is a claim.
- Open recall of taught right inferences fades with age (written at visit 2: 0.10 -> 0.08 over visits 8-10; visit 6: 0.01-0.02): the same pool-growth fade as the facts.
- kra18 adds a READOUT (no model change): [REAL-TEXT INFERENCE IN CONTEXT, MATCHED] - the same inferences and the same distractors scored closed book, after their OWN premises and after ANOTHER inference's premises, with the paired difference and its SE.

### cg27a_kra16 local safety run (20 Nov 17:20; tools/safety_run.sh)
- PASS: HEADLINE -0.040 / 2.616 (fast store -0.039 / 2.617) vs kra15 -0.041 / 2.615; no Traceback. Chain dose rows at visit 4: 277 for 420 inferences, beside 861 plain rows. (Superseded on the GPU by kra17, which includes it.)

### cg27a_kra18 GPU, visits 1-4 (started 17:05, before the matched readout was pushed at 17:14: this run does not print it)
| kra18 vs kra17 | visit 2 | visit 3 | visit 4 |
|---|---|---|---|
| held-out bits, domains read | 2.0261 vs 2.0359 | 2.0052 vs 2.0180 | 1.9223 vs 1.9343 |
| name memory entries (facts) | 305 vs 300 | 495 (469) vs 384 (357) | 634 (594) vs 384 |
| taught facts right after the visit | - | - | 0.97 vs 0.89 |
| facts ever taught | - | - | 0.85 vs 0.83 |
| gated answers right overall / never-read given a stored name | - | 0.41 / 0.01 vs 0.41 / 0.07 | 0.42 / 0.02 vs 0.47 / 0.02 |
- Eight question rows teach harder (0.97 vs 0.89 just after the visit) at no cost in bits so far (about -0.012 each visit). The test is visits 8-10.
- READOUT FIX (20 Nov 18:45): the rank readouts score candidates with distill._dw_cand_mc, which cuts text at the model's window (seq_len). A prompt that does not fit leaves every candidate unscored, they tie, and the first - always the true answer - wins. Locally (seq_len 64) every IN CONTEXT prompt overflows: the local IN CONTEXT 1.00 values (kra17 and kra18 safety runs) and the matched readout's local 1.00 / 1.00 are this artifact. On the GPU (seq_len 1024) the premises fit and the GPU readouts stand. Rank, IN CONTEXT and MATCHED now skip any prompt that does not fit.

### cg27a_kra18 local safety run (20 Nov 19:10; tools/safety_run.sh)
- PASS: HEADLINE -0.041 / 2.612 (fast store -0.040 / 2.614) vs kra17 -0.040 / 2.616; no Traceback. 8 question rows per step; facts ever taught 0.30 vs 0.29 at visit 4. The name memory (70-110 entries locally) never reaches either cap, so that change is GPU-only. Stopped after the headline (end-of-run diagnostics) to free the machine.

### cg27a_kra18 GPU, visits 5-8 (kra17 beside it, same seed)
| kra18 vs kra17 | visit 5 | visit 6 | visit 7 | visit 8 |
|---|---|---|---|---|
| facts ever taught | 0.86 vs 0.85 | 0.81 vs 0.78 | 0.80 vs 0.72 | **0.75 vs 0.72** |
| gated answers right overall | 0.50 vs 0.44 | 0.30 vs 0.41 | 0.31 vs 0.39 | 0.48 vs 0.50 |
| name memory: facts / conclusions | 755 / 55 | 921 / 86 | 921 / 103 (cap 1,024 reached) vs 280 / 104 | - |
| held-out bits | 1.8560 vs 1.8638 | 1.9101 vs 1.9222 | 1.8852 vs 1.8956 | - |
| inferences open: TAUGHT right / wrong vs HELD BACK right / wrong | - | 0.03 / 0.02 vs 0.00 / 0.00 | 0.04 / 0.01 vs 0.01 / 0.00 | 0.07 / 0.04 vs 0.03 / 0.05 |
| rank: TAUGHT right vs HELD BACK right | - | 0.20 vs 0.15 | 0.21 vs 0.15 | 0.33 vs 0.22 |
| IN CONTEXT HELD BACK right (closed book) | - | 0.16 (0.15) | 0.24 (0.15) | 0.31 (0.22) |
- Facts hold better from visit 6 (+0.03 to +0.08) and reach 0.75 at visit 8; bits stay about 0.01 better. The bigger name memory holds 3x the facts but gated answers do not rise (0.30-0.48): the gate's coverage, not the memory's size, now limits them.
- False memories stay at the held-back level (taught wrong 0.04 vs held back 0.05). IN CONTEXT minus closed book on held-back right: +0.01 / +0.09 / +0.09 at visits 6-8, as in kra17.

### Fact fading on the stand-in (tools/fact_fade.py; A6 trunk, 150 new facts a visit to 1,200, 250 steps a visit) - INCONCLUSIVE so far
- FLOOR / DOUBLE / TARGET all fall to near chance by visit 4 (all facts 0.13-0.15; the newest cohort 0.07-0.15): at the floored weight (0.25 x 0.125) 250 steps teach the stand-in almost nothing, whereas the GPU trunk learns its newest facts at 0.97. The regime does not match the GPU; NODECAY (pending) shows whether full weight teaches at all. Not a basis for kra19.

### cg27a_kra19 / kra19b prepared (20 Nov 20:35)
- kra19 = kra18 + inf_rti_rows 32 (was 16): twice the dose of confirmed real-text inferences (they reach the trunk as recognition and fade with age). Includes the matched in-context readout and the window fix.
- kra19b = kra19 + inf_distill_wake_decay_min 0.25: the fallback if kra18's facts fade at visits 9-10.
- Evidence replay (inf_rti_evidence) is NOT used: it replays w['ev'], which only the older rti_write path fills (the code-written inferences have none: it would replay nothing), and it was negative on the stand-in (14 Nov).
- The stand-in fact-fading test is set aside: even NODECAY fades (0.45 -> 0.19 by visit 6); the A6 trunk cannot hold a growing pool at any schedule, so it says nothing about the GPU trunk.

### Pretrained-base reader test (20 Nov 20:30, Colab GPU; tools/pretrained_reader.py, 20,000 documents, 30,000 held-out pairs, base rate 0.133, 0.86 seen once)
| reader | label-free (usefulness) AUC all / seen once | TRUE labels (ceiling) AUC all / seen once | ceiling top 5% precision all / seen once |
|---|---|---|---|
| word counts (ours) | **0.634 / 0.622** | 0.719 / 0.718 | 0.51 / 0.49 |
| GPT-2 124M, mid / last | 0.584 / 0.596, 0.572 / 0.576 | 0.781 / 0.771, **0.784 / 0.772** | 0.57 / 0.51, 0.60 / 0.50 |
| SmolLM2-360M, mid (last: NaN states, failed) | 0.587 / 0.594 | 0.800 / 0.790 | 0.60 / 0.51 |
| Qwen2.5-0.5B, mid / last | 0.588 / 0.596, 0.577 / 0.576 | **0.806 / 0.797**, 0.788 / 0.776 | 0.62 / 0.55, 0.60 / 0.53 |
| Qwen2.5-1.5B, mid / last | 0.587 / 0.594, 0.580 / 0.584 | 0.798 / 0.789, 0.778 / 0.765 | 0.58 / 0.49, 0.57 / 0.49 |
- **Label-free, pretrained states LOSE to word counts (0.58-0.60 vs 0.62-0.63):** the usefulness pseudo-labels (related 0.42 vs 0.11) are too noisy for a rich representation; the teacher, not the representation, limits the label-free reader.
- **With true labels, pretrained states WIN by +0.06 to +0.09 AUC, on pairs seen once too (0.77-0.80 vs 0.72).** Size barely matters (GPT-2 124M 0.77 seen once, Qwen 1.5B 0.77-0.79), so the gain is mostly READING (relation cues), not KNOWING (a 12x larger model knows far more and gains nothing).
- Implication: the package's seeded reader (20,000 labelled records, words) sits at the word ceiling (about 0.72); the same labels on a frozen small pretrained model's states would give about 0.78-0.80. A PRETRAINED READER (a component, the trunk unchanged) captures the measured reading gain; swapping the trunk is not needed for it.

### Pretrained reader IN THE PIPELINE (20 Nov 20:45, Colab; tools/confirm_pre.py, 50,000 documents, 20,000 labelled seed records, 2,400 code writes): GPT-2 - NEGATIVE
| | WORDS (the package's seeded reader) | STATES GPT-2 (layer 6) |
|---|---|---|
| held-out AUC (30,000 later records) | 0.757 | 0.770 |
| written right (2,400) | 0.426 | 0.438 |
| confirmed right / right kept, all read: >= 1 record | 0.604 / 0.62 | 0.594 / 0.60 |
| 1 rated record (kra17) | **0.659 / 0.56** | 0.648 / 0.56 |
| 2 rated records | **0.749 / 0.34** | 0.701 / 0.35 |
- The package's word reader uses 3 CONTEXT words each side as well as the words between: AUC 0.757, not the 0.718 of words-between alone in tools/pretrained_reader.py. Against it GPT-2's states add +0.013 AUC, +0.012 written right, and confirm WORSE (-0.011 / -0.048). The +0.05-0.08 edge estimated at 20:34 compared against the weaker word reader; it does not exist against the package's own.
- **Qwen2.5-0.5B (layer 12) in the same test:** held-out AUC 0.793 vs 0.757; written right **0.455 vs 0.426** (1,091 vs 1,023 right of 2,400); confirmed right, all read: >= 1 record 0.611 vs 0.604, **1 rated record 0.661 vs 0.659 (right kept 0.58 vs 0.56; about 629 vs 569 right ones confirmed)**, 2 rated 0.721 vs 0.749. A larger pool of right inferences (+7% written, +10% confirmed) at the SAME precision; confirmation precision does not rise. Below the bar set for kra20 (+0.03 confirmed right); not built. The precision of what is confirmed is limited by the confirmation rule and the evidence on later documents, not by the reader.

### cg27a_kra19 local safety run (20 Nov 22:00; tools/safety_run.sh)
- PASS: HEADLINE -0.041 / 2.612 (fast store -0.040 / 2.614), the same as kra18's; no Traceback. The dose rows train only at visit 5 here, so the doubled dose is barely exercised locally. The matched readout prints nan (0) locally as intended: with a 64-byte window every in-context prompt is skipped (the window fix).
- kra19b GPU (started 21:06) to visit 3: bits 2.0340 / 2.0062 vs kra18 2.0261 / 2.0052; gated answers 0.41 = kra18; no errors. The matched readout prints (nan before visit 4, as expected).

### REVIEW (20 Nov 22:30): what the inference and retention readouts actually measure - several earlier readings were wrong
1. **Almost no confirmed inference reaches the trunk.** inf_kmc_useful 1 (cg27a_kus, 16 Nov) is still on under the kra12+ confirm pipeline: a row of a written inference trains the trunk only if it is judged-kept AND "useful" (the keyed memory saved bits on its later text). [USEFULNESS] in kra17: useful AND judged-kept 6 (visit 6), 14 (visit 8), 25 (visit 10) of 668 / 1,022 / 1,344 confirmed - about 2%. distill._kmc_pending skips every other row, in the pool's rows AND the dose rows (kmc2). Consequences: the dose fires only on steps whose sampled rows include an eligible one (255 -> 28 steps a visit as the rows grow), its loss falls to ~0 because it memorises a handful of inferences, and kra15 (no decay), kra16 (chain dose), kra19 (dose x2) changed the treatment of about 2% of the "taught" group. "Useful" is also the WORSE filter: useful right 0.56-0.62 vs not useful 0.69-0.73; "NOT HARMFUL AND judged-kept" (inf_kmc_useful 2) is right 0.775 on 636-787.
2. **"Taught vs held back" for real-text inferences is therefore not a teaching contrast.** Both groups are confirmed (both had a relational record read); ~98% of "taught" were never trained. The rank gap (0.33-0.35 vs 0.19-0.22) comes from another path - most likely the source/real-text replay of the pool's entries (to check). The "no false-memory excess" result holds only because almost nothing was taught; it is not yet evidence that taught wrong inferences stay out.
3. **GATED ANSWERS is the TRUNK alone** (memory off) with a confidence threshold calibrated to keep 75% of the previous sleep's right answers, on 64 new questions a visit: right overall is about 0.75 x the trunk's recall on those questions, SE about 0.06. The name memory never enters it, so kra18's larger memory could not move it; "the gate opens too rarely" was a misreading. It is a noisy view of fact recall, not a separate gap.
4. **IN CONTEXT is confounded by copying:** the premises 'A r1 B. B r2 C.' contain the answer C, and the chain rows teach 'the answer is the last name of the sentence before'. The matched readout's 'other premises' arm does not remove this (C is absent there). The control needed: the inference's own B-C premise with another inference's A-B premise (C present, the chain broken).
5. **The two fact readouts disagree:** EVER TAUGHT (ledger sample, test wording) 0.75-0.86 at visits 5-8; the INTERLEAVED probe (32 pool facts, test wording + paraphrase) 0.44-0.65 with never-taught control 0.25-0.29. The 0.75 target is met on one readout only, and the control shows about 0.25-0.29 comes from reading alone.
6. The dose weight algebra is as intended: the wake weight carries the floored decay, inf_rti_nodecay divides it back out, and rows / rows_ref scales the push (kra19: 4x the reference, about the real-text loss's own weight).

### cg27a_kra20 prepared (20 Nov 22:25; kra19b stopped by the user after the review)
- kra18 + inf_kmc_useful 2 (confirmed inferences train the trunk unless measured not useful: about 640-790 at right 0.775, against about 2% under 1) + inf_distill_wake_decay_min 0.25. Dose rows stay 16 (kra18).
- Readouts: [REAL-TEXT INFERENCE TRAINED] - how many 'taught' inferences pass the keep rule, rank recall TRAINED vs NOT TRAINED vs HELD BACK; the matched readout's BROKEN CHAIN arm (another inference's A-B premise + this one's B-C premise: the answer in context, the chain broken) with own - broken and its SE.
- What it tests for the first time: whether taught confirmed inferences consolidate (TRAINED vs HELD BACK) and whether taught WRONG ones become false memories.

### cg27a_kra20 GPU, visit 4 (the first readout with the keep rule fixed)
- **[REAL-TEXT INFERENCE TRAINED]: 219 of 219 taught inferences pass the keep rule (right 0.80)** - against 6-25 in kra17. They are first trained during visit 5 (the dose rows are built at this sleep): rank TRAINED right 0.21 / wrong 0.16 vs HELD BACK 0.15 / 0.19 is the pre-training baseline.
- **MATCHED, BROKEN CHAIN (before any chain training):** HELD BACK right (n 51): closed 0.20 | own premises **0.29** | other premises 0.22 | broken chain (the answer in context, A not linked) **0.22** | own - broken +0.08 (SE 0.04), own - closed +0.10 (SE 0.04). TAUGHT right (n 114): closed 0.22 | own 0.24 | other 0.17 | broken 0.18 | own - broken +0.05 (SE 0.03). Pooled own - broken about +0.06 (about 2.3 SE).
- Reading: the answer's presence in the context does NOT lift it (broken 0.18-0.22 = other 0.17-0.22 = closed 0.20-0.22); only the LINKED chain does. A first sign that the trunk uses the chain, not copying - one visit, n 165, before chain teaching.
- Facts ever taught 0.84 (kra18 0.85); gated 0.36 (0.42); bits 1.9243 (kra18 1.9223).

### cg27a_kra20 GPU, visit 5 (after ONE visit of real inference teaching) + local safety run
| visit 5 | TAUGHT (trained, 219) right / wrong | HELD BACK right / wrong |
|---|---|---|
| open recall from the trunk | **0.96 (176) / 0.95 (43)** | 0.05 (91) / 0.06 (16) |
| rank among 10 | 0.99 / 0.98-1.00 | **0.31-0.34 / 0.06** (visit 4: 0.15 / 0.19) |
| MATCHED: closed / own premises / other / broken chain | 1.00 / 1.00 / 0.96 / 1.00 | right (n 51): 0.35 / **0.41** / 0.25 / 0.29; own - broken **+0.12 (SE 0.06)** |
- **CONSOLIDATION WORKS: confirmed inferences taught for one visit are recalled open-ended at 0.96 (held back 0.05).** The ~2% keep rule, not the trunk, was the limit in kra12-kra19.
- **FALSE MEMORIES, as expected once wrong ones are taught: taught WRONG inferences recalled 0.95 (held back 0.06).** What the trunk learns is now exactly as right as what confirmation passes (0.80 of the 219; 59 wrong of 326 confirmed). Distant labels are incomplete, so 'wrong' is an upper bound on false memories. The next lever is precision before teaching (two rated records: 0.749 offline) and retraction afterwards (inf_rret is on; no inference flagged yet at visit 4).
- **Held-back RIGHT inferences rise without being taught (rank 0.15 -> 0.31-0.34) while held-back WRONG ones fall (0.19 -> 0.06):** teaching right inferences generalises to other right ones, not to wrong ones. With their own premises in context 0.41 vs broken chain 0.29.
- No cost: bits 1.8606 (kra18 1.8560); facts ever taught 0.85 (0.86); gated answers 0.52 (0.50); never-read 0.00.
- kra20 local safety run: HEADLINE -0.031 / 2.621 (fast store -0.029 / 2.623) vs kra18 -0.041 / 2.612: within noise; no Traceback; 28 of 28 taught inferences train locally at visit 4. PASS.

### kra20 GPU, visit 6 (part) + a second review finding: RETRACTION NEVER RUNS under the confirm pipeline
- Visit 6 so far: facts just taught 0.94 (control 0.29), ever taught 0.83 (kra18 0.81); gated answers 0.44 (kra18 0.30), never-read 0.01. [RETRACTION] 0 retracted, 0 unlikelihood rows at every visit; FLAGGED, KEPT none.
- Code: with inf_rti_tentative on, premises' judge returns after the confirm/fade block, before the retraction trigger (the docs / rel / typed-confirmation score and W[k]['ret']). A confirmed inference is never re-judged: once taught it stays, right or wrong. Faded ones are flagged but were never taught. So the unlikelihood rows (inf_rret) have nothing to act on, and kra20's taught wrong inferences (recalled 0.95) have no path out.
- For kra21: a retraction path for CONFIRMED inferences (e.g. the existing typed-confirmation score _cc9: head-first records since the write naming another tail are contradictions; contradictions > confirmations -> flag and retract with the unlikelihood rows), alongside a stricter confirmation.

### Confirm, teach, retract - offline (21 Nov 00:40; tools/retract_min.py, distant_50k, 2,400 code writes from 40% of the documents, confirmation on the next 20%, retraction on the rest)
| confirmation rule | confirmed | right | right writes kept |
|---|---|---|---|
| any relational record (kra14) | 602 | 0.654 | 0.39 |
| 1 record rated top half (kra17-kra20) | 473 | 0.708 | 0.33 |
| **1 record rated top quarter** | 307 | **0.766** | 0.23 |
| 1 record rated top tenth | 139 | 0.827 | 0.11 |
| 2 records rated top half | 166 | 0.771 | 0.13 |
| RETRACTION of confirmed inferences (on top of top-half / top-quarter confirmation) | wrong removed | right lost | kept right |
| NO relational record of the pair in the later 40% | 0.38 / 0.35 | 0.19 / 0.19 | 0.759 / 0.803 |
| ... in the later 30% | 0.42 / 0.39 | 0.26 / 0.25 | 0.756 / 0.800 |
| **... in the later 20% (about 2 visits)** | 0.55 / **0.53** | 0.36 / **0.34** | 0.773 / **0.821** |
| contradiction, own typed wording (1, > 2x confirmations) | 0.19 / 0.19 | 0.09 / 0.09 | 0.728 / 0.786 |
| contradiction, any wording | 0.67-0.95 | 0.55-0.87 | near random |
- RE-CONFIRMATION (the pair recurs relationally after confirmation) is the useful retraction signal; contradiction by another tail is near random in any wording and small in the typed wording. Top-quarter confirmation + re-confirmation within about 2 visits: kept right 0.708 -> 0.821 offline, at about half the right inferences.

### cg27a_kra21 prepared (21 Nov 00:50)
- kra20 + inf_rti_conf_q 0.75 + inf_rti_reconf 2 (new option: [RE-CONFIRMATION] retracts confirmed inferences with no relational record of the pair since confirmation, 2 visits on; retracted ones leave the pool and get unlikelihood rows, inf_rret 1). Unit test: the unrecorded pair retracted, the recorded one kept, the top-quarter record confirms.
- Expected on the GPU: confirmed right about 0.80-0.82 -> about 0.88-0.90 by visit 8; taught WRONG recalled falls after retraction at visits 6 / 8 / 10 ([RETRACTION RECALL]); about half as many right inferences taught.
- kra21 adds a READOUT (21 Nov 01:00): [REAL-TEXT INFERENCE PRIOR CONTROL] - rank among 10 with the question's own subject vs another inference's subject, same candidates, for trained and held-back right / wrong. kra20's held-back RIGHT rise (0.15 -> 0.33) could be a preference for answer names that taught inferences made familiar (states, countries) rather than knowledge of A: if it survives the swap, it is a name prior. Guarded by its own try (a readout failure must not skip the visit's writes).

### The two fact readouts reconciled (21 Nov 01:10; logs of kra17, kra18, kra20)
- Both score with distill._dt_recall (test wording + paraphrase, open-ended). They differ in WHICH facts: [INTERLEAVED] probes 32 random facts of the CURRENT pool (recent entries, one or two visits of teaching); [EVER TAUGHT] samples about 240 ever taught, dominated by long-taught ones. The probe therefore tracks the current teaching weight; the ledger tracks retention.
- Probe, visits 4-6 / 8: kra17 0.89 / 0.61 / 0.76 ... 0.47; kra18 0.97 / 0.61 / 0.59 ... 0.44; **kra20 0.84 / 0.88 / 0.94** (control 0.29-0.40). kra20's 0.25 floor teaches new facts 2x (visit 5) to 4x (visit 6) harder than kra18's 0.125, and the probe shows it; the ledger agrees in kra20 (0.83-0.85).
- Review point 5 was a composition difference, not a scoring conflict. The fact target should hold on BOTH: the ledger (retention) and the probe (newly taught facts) at 0.75 or more. kra20 meets both through visit 6.

### kra21 local safety run (21 Nov 02:25; tools/safety_run.sh kra21, 150 steps a domain, distant_10k)
- HEADLINE weights only: FORGETTING **-0.030** | FINAL BPC **2.622** (fast store on -0.028 / 2.624). kra20: -0.031 / 2.621. Within noise: PASS. No errors; the run was stopped at its time limit in the end-of-run diagnostics, after the HEADLINE.
- [RE-CONFIRMATION] runs at visit 4 (0 checked: locally nothing is confirmed 2 visits before; it first acts at visit 6 on the GPU). [REAL-TEXT INFERENCE TRAINED]: 22 of 22 taught pass the keep rule. [PRIOR CONTROL] prints at visits 4-5 (n 4-15 locally, too few to read). The MATCHED in-context readout is empty locally (every prompt exceeds the 64-byte window; it reads on the GPU's 1024).
- kra21 is cleared for the GPU.

### Write budget for kra21's rules, offline (21 Nov 02:30; tools/retract_min.py, RT_W 2400 / 4800 / 7200, distant_50k, same windows as above)
| written | right | top-quarter confirmed (right) | + retract if NO later record: kept (right, count right) | top-tenth confirmed (right, count right) |
|---|---|---|---|---|
| **2,400 (kra21)** | 0.426 | 307 (0.765) | **238 (0.803, 191)** | 139 (0.827, 115) |
| 4,800 | 0.310 | 441 (0.703) | 333 (0.745, 248) | 194 (0.830, 161) |
| 7,200 | 0.252 | 550 (0.662) | 413 (0.709, 293) | 221 (0.819, 181) |
- More writes dilute: each extra write is less likely right, and the confirmation rules pass them in proportion. Volume of right inferences kept rises 191 -> 293 (x1.5 for x3 writes) but precision falls 0.80 -> 0.71. Only top-tenth confirmation holds precision (0.82-0.83) as the budget grows, at 115-181 right.
- Do not raise the write budget in kra22. If volume is needed later, 7,200 writes + top-tenth confirmation (181 right at 0.82) matches kra21's 2,400 + top quarter + re-confirmation (191 at 0.80) without retraction - a fallback if kra21's retraction misbehaves on the GPU.

### Name-prior forecast for kra20's held-back rise (21 Nov 02:30; tools/tail_prior.py, distant_50k, kra20's confirmation rule)
- 473 confirmed (right 0.708): 386 taught, 87 held back (52 right, 35 wrong). A held-back RIGHT inference's answer is more often also a taught answer (0.88 vs 0.74; taught 6.8 vs 4.2 times on average; AUC 0.64). Store degree does not separate them (AUC 0.52).
- A PURE NAME PRIOR (the candidate taught most often wins) gives rank among 10: held-back right **0.29**, wrong **0.14**. kra20 GPU visit 5: right 0.31-0.34, wrong 0.06 (visit 4, before teaching: 0.15 / 0.19).
- So most of kra20's held-back right rise can be explained by answer-name familiarity from teaching, without the trunk knowing anything about the subject. **The held-back rise is NOT evidence of generalisation** until kra21's [PRIOR CONTROL] (own vs swapped subject, same candidates) shows own > swapped for held-back right. The matched in-context own-vs-broken gap (+0.12, SE 0.06) is a separate measure and stands.

### Stand-in chain test with the BROKEN-CHAIN control (21 Nov 03:50; tools/chain_scale.py, A6 stand-in trunk, 150 taught + 150 control two-hop chains, premises first 4,000 steps, then 3,000 steps an arm)
| arm (step 3000) | taught: closed book | taught: premises in context | taught: BROKEN chain | in context - broken | control (untaught chains): in context / broken |
|---|---|---|---|---|---|
| premises only (before the arms) | 0.160 | 0.187 | 0.180 | +0.007 | 0.147 / 0.147 |
| DIRECT (A r C taught) | 0.980 | 0.240 | 0.247 | -0.007 | 0.160 / 0.153 |
| CHAIN (P1. P2. So A r C taught) | 0.653 | 0.993 | **0.993** | **+0.000** | 0.213 / 0.200 |
| BOTH | 0.993 | 0.973 | 0.947 | +0.027 | 0.187 / 0.187 |
- Rank among 10. The premises are recalled at 0.94-0.99 throughout. Forgetting -0.04 to -0.07 on every arm (within the stand-in's range).
- **Training on worked chains teaches the taught answers, not a skill of reading premises.** CHAIN scores 0.993 with the right premises AND with a broken chain (one premise from another inference): the model answers from "So A r", i.e. it memorised A -> C. On untaught chains, in context equals broken (0.21 / 0.20). No arm learns to compose from premises in context at this scale.
- **No latent two-hop**: with both premises learned (0.96), untaught conclusions stay at chance closed book (0.14 vs 0.12 at the start). The two-hop curse holds for the stand-in trunk, as in the literature; Platypus composes outside the trunk (route J) and teaches A -> C.
- Against the GPU: kra20's matched readout (own +0.12 over broken, SE 0.06, held-back right, visit 5) is the only positive in-context sign so far; this test does not reproduce it at stand-in scale. Treat it as unconfirmed until kra21's visits 6-10 repeat it.

### cg27a_kra21 GPU, visits 4-5 (21 Nov 07:05; log to the end of visit 5, Phase 6 starting)
| visit 5 | TAUGHT (trained, 188) right / wrong | HELD BACK right / wrong |
|---|---|---|
| open recall from the trunk | 0.97 (154) / 0.97 (34) | 0.05 (81) / 0.00 (13) |
| rank among 10 | 0.99-1.00 / 1.00 | 0.21-0.28 / 0.15 (visit 4: 0.05-0.15 / 0.08) |
| **PRIOR CONTROL: own subject / SWAPPED subject (same candidates)** | 1.00 / 0.23 | **right 0.26 / 0.11, own - swapped +0.15 (SE 0.05)**; wrong 0.08 / 0.08, +0.00 |
| MATCHED: closed / own premises / other / broken chain | 1.00 / 1.00 / 0.93 / 1.00 | right (n 45): 0.31 / **0.44** / 0.18 / 0.27; own - broken **+0.18 (SE 0.07)**, own - closed +0.13 (SE 0.07) |
- **THE NAME-PRIOR CHECK COMES OUT AGAINST THE NAME PRIOR.** A pure name prior would score the swapped subject as high as the own subject. Held-back RIGHT inferences: own 0.26 vs swapped 0.11 (about chance), +0.15 at 3 SE. At visit 4, before any inference teaching, the same gap was -0.01 (SE 0.03). So one visit of teaching right inferences makes the trunk rank OTHER, never-taught right answers higher for their own subject. Held-back WRONG ones get no such lift (+0.00, n 13). This is the first measure that separates subject-specific transfer from answer familiarity; it replaces the 02:30 forecast's reading for kra21.
- Where the transfer comes from is not yet known (shared subjects or premises with taught inferences, or a learned composition step). It is recall (rank), not open-ended recall (0.05). Recall and consolidation claims stay separate: consolidation of TAUGHT inferences is 0.97; this row is generalisation evidence only.
- **MATCHED in-context repeats kra20**: held-back right own premises 0.44 vs broken chain 0.27, +0.18 (SE 0.07); kra20 +0.12 (SE 0.06). Two builds, same sign. The stand-in test (03:50) still does not reproduce it.
- False memories: taught WRONG recalled 0.97 (34), as expected before retraction. RE-CONFIRMATION at visit 4 checked 0 (acts from visit 6). CONFIRMED 282 (right 0.83), tentative 318 (0.66).
- No cost: prequential bits on read domains 1.8587 (kra20 1.8606); facts ever taught 0.87 (taught 1 visit 0.95, 2-3 visits 0.86, 4+ 0.87; control 0.37); gated answers smoothed 0.44 right when answered 1.00 (kra20 0.52); never-read given a name 0.00. Route D two-step 0.73 vs direct 0.05. Sleep 1070 s (> 600 s budget: optional readouts skipped next sleep). GPU peak 56.7 of 95 GB.

### cg27a_kra21 GPU, visit 6 (21 Nov 07:35): retraction fires but is NOT selective; the subject and in-context gaps hold
- **[RE-CONFIRMATION]: 282 confirmed inferences checked, 72 RETRACTED (right 0.82) vs 210 re-confirmed (right 0.84).** No separation: offline (tools/retract_min.py) the same rule removed 0.53 of the wrong and 0.34 of the right. On the GPU almost every pair recurs somewhere ("any later record of the pair" 779 of 918), so the absence of a relational record in 2 visits is close to random. 47 retracted taught ones (38 right, 9 wrong) get 987 unlikelihood rows in visit 7: about 4 right unlearned per wrong one. [RETRACTION RECALL] before those rows: retracted right 0.92 / wrong 1.00.
- **kra22 turns re-confirmation off** (inf_rti_reconf 0) and keeps top-quarter confirmation, which is already 0.82 right on the GPU (confirmed 509 at 0.82, faded 221 at 0.59, tentative 398 at 0.60). False memories stay at roughly 0.18 of taught inferences (an upper bound: distant labels are incomplete) until a selective signal is found.
- **PRIOR CONTROL holds:** held-back RIGHT own 0.23 vs swapped 0.13, **+0.09 (SE 0.03, n 150)**; held-back wrong +0.03 (SE 0.06, n 29). Trained right +0.20 (SE 0.04).
- **MATCHED holds a third time:** held-back right own premises 0.40 vs broken chain 0.28, **+0.12 (SE 0.04, n 86)**; own - closed +0.15 (SE 0.05).
- **Taught inferences written at visit 2 fall 0.97 -> 0.66 open (right) and 0.97 -> 0.63 (wrong)** in one visit; those written at visit 4 (just confirmed, not yet trained) 0.09. Candidates: dose rows shared by 352 taught (was 188), the Github visit (bits on read domains 1.8587 -> 1.9049). Rank among 10 for taught right 0.73. Watch visit 7.
- **Gated answers reach the goal with the old rule:** t fell to 0.94-0.96; known answered 0.58-0.61, right when answered 0.97, right overall **0.56-0.59**; never-read given a name 0.01-0.02.
- Facts ever taught 0.83 (kra20 0.83 at visit 6); taught 1 visit 0.72, 4+ visits 0.85; control 0.33. No Traceback.

### cg27a_kra21 GPU, visit 7 (21 Nov 08:10): unlearning works, the visit-6 dip recovers, the subject gap holds
- **[RETRACTION RECALL]: the 47 retracted taught inferences fall from 0.92 / 1.00 (right / wrong) to 0.03 / 0.00 after one visit of unlikelihood rows** (unlikelihood loss 4.46 -> 0.18); judged, not flagged 0.80 / 0.84. The unlearning step is strong and contained; the SELECTION was the failure (38 of the 47 were right). Watch visits 8-10 for the retracted ones coming back (benign relearning).
- **The visit-6 dip was transient:** taught inferences open recall right 0.43 -> **0.83**, wrong 0.36 -> 0.80; written at visit 2: 0.66 -> 0.75; written at visit 4 (first trained this visit) 0.95. Rank among 10: taught right 0.93. Visit 6 was Github; bits on read domains 1.9049 -> 1.8856.
- **PRIOR CONTROL:** held-back right own - swapped **+0.09 (SE 0.04, n 150)**; held-back wrong +0.07 (SE 0.05, n 29) - the wrong ones now rise too, so the right-vs-wrong asymmetry of visits 5-6 is not clear at visit 7.
- **MATCHED weakens:** held-back right own premises 0.44 vs broken chain 0.40, +0.05 (SE 0.03); own - closed +0.20 (SE 0.04). The answer in context alone now lifts it (broken 0.40 vs closed 0.24).
- **Gated answers swing back down:** 75% rule, t 0.96: right overall 0.36 (visit 6: 0.59); ungated 0.73; never-read 0.00. The calibrated rule is unstable from sleep to sleep - kra22's 0.9 rule (with the 0.75 printed beside it) stands.
- Facts: interleaved probe (newly taught) 0.89 / 0.91 / 0.87 / **0.79** at visits 4-7 (kra20 0.84 / 0.88 / 0.94 at 4-6); ledger ever taught 0.82, 4+ visits 0.83, control 0.37. No Traceback.

### Bridge test (21 Nov 08:40; tools/bridge_probe.py, A6 stand-in, 1,256 chains from distant_50k code writes, TRAIN 400 / TEST 200 on bridges never in a train chain)
- Premises first (5,000 steps): A-B and B-C recalled 0.84-0.85 (rank among 10); closed-book two-hop 0.125 (train) / 0.155 (test); forgetting -0.038.
- **PART 1 - the bridge B is NOT formed before it is spelled out.** Ridge probe from each layer's last-byte state to B as read (layer 1 at B's last byte), B among 10 (chance 0.10), random / same-first-byte distractors: after 'A w1 ' (hop 1 asked) 0.09-0.15 / 0.05-0.12; after the two-hop question 0.09-0.12 / 0.09-0.13; after 'A w1 B' (B spelled, reference) **0.75-0.84 / 0.69-0.87**; 'A' alone 0.08-0.10. The trunk recalls B byte by byte, but its state before B carries no identity of B.
- **PART 2 - a new top layer does not compose** (3,000 steps, closed book, TRAIN taught / TEST never taught): NONE (all weights) 0.890 / 0.190 | RANDOM new block + adapter 0.948 / 0.215 | COPY0 frozen copy of block 0 + adapter 0.258 / 0.185 | COPY1 0.610 / 0.180; forgetting -0.06 to -0.07. The copied layers cannot even learn the taught chains through an adapter (no 'B as predicted' to map); the random layer memorises taught chains best and lifts never-taught ones by +0.025 at most.
- Conclusion: depth (neurogenesis) does not give latent two-hop composition at this scale; B must exist as text first (spell the bridge: route J outside the trunk, or 'A w1 B, and B w2 C' in the trunk).

### cg27a_kra21 GPU, visit 8 (21 Nov 08:40)
- **RE-CONFIRMATION again not selective:** 509 checked, 98 retracted (right 0.81) vs 411 kept (0.82); 170 retracted so far. Confirmations this visit 322 at right 0.76; all 733 at 0.79 (precision slipping: 0.83 -> 0.82 -> 0.79).
- [RETRACTION RECALL]: retracted (118, mixing the 47 unlearned in visits 7-8 with 71 newly retracted) 0.55 / 0.57 right / wrong; judged, not flagged 0.48 / 0.30.
- **PRIOR CONTROL: the asymmetry is back** - held-back RIGHT +0.13 (SE 0.04, n 150), held-back WRONG +0.00 (SE 0.04, n 51). Trained right +0.45 vs trained WRONG +0.21 (own 0.66 vs 0.38): wrong taught inferences have half the subject-specific support of right ones - a per-inference signal worth testing (select_min part B).
- MATCHED held-back right own - broken +0.05 (SE 0.02), own - closed +0.12 (SE 0.03).
- **Taught open recall dips again:** right 0.48 / wrong 0.35 (rank 0.89 / 0.57); written at visit 4: 0.95 -> 0.62; visit 2: 0.75 -> 0.64. Dips at visits 6 and 8 (Github, USPTO), recovery at 7. kra22's [TAUGHT RECALL BREAKDOWN] will split it.
- **Gated answers:** the keep-75% rule at t 0.85: right overall 0.61, right when answered 0.93, but NEVER-READ given a stored name **0.11** (smoothed 0.56 / 0.09). The rule trades coverage for false answers from sleep to sleep. **kra22 changes the gate to calibrate on never-read questions** (inf_distill_gate_leak 0.02: the lowest threshold at which at most 2% of the previous sleep's never-read questions get a name), the keep-75% rule printed beside it; the keep-90% plan is dropped (it would let more never-read through).
- Facts: probe 0.81, ledger ever taught 0.82, 4+ visits 0.84, control 0.38. Bits on read domains **1.8292** (best of the run). No Traceback.

### Signals test: teach-gates on the confirmed set (21 Nov 08:55; tools/select_min.py part A, distant_50k, 2,400 code writes from 20k documents, top-quarter confirmation on the next 10k: 306 confirmed, right 0.768, 71 wrong)
Scored by RIGHT LOST PER WRONG REMOVED (random removal: 3.31).
| teach only if | taught (right) | wrong removed | right lost | right lost per wrong |
|---|---|---|---|---|
| all confirmed (kra21) | 306 (0.768) | - | - | - |
| **2+ documents confirm OR a second look (top-half record in the next 10k)** | 220 (**0.823**) | 0.45 | 0.23 | **1.69** |
| second look: any relational record again | 190 (0.821) | 0.52 | 0.34 | 2.14 |
| second look: top-quarter record again | 142 (0.852) | 0.70 | 0.49 | 2.28 |
| 2+ distinct documents confirm | 159 (0.824) | 0.61 | 0.44 | 2.42 |
| best record in the top tenth | 139 (0.827) | 0.66 | 0.51 | 2.55 |
| 2+ records rated top quarter | 81 (0.889) | 0.87 | 0.69 | 2.63 |
| VERBATIM: a confirming record in the inference's own wording | 57 (0.772) | 0.82 | 0.81 | 3.29 (random) |
| NOT contradicted later in own wording (another tail) | 161 (0.733) | 0.39 | 0.50 | **4.18 (worse than random)** |
- **Independent evidence helps; exact wording and contradiction do not.** A second document or a later second look removes wrong inferences about twice as efficiently as random (1.7-2.4 right lost per wrong vs 3.3). The best rule keeps 0.82 precision at 77% of the right inferences.
- Own-wording ('verbatim') confirmation is random here: the typed wording is the commonest wording of the type pair, rarely the record's own. Contradiction by another tail is WORSE than random: prominent (often right) subjects simply get more records naming other tails. The one-answer filter was degenerate in this run (heads with one record counted as one-answer); fixed, to re-run with part B.
- No rule loses fewer right than wrong ones: document signals alone cannot make retraction cheap.

### Hand audit of 'wrong' confirmed inferences (21 Nov 09:00; Claude's judgement from general knowledge, NOT gold; 60 labelled wrong, 30 labelled right, sampled from the 306 above)
- **Labelled WRONG (60): 26 true (43%), 9 ambiguous (15%), 25 false (42%).** True but unlabelled: 'AIR Studios in London', 'British Museum in London', 'Henry VIII of England', 'Memphis in Tennessee', 'Cleveland in Ohio', 'Times Square in New York City', 'Oxford University in England', 'Portland in Oregon', ... Distant labels miss them.
- **Of the 25 false, 13 have a malformed entity** (a number, a year or a fragment as A or C: '12 in US', 'Telugu in 2015', 'CDP in Washington', '1893 ) was Chicago'); the other 12 are real confusions ('South America in Mexico', 'Namibia in South Africa', 'Northern Ireland in England', 'Shenzhen in Hong Kong').
- Labelled RIGHT (30): 29 true, 1 questionable.
- **So the true false-memory rate is about half the labelled one:** GPU confirmed labelled wrong 0.18-0.21 -> truly false about 0.08-0.12 (0.42-0.57 of labelled wrong). Half of those are extraction errors that a name-shape check (no numbers, years or fragments as names) would remove before writing.

### Correction test: erase, suppress or replace (21 Nov 09:20; tools/correct_min.py, A6 stand-in, 150 made-up one-answer facts taught WRONG (Z), 150 bystanders right)
| arm (600 steps) | false Z first, right after (test / new wording) | after 400 steps of other training | true Y first (after / later) | savings: Z re-taught 30 steps, corrected vs fresh |
|---|---|---|---|---|
| taught (start) | 0.55 / 0.63 | - | 0.08 | - |
| NONE (stop teaching) | 0.13 / 0.15 | 0.17 / 0.18 | - | 0.33 vs 0.27 |
| UNLIKE (kra21) | **0.07 / 0.08** | **0.11 / 0.08** | - | 0.27 vs 0.25 |
| REPLACE (teach Y) | 0.27 / 0.21 | 0.15 / 0.17 | 0.56 / 0.26 | 0.27 vs 0.25 |
| NOTREC ('nowhere on record') | 0.23 / 0.25 | 0.15 / 0.16 | - | 0.34 vs 0.29 |
| REPLACE + UNLIKE | **0.05 / 0.05** | 0.11 / 0.13 | 0.44 / 0.25 | 0.27 vs 0.28 |
- **Unlikelihood suppresses best** (with or without the replacement); **replacing alone keeps the false answer HIGHER right after** (0.27 vs 0.13 for doing nothing): rehearsing the question keeps both answers active. The misinformation finding (a retraction needs an alternative) does not carry over to this trunk. Savings are small everywhere (+0.00 to +0.06): no strong hidden suppression. Caveat: this stand-in forgets unrehearsed facts fast (NONE 0.55 -> 0.13); Platypus replays, so stopping alone would help less there.
- Decision: unlikelihood rows stay the correction tool; the open problem is only WHICH inferences to correct.

### kra22 selection rules checked (21 Nov 09:25; tools/select_min.py part A, 307 confirmed at right 0.769)
- **NAME SHAPE (no digits, brackets or quotes in A or C): wrong removed 0.20, right lost 0.01 - 0.21 right lost per wrong (random 3.3).**
- 2+ distinct documents (confirm + delay windows): 0.41 / 0.22 (1.76). BOTH: taught 217, right **0.839** (182 right of 236), 1.50 right lost per wrong.
- One-answer wordings: none of the typed wordings ('in', 'of', ...) has heads with one tail 60%+ of the time (fixed measure), so contradiction cannot be restricted to one-answer relations with these wordings; it stays off.
- **kra22 takes both: inf_rti_nameshape 1, inf_rti_conf_docs 2** (with retraction off and the never-read-calibrated gate).

### cg27a_kra21 GPU, visit 9 (21 Nov 09:20)
- [RETRACTION RECALL]: all 118 retracted (95 right, 23 wrong) now 0.13 / 0.09 open (rank 0.22 / 0.26); judged, not flagged 0.74 / 0.74. No sign yet of the visit-7 cohort coming back.
- Taught inferences recover again at an odd visit: right 0.74 / wrong 0.73 open, rank 0.85 / 0.87. By write visit: 2 -> 0.55 (0.97, 0.66, 0.75, 0.64, 0.55 at visits 5-9: the oldest cohort decays slowly), 4 -> 0.82, 6 -> 0.91.
- PRIOR CONTROL: held-back right **+0.13 (SE 0.04)**, wrong +0.06 (SE 0.05). MATCHED held-back right own - broken **+0.11 (SE 0.03)**, own - closed +0.20 (SE 0.03): both back to their visit-5/6 sizes.
- Gated (keep-75% rule, t 0.91): right overall 0.44, right when answered 0.90, never-read 0.02.
- Facts: probe 0.80; ledger ever taught **0.76** (0.82 at visits 7-8), taught 1 visit 0.37 (n 30), 2-3 visits 0.79, 4+ 0.83; control 0.42. Visit 9 is DM Mathematics. Bits on read domains **1.7735** (best yet). No Traceback.

### cg27a_kra22 GPU, visits 2-4 (21 Nov 10:35; started 09:17, the final preset confirmed in the log)
- **Selection before teaching works as predicted: CONFIRMED 200 (right 0.88)** vs kra21 visit 4: 282 (0.83). Still tentative 400 (0.70). Taught 136 (right 0.87) vs kra21 188 (0.82). Offline forecast: 0.77 -> 0.84 at 77% of the right ones.
- Gate (never-read calibration, before the floor fix): visit 2 t 0.09, never-read given a name 0.43 (the flaw); visit 3 t 0.99, right overall 0.45, never-read 0.02; visit 4 t 0.99, 0.33 / 0.02. Keep-75% rule beside it: 0.45 / 0.02 and 0.25 / 0.00. The smoothed line carries visit 2's 0.09 (never-read 0.12-0.27): ignore it in kra22.
- Facts: probe 0.83 / 0.92 / 0.94 (visits 2-4), ledger 0.87 / 0.88 / 0.87; control 0.36-0.55. Bits on read domains 2.0357 / 2.0106 / 1.9248.
- Visit 4 is the pre-teaching baseline for the new readouts (the 136 are first trained in visit 5): PRIOR CONTROL held-back right +0.02 (SE 0.04); TRANSFER SOURCE +0.00 in every split; TAUGHT RECALL BREAKDOWN 0.17-0.21 (CONFLICT 0.10, n 10, vs none 0.21). No Traceback.

### cg27a_kra22 GPU, visits 5-6 (21 Nov 11:35)
- **TRANSFER SOURCE - the lift on never-taught inferences comes through ANSWERS that were also taught.** Held-back RIGHT, own - swapped subject (visit 5 / 6): all +0.14 (SE 0.06, n 58) / +0.07 (SE 0.03, n 119); **answer is a taught answer +0.21 (SE 0.08, n 39) / +0.11 (SE 0.04, n 75) vs answer NOT taught +0.00 (n 19) / +0.00 (n 44)**; subject a taught name +0.33 (n 9) / +0.09 (n 23) vs not +0.10 / +0.06; shares a bridge +0.15 / +0.07 vs not +0.08 / +0.07; NOTHING SHARED +0.00 (n 7) / +0.00 (n 14). Held-back wrong +0.00 / +0.05.
- Reading: teaching 'X r C' makes the trunk prefer C for OTHER subjects of the same kind (own subject over a swapped one), but only when C itself was taught. Subject-aware transfer of taught answers (a learned subject-kind -> answer association), not composition: no lift where the answer was never taught or nothing is shared.
- **TAUGHT RECALL BREAKDOWN at the visit-6 dip:** bare training form and test wording equal (0.74 / 0.76 for the visit-2 cohort; not a wording effect). **CONFLICT (another taught answer for the same subject and wording) 0.44-0.46 vs none 0.78 (n 39 / 120)** - competing answers under the generic wording 'in' (often both right: 'X in Germany', 'X in Europe') split the trunk's answer. Keep rule: those it stops training are recalled 0.93-0.96 (already known); the ones it trains include the 61 just confirmed (0.11-0.13, not yet trained).
- Taught inferences: visit 5 right 0.97 / wrong 1.00 open vs held back 0.03 / 0.00; visit 6 0.53 / 0.33 (visit-2 cohort 0.69 / 0.55; the visit-6 dip again, as kra21's 0.43).
- Confirmation: visit 6 223 confirmed (right 0.78); all so far 423 (0.83; kra21 at visit 6: 509 at 0.82); faded 299 (0.67; kra21 0.59). The two-document rule's gain shrinks at the second judgement: it fades more right ones.
- Gate (never-read calibration): visit 5 right overall 0.34 / never-read 0.00 (keep-75%: 0.53 / 0.01); visit 6 0.59 / 0.02 (same as keep-75%).
- Facts: probe 1.00 / 0.84, ledger 0.86 / 0.83 (taught 1 visit 0.69 at visit 6), control 0.26-0.46. Bits on read domains 1.8614 / 1.9094 (kra21 1.8587 / 1.9049). No Traceback.

### Bridge teaching on real-text chains (21 Nov 11:50; tools/bridge_teach.py, A6 stand-in, 872 chains from distant_50k code writes, TRAIN 400 / TEST 200 on bridges never in a train chain)
- 606 distinct first-hop wordings for 13 conclusion wordings; premises recalled 0.67 after 3,000 steps (forgetting -0.04 to -0.05).
- DIRECT ('Q: A tw ? So A tw C.'): taught C first among 10 0.79 / 0.92 at steps 1,500 / 3,000; never taught 0.19 / 0.14 (chance 0.10).
- **BRIDGE ('Q: A tw ? A w1 B. B w2 C. So A tw C.'), the model writing its own chain (greedy): taught C first 0.18 / 0.33 (exact 0.12 / 0.27), first hop written exactly 0.31 / 0.51; NEVER TAUGHT C first 0.015 / 0.010, first hop 0.005 / 0.005, chain format written 0.08 / 0.105.** No composition: it memorises (slowly) the taught chains and cannot even write the format for new subjects.
- The samples show why: real-text premises are mostly list fragments ('Bergen , norway and Iceland', 'Belize , guatemala , Nicaragua', 'Frankfurt , sweden ...'), not relations - there is no consistent first hop to learn, unlike the synthetic world's one rule. (A first run scored 0.000 everywhere because greedy decoding fed unshifted input; fixed, 4672a46.)
- Next (route A): the same restricted to clean relation families (containment wordings; word-only relational wordings).

### Route B dose test: how many taught chains make a trunk compose (21 Nov 12:05; tools/compose_synth.py CS_COTLATE 1, CHAIN arm, pretrained cs_impl0_s0, seed 0)
| taught people (chain form 'A is a citizen of B > C.') | never taught B / C at 300 steps | 750 | 1,500 |
|---|---|---|---|
| 10 | 0.19 / 0.19 | 0.12 / 0.15 | 0.35 / 0.30 |
| 30 | 0.47 / 0.46 | 0.51 / 0.52 | 0.61 / 0.62 |
| **100** | 0.70 / 0.74 | 0.87 / 0.86 | **0.88 / 0.87** |
| 300 (19 Nov) | 0.85 / 0.83 | - | 0.87 / 0.85 (3,000) |
- Taught C 1.00 in every arm. The decomposition is learned from about 100 worked chains; 30 give 0.62.
- On the GPU the rule world's planted composites are DIRECT ('X is a grandchild of Z'), which never composes, and route J's chain-form writes are a few dozen: held-back composites 0.00. **kra23 plants the composites with their bridge (inf_rule_world_chainplant 1) and doubles the rule world (96: about 118 chain composites per rule)** - the first GPU test of trunk-level inference on the rule world.

### cg27a_kra22 GPU, visit 7 (21 Nov 12:10)
- TRANSFER SOURCE, third visit running: held-back right own - swapped +0.14 (SE 0.04, n 119); **answer a taught answer +0.23 (SE 0.06, n 70) vs not +0.02 (SE 0.02, n 49); NOTHING SHARED +0.00 (n 15)**; subject a taught name +0.32 (n 22). Held-back wrong +0.11 (SE 0.07, n 19).
- Taught inferences recover at the odd visit: open right 0.85 / wrong 0.80, rank 0.97 / 0.91; visit-2 cohort 0.82, visit-4 cohort 0.92. Breakdown: CONFLICT 0.90-0.92 vs none 0.97-0.99 (the conflict cost is largest at the dip visits: 0.44-0.46 vs 0.78 at visit 6).
- MATCHED held-back right own - broken +0.09 (SE 0.05), own - closed +0.21 (SE 0.05).
- Gate (never-read calibration): right overall 0.52, right when answered 0.92, never-read 0.01; keep-75%: 0.47 / 0.00.
- Facts: probe 0.88, ledger 0.82 (taught 1 visit 0.65, 4+ visits 0.84), control 0.24-0.31. Bits on read domains 1.8832 (kra21 1.8856). No Traceback.

### cg27a_kra22 GPU, visit 8 (21 Nov 12:40; stopped here to start kra23)
- TRANSFER SOURCE: held-back right +0.12 (SE 0.04, n 150); answer a taught answer +0.16 (SE 0.05, n 93) vs not +0.05 (SE 0.04, n 57); nothing shared -0.07 (n 15). Held-back WRONG now also +0.13 (SE 0.06, n 30). Four visits: the lift follows taught answers; nothing shared never lifts.
- Taught inferences dip again at the even visit: open right 0.54 / wrong 0.55, rank 0.92 / 0.63; CONFLICT 0.64-0.74 vs none 0.93-0.97 (n 76 / 120) - conflicts grow with the pool (39 at visit 6, 76 now).
- Confirmed 639 so far at right 0.82 (kra21 at visit 8: 733 at 0.79); faded 662 (0.65).
- Gate: both rules t 0.91-0.93, right overall 0.48, never-read given a name **0.07** - neither rule holds 2% out of sample at visit 8.
- Facts: probe **0.68** (newly taught; first time below 0.75), ledger 0.81 (4+ visits 0.86). Bits on read domains 1.8250 (kra21 1.8292). No Traceback.
- kra23 local check, visits 1-2: no errors; [RULE WORLD, THE TRUNK'S OWN CHAIN] prints (0.00 everywhere at 150 local steps, as expected); the gate floor was skipped when fewer than 4 right answers existed (t 0.09 at visit 2) - now floored at 0.9 then.
- kra23 local check, visits 3-4 (stopped at visit 4 to free the CPU; the GPU run is under way): no errors; every new readout prints - SELF-SUPPORT JUDGE (first values at visit 4: just confirmed margin AUC 0.660 on n 15, trained 0.550), AUDIT SAMPLE (16 printed), the floored gate, and RULE WORLD, THE TRUNK'S OWN CHAIN (0.00 everywhere at 150 local steps a visit: the local harness does not learn the planted facts).

### cg27a_kra23 GPU, visits 1-4 (21 Nov 13:35): route B - the trunk writes chains for composites never planted; first clean sign of composition
- Reader fix works: the relational module finds both rules from visit 1, 0 false; 125 of 128 planted rule-world facts recorded at visit 1.
- **[RULE WORLD, THE TRUNK'S OWN CHAIN]** (greedy, store and memory off, parsed 'B > C'), visits 2 / 3 / 4:
  - PLANTED (read in chain form): chain format 0.46 / 0.29 / 0.31, C right 0.46 / 0.71 / 0.62 (n 13 / 17 / 26).
  - HELD BACK, never planted: chain format 0.27 / 0.37 / **0.77**, bridge right 0.09 / 0.16 / 0.37, C right 0.09 / 0.05 / **0.30** (n 11 / 19 / 30; direct 0.00).
  - PURE FAMILIES: chain format 0.17 / 0.30 / **0.71**, bridge right 0.08 / 0.20 / 0.36, C right 0.00 / 0.00 / **0.29** (n 12 / 20 / 28; direct 0.00).
  - **CONTAMINATED:** route J also writes never-planted composites (76 by visit 3, 128 by visit 4; 68 taught in chain form), and this readout did not exclude them. Fixed for kra24 (NEVER WRITTEN BY ROUTE J).
- **The clean number - route J's written composites HELD BACK as its control (never planted, never taught):** recalled from the trunk at visit 4 **0.17-0.21 (n 24)** (edits subtracted 0.17), greedy C right 2 of 24; pure families 0.08 (n 12); taught 0.98-1.00. kra21 / kra22: 0.00 held back at every visit. A first, small sign that the trunk composes from its own chain; n is small and it is one visit.
- Route D (explicit two steps, store off): two-step C 0.36 / 0.47 vs direct 0.03 / 0.06. Route A (memory) 0.71-0.82.
- Facts: probe 0.95 / 0.97, ledger 0.88 / 0.92. Gate visit 2 t 0.09 (the floor needs 4 right answers; patched 5a63346 after this run started), visit 3 keep-75% 0.42 / never-read 0.06. Bits 2.0216 / 2.0077 (kra22 2.0357 / 2.0106). No Traceback.

### cg27a_kra23 GPU, visit 5 (21 Nov 14:00): the trunk composes for new SUBJECTS, not yet through new BRIDGES
- Route J's held-back written composites (never planted, never taught): **0.19 (n 54)**, never planted 0.20 (n 51); taught 0.97-0.99 (67). **But PURE FAMILIES held back 0.00 (n 26)** (test wording, _dt_recall).
- Reading: the held-back composites answered right are in families where another leaf's composite was planted as a chain ('Y is a grandchild of M > Z'). The trunk takes the first hop from the read premise ('X is a child of M', never in any chain for X) and the second hop from chains it read about M's other grandchildren ('M > Z'). Where the bridge M never appears in any chain (pure families), the second hop would have to come from the premise 'M is a child of Z' alone - and it does not (0.00), although route D, asking the two hops explicitly, gets pure families 0.42.
- So: composition over new SUBJECTS (a genuine two-hop answer for a person never described), not yet over new BRIDGES. Consistent with every test so far: the synthetic world's held-out people lived in cities that appeared in taught chains (0.87); the bridge tests held out the bridges and failed (0.01-0.02); the dose test fits either.
- [RULE WORLD, THE TRUNK'S OWN CHAIN] (contaminated by route J teaching, see visit 4): held back C right 0.36, pure 0.43; chain format 0.71-0.76; bridge right 0.67-0.68 (premise recall in the chain is good).
- SELF-SUPPORT JUDGE (visit 4, first GPU values): just confirmed, margin AUC 0.559 (n 149, 20 wrong) - too weak for a teach-gate; trained 0.43-0.57; held back 0.61-0.65 (n 64, 6 wrong). AUDIT SAMPLE printed (40 of 200 at visit 4).
- Confirmed at visit 4: 200 (right 0.88), as kra22. Taught inferences 0.93 vs held back 0.03. Facts: probe 0.94, ledger 0.88. Gate (never-read calibration) t 1.00 at visit 5: right overall 0.17 / never-read 0.00 - too strict; keep-75%: 0.48 / 0.02.

### Route A: bridge teaching on real-text containment chains (21 Nov 14:10; tools/bridge_teach.py BT_SOURCE=records, 5,734 chains built from records 'X in Y', 'Y in Z' read in distant_50k; 95% of the conclusions never stated in a record; TRAIN 400 / TEST 200 on bridges never in a train chain)
- Premises recalled 0.56 after 3,000 steps. DIRECT: taught 0.68 / 0.90, never taught 0.05 / 0.035 (C first among 10, steps 1,500 / 3,000). **BRIDGE: taught 0.29 / 0.64 (exact 0.25 / 0.60), first hop written 0.42 / 0.68; NEVER TAUGHT 0.00 / 0.00, first hop 0.01, format 0.01-0.015.**
- No composition through held-out bridges, as kra23 visit 5 predicted (pure families 0.00). Also, records 'X in 2007' make years into bridges - the name-shape check belongs on the premises too.
- The earlier filters on the code writes left too few chains: containment 8, word-only relations 116 (78 first-hop wordings).
- Next: tools/compose_synth.py CS_NEWB - in the synthetic world, never-taught people in cities that appear in a taught chain (seen bridge) vs cities that never do (new bridge), with taught chains concentrated or one per city.

### cg27a_kra23 GPU, visit 6 (21 Nov 14:45) + the synthetic new-bridge test (first arm)
- Route J held-back composites (never planted, never taught): 0.17 (n 65), never planted 0.18 (n 62); **pure families held back 0.00 (n 29)** again; held back answered with the middle hop 8 of 65. Route D (two hops asked explicitly, store off) 0.60, pure families 0.58 vs direct 0.00-0.03: the trunk knows both premises of pure families but its own chain does not use the second.
- **Re-wording halved the conflicts (20 vs kra22's 39 at visit 6) and they are no longer recalled worse (0.75-0.80 vs 0.71-0.73), yet the visit-6 dip is unchanged** (taught right open 0.50, kra22 0.53; visit-2 cohort 0.97 -> 0.74). The dips at visits 6 and 8 (Github, USPTO) are domain interference that recovers at the next visit, not conflicting answers.
- TRANSFER SOURCE: held-back right +0.03 (SE 0.03, n 119); answer taught +0.02 vs not +0.03 - the answer-transfer lift of kra22 (+0.11 to +0.23) is gone after re-wording (answers now in varied wordings).
- SELF-SUPPORT JUDGE at visit 6: just confirmed AUC 0.424 (n 150, 37 wrong), trained 0.416 - below 0.5. Not a teach-gate.
- Confirmed 423 so far at right 0.83 (as kra22). Facts: probe 0.91, ledger 0.88. Gate: never-read calibration 0.48 / never-read 0.06; keep-75% 0.56 / 0.07. Bits on read domains 1.8954 (kra22 1.9094). No Traceback.
- **Synthetic new-bridge test (compose_synth CS_NEWB 1, 100 taught people in even-numbered cities, 73 cities):** never-taught people in cities NEVER in a taught chain compose at **B 0.84 / C 0.84** (3,000 steps), seen-bridge 0.92 / 0.96. **Confounded:** synthetic city names carry their country's code 80% of the time ('Name-qX'), so the second hop can be read off the bridge's name - which rule-world and real names cannot. Next: the same with uninformative codes (re-pretrained).

### Synthetic new-bridge test, second arm + GPU audit of kra23 (21 Nov 14:55)
- compose_synth CS_NEWB 1 CS_NEWB_DIV 1 (100 taught people, one per city): never taught, NEW bridge (n 70) B / C 0.83 / 0.83 at 3,000 steps; seen bridge (n 30) 0.93 / 0.93. Spreading the chains over 100 cities changes little against concentrating them over 73 (0.84). Same confound (city codes); the clean run (CS_CODEINF 0, re-pretrained) is under way.
- **Audit of kra23's printed sample** (80 inferences confirmed at visits 4 and 6; the AI model's judgement from general knowledge, not gold):
  - Labelled WRONG 11 (14%): 5 true (Seattle in Washington, Cleveland in Ohio, Henry VIII of England, Union Army during American Civil War, Taliban in Afghanistan), 4 ambiguous (Basque in Spanish, Jewish state at Jerusalem, Jewish 's Holocaust, Hindi as Bollywood), 2 false (Central America in Mexico, Peru in administration of Peruvian).
  - Labelled RIGHT 69: **5 are stated BACKWARDS** after re-wording on confirmation ('Texas are located in Dallas', 'India is located in Himachal Pradesh', 'Puerto Rico in San Juan', 'Italy region of Lombardy', 'Pakistan they have settled in Karachi'): the pair is related, but the taught sentence is false. The record's word order is not the relation's direction.
  - **False as taught: 7 of 80 (9%), plus 4 ambiguous (5%).** Without re-wording it would have been about 2 of 80 (3%) plus the ambiguous. Re-wording did not fix the dips either (kra23 visit 6): drop it in kra24.

### cg27a_kra23 GPU, visit 7 (21 Nov 15:10)
- Route J held-back composites: 0.11 (n 70), never planted 0.12 (n 67) - down from 0.19-0.20 at visits 4-5 as the held-back set grows with later writes; **pure families held back 0.06 (n 33)**, the first non-zero (2 of 33); greedy: right 4, middle hop 4 of 70. Route D explicit two steps 0.54, pure 0.53.
- Taught inferences recover at the odd visit: open right 0.82 / wrong 0.85, rank 0.90 / 0.93.
- Facts: probe 0.82, ledger 0.79 (taught 1 visit 0.58). Gate: never-read calibration too strict (t 0.99, right overall 0.19, never-read 0.00); keep-75% 0.48 / 0.01. Bits on read domains 1.8676 (best of kra21-23 at visit 7: 1.8856 / 1.8832). No Traceback.

### Synthetic new-bridge test, CLEAN (21 Nov 15:45; compose_synth CS_CODEINF 0 - city names carry no country code; re-pretrained, city's country 1.00, person's city 1.00; 100 taught people in even-numbered cities)
| arm | never taught, NEW bridge B / C (300 / 750 / 1,500 / 3,000 steps) | SEEN bridge B / C at 3,000 |
|---|---|---|
| concentrated (73 cities) | 0.82 / 0.55, 0.82 / 0.60, 0.85 / 0.67, **0.90 / 0.73** (n 78) | 0.73 / 0.73 (n 22) |
| one per city (100 cities) | 0.77 / 0.55, 0.82 / 0.62, 0.87 / 0.68, **0.87 / 0.82** (n 71) | 0.97 / 0.93 (n 29) |
- **Composition through bridges never seen in a chain works without the name shortcut:** the second hop comes from the plain premise 'B lies in C' (direct teaching: 0.08). Diverse chains help a little (0.73 -> 0.82).
- So the rule world's pure-family failure (kra23: 0.00-0.06) is specific. Candidate causes: the synthetic second hop chooses among 20 countries (rule world: about 190 roots), and its premise is known perfectly (rule world: recalled 0.53-0.58). Next: CS_NC 150.

### cg27a_kra23 GPU, visit 8 (21 Nov 16:10; the user's log ends at the start of visit 9)
- Route J held-back composites 0.12 (n 84; never planted 0.12, n 81); taught 0.73-0.74 (the visit-8 dip). **Pure families held back 0.02 (n 41)** (visit 7: 0.06, n 33). Route D explicit two steps 0.48, pure 0.47; direct 0.00-0.01.
- Route B over kra23, held back (all / pure): visit 4 0.17-0.21 / 0.08, 5 0.19-0.20 / 0.00, 6 0.17-0.18 / 0.00, 7 0.11-0.12 / 0.06, 8 0.12 / 0.02. Composition over new subjects at about 0.12-0.20; through new bridges at about chance.
- Confirmed 639 so far at right 0.82 (kra22 identical). Taught inferences 0.55 open at the even visit. Facts: probe 0.81, ledger 0.75. Gate: never-read calibration 0.58 / never-read 0.07; keep-75% 0.61 / 0.08. Bits on read domains 1.8125 (best yet; kra22 1.8250). No Traceback.
- **150 countries (CS_NC 150, clean, one taught person per city, 100 taught): never taught, NEW bridge (n 66) B / C 0.95 / 0.92 at 300 steps, 0.94 / 0.94 at 3,000; seen bridge 0.88 / 0.88.** A many-valued second hop is no obstacle. What remains different from the rule world is premise strength (synthetic: the city's country recalled 1.00; rule world: about 0.5-0.7). **kra25 plants the rule world's premises three times as often (inf_rule_world_premk 3).**

### cg27a_kra24 GPU, visits 5-6 (21 Nov 17:25)
- **Pooled gate (keep-75% over 3 sleeps): right overall 0.64 / 0.62, right when answered 1.00, never-read given a name 0.03 / 0.02** (visit 4: 0.55 / 0.03). One-sleep keep-75% beside it: 0.64 / 0.03 and 0.58 / 0.02.
- Facts: probe 0.94 / 0.88, ledger 0.89 / 0.89. Taught inferences visit 5: 0.96 vs held back 0.05.
- Route D (explicit two steps, store off): first step B 0.80, two-step C **0.74**, pure families 0.71 (kra23 at visit 5-6: 0.47-0.60) - the 12 question rows a step raise premise recall. Route J held back: pure families 0.00 (n 25).
- Bits on read domains 1.8496 at visit 5 (kra22 1.8614). No Traceback.
- kra25 local check: visits 1-2 run without errors with the premises planted three times (domain 0: 726 rule-world facts in 25,290 planted sentences).

### cg27a_kra24 GPU, visits 6-7 (21 Nov 18:10)
- **Pooled gate: never-read given a name 0.02 / 0.01** (right overall 0.62 / 0.45, right when answered 1.00 / 0.97). kra22 / kra23 at visit 7: 0.01-0.07.
- **Facts at visit 7: ledger ever taught 0.84 (kra23 0.79, kra22 0.82); probe (newly taught) 0.76 (kra23 0.82)**; control 0.29-0.31.
- Route B, clean readout (never planted, never written by route J), visits 6 / 7: held back C right 0.33 (n 36) / 0.03 (n 37); **pure families 0.14 (n 35) / 0.08 (n 26)**; route J pure held back 0.06 / 0.03. Noisy and small. Route D explicit two steps 0.65-0.69 (pure 0.62-0.65).
- Confirmed 322 so far at right 0.84. Taught inferences 0.65 / 0.80 (even / odd visit). Bits on read domains 1.8944 / **1.8603** (kra23 1.8954 / 1.8676). No Traceback.

### Route A with an ORACLE reader (21 Nov 18:20; bridge_teach BT_SOURCE=records BT_ORACLE=P131,P17,P30,P361,P150: containment chains only from records whose pair carries a gold containment relation; 429 chains, TRAIN 312 / TEST 117 on bridges never in a train chain)
- Premises recalled 0.95 (the noisy records: 0.56). BRIDGE (restated form 'Q: A r ? A w1 B. B w2 C. So A r C.'): taught C first 0.54 / 0.64; NEVER TAUGHT 0.000, first hop 0.034, format 0.017.
- **A test-format failure, not a reader one:** for a new subject the stand-in writes a TAUGHT subject with similar letters ('Q: Nagpur Division in ?' -> 'Nazca Lines in Peru ...', 'Bangalore' -> 'Basel', 'Haryana' -> 'Hazarajat'): it cannot copy a new name into the restated chain. The synthetic and rule worlds never restate the subject ('A r B > C.'). P150 ('contains') also reverses direction in these labels ('Germany in Berlin'). Re-run in the arrow form without P150 (BT_FORMAT=arrow).
- The earlier route A runs (writes, records) used the same restated form, so their 0.00 on held-out bridges is confounded the same way.

### cg27a_kra24 GPU, visit 8 (21 Nov 18:50; stopped here to start kra25)
- **Pooled gate at the patent visit: right overall 0.56, right when answered 1.00, never-read given a name 0.03** (kra23 0.07, kra22 0.07). Visits 4-8: 0.03 / 0.03 / 0.02 / 0.01 / 0.03.
- **Facts: ledger ever taught 0.83 (kra23 0.75, kra22 0.81); probe 0.78 (kra23 0.81, kra22 0.68).** Both above 0.75 at visits 2-8.
- **Bits on read domains 1.8040, the best of any build at visit 8** (kra23 1.8125, kra22 1.8250): 12 question rows a step cost nothing.
- Route B: clean readout pure families 0.11 (n 46), never written 0.22 (n 46); route J pure held back 0.02 (n 53). Route D two-step 0.62, pure 0.57.
- Confirmed 423 so far at right 0.84. No Traceback.
- **kra24 meets the facts and gated-answers goals through visit 8** (one seed).
- **Re-run in the arrow form ('A in B > C.', without P150; 21 Nov 19:00):** premises recalled 1.00. DIRECT never taught 0.050 / 0.075 (C first among 10, steps 1,500 / 3,000; chance 0.10). **BRIDGE never taught: C first 0.175 / 0.150, C exact 0.100 / 0.050, own bridge right 0.250 / 0.175, chain format 0.300 / 0.275** (n 117); taught 0.92-0.98.
- **A weak positive with a clean reader:** bridge teaching beats direct teaching on chains through bridges never in a taught chain by about +0.10 (about 2-3 SE), but far below the synthetic world (0.73-0.94): the stand-in writes the chain form for under a third of new subjects and often answers in a single hop. Real containment is many-levelled (city in region in country in continent), so 'in' has several right answers.

### cg27a_kra25 GPU, visits 1-5 (21 Nov 19:50; premises planted three times, inf_rule_world_premk 3)
- **Premise recall rises:** route D premises, first step B 0.59 / 0.63 / 0.69 / **0.86** at visits 2-5 (kra24 at visit 5: 0.80; kra23: 0.50); explicit two steps 0.81, **pure families 0.80** (kra24 0.71, kra23 0.42).
- **But the trunk's own chain does not use them for new bridges:** pure families, never written: chain format 0.89, **bridge right 0.22**, C right **0.00** (n 9); route J pure held back 0.00 (n 24). Non-pure held back: bridge right 0.54, C right 0.15 (n 13).
- So premise strength is not the remaining gap: with premises recalled at 0.86 and the two hops answered 0.80 when asked one by one, the chain's FIRST hop is wrong for pure-family leaves (0.22) - asked 'X is a grandchild of ', the trunk does not retrieve X's parent the way it does when asked 'X is a child of '.
- Cost: bits on read domains 1.8516 at visit 5 (kra24 1.8496); facts probe 0.97, ledger 0.90; gate 0.56 / never-read 0.02. No Traceback.

### cg27a_kra25 GPU, visit 6 (21 Nov 20:25; stopped here to start kra26) + kra26 local check
- Pure families: trunk's own chain (never written) bridge right 0.18, **C right 0.04 (n 28)**; route J pure held back 0.07 (n 27). Non-pure never written 0.34 (n 29). Premise recall at visit 6: first step 0.75 (0.86 at visit 5); explicit two steps 0.72, pure 0.68.
- **Tripled premises raised premise recall (visit 5) but not composition through new bridges.** Cost: none - bits on read domains 1.8945 (kra24 1.8944); facts probe 0.91, ledger 0.87; gate 0.52 / never-read 0.00.
- kra26 local check (spelled steps, inf_rule_world_chainplant 2): visits 1-2 run without errors; the reader records 192 rule-world facts by visit 2 (the local harness learns too little to show chains).

### cg27a_kra26 GPU, visits 1-2 (21 Nov 21:05; spelled steps, inf_rule_world_chainplant 2)
- The reader reads the spelled composites: both rules found from visit 1, 0 FALSE; route J composite 0.99, pure 1.00, writes precision 0.98 (81).
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 2: PLANTED chain format 0.69, bridge right 0.46, C right 0.38 (n 13); HELD BACK never written format 0.45, bridge 0.00, C 0.00 (n 11); **PURE FAMILIES never written: format 0.83 (kra23 at visit 2: 0.17), bridge right 0.00, C 0.00 (n 12)**. The spelled form is taken up fast; the first step for new subjects is wrong so far (small n, visit 2).
- Route D: first step 0.59, two-step 0.52, pure 0.48 (direct 0.00-0.05) - as kra25 at visit 2. Facts taught one visit 0.87; control 0.20. Bits on read domains 2.0223. No Traceback.

### cg27a_kra26 GPU, visit 3 (21 Nov 21:28)
- [RULE WORLD, THE TRUNK'S OWN CHAIN]: PLANTED format 0.88, bridge right 0.82, **C right 0.76** (n 17; kra23 0.71). **HELD BACK, never planted, never written (new subjects): bridge right 0.25, C right 0.17 (n 12; direct 0.00)** - a clean readout (kra23's was contaminated). **PURE FAMILIES never written: format 0.75, bridge right 0.00, C 0.00 (n 12)**; kra23 (arrow form, subject not restated) had bridge right 0.20 at visit 3.
- The trunk knows the first step when asked alone (route D first step 0.66, pure two-step 0.54) but writes a wrong bridge inside the spelled chain: consistent with the stand-in's copy failure (step 1 restates a NEW subject). kra27 prepared: step 1 without the subject (inf_rule_world_chainplant 3); and a readout of the subject the trunk writes in step 1 (subject copied / another subject with ITS true bridge) + 4 examples.
- Facts taught 0.91 (one visit 0.93); gate never-read 0.02, right when answered 1.00; route J 0.96, writes 0.98; bits on read domains 2.0008. No Traceback.

### cg27a_kra26 GPU, visit 4 (21 Nov 21:43)
- [RULE WORLD, THE TRUNK'S OWN CHAIN]: PLANTED format 1.00, bridge 0.73, C 0.69 (n 26); HELD BACK never written: bridge 0.09, C 0.09 (n 11; visit 3: 0.25 / 0.17); **PURE FAMILIES never written: format 0.75, bridge right 0.17, C right 0.08 (n 12)** - kra23's arrow form at visit 4 (contaminated): bridge 0.36, C 0.29. Not above the 0.00-0.14 band of kra23-kra25.
- Route D: first step 0.69, two-step 0.66, pure 0.57; "asked DIRECTLY" 0.29 / pure 0.26 (earlier builds 0.00-0.05) - route D's set does not exclude route J's written composites, which kra26 teaches directly (inf_distill_chainform 0), so this is likely their recall, not composition.
- Facts taught 0.88 (2-3 visits 0.89); gate right when answered 1.00, never-read 0.02; route J 0.96, writes 0.96 (188); bits on read domains 1.9202. No Traceback.
- kra27 local check, visit 1: no errors; new readout prints (subject copied 1.00 on planted; examples 'Anpelqui -> is a child of the could not ...' - the 150-step stand-in learns the form's start only).

### cg27a_kra27 GPU, visit 1 (21 Nov 22:07; spelled steps without the restated subject, inf_rule_world_chainplant 3) + local check visits 1-2
- No Traceback (GPU visit 1, local visits 1-2). The reader reads the new form: both rules found from visit 1, 0 FALSE. Bits on read domains 2.0086 (kra26 2.0043).
- Trunk's own chain at visit 1: 0.00 everywhere, as every build; examples for pure families: 'Tevmizo -> is a child of Zonorbri', 'Vinquisen -> is a child of Zofaris' - the step-1 form is taken up already, with a common name in the slot.

### cg27a_kra27 GPU, visits 2-3 (21 Nov 22:44; step 1 without the restated subject)
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 2 / 3: PLANTED bridge 0.62 / 0.88, C 0.38 / 0.82 (n 13 / 17); **HELD BACK never written: bridge 0.64 / 0.17, C 0.36 / 0.08 (n 11 / 12)** (kra26: 0.00 / 0.25, 0.00 / 0.17); **PURE FAMILIES never written: bridge 0.33 / 0.10, C 0.00 / 0.00 (n 12 / 10)** (kra26: 0.00 / 0.00).
- Examples, pure families: 'Tevmizo -> is a child of Zogalka; Zogalka is a child of Zogalka', 'Pelsenzo -> is a child of Orzopel; Orzopel is a child of Orzopel', 'Quibrisal -> is a child of Quibeld; Quitevbri'. Step 1 is now sometimes right (no subject to copy), but STEP 2 fails inside the chain: the bridge is copied correctly and then repeated as its own parent, or step 2 is dropped - while asked on its own (route D) the second hop is right about 0.5. The same pattern as step 1 in kra26: a premise recalled in its own sentence but not inside the chain (premises planted in one wording; synthetic test running).
- Facts taught 0.92 (one visit 0.93); gate never-read 0.00, right when answered 1.00; route J writes 0.97-0.98; bits on read domains 1.9977 (kra26 2.0008). No Traceback.

### KNOWLEDGE AUGMENTATION OF THE PREMISES IS THE MISSING PIECE (synthetic, 21 Nov 22:55; compose_synth CS_PREMW 1 x12, CS_PTW 1 x6 - one wording per premise, same sentence count - CS_CODEINF 0, CS_NEWB 1, 100 taught people)
- Premises recalled perfectly in their wording either way (city's country 1.00, person's city 1.00 from step 2,000).
- **Chain teaching, never taught, NEW bridge B / C at 300 / 750 / 1,500 steps: one wording 0.12 / 0.15, 0.18 / 0.19, 0.19 / 0.18 (n 78) vs four / three wordings 0.82 / 0.55, 0.82 / 0.60, 0.85 / 0.67.** Seen bridge 0.18-0.27 (four wordings 0.73-0.91). Direct teaching 0.08-0.15 either way.
- **The first hop fails inside the chain** (B 0.12-0.19) although asked alone it is 1.00 - the rule world's pattern exactly (kra26 step 1, kra27 step 2). A premise learned in one wording is keyed on that wording's state and is not retrieved inside the chain; varied wordings key it on the subject (Allen-Zhu & Li's knowledge augmentation, here for composition).
- The rule world planted each premise in ONE wording (kra25: the same sentence three times). **kra28: kra27 + inf_rule_world_premw 6** (six wordings, subject and relation in varied contexts; no ':' or ' > ', which the chain readers rewrite).

### cg27a_kra28 GPU, visits 1-2 (21 Nov 23:29; premises in six wordings, inf_rule_world_premw 6) + local check
- No Traceback (GPU visits 1-2; local visits 1-2). Reader on the GPU: planted rule-world facts recorded 109 / 128 and 206 / 232 at visits 1 / 2 (kra27 128 / 128, 230 / 232) - the comma wording ('X, who is a child of M, lives nearby.') is split by the clause reader; locally the loss is about half (57 vs 110). Rules: 0 found at visit 1, both at visit 2, 0 FALSE (kra27: both from visit 1).
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 2: PLANTED bridge 0.46, C 0.31 (n 13); HELD BACK never written bridge 0.55, C 0.36 (n 11); **PURE FAMILIES bridge 0.25, C 0.00 (n 12)** - as kra27 at visit 2 (0.64 / 0.36; 0.33 / 0.00). Step 2 inside the chain still repeats the bridge in some examples ('Tarquira -> is a child of Zogalka; Zogalka is a child of Zogalka'). Visits 3-4 decide.
- Route D: first step 0.65, two-step 0.56, pure 0.48. Bits on read domains 2.0339 (kra27 2.0187 at visit 2; +0.015, about seed noise).

### cg27a_kra28 GPU, visit 3 (21 Nov 23:51)
- [RULE WORLD, THE TRUNK'S OWN CHAIN]: PLANTED bridge 0.88, C 0.59 (n 17); HELD BACK never written bridge 0.27, C 0.00 (n 11); **PURE FAMILIES never written: format 0.83, bridge right 0.33, C right 0.00 (n 12)** (kra27 visit 3: 0.10 / 0.00; kra26: 0.00 / 0.00).
- Examples: step 2 is now mostly a real second statement with a NEW name ('Quibrisal -> is a child of Quibelris; Quibelris is a child of Quibelor'; 'Quirasal -> ... Quiratev is a child of Quiedzo; Quiedzo is a c...' - a third step), not kra27's 'M is a child of M'; but the root is wrong (C 0.00), often a name sharing the bridge's first letters. One restates the subject ('Tozomon -> is a child of Ratozo; Tozomon is a child of Ratozo').
- Route D (premises asked directly): first step 0.70, two-step 0.62, **pure 0.58** (kra27 0.62 / 0.53 / 0.49) - six wordings raise premise recall in the canonical wording too. Facts taught 0.90; gate right when answered 1.00; bits on read domains 2.0152 (kra27 1.9977). No Traceback.

### THE SPELLED FORM DOES NOT COMPOSE THROUGH NEW BRIDGES; THE ARROW FORM DOES (synthetic, 22 Nov 00:10; compose_synth CS_SPELL 1 - 'A is a citizen of: lives in B; B lies in C.', kra27 / kra28's form - premises in 4 / 3 wordings, checkpoint of the clean run, 100 taught)
| form (varied premises) | NEW bridge B / C at 300 / 750 / 1,500 steps (n 78) | SEEN bridge C |
|---|---|---|
| arrow 'A is a citizen of B > C.' (clean run) | 0.82 / 0.55, 0.82 / 0.60, 0.85 / 0.67 | 0.68-0.91 |
| **spelled (kra27 / kra28)** | **0.77 / 0.12, 0.86 / 0.14, 0.86 / 0.12** | 0.55-0.86 |
| spelled + JEPA alignment of STEP 1's state to the premise's (weight 1.0) | 0.24 / 0.06, 0.44 / 0.08, 0.47 / 0.12 (taught C 0.45-0.89) | 0.23-0.41 |
- In the spelled form the FIRST step works for new bridges (B 0.86) and the SECOND fails (C 0.12-0.14): 'B lies in' inside the chain is answered only for bridges seen in taught chains - **exactly kra28 at visit 3** (pure bridge 0.33, C 0.00; step 2 a real statement with the wrong root). Aligning step 1 (which was not the problem) hurts.
- **kra29: the arrow form back (inf_rule_world_chainplant 1) with the six premise wordings** - the combination that composes in the synthetic world. Queued locally: JEPA alignment of the SECOND step's state (spelled and arrow, weight 0.3), alignment with one wording, two wordings.

### cg27a_kra28 GPU, visit 4 (22 Nov 00:16) - the synthetic spelled-form prediction, on the GPU
- [RULE WORLD, THE TRUNK'S OWN CHAIN]: PLANTED bridge 0.77, C 0.50 (n 26); **HELD BACK never written (new subjects): bridge 0.65, C 0.35 (n 20)**; **PURE FAMILIES never written: bridge right 0.56, C right 0.00 (n 18)** (kra26 visit 4: 0.17 / 0.08; kra27 visit 3: 0.10 / 0.00).
- **Six wordings fixed the FIRST step for new bridges (0.00-0.17 -> 0.56) and the premises (route D first step 0.91, two-step 0.84, pure 0.78; kra26 visit 4: 0.69 / 0.66 / 0.57); the spelled form's SECOND step fails** - examples 'Tozosen -> is a child of Risquiris; Loquiris is a child of Quihunis', 'Quimontar -> is a child of Riszohun' (step 2 dropped): step 2 does not even restate the bridge. The synthetic world's spelled form: B 0.86, C 0.12-0.14. kra29 (arrow form + six wordings) replaces it.
- Gate never-read 0.02; bits on read domains 1.9199 (kra26 visit 4 1.9202). No Traceback.

### JEPA ALIGNMENT OF THE SECOND STEP (synthetic, 22 Nov 00:30; CS_ALIGN 0.3, CS_ALIGN_STEP 2: the state where the chain predicts the root ('A is a citizen of B > ' / '...; B lies in ') pulled to the state in the bridge's own premise 'B lies in ' (stop-gradient), taught people only; premises in 4 / 3 wordings, same checkpoint)
| form | NEW bridge B / C at 300 / 750 / 1,500 steps (n 78) | taught C |
|---|---|---|
| arrow, no alignment | 0.82 / 0.55, 0.82 / 0.60, 0.85 / 0.67 | 1.00 |
| **arrow + step-2 alignment** | 0.67 / **0.62**, 0.82 / **0.85**, 0.82 / **0.78** | 0.93-1.00 |
| spelled, no alignment | 0.77 / 0.12, 0.86 / 0.14, 0.86 / 0.12 | 1.00 |
| spelled + step-2 alignment | 0.71 / 0.08, 0.77 / 0.08, 0.74 / 0.14 | 0.96-0.99 |
- **JEPA alignment of the second step adds to the arrow form: composition through new bridges 0.67 -> 0.78 at 1,500 steps (0.85 at 750)** - the C slot's state is keyed like the premise, so the premise's answer is retrieved for bridges never in a chain. It does NOT rescue the spelled form (0.08-0.14). Alignment with ONE wording per premise: running.
- **Alignment with ONE wording per premise (22 Nov 01:00; arrow form, CS_ALIGN 0.3 step 2, re-pretrained one-wording checkpoint cs_onew_s0.pt):** NEW bridge B / C 0.14 / 0.21, 0.26 / 0.28, 0.17 / 0.22 at 300 / 750 / 1,500 (one wording, no alignment: 0.12 / 0.15, 0.18 / 0.19, 0.19 / 0.18). A small gain (+0.04-0.09) - **alignment does not replace varied wordings**: with one wording the FIRST step fails (B 0.14-0.26), and step-2 alignment cannot fix it. Varied premises fix step 1; alignment boosts step 2. Queued: alignment of BOTH steps with one wording (CS_ALIGN_STEP 3).
- **TWO wordings are enough (22 Nov 01:35; arrow form, no alignment, CS_PREMW 2 x6, CS_PTW 2 x3, re-pretrained; premises 1.00):** NEW bridge B / C **0.83 / 0.74, 0.71 / 0.62, 0.79 / 0.77** at 300 / 750 / 1,500 (n 78) - as good as four / three wordings (0.82 / 0.55, 0.82 / 0.60, 0.85 / 0.67); one wording 0.12 / 0.15, 0.18 / 0.19, 0.19 / 0.18 (re-run from the saved checkpoint: identical). **The step is from one wording to two**: any second context keys the premise on its subject. kra28-kra30's six wordings are ample; for real text, a recorded fact needs only one rehearsal in a second frame.

### kra30 local check (22 Nov 02:05; three visits, 150 steps a task)
- No Traceback through visit 3. [JEPA STEP-2 ALIGNMENT] prints each visit: 38 / 82 / 108 planted chains; 149 alignment steps a visit; mean 1 - cosine 0.028 -> 0.009 (visit 2), 0.018 -> 0.011 (visit 3) - the term trains. Trunk's own chain 0.00-0.02 locally (as every build at this scale). Relational module 2 relations, 0 rules locally (as kra25-kra28 locally; the GPU finds them). **kra30 clear to run.**

### Overnight synthetic tests, part 2 (22 Nov 02:51-03:20; arrow form; NEW bridge B / C at 300 / 750 / 1,500 steps)
- **JEPA ALIGNMENT OF BOTH STEPS rescues ONE-wording premises (CS_ALIGN 0.3, CS_ALIGN_STEP 3; step 1 'A is a citizen of ' ~ 'A lives in ', step 2 'A is a citizen of B > ' ~ 'B lies in '):** 0.55 / **0.50**, 0.72 / **0.68**, 0.64 / **0.55** (n 78) vs no alignment 0.12 / 0.15, 0.18 / 0.19, 0.19 / 0.18 and step 2 only 0.14 / 0.21, 0.26 / 0.28, 0.17 / 0.22. Taught C 0.96 / 0.99 / 0.85 (a dip at 1,500). **Aligning the chain's steps to their premises' own states does what varied wordings do** - a label-free, general mechanism (no generated wordings needed), close to two wordings (0.62-0.77).
- **FEWER TAUGHT CHAINS (30, the GPU's dose), four wordings:** no alignment 0.66 / 0.34, 0.60 / 0.43, 0.63 / 0.30; **step-2 alignment 0.54 / 0.48, 0.58 / 0.52, 0.65 / 0.65** (n 91). With few chains alignment roughly doubles composition at 1,500 steps - it matters most at the rule world's dose (13-26 planted chains by visit 4).
- Chain teaching + step-2 alignment, 100 taught (CS_COTLATE_X run): 0.67 / 0.63, 0.81 / 0.78, 0.76 / 0.76 - false-memory and forgetting lines pending.
- **COST CHECKS with step-2 alignment (CS_COTLATE_X / CS_COTLATE_REV, seed 0, clean checkpoint, 1,500 steps; the 19 Nov no-alignment runs used 3,000 steps and another checkpoint, so the comparison is indicative):**
  - FALSE MEMORY: NOVEL names (never read) given a well-formed made-up chain **0.90** (19 Nov, no alignment: 0.93); abstain 0.00; confidence separates known from novel AUC 0.81 (0.83). **Alignment neither causes nor cures confabulation - the trunk's own chain still needs the gate** (as concluded 19 Nov: composite questions answered through the gated memory / margin gate).
  - FORGETTING on fixed pretraining windows: 0.435 -> 0.463 bits per byte (+0.028; 19 Nov no alignment +0.021). A small extra cost at most.
  - Premises only READ (never taught, never asked): chain C 0.04 / 0.08 / 0.07 / 0.16 for 1 / 2 / 4 / 8 reads (19 Nov 0.08 at 8 reads) - reading alone still does not make a premise chainable.
  - **BELIEF REVISION (move to a city in another country, new premise taught 600 steps): taught people NEW / STALE country 0.32 / 0.42 (19 Nov seed 0, no alignment: 0.19 / 0.65); never-taught 0.39 / 0.24 (0.14 / 0.52).** Aligned chains follow the new premise more often - the chain step is keyed on the premise, so changing the premise changes the conclusion. Partial, not complete: retraction and re-derivation of written composites stays necessary.

### Overnight synthetic tests, part 3 (22 Nov 03:55-05:35)
- **Four wordings + alignment of BOTH steps (seed 0):** NEW bridge B / C 0.74 / 0.68, **0.40 / 0.37**, 0.90 / 0.81 at 300 / 750 / 1,500; taught C 0.85 / 0.81 / 0.96 - unstable and costly to taught chains; no gain over step 2 alone (0.62 / 0.85 / 0.78, taught 0.93-1.00). **Both-step alignment is for one-wording premises only; kra31 not recommended.**
- **SECOND SEED (seed 1, re-pretrained, four / three wordings, premises 1.00):** arrow form, no alignment NEW bridge B / C **0.86 / 0.75, 0.92 / 0.84, 0.88 / 0.86** (n 85) - **the main finding replicates** (seed 0: 0.55 / 0.60 / 0.67). With step-2 alignment: 0.86 / 0.82, 0.71 / 0.71, 0.81 / 0.65, taught C 0.98 / 0.97 / 0.86 - **alignment does not help on seed 1** (+0.07 at 300, -0.21 at 1,500).
- **Real text with step-2 alignment (bridge_teach arrow, oracle relations, BT_ALIGN 0.3):** TEST never taught C first 0.150 / 0.150, exact 0.150 / 0.050, own bridge 0.150 / 0.175 at 1,500 / 3,000 (no alignment: 0.175 / 0.150, 0.100 / 0.050, 0.250 / 0.175). No change; the stand-in's real-text limit is unchanged.
- **Summary of the night:** robust on two seeds - varied premise wordings (two suffice) + the arrow form compose through new bridges (0.67-0.86); one wording or the spelled form do not (0.12-0.19). JEPA step-2 alignment: helps on seed 0 (0.67 -> 0.78; at 30 chains 0.30 -> 0.65), hurts late on seed 1 (0.86 -> 0.65), no change on real text; no false-memory cost, revision better. Both-step alignment rescues one-wording premises (0.15-0.19 -> 0.50-0.68) but destabilises varied ones. **Morning GPU: kra29 (the robust combination); kra30 (+ step-2 alignment) second.** Seed 1 at 30 chains with / without alignment: running.
- **Seed 1 at 30 taught chains (the GPU's dose), four wordings:** NEW bridge C no alignment 0.56 / 0.51 / 0.64; **step-2 alignment 0.56 / 0.77 / 0.63** at 300 / 750 / 1,500 (n 94; taught C 0.90-1.00). At the low dose alignment never hurts on either seed (seed 0: 0.34 / 0.43 / 0.30 -> 0.48 / 0.52 / 0.65; seed 1: equal at the end, +0.26 mid-way); it hurt only at 100 chains on seed 1. kra30 is a reasonable second run after kra29.

### cg27a_kra29 GPU, visit 1 (22 Nov 06:56; arrow form + premises in six wordings)
- No Traceback. Relational module: 242 names, 12 relations, 179 facts; **both rules found at visit 1, 0 FALSE** (kra28: at visit 2). Planted rule-world facts recorded 118 / 128 (kra28 109 / 128). Bits on read domains 2.0096 (kra27 2.0086).
- Trunk's own chain at visit 1: 0.00 everywhere (as every build); examples answer one name with no '>' ('Tevmizo -> Zofaris', 'Quisenhun -> Zonorbri'). Local check (two visits): no Traceback.

### cg27a_kra29 GPU, visits 2-4 (22 Nov 07:43): the arrow form is not taken up from the planted chains alone
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 2 / 3 / 4: PLANTED format 0.46 / 0.29 / 0.46, bridge 0.38 / 0.24 / 0.27, C 0.46 / 0.59 / 0.58 (direct 0.08 / 0.41 / 0.38 - planted composites mostly answered DIRECTLY); HELD BACK never written C 0.00 / 0.00 / 0.00 (n 11-13); **PURE FAMILIES never written: format 0.17-0.20, bridge 0.08-0.10, C 0.00 / 0.10 / 0.00 (n 10-12).** Examples: single names ('Quibrisal -> Quibelor'), drift ('Quihunis and Technology and Technology').
- Premises are strong (route D first step 0.64 / 0.68 / 0.86, two-step 0.51 / 0.59 / 0.73, pure 0.46 / 0.56 / 0.69 - the best yet), but chain FORMAT uptake is low: kra29 teaches route J's composites DIRECTLY (inf_distill_chainform 0 since kra26), so arrow chains come only from 13-26 planted sentences. The synthetic world taught 100 chains intensively. kra23-kra25 taught route J writes in the chain form (format and composition for new subjects 0.12-0.29) but with one-wording premises. **Neither GPU build had both: kra33 = kra29 + inf_distill_chainform 1.** kra32 (real text, seed 5) now builds on kra33.
- Gate never-read: 0.51 at visit 2 (!), 0.03, 0.02 at visits 3-4. Facts and bits normal (2.0326 / 2.0060). No Traceback.

### kra32 local check (real-text chains; 22 Nov 08:30; two visits, inf_premises_visits 2 locally)
- No Traceback. Chains built from the record store's reading: visit 1 276 (TAUGHT 227, TEST 22), visit 2 832 (TAUGHT 400, TEST 75; 0.17 of test chains have both premises gold-labelled as containment), places only ('1 Carlton Gardens in London > UK.', 'Talyllyn Railway in Wales > United Kingdom.'). Rows: 588 / 1,164 premise rows (two wordings), 227 / 400 chain rows; the rows' loss falls 1.785 -> 0.712 over visit 2's 149 steps.
- [REAL-TEXT CHAINS, THE TRUNK'S OWN | visit 2] prints: 0.00 everywhere locally (the 150-step stand-in recalls neither premises nor chains; guessing the commonest taught C 0.34 on taught, 0.05 on test). **kra32 clear to run** (the GPU is the test).

### cg27a_kra33 GPU, visit 3 (22 Nov 08:47): chain-form teaching starts
- [RELATIONAL MODULE WRITES | visit 3] 93 written (precision 0.99) -> 59 taught, 34 held back; CHAIN FORM 93.
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 3 (kra29 visit 3): PLANTED format 0.59 (0.29), bridge 0.29, C 0.47; HELD BACK never written format **0.64** (0.18), bridge 0.18 (0.00), C 0.00; **PURE FAMILIES format 0.42 (0.20), bridge 0.08, C 0.00 (n 12)**. Examples now in the arrow form: 'Tozomon -> Tozodra > Quizotev', 'Pelsenzo -> Ratozo > Ratozo'. Format taken up; the bridge and root not yet (one visit of chain teaching).
- Route D: first step 0.62, two-step 0.48, pure 0.46. Gate never-read 0.40 (visit 2), **0.16 (visit 3; kra29 0.03)**. No Traceback.

### SLOW CHANNELS, ONE WORDING (synthetic, 22 Nov 08:50; CS_SLOW 0.1, 32 channels, slowness within a sentence + SVReg variance floor, during pretraining and teaching; after Ren, Wang, Hoang, Kuang & LeCun 2026)
- Pretraining unchanged: 0.429 bits per byte at step 6,000 (one wording without: 0.428); premises 1.00.
- **NEW bridge B / C at 300 / 750 / 1,500: 0.29 / 0.22, 0.35 / 0.29, 0.32 / 0.32** (n 78) vs one wording alone 0.12 / 0.15, 0.18 / 0.19, 0.19 / 0.18; two wordings 0.83 / 0.74, 0.71 / 0.62, 0.79 / 0.77. **A partial rescue at no cost: +0.07 to +0.14 on C, the first step roughly doubled.** Weight 0.3 running.
- **Slow channels at weight 0.3 (one wording, 22 Nov 09:20):** pretraining 0.429 bits per byte (no cost); NEW bridge B / C **0.18 / 0.22, 0.14 / 0.17, 0.21 / 0.23** - no better than one wording alone (0.15-0.19) and below weight 0.1 (0.22-0.32). **No dose-response: slow channels are at best a weak, uncertain help (n 78, about 0.05 standard error); not a candidate for the GPU.** Varied wordings (two suffice) stay the fix.

### cg27a_kra33 GPU, visits 4-5 (22 Nov 09:22): THE TRUNK COMPOSES FOR NEW SUBJECTS; the second step fails through bridges never in a chain
- [RULE WORLD, THE TRUNK'S OWN CHAIN] visit 4 / 5: PLANTED format 0.46 / 0.55, bridge 0.35 / 0.48, C 0.69 / 0.84; **HELD BACK, never planted, never written (new subjects, known bridges): format 0.82 / 0.85, bridge 0.53 / 0.46, C right 0.41 / 0.31 (n 17 / 13; direct 0.00 / 0.00)** - the best clean composition by the trunk yet (kra28 0.35, kra29 0.00); **PURE FAMILIES never written: format 0.46 / 0.89, bridge 0.08 / 0.67, C 0.08 / 0.00 (n 13 / 9).**
- Examples, pure families: 'Raquiul -> Quiulsal > Quigalcor', 'Tartarqui -> Quiulsal > Quigalcor', 'Quivinzo -> Quiquihun > Quigalcor' - step 1 now right (0.67), step 2 answers a common root for bridges never seen in a chain. Route D (asked step by step): first step 0.92, two-step 0.82, pure 0.74 - the premise B -> C is known when asked alone.
- Gate recovered: never-read 0.02 / 0.01 (t 0.84 / 0.87). Bits on read domains 1.9191 (visit 4). No Traceback.
- **kra34 = kra33 + JEPA step-2 alignment (inf_rule_world_align 0.5) + gate floor 0.8.**

### cg27a_kra32 GPU (seed 5, real-text chains), visits 1-4 (22 Nov 09:29)
- Real-text chains on the GPU: 848 / 2,088 / 3,345 chains at visits 1-3 (TAUGHT 400, TEST 80 / 150 / 150; 0.16-0.23 of test chains gold-labelled). Rows' loss 1.147 -> 0.108 (visit 2), 0.826 -> 0.083 (visit 3).
- **[REAL-TEXT CHAINS, THE TRUNK'S OWN | visit 3]: TAUGHT (80) format 0.93, bridge 0.90, C 0.89 (given the right bridge 0.99), premises alone 0.94 / 0.70 - the trunk learns real chains and premises. TEST, bridge never in a taught chain (150): format 0.40, bridge 0.03, C 0.03 (given the right bridge 0.25; commonest-C guess 0.07; direct 0.00); premises asked alone: first hop 0.55, second hop 0.69.** Composition through unseen bridges on real text: not yet. The first hop is known in its own wording (0.55) but not inside 'A in ' (0.03) - many first premises are worded 'region of', 'located in', 'province of', never 'in'. Prepared: inf_rchain_ownw (chains start with the first premise's own wording).
- Rule world, seed 5, visit 4: PLANTED C 0.65; HELD BACK never written bridge 0.41, C 0.09 (n 22); PURE FAMILIES format 0.95, bridge 0.58, C 0.05 (n 19) - the same step-2 failure as kra33 (seed 4), weaker for new subjects (0.09 vs 0.41).
- Gate never-read: 0.07 (visit 3, t 0.94), 0.10 (visit 4, t 0.77) - the pooled threshold fell below 0.8; kra34's floor (0.8) addresses it.

### cg27a_kra32 GPU, visits 4-5 (22 Nov 10:00): the real-text test was confounded by its prompt
- [REAL-TEXT CHAINS, THE TRUNK'S OWN] visit 4 / 5: TAUGHT C 0.81 / 0.85 (premises alone 0.91-0.96 / 0.75-0.82); **TEST format 0.21 / 0.20, bridge 0.01 / 0.01, C 0.01 / 0.01** (premises alone 0.75 / 0.81, 0.71 / 0.67). Examples: 'Amargosa River -> Nye County', 'Nuffield Theatre -> Southampton' - the right first premise, then a full stop. **Cause (design): the chain was asked as 'A in ', the same prompt as a first premise worded 'in', and every test premise was taught as 'A in B.' - so for test heads the trunk answers the premise, as taught. (inf_rchain_ownw would have made it universal; replaced by inf_rchain_crel - the composite gets its own relation, 'A is within B > C.', as the rule world's 'grandchild' differs from 'child'.)**
- Rule world, seed 5, visit 5: PLANTED C 0.80; HELD BACK never written bridge 0.48, **C 0.22** (n 23); PURE FAMILIES format 0.92, bridge 0.38, C 0.00 (n 13). Gate never-read 0.02 (visit 5, t 0.90).
- **kra35 = kra34 + real-text chains as 'A is within B > C.' + JEPA step-2 alignment on them; seed 5.**

### cg27a_kra35 GPU (seed 5; kra34 + real-text chains 'A is within B > C' + step-2 alignment), visits 1-3 (22 Nov 11:17)
- **[REAL-TEXT CHAINS, THE TRUNK'S OWN | visit 3]: TAUGHT (80) format 0.96, bridge 0.96, C 0.94; TEST, bridge never in a taught chain (150): chain format 0.97 (kra32: 0.20 - the prompt fix works), bridge right 0.09, C right 0.03 (given the right bridge 0.29; commonest-C guess 0.07; direct 0.01); premises asked alone 0.66 / 0.51.** Test heads now chain, but step 1 inside the chain is mostly wrong ('Amargosa River -> India > Andhra Pradesh' for Nye County > U.S.; 'Permesta rebellion -> Indonesia > Andhra Pradesh' - bridge right, root a taught one). Rows' loss 0.767 -> 0.064; alignment 1 - cos 0.046 -> 0.035.
- Rule world, seed 5, visit 3: PLANTED C 0.38; HELD BACK C 0.06 (n 16); PURE bridge 0.00, C 0.00 (n 16). JEPA step-2 alignment 0.069 -> 0.009 within visit 3 (399 steps; 64 planted chains next).
- Gate: visit 3 t 0.84 (floor held), never-read 0.07 (ungated 0.65). No Traceback.

### Real-text stand-in with the composite relation (22 Nov 11:55; bridge_teach BT_FORMAT arrow, BT_CREL 'is within', oracle relations P131/P17/P30/P361, BRIDGE arm; same 237 chains: TRAIN 197, TEST 40)
- TEST (bridges never in a train chain) at 1,500 / 3,000 steps: **chain format 0.925 / 0.925** (with 'A in ': 0.275 / 0.625), **own bridge right 0.300 / 0.300** (0.250 / 0.175), **C first among 10 0.250 / 0.250** (0.175 / 0.150; direct teaching 0.050 / 0.075; chance 0.10), C exact 0.150 / 0.100 (0.100 / 0.050). TRAIN 0.97-1.00.
- **The prompt clash was part of the stand-in's weak real-text result: with the composite's own relation, composition through unseen bridges rises to 0.25 (2.5x chance, 3-5x direct teaching; n 40, about 0.07 standard error).** Modest and on the small stand-in; examples 'Fraser River is within British Columbia > Canada', 'Museo del Prado is within Madrid > England' (bridge right, root wrong).
- **kra35 visit 4 (11:55):** REAL-TEXT TEST (150): format 0.94, **bridge right 0.34** (visit 3: 0.09), C right 0.04 (**given the right bridge 0.10**; guess 0.05); premises alone 0.79 / 0.77. Step 1 improving, step 2 failing: 'Lateran University -> Rome > Georgia' (Rome > Gianicolo), 'Permesta rebellion -> Indonesia > Andhra Pradesh'. TAUGHT C 0.86. Rule world, seed 5: HELD BACK C 0.14 (n 14; kra32 visit 4 0.09); PURE bridge 0.10, C 0.10 (direct 0.10; n 20; kra32 visit 4: bridge 0.58, C 0.05). Gate 0.06 (t 0.84). Step-2 alignment trains (0.045 -> 0.010) but has not fixed step 2 on seed 5 by visit 4.

### cg27a_kra35 GPU, visit 5 (22 Nov 12:13): FIRST COMPOSITION THROUGH BRIDGES NEVER IN A CHAIN (rule world, seed 5, with JEPA step-2 alignment)
- [RULE WORLD, THE TRUNK'S OWN CHAIN | visit 5]: PLANTED C 0.76; HELD BACK never written: format 0.96, bridge 0.39, **C 0.26** (n 23; direct 0.09); **PURE FAMILIES, NEVER WRITTEN: format 1.00, bridge right 0.36, C right 0.36 (n 11; direct 0.00)** - kra32 (seed 5, same settings without alignment) at visit 5: pure bridge 0.38, C 0.00; held back C 0.22. When the bridge is right the root is right. Examples: 'Razolo -> Drazo > Monzocor', 'Tevulqui -> Zofapel > Zorapel', 'Ormonqui -> Zorisbri > Drazoto' (varied roots, no single repeated guess). n 11 (about 0.14 standard error): one visit, to be confirmed at visits 6-8 and on seed 4 (kra34 disconnected at visit 2).
- Real text, visit 5: TEST format 0.96, bridge 0.30, C 0.04 (given the right bridge 0.13; guess 0.07); TAUGHT C 0.84. Step 2 still fails on real text: 'Barangay Tinago -> China > Liaoning', 'Mont Perdito -> London > UK' (roots that are taught chain answers).
- Gate never-read 0.05 (t 0.91). No Traceback.
- **Real-text stand-in, composite relation + JEPA alignment of BOTH steps (BT_ALIGN 0.3, BT_ALIGN_STEP 3; 22 Nov 12:35):** TEST at 1,500 / 3,000: C first among 10 **0.375** / 0.275 (no alignment 0.250 / 0.250), C exact **0.275** / 0.150 (0.150 / 0.100), own bridge 0.325 / 0.250 (0.300 / 0.300); TRAIN 0.93-0.95. A gain at 1,500 steps that fades by 3,000 (n 40, about 0.07 standard error): suggestive, not established.
- **kra35 visit 6 (12:52): CONFIRMED on a second visit.** PURE FAMILIES, NEVER WRITTEN: format 0.94, bridge 0.59, **C right 0.35 (n 17; direct 0.00)** - visits 5-6 together 10 / 28 = 0.36. HELD BACK never written: C 0.32 (n 34; direct 0.03). PLANTED C 0.88. Examples: 'Briedqui -> Zoraed > Kaquiul', 'Hunquiul -> Mizolo > Loquian' (siblings share bridge and root). Real text: TEST C 0.09 (given the right bridge 0.26; guess 0.07), bridge 0.31; TAUGHT C 0.86. Gate never-read 0.01 (t 0.91). Bits on read domains 1.8622 (visit 5), 1.9103 (visit 6). No Traceback.

### Root novelty does not block step 2 (synthetic, 22 Nov 13:10; 150 countries, four wordings, arrow form, 100 taught, re-pretrained)
- NEW bridge, C at 300 / 750 / 1,500: **root SEEN as an answer in a taught chain (n 34) 0.56 / 0.50 / 0.56 | root NEVER an answer (n 43) 0.44 / 0.44 / 0.49**; seen bridge 0.78-0.91; never taught overall 0.54-0.60.
- **Hypothesis rejected (mostly):** roots never written after '>' compose nearly as well (a 0.07-0.12 gap, within noise at n 34-43). Real-text step 2 (C given the right bridge 0.10-0.26 in kra35) fails for another reason - candidates: a much larger answer set, noisy and multi-word names, fewer visits of teaching (it is still rising: 0.10 -> 0.13 -> 0.26 at visits 4-6).
- **kra35 visit 7 (13:21): holds with a larger sample.** PURE FAMILIES, NEVER WRITTEN: format 0.98, bridge 0.39, **C right 0.28 (n 46; direct 0.00)**; visits 5-7 together 23 / 74 = **0.31**. HELD BACK never written: **C 0.40** (n 50; direct 0.02). PLANTED C 0.77. Real text TEST: bridge 0.35, C 0.11 (given the right bridge 0.23; commonest-C guess 0.15 - the baseline rose with more chains). Gate never-read 0.02 (t 0.92). Bits on read domains 1.8802. No Traceback.

### THE ARROW PREMISE FRAME (synthetic, 22 Nov 13:45; 150 countries, four wordings + every city's premise also as '<B> > <C>.' x3 in pretraining, CS_ARROWPREM 3, re-pretrained; arrow chains, 100 taught, no alignment)
- NEW bridge B / C at 300 / 750 / 1,500: **0.84 / 0.73, 0.77 / 0.73, 0.81 / 0.81** (without the frame, same world: 0.88 / 0.49, 0.79 / 0.47, 0.77 / 0.52). Root NEVER an answer: **0.65 / 0.70 / 0.77** (without: 0.44 / 0.44 / 0.49); root seen 0.82 / 0.76 / 0.85. Taught 1.00.
- **The premise written in the chain's own step-2 context ('B > C') lifts step 2 by about 0.25-0.30** - for roots never an answer most of all. Not the root-vocabulary hypothesis (which failed), but the same lesson as the wordings: a premise is retrieved in the contexts it was learned in.
- **kra36 = kra35 + arrow premise frames (rule world 'M > Z.' x2, inf_rule_world_premarrow 2; real text 'B > C.', inf_rchain_arrowprem 1), seed 4.**

### GATED CHAINS (synthetic, 22 Nov 13:30; 20 countries, CS_COTLATE_X, step-2 alignment 0.3, 100 taught chains)
- Chains: new bridge C 0.63 / 0.78 / 0.76 at 300 / 750 / 1,500 steps; taught 0.92-1.00.
- FALSE MEMORY on made-up chains: after chain teaching, NOVEL (never read) names given a well-formed chain to a known city and country **0.90** (held-out known people 0.93) - chain teaching makes the trunk invent chains. Confidence (mean byte log-prob of its own chain) separates known from novel AUC 0.81 (0.54 before teaching). **GATED (t keeps 75% of right known chains): novel names given a chain 0.11, known answered 0.58, right when answered 1.00.**
- Premises only read (never taught or asked): chain C 0.04 / 0.08 / 0.07 / 0.16 at 1 / 2 / 4 / 8 reads - reading alone does not build chains. Forgetting 0.435 -> 0.463 bits/byte.
- **The confidence gate that fixed false memory for single facts also works for chains: invention falls from 0.90 to 0.11 while every answered chain is right.**

### kra35 VISIT 8 (GPU, seed 5; 22 Nov 13:49)
- RULE WORLD, THE TRUNK'S OWN CHAIN, PURE FAMILIES (bridge never in a chain, never written): **visit 8 C right 0.40 (n 60), bridge right 0.52, direct 0.00**. Visits 5-8: 0.36 (11), 0.35 (17), 0.28 (46), 0.40 (60) - pooled **0.35 (n 134)**, direct 0.00 throughout. Held back 0.33 (46). Planted 0.83 (direct 0.42).
- Route D (two steps prompted), pure families 0.93 (n 67) vs direct 0.00; route J 0.98 (n 292).
- REAL-TEXT CHAINS: taught C 0.91 (given the bridge 0.95). **TEST (bridge never in a taught chain): bridge right 0.35 -> 0.49, C right 0.11 -> 0.19 (commonest-C baseline 0.14), given the right bridge 0.23 -> 0.35**, direct 0.00. Premises asked alone 0.89 / 0.89. Rising, but C only 0.05 above the baseline: step 2 on real text is still the gap kra36's arrow frames target.
- Gate: never-read questions given a stored name 0.03 (ungated 0.57); known answered 0.75, right when answered 0.98.

### kra34 VISIT 4 (GPU, seed 4 restart; 22 Nov 13:51)
- RULE WORLD, THE TRUNK'S OWN CHAIN: **HELD BACK (never planted, never written by route J) C right 0.44 (n 18), bridge 0.44, direct 0.00; PURE FAMILIES C right 0.14 (n 14), bridge 0.29, direct 0.00.** Same visit on seed 5 (kra35): held back 0.14 (n 14), pure 0.10 (n 20). Seed 4 is ahead of seed 5 at visit 4 - the composition through bridges never in a chain is appearing on a second seed; the verdict needs visits 5-6 (n is small).
- Route D pure families two-step 0.83 (n 54) vs direct 0.00. Gate: never-read given a stored name 0.06 (ungated 0.68).

### JEPA STEP-2 ALIGNMENT FORMS (synthetic, 22 Nov 13:55; 150 countries, same checkpoint, alignment 0.3, step 2, 100 taught arrow chains; new bridge n 77, root seen 34 / never an answer 43)
| form | NEW bridge C at 300 / 750 / 1,500 | root never an answer | taught C |
|---|---|---|---|
| none | 0.49 / 0.47 / 0.52 | 0.44 / 0.44 / 0.49 | - |
| cosine (current GPU builds) | 0.61 / 0.69 / 0.53 | 0.56 / 0.74 / 0.58 | 0.96-0.97 |
| **contrastive (InfoNCE over the batch)** | **0.68 / 0.77 / 0.64** | **0.79 / 0.77 / 0.65** | 1.00 |
| answer-space KL | 0.55 / 0.35 / 0.60 | 0.58 / 0.37 / 0.58 | 0.63-0.79 (damaged) |
| residual (centred, the counterfactual world-model rule; 15:35) | 0.57 / 0.64 / 0.71 | 0.63 / 0.63 / 0.74 | 0.92-0.97 |
- **Contrastive alignment is best at every step count: +0.12 to +0.30 over none, +0.07 to +0.11 over cosine; for roots never an answer +0.16 to +0.35 over none.** Fits the maths note: cosine on raw states is dominated by shared directions; the negatives force the step-2 state to carry the bridge's identity. Answer-space KL harms the taught chains. Residual mode and alignment on top of the arrow frame queued.

### kra34 VISIT 5 (GPU, seed 4; 22 Nov 14:05) - THE SEED-5 RESULT REPLICATES
- RULE WORLD, THE TRUNK'S OWN CHAIN: **HELD BACK C right 0.62 (n 13), bridge 0.62, direct 0.00; PURE FAMILIES C right 0.58 (n 12), bridge 0.50, direct 0.00.** Visits 4-5 pooled: held back 0.52 (16 / 31), pure families 0.35 (9 / 26), direct 0.00. Seed 5 (kra35) visits 5-8: pure families 0.35 (n 134). Planted 0.84 (direct 0.35).
- Caveat: the pure-family examples at visit 5 mostly end in the same root (Zototo), C right (0.58) is above bridge right (0.50), and n is small - some C may be the family's common root rather than a composed answer; the bridge-right figure (0.50, direct 0.00) is the cleaner evidence.
- Route D pure families two-step 0.88 (n 65) vs direct 0.02. Gate: never-read given a stored name 0.02 (ungated 0.60).
- **Composition by the trunk's own written chain through bridges never in a chain is now seen on two seeds (4 and 5), with the direct answer at 0.00 on both.**

- RESIDUAL (15:35): between cosine and contrastive on average (new bridge mean 0.64 vs cosine 0.61, contrastive 0.70); the only form still rising at 1,500 steps (0.71, best there; contrastive 0.64). Contrastive stays the choice for kra37; one seed, about +/-0.055 per point.

### ARROW PREMISE FRAME + ALIGNMENT (synthetic, 22 Nov 15:50; 150 countries, arrow frame x3 checkpoint, alignment 0.3 step 2)
| | NEW bridge C at 300 / 750 / 1,500 (mean) | root never an answer | taught C |
|---|---|---|---|
| arrow frame alone | 0.73 / 0.73 / 0.81 (0.76) | 0.65 / 0.70 / 0.77 | 1.00 |
| arrow + contrastive | 0.79 / 0.78 / 0.71 (0.76) | 0.77 / 0.77 / 0.67 | 1.00 |
| arrow + cosine | 0.71 / 0.68 / 0.71 (0.70) | 0.65 / 0.70 / 0.60 | 0.93-0.99 |
- **The gains do not stack on average**: with the arrow frame, contrastive is faster early (+0.06 / +0.05) and lower at 1,500 (-0.10); cosine is slightly worse than no alignment. Both fixes address the same failure (the second premise not retrieved in the chain's context), and the frame alone reaches most of it.
- **For the GPU: kra37 (arrow + contrastive 0.3) is at least as good as the frame alone and better than kra36 (arrow + cosine 0.5)** - run kra37 first if only one can run. One seed, about +/-0.05 per point.

### kra36 VISITS 1-6 (GPU, seed 4, arrow premise frames; 22 Nov 16:09)
- REAL-TEXT CHAINS, TEST (bridge never in a taught chain, n 150), kra36 vs kra35 (no frame, seed 5) at the same visit:
  | visit | bridge right | C right | C given the right bridge | commonest-C baseline | GOLD-TRUE C right |
  |---|---|---|---|---|---|
  | 3 | 0.13 (0.09) | 0.06 (0.03) | **0.37** (0.29) | 0.07 | 0.12 (n 24) |
  | 4 | 0.29 (0.34) | **0.15** (0.04) | **0.50** (0.10) | 0.05 | 0.21 (n 28) |
  | 5 | 0.26 (0.30) | 0.09 (0.04) | **0.33** (0.13) | 0.07 | 0.17 (n 23) |
  Direct 0.00-0.01. Taught 0.88-0.93. Premises asked alone, test heads: first hop 0.60-0.69, second hop 0.43-0.71.
- **The arrow frame lifts real-text step 2 (C given the right bridge) from 0.10-0.29 to 0.33-0.50 at the same visits, and C to 1.3-3x the baseline (pooled visits 3-5: 0.10 vs baseline 0.06; kra35 0.04).** Different seeds, so partly confounded.
- **The bottleneck has moved to STEP 1 on real text: the bridge is right 0.26-0.29 inside the chain although the first premise asked alone is right 0.60-0.69.** The chain prompt 'A is within ' does not retrieve the premise learned as 'A in B' / 'A, B' - the step-1 analogue of what the arrow frame fixed for step 2. Next: first premises also taught in the composite frame ('A is within B.', true by transitivity of containment).
- RULE WORLD, THE TRUNK'S OWN CHAIN: pure families visit 5 C right 0.42 (n 12), bridge 0.33; held back 0.55 (n 11), direct 0.00 (kra34, same seed without the frame: 0.58 / 0.62). Route D pure families two-step 0.98 (n 65) vs direct 0.00. Gate: never-read given a stored name 0.01 (ungated 0.60).

### kra36 VISITS 6-7 (GPU, seed 4; 22 Nov 16:45)
- REAL-TEXT TEST (n 150): visit 6 bridge 0.32, C 0.13 (given the right bridge 0.38; baseline 0.07); visit 7 bridge 0.26, C 0.10 (given the right bridge 0.31; baseline 0.15). Premises asked alone (test heads): first hop 0.78 / 0.68.
- Visits 3-7: C given the right bridge mean **0.38 (kra35, no frame: 0.20)**; C mean 0.11 vs baseline mean 0.08; bridge right flat at 0.26-0.32 against the first premise alone at 0.60-0.78. **Step 2 doubled by the arrow frame and holding; step 1 is the bottleneck (kra38's paraphrased wordings target it).**
- RULE WORLD, own chain: pure families visit 6 0.11 (n 28), visit 7 0.35 (n 20); visits 5-7 pooled **0.25 (15 / 60)**, held back 0.40 (25 / 62); direct 0.00 for pure families (held back 0.03-0.10). Route D pure families two-step 0.86-0.87 vs direct 0.00. Gate: never-read given a stored name 0.02.

### realrules (offline, real records; 22 Nov 17:05; 20,000 documents, 31,081 facts in the 30 commonest raw wordings, 32,189 names)
- RULES on raw wordings: only one above the shuffled null ('on' o 'and in' => 'and in' 0.135, 5 / 37; null best 0.111). **Raw connecting words ('in', 'on', 'of', 'by', 'until') are too ambiguous to compose symbolically** - the real-text chains work because they are filtered to LOC containment first.
- ROUTE J on real records: one-hop TRAIN 0.81, HELD-OUT (never trained) 0.010 (chance 0.00003). **Real records are about one fact per name (31k facts, 32k names): route J memorises but has almost no neighbours to generalise from.** Composites mostly noise (e.g. 'Gwadar Port -> Spain'), gold containment 0.73 among the 52 pairs that have any gold relation (selection).
- RETRACTION proxy (gold-labelled vs unlabelled held-out records): rest-of-knowledge log p AUC 0.538, documents 0.522 - near chance; the proxy is weak (unlabelled is not wrong). The planted-error test (tools/retract_jepa.py, running) gives a clean ground truth and reports facts whose head has other facts separately.

### kra38 VISITS 2-3 (GPU, seed 4; 22 Nov 17:16) - EARLY WARNING
- Contrastive alignment converges (1.02 -> 0.07 at visit 2; 0.26 -> 0.15 at visit 3); paraphrased premise rows in (4,123 -> 5,117).
- REAL-TEXT TEST visit 3: bridge 0.17 (kra36 0.13), C 0.03 (kra36 0.06), **C given the right bridge 0.04 (kra36 0.37)**; premises asked alone 0.53 / 0.37 (kra36 0.60 / 0.43). Taught 0.90.
- RULE WORLD own chain visit 3: chain format 0.08 held back / 0.08 pure (kra36 same seed 0.46 / 0.88; kra34 0.42 / 0.25).
- Step 2 lost on real text and the chain format slower in the rule world. Suspect: the contrastive loss at 0.3 is about 15x the cosine term's gradient scale (InfoNCE ~1 vs 1 - cos ~0.04) in a loss shared with many other rows. Visits 4-5 decide. Contingency built: **kra39 = kra36 + paraphrased wordings, cosine alignment** (separates the two changes).

### JEPA RETRACTION ON PLANTED ERRORS (offline, real records; 22 Nov 17:45; tools/retract_jepa.py; 31,081 facts, 5% corrupted with frequency-matched tails, 4-fold x 2 seeds)
- AUC corrupted vs clean (0.5 = chance): S1 surprise 0.44 | S2 normalised 0.46 | S3a latent path 0.44 | S3b symbolic paths 0.50 | S4 confident contradiction 0.51 | rank sum 0.46 | baselines: documents 0.50, tail prior 0.45, head degree 0.49. Same on heads with other facts (12,104). Real-data proxy (gold vs unlabelled) 0.49-0.55.
- **NEGATIVE: no signal detects planted errors on real records.** Surprise is even slightly inverted (frequency-matched wrong tails are common names, easier to predict than true rare ones). Cause, with realrules: about one fact per name per wording - route J predicts held-out real facts 0.01 - so there is no 'rest of knowledge' to contradict a fact. JEPA surprise can only retract where knowledge is dense and redundant. Next: (1) relation merging (denser relations; tools/relmerge.py tonight), (2) the mechanism in a dense world (two views of the same fact).

### LATENT SEARCH vs WRITTEN CHAINS (offline, synthetic; 22 Nov 17:45; tools/latent_search.py; person -> city -> region -> country -> continent, two wordings, 945 names)
- LATENT (route J on the trunk's codes, programs found by search over 340 operator sequences scored on endpoints only): 'citizen of' -> **lives o city_in o region_of found, right 1.00 on 100 endpoints (null, answers shuffled, 0.15)**; HELD-OUT people in cities in no taught example (79) **1.00**; margin gate: answered 0.70, right when answered 1.00; **never-read names given an answer 0.00**. 'continent' (4 hops, never taught to the trunk) from 20 endpoints: **lives o city_in o region_of o country_in found (null 0.50), held-out 1.00.**
- TRUNK (3-hop arrow chains taught for 100 people; premises recalled 1.00): taught 1.00, **NEVER TAUGHT 0.01-0.03** (format 0.41-0.68, bridge 0.08-0.10, region 0.00-0.01); never-read names given a well-formed 3-hop chain **0.77** (invented).
- **Inference as search over latent operators (LeCun's direction) solves 3- and 4-hop composition from endpoints alone, abstains on unknown names, and finds a relation never taught; the autoregressive trunk does not generalise 3-hop chains here and invents chains for unknown names.** Caveat: route J's operators are fitted on the premises (the random-code control in tonight's pure-JEPA test checks how much is a learned table).

### kra38 VISITS 4-6 (GPU, seed 4; 22 Nov 18:25) - NEGATIVE for its two changes
- REAL-TEXT TEST, kra38 vs kra36 (same seed): bridge 0.36 / 0.25 (0.29 / 0.26) | C 0.05 / 0.07 (0.15 / 0.09) | **C given the right bridge 0.11 / 0.27 (0.50 / 0.33)** | baseline 0.05 / 0.07 | premises alone, first hop 0.67 / 0.65.
- **Paraphrased containment wordings did not lift step 1** (bridge within noise of kra36); **contrastive alignment on the GPU is worse for step 2** than cosine (visits 3-5 C given the bridge mean 0.14 vs 0.40), unlike the synthetic test (where it was the only extra loss).
- RULE WORLD own chain, pure families visits 4-6: 0.00 (12), 0.62 (8), 0.08 (24) - pooled 0.16 (kra36 same visits 0.17); held back 0.27 / 0.46 / 0.18. Route D pure families 0.78-0.94 vs direct 0.00.
- TEST EXAMPLES show the readout undercounts and the records are noisy: 'Vancouver International Airport -> British Columbia > Canada' (true) scored wrong against the record's 'Richmond > North Yorkshire'; records such as 'Southampton > Suffolk County', 'Manchester > New England' (name ambiguity). Many wrong bridges are frequent places ('New York City', 'Canada') - guessing.
- Next: **kra39 = kra36 on seed 5 + an ANY VALID CHAIN readout** (written chain valid when both links are recorded in any wording; written conclusion checked against Wikidata containment).

### SVReg AS THE DREAM SELECTOR'S REGULARISER (local, kra38 preset, 22 Nov 18:40; vs SIGReg, same 50-minute limit)
- No errors. Dream mixing identical on the local stand-in: 644 spliced pairs tested, 0 accepted in both; known linked pairs pass 2/2 (weaker-side gain +4.3 vs +3.9 bits). The SVReg-trained selector's embeddings sit much closer to Gaussian by the SIGReg statistic (0.38-0.79 vs 3.6-10.1).
- **Inconclusive locally** (the local stand-in accepts no dreams either way); harmless; a GPU comparison is needed to judge it, low priority beside real text and retraction.
- NOTE (19:00): the GPU run named kra39 (started 18:24, seed 4) used the FIRST kra39 definition (kra36 + paraphrased wordings, COSINE alignment) - it isolates the paraphrases from contrastive alignment. The redefinition (kra36 on seed 5 + ANY VALID CHAIN readout) is now **kra40**; kra39's files are restored to what ran.

### REASONING WITHOUT THE LANGUAGE MODEL (offline, synthetic; 22 Nov 19:00; tools/jepa_reason.py; five-level world, 945 names)
- ARM D, PURE JEPA (bidirectional byte encoder trained ONLY by masked-word latent prediction, InfoNCE in the batch, EMA target; hops = the predictor on masked text, snapped to name codes; no next-byte objective, no fitted operators, no labels): **one-hop recall 0.01 / 0.01 / 0.04 / 0.12 / 0.11 at 1k-5k steps** (the causal trunk recalls the same premises 0.97-1.00 by 2k steps); 3 hops 0.00 (null 0.01), 4 hops 0.00; margin gate useless; never-read names given an answer 0.55. Loss flat about 1.0 from 1k steps.
- ARM B, ROUTE J on the JEPA encoder's codes: 3 hops held-out 1.00, gated 0.75 answered right 1.00, never-read 0.00, 4 hops 1.00. ARM C, ROUTE J on RANDOM codes: the same (1.00; gated 0.82 / 1.00; never-read 0.00; 4 hops 1.00).
- **Two findings: (1) route J's reasoning needs no knowledge in the codes - its fitted operators are the fact table, and latent search composes them (fully separable from the language model); (2) a JEPA trained only by latent prediction on these sentences does NOT store arbitrary facts (who lives where) - it learned within-batch discrimination (coarse categories), not 945-way identity.** In this design the next-byte trunk (or an explicit relational memory) stores facts; the JEPA side reasons over them - LeCun's hybrid, not a JEPA-only replacement. One configuration (word masks, 48-sentence batches, 5k steps); recall was still rising slowly.

### kra39 VISIT 3 (GPU, seed 4; paraphrased wordings + COSINE alignment; 22 Nov 19:18)
- REAL-TEXT TEST visit 3: bridge 0.18, C 0.02, **C given the right bridge 0.04** - the same as kra38 (paraphrases + contrastive: 0.17 / 0.03 / 0.04) and below kra36 (no paraphrases, cosine: 0.13 / 0.06 / 0.37). Premises alone 0.57 / 0.37.
- **So the PARAPHRASES, not the contrastive alignment, cost step 2 at visit 3.** Likely mechanism: half of each step's real-text rows are premise rows; with 4 paraphrases per premise the arrow-frame rows ('B > C.', the step-2 fix) are diluted about 3x. (The contrastive question is reopened; it may still be weaker than cosine on the GPU.) Visits 4-5 to confirm; kra40 (kra36 recipe, seed 5) unaffected.
- Rule world own chain visit 3: held back 0.00 (n 11), pure families 0.00 (n 10), format 0.73 / 0.50 (kra36 same visit 0.15 / 0.00).
- kra39 VISIT 4 (19:51): REAL-TEXT TEST bridge **0.42** (kra38 0.36, kra36 0.29), C 0.09 (kra36 0.15; baseline 0.05), C given the right bridge 0.21 (kra38 0.11, kra36 0.50); premises alone first hop 0.79 (kra36 0.69). Rule world held back 0.37 (19), pure families 0.08 (12).
- **Revised reading: the paraphrases DO lift step 1 (bridge at visit 4: 0.36-0.42 with them vs 0.29 without) but dilute the arrow rows and cost step 2.** A trade-off, not a failure: paraphrases with the arrow rows scaled up to keep their share (e.g. inf_rchain_arrowprem 3-4) should keep both. Candidate after kra41.

### RELATION MERGING v1 (offline, real records; 22 Nov 20:05; 32,014 facts, 40 raw wordings, 33,038 names)
- Raw wordings: route J held-out one-hop 0.0097 (heads with other facts 0.0041); rules above the shuffled null 0. **The merge thresholds (JEPA operator agreement 0.50 and 0.35) merged nothing** - every pair of wordings agrees below 0.35 (operators fitted on about one fact per name rarely see the same heads) - so v1 is uninformative. v2 merges to fixed numbers of relations (20, 10) and prints the most similar pairs.

## Summary, 20-22 November

Every figure below is in the entries above, with its conditions; GPU runs are pass 1, one run per build unless two seeds are named. Synthetic = tools/compose_synth.py and the other offline worlds; rule world = the planted families and org charts inside the GPU build.

**Composition by the trunk itself (criterion 2).**
- The trunk composes by writing its own intermediate step ('A is a grandchild of B > C'), taught on route J's written composites in that arrow form (kra23 onwards). First for new subjects (kra23, held back 0.19-0.30), then through bridges never in any chain ('pure families'): kra35, seed 5, visits 5-8 pooled **0.35 (n 134)**; kra34, seed 4, visits 4-5 pooled 0.35 (9 / 26; visit 5 0.58); **the direct one-step answer 0.00 throughout**. Prompted step by step, pure families 0.78-0.98.
- What made it work, each found in the synthetic world first and then on the GPU: premises in at least **two wordings** (one wording: new-bridge composition 0.12-0.19; two or more: 0.62-0.77 - a premise learned in one context is not retrieved inside a chain); the **arrow form**, not the spelled form (spelled 0.12-0.14); enough **taught chains**; and **JEPA alignment of the second step** (synthetic 0.67 -> 0.78; on the GPU kra33 without it: bridge 0.67, root 0.00).
- Alignment forms (synthetic, 150 countries): contrastive 0.64-0.77 > residual 0.57-0.71 > cosine 0.53-0.69 > none 0.47-0.52; answer-space KL damages taught chains. On top of the arrow premise frame the gains do not stack.
- The arrow premise frame ('B > C.' as an extra premise wording) adds about 0.25-0.30 for new bridges (synthetic 0.73-0.81 vs 0.47-0.52).

**False memories.**
- The trunk's confidence gate with a start-up floor (inf_distill_gate_min 0.8): never-read questions given a stored name **0.01-0.07** (ungated 0.55-0.69), right when answered 0.93-1.00, on seeds 4 and 5.
- Chains: teaching chains makes a synthetic trunk invent chains for 90% of unknown names; the same gate cuts that to 11% with every answered chain right.

**Real text.**
- Containment chains from the distant Wikipedia records ('A is within B > C'; LOC filter, composite relation of its own after kra32's prompt clash): taught chains learned 0.86-0.94.
- Through bridges never in a taught chain: kra35 C 0.03-0.19 (baseline 0.07-0.15); **kra36 (arrow premise frames) doubled step 2** (C given the right bridge 0.31-0.50 vs 0.10-0.29 at the same visits), C 0.06-0.15 vs baseline 0.05-0.15 - above baseline, not yet clearly. **Step 1 is the bottleneck**: the bridge inside the chain 0.25-0.36 while the first premise asked alone is 0.60-0.79.
- Paraphrased premise wordings (kra38, kra39) lift step 1 (bridge 0.36-0.42 at visit 4) but dilute the arrow rows and cost step 2; contrastive alignment did not help on the GPU (kra38). Next: the step-1 frame ('A is within ' -> 'B >', kra41), kra36 on seed 5 with an ANY VALID CHAIN readout (kra40) - test examples show valid alternative bridges scored wrong and noisy records ('Southampton > Suffolk County').

**Latent reasoning and the JEPA (offline, synthetic).**
- Inference as search over route J's operators (latent_search.py): finds the 3-hop and a never-taught 4-hop relation from endpoints alone, held-out 1.00, abstains on unknown names; the trunk's written 3-hop chains reach 0.01 on people never taught (100 taught chains).
- Route J works as well on random name codes: its operators are the fact table. A JEPA trained only by latent prediction stores few facts (recall 0.11 vs 1.00 for the next-byte trunk): the language model stores, the JEPA side reasons.
- Slow channels (Slow SVReg): no dose-response; dropped. SVReg as the dream selector's regulariser: harmless, inconclusive locally.

**Retraction (open).**
- Offline: re-confirmation removed 0.53 of wrong confirmed inferences and 0.34 of right ones; on the GPU (kra21) it was not selective (retracted right 0.82 vs kept 0.84) because almost every pair recurs; unlearning itself works (retracted inferences 0.92-1.00 -> 0.00-0.03 in one visit).
- JEPA surprise on real records with planted errors: AUC 0.44-0.51 for every signal (surprise, normalised, path consistency, confident contradiction) - real records hold about one fact per name, so nothing contradicts a wrong one (route J held-out one-hop 0.01; rules over raw wordings: none above a shuffled null). A dense-world test (redundant facts) and relation merging by JEPA operator agreement are running.

### kra40 VISITS 2-3 (GPU, seed 5; kra36 recipe + ANY VALID CHAIN readout; 22 Nov 20:19)
- The new readout runs. REAL-TEXT TEST visit 3 (n 150, 148 well-formed): bridge right 0.21, C 0.03, C given the right bridge 0.13 (baseline 0.07); **ANY VALID CHAIN: first link recorded (any recorded container of A, any wording) 0.21, whole chain recorded 0.03** - the same as the strict scores; written conclusions with a gold label 5, all true containment. Taught (77 well-formed): valid 0.96.
- **The measurement hypothesis is rejected:** wrong bridges are almost never another recorded container of A - they are guesses ('Woodbine Racetrack -> Australia > Tasmania', 'Oland -> Brooklyn > New York City'). The strict score was not undercounting; step 1 genuinely fails inside the chain (premise alone 0.51).
- Same visit, other runs: kra35 (seed 5, no arrow frame) 0.09 / 0.03 / 0.29; kra36 (seed 4) 0.13 / 0.06 / 0.37. Rule world visit 3: held back 0.00 (14), pure 0.00 (12) - as kra35 at visit 3 (composition appeared from visit 5 on seed 5).

### New readout (22 Nov 20:40): STEP BY STEP on real text (route D for real-text chains; the combined design's test)
- In [REAL-TEXT CHAINS, THE TRUNK'S OWN], for taught and TEST chains: the trunk answers hop 1 ALONE in its premise wording ('A in '), then hop 2 on its OWN hop-1 answer ('B' lies in ' -> C'); hop 1 right, C right, valid chain (both links recorded), gold-true written conclusions; GATED by the weaker hop's mean byte log-probability, the threshold keeping 75% of the right TAUGHT step-by-step answers: answered, right when answered. Compares the trunk's one-chain answer (bridge 0.21-0.32 inside the chain vs the premise alone 0.51-0.79) with the plan-then-answer-each-step design. Readout only, guarded; in every build from commit below (kra42 onwards if started after it).
- kra40 VISIT 4 (20:36): REAL-TEXT TEST bridge 0.32, C 0.07 (baseline 0.05), C given the right bridge 0.21 (kra36 visit 4, seed 4: 0.29 / 0.15 / 0.50; kra35 visit 4, seed 5 without frames: 0.34 / 0.04 / 0.10); premises alone 0.74 / 0.71. Rule world: held back 0.23 (13), **pure families 0.11 (18), bridge 0.44**, direct 0.00 (kra35 same visit 0.10). On seed 5 the arrow frame's step-2 gain is smaller so far (0.13-0.21 vs 0.10-0.29 without it at visits 3-4).

### RETRACTION IN A DENSE WORLD (offline, synthetic; 22 Nov 20:53; tools/retract_dense.py; five-level world, 500 people, wrong facts planted in 'lives' (person -> wrong city) and 'city_in' (city -> wrong region), a redundant recorded country for 0 / 50 / 100% of people; route J 4-fold x 2 seeds)
| signal (AUC; precision at the error count) | 0% redundant | 50% | 100% |
|---|---|---|---|
| 'lives' S1 surprise (leave-fold-out) | 0.54 | 0.41 | 0.51 |
| 'lives' S4 confident contradiction | 0.46 | 0.48 | 0.42 |
| **'lives' S5 path contradiction, symbolic** (recorded city -> region -> country vs the recorded country) | - | **0.94; 0.60** | **0.95; 0.69** |
| 'lives' S5 path contradiction, latent (route J composite) | - | 0.84; 0.10 | 0.83; 0.34 |
| 'city_in' S1 surprise | 0.53 | 0.57 | 0.55 |
| **'city_in' S5 path contradiction** (share of the city's residents whose recorded country disagrees) | - | **0.82; 0.67** | **0.998; 0.90** |
- BLAME (which link of a contradicted chain is planted, by surprise): 0.35-0.49 vs chance 0.33 - surprise does not localise.
- **Retraction works by CONSISTENCY BETWEEN REDUNDANT VIEWS, not by surprise.** Leave-one-out surprise and confident contradiction stay at chance even in a dense world (a held-out fact's subject has nothing else that predicts it). A chain that contradicts an independent record finds wrong facts at AUC 0.94-0.998 - but only where that redundant record exists (0% redundancy: no signal possible). A fact contradicted by MANY independent chains (a wrong city_in, contradicted by its residents) is found and localised at 0.90 precision; a single contradicted chain flags all its links (lives precision 0.60-0.69).
- For real text (one fact per name, no redundancy): retraction needs redundant views - the same fact reached by two routes (another relation, another document, a composite) - which relation merging and the trunk filling in facts (the combined design) would supply; and aggregation over many contradicting chains to localise the culprit.

### kra41 local check (22 Nov 20:50): PASS
- No Traceback; the step-1 frame adds one premise row per chain (visit 1 64 vs kra36's 51; visit 2 318 vs 249; visit 3 587 vs 459); ANY VALID CHAIN prints; taught real-text chains at visit 3 as kra36 (format 0.96, C given the bridge 0.92).
- kra40 VISIT 5 and END (21:05; stopped by the user after visit 5): REAL-TEXT TEST bridge 0.22, **C 0.10 (baseline 0.07), C given the right bridge 0.42**; premises alone 0.70 / 0.69. Visits 3-5 on seed 5: C given the bridge 0.13 / 0.21 / 0.42 (kra35, same seed, no frames: 0.29 / 0.10 / 0.13) - **the arrow frame's step-2 gain replicates on seed 5 by visit 5**; C stays close to baseline because step 1 (bridge 0.21-0.32) is the bottleneck on both seeds. Rule world: held back **0.48 (n 23)**, pure families 0.11 (n 9) (visits 4-5 pooled 0.11, n 27), direct 0.00; route D pure families 0.88 vs direct 0.00. Gate: never-read given a stored name 0.03 (ungated 0.57).

### kra41 VISITS 3-5 (GPU, seed 4; kra36 + the STEP-1 FRAME 'A is within ' -> 'B >'; 22 Nov 21:40) - STEP 1 FIXED
| REAL-TEXT TEST (n 150) | kra41 visit 3 | kra41 visit 4 | kra36 (same seed) visit 3 / 4 | kra39 (paraphrases) visit 3 / 4 |
|---|---|---|---|---|
| **bridge right (step 1)** | **0.40** | **0.66** | 0.13 / 0.29 | 0.18 / 0.42 |
| C right | 0.04 | **0.16** | 0.06 / 0.15 | 0.02 / 0.09 |
| C given the right bridge (step 2) | 0.07 | 0.23 | 0.37 / 0.50 | 0.04 / 0.21 |
| baseline (commonest C) | 0.07 | 0.05 | 0.07 / 0.05 | 0.07 / 0.05 |
| premises alone, hop 1 / hop 2 | 0.53 / 0.21 | 0.77 / 0.49 | 0.57-0.69 / 0.37-0.62 | 0.57 / 0.37, 0.79 / 0.66 |
- **The step-1 frame closes the step-1 gap: the bridge inside the chain 0.66 against the premise alone 0.77 (kra36: 0.29 vs 0.69).** The C answer at visit 4 is 0.16, three times the baseline (0.05), the best real-text visit-4 figure so far; ANY VALID CHAIN agrees (149 well-formed).
- Step 2 is now the weaker step (0.07-0.23 vs kra36's 0.37-0.50), and the second premise asked alone is lower (0.21-0.49). kra42 (step-1 frame + arrow rows x4 + paraphrases + 800 taught chains) adds weight to step 2 - the right combination on paper.
- Rule world: held back 0.29 / 0.50 (n 14 each), pure families 0.00 (12) / 0.12 (8), direct 0.00-0.07.

### kra42 local check (22 Nov 21:45): PASS
- No Traceback (four visits within the 50-minute limit); premise rows with the paraphrases and the arrow rows x4 (visit 1: 183 vs kra41's 64); the STEP BY STEP readout prints on the full system (stand-in, visit 3 taught: hop 1 right 0.43, C 0.13, gated answered 0.58, right 0.23) - readout only. Ran with 400 taught chains (preset loaded before the 800 change; a count only).

### THREE HOPS WITH TWICE THE TAUGHT CHAINS (offline, synthetic; 22 Nov 22:05; tools/latent_search.py LS_NT 200)
- TRUNK, 3-hop arrow chains taught for 199 people (was 100): taught 1.00; **NEVER TAUGHT (68, cities in no taught chain): bridge 0.15-0.22, region 0.00-0.01, country 0.01-0.03** (100 taught: 0.08-0.10 / 0.00-0.01 / 0.01-0.03); never-read names given a well-formed chain 0.80. Premises recalled 1.00.
- **More chains do not help at three hops; the first hop fails inside the chain (0.15-0.22) although the premise is recalled perfectly** - the real-text step-1 failure in the synthetic world. kra41's step-1 frame fixed exactly that on real text (bridge 0.29 -> 0.66): the next local test teaches the premises in the chain's own frames at every step ('A is a citizen of ' -> 'B >', 'B > ' -> 'R >', 'R > ' -> 'K.').
- LATENT ARM (route J search) again: 3 hops held-out 1.00, gated 0.63 answered right 1.00, never-read 0.00; 4 hops 1.00.

### RELATION MERGING v2 (offline, real records; 22 Nov 23:00; 32,014 facts, 40 raw wordings; merged to fixed numbers of relations)
- Most similar pairs by JEPA operator agreement: 'state of' ~ 'region of', 'and in' ~ 'region of', 'since' ~ 'in early', 'until' ~ 'since', 'is' ~ 'are' (0.13-0.17); by Jaccard of (head, tail) pairs: all below 0.005 (the wordings share almost no pairs).
| merged to | method | gold purity | rules above the null | route J held-out (heads with other facts) |
|---|---|---|---|---|
| 40 (raw) | - | 1.000 (trivial) | 0 | 0.010 (0.004) |
| 20 | **JEPA operator agreement** | **0.826** | 0 | 0.003 (0.005) |
| 20 | symbolic Jaccard | 0.640 | 0 | 0.004 (0.007) |
| 10 | **JEPA operator agreement** | **0.667** | 1 (0.125, 3 / 24) | 0.003 (0.007) |
| 10 | symbolic Jaccard | 0.576 | 4 (all <= 0.034) | 0.004 (0.007) |
- **JEPA operator agreement merges wordings into purer relations than co-occurrence** (0.83 vs 0.64 at 20; groups such as {in, until, and in, since, in early}, {at, died in, was born in}, {film, rock band, album}) - a label-free latent criterion that finds what the wordings mean.
- **But merging does not create density**: route J's held-out one-hop stays at 0.003-0.007 and no strong rules appear. The sparsity is in the NAMES (about one fact per name), not in the wordings. Redundancy for retraction and inference on real text has to come from elsewhere: more documents per entity, the trunk filling in facts it read (the combined design), or chains through the containment hierarchy (where kra36-kra41 work).

### THREE STEPS TOWARD LECUN'S DESIGN (offline, synthetic; 23 Nov 00:00; tools/lecun_steps.py; five-level world, codes learned jointly, 3,000 steps)
- **A. Objectives:** InfoNCE, VICReg, SIGReg, SVReg and plain regression ('none') ALL find 'citizen of' (3 hops) and 'continent' (4 hops) by search, held-out 1.00, never-read names answered 0.00; codes' effective rank 59.8-64 of 64 (no collapse, even unregularised, in this small world). **Non-contrastive objectives answer more under the same gate (0.72-0.73 vs 0.56 for InfoNCE) at the same precision (1.00)** - route J can drop its negatives without loss; the world is too easy to separate the regularisers.
- **B. Ensemble of 5 seeds:** known held-out people right 1.00, seeds agree 1.00; never-read names: seeds agree 0.34 (uncertain). Unknown vs known AUC 1.000 (one model's margin also 1.000 - this world is too clean to show a gain). **Claims about known people: false claims CONTRADICTED (seeds agree >= 0.8 on another answer) 1.00, true claims 0.00; never-read names called contradicted 0.02.** The ensemble separates "contradicted" (retract) from "unknown" (abstain) - the distinction retraction needs.
- **C. Hierarchy by cached programs:** 'continent' within a 2-step budget: relations only 0.40 (= null 0.40); **with the cached 'citizen of' program: citizen* o country_in found, 1.00, held-out 1.00** (30 programs searched). A distilled macro (one MLP on the cached program's 100 endpoints): held-out 0.06 - a fitted shortcut is a table; the cached composition generalises.

### THREE HOPS WITH STEP FRAMES (offline, synthetic; 23 Nov 00:05; latent_search LS_FRAMES 1, 100 taught people)
- Premises also taught in the chain's own frame at every step ('A is a citizen of ' -> 'B >', 'B > ' -> 'R >', 'R > ' -> 'K.'). NEVER-TAUGHT people (79, cities in no taught chain): **bridge 0.86-0.99 (without frames 0.08-0.22)**, region 0.00-0.01, country 0.00-0.04; format 0.42-0.70. Taught 1.00. Never-read names given a chain 0.70.
- **The step-1 frame fixes step 1 at three hops too (kra41's real-text result replicated in the synthetic world); the MIDDLE step does not follow** - 'B > R >' as a standalone row does not carry into the chain after 'A is a citizen of B >'. Two-hop chains end in the arrow frame 'B > C.' (which works); the middle step of a longer chain needs its own fix.

### THREE HOPS, MIDDLE-STEP FRAME IN CHAIN CONTEXT (offline, synthetic; 23 Nov 06:55; latent_search LS_FRAMES 2)
- Step frames as before plus each person's chain PREFIX up to the middle step ('A is a citizen of B > R >': premises only, the country never shown). NEVER-TAUGHT people (79): **bridge 0.80 / 0.95 / 1.00, region 0.15 / 0.46 / 0.81, country 0.01 / 0.06 / 0.14** at 300 / 750 / 1,500 steps (standalone frames only: region 0.00-0.01, country 0.00-0.04); format 0.94.
- **The middle step is fixed by its frame IN CHAIN CONTEXT (region 0.81 vs 0.01); the last step now limits (0.14, still rising).** The pattern: each step must be learned in the context in which the chain reaches it. The last step's in-context frame would show the answer, so it has to come from the standalone arrow frame ('R > K.') - which works for two-hop chains - and more teaching.

### kra42 VISIT 3 (GPU, seed 4; 23 Nov 07:15) - DILUTED
- Real-text premise rows 9,454 (visit 1) -> 13,302 (visit 2) from 4 paraphrases + arrow rows x4 + 800 taught chains, at 8 rows a step (kra41: 2,247). TEST (n 150) visit 3: **premises asked alone 0.23 / 0.13 (kra41 0.53 / 0.21)**, bridge 0.11 (kra41 0.40), C 0.02 (baseline 0.07). TAUGHT chains fine (C 0.72). STEP BY STEP prints on the GPU: taught hop 1 0.71, C 0.38 (valid 0.38, gold-true 7 / 7); test hop 1 0.23, C 0.05.
- **Too many premise rows for the rows a step: each test premise is seen about a fifth as often, so the trunk has not learned them by visit 3.** kra43 redefined (it had not run): kra41 + arrow rows x2 + 16 real-text rows a step + 400 test chains, no paraphrases, seed 5.
- kra42 VISIT 4 (07:41): TEST premises alone 0.41 / 0.22 (rising from 0.23 / 0.13), bridge 0.25, **C 0.02 (baseline 0.05), C given the right bridge 0.03** - step 2 inside the chain collapses with the paraphrases even with arrow rows x4 (as kra38 / kra39). **STEP BY STEP on the same test chains: hop 1 0.41, C 0.10, valid 0.07, gold-true 8 / 8** - asking each hop separately beats the chain 5x when the chain's step 2 fails. Taught: chain C 0.76, step by step C 0.51. Rule world: held back 0.40 (15), pure families 0.12 (8), direct 0.00. Verdict: stopped; paraphrases out of the recipe.

### CAN A JEPA STORE KNOWLEDGE? (offline, synthetic; 23 Nov 08:27; tools/jepa_store.py; five-level world, 885 names, 1,760 premise sentences x3; no next-byte objective, no fitted operators, no labels; 8,000 steps each)
| JEPA (bidirectional encoder, EMA target, predictor) | step | one-hop recall lives / city_in | chained by its own hops: 2 / 3 hops |
|---|---|---|---|
| WORD masks, in-batch negatives (v1 recipe) | 2k / 4k / 6k / 8k | 0.07 / 0.23 / 0.38 / 0.40 ; 0.03 / 0.24 / 0.36 / 0.37 | 0.16 / 0.07 at 8k |
| NAME masks + 4,096-target memory bank | 8k | 0.01 ; 0.22 | 0.00 / 0.00 |
| **SLOT: the JEPA predicts a KEYED MEMORY** (each name string owns a slot, an EMA of the target encoder's states of that name; InfoNCE over all slots) | 2k / 4k / 6k / 8k | **0.73 / 0.98 / 0.97 / 1.00 ; 0.43 / 1.00 / 1.00 / 0.98** | **0.36 / 0.98 / 0.95 / 1.00 ; 0.08 / 0.90 / 0.95 / 1.00** |
- **A JEPA stores knowledge when it predicts into a keyed memory: recall 0.98-1.00 and two- and three-hop answers by chaining its own predictions 1.00 - from raw sentences, without a generative objective.** The next-byte trunk reaches 1.00 recall by 2k steps; the slot JEPA by 4k.
- The plain JEPA learns slowly (0.40 at 8k; 0.11 at 5k in v1) - unstable, few targets. A memory bank of stale targets makes it worse.
- Caveat: predicting a slot over the named entities is close to choosing from an entity vocabulary (masked entity prediction); the targets are latent (EMA of the encoder's own states), not bytes or words. The design point: **memory as the store, the JEPA as reader and predictor** - Platypus's keyed memory is that store.

### LLM-JEPA IN THE TRUNK (offline, synthetic; 23 Nov 09:08; latent_search LS_FRAMES 2 + LS_LLMJEPA 0.5: during pretraining the trunk's last-byte state of one wording of a premise predicts, through a small predictor, its state of the other wording)
- Never-taught 3-hop chains at 300 / 750 / 1,500 steps, WITH vs WITHOUT ($S/latent_search_frames2.log): bridge 0.84 / 1.00 / 1.00 vs 0.80 / 0.95 / 1.00; **region 0.27 / 0.75 / 0.96 vs 0.15 / 0.46 / 0.81**; country 0.01 / 0.03 / 0.06 vs 0.01 / 0.06 / 0.14; never-read names given a well-formed chain 0.90 vs 0.77. Premises recalled 1.00 by 3k steps either way.
- **Mixed: the latent-prediction loss speeds and raises the middle step (region 0.96 vs 0.81) but not the last step (0.06 vs 0.14), and the trunk invents more chains for unknown names.** One seed; not a recommendation for the GPU build yet.
- REPLICATION (23 Nov 13:05; two new worlds, LS_SEED 1 and 2, slot JEPA only): world 1 - 2k 0.34 / 0.05, 4k 1.00 / 0.96 (chained 0.94 / 0.81), 6k 0.97 / 1.00 (0.96 / 0.96), **8k 0.99 / 1.00 (chained 2 hops 0.99, 3 hops 0.99)**; world 2 - 2k 0.13 / 0.07, 4k 0.98 / 0.93 (0.89 / 0.71), **6k 1.00 / 1.00 (1.00 / 1.00)**, **8k 1.00 / 1.00 (1.00 / 1.00)**. **Replicates on three worlds**: the keyed-memory JEPA learns every fact and chains three hops by 6-8k steps (slower start than world 0).

### CONTINUAL KEYED-MEMORY JEPA (offline, synthetic; 23 Nov 14:23; tools/jepa_store.py JS_CONT=1 JS_PHASE=2500 JS_REPLAY=0.25; slot JEPA; three domains of people learned one after another, 334 / 334 / 332 'lives' sentences each, the 760 geography sentences in every domain)
| replay of earlier domains | after domain 0: D0 / D1 / D2 | after domain 1 | after domain 2 |
|---|---|---|---|
| none | 1.00 / 0.00 / 0.01 | 0.08 / 1.00 / 0.00 | 0.03 / 0.30 / 1.00 |
| 25% of the stored sentences | 1.00 / 0.00 / 0.01 | **1.00 / 1.00** / 0.01 | **1.00 / 1.00 / 0.99** |
- **Without replay the keyed-memory JEPA forgets catastrophically (D0 1.00 -> 0.08 after one more domain), although the memory slots of the earlier people are never touched: the forgetting is in the reader (encoder and predictor), not the store.** 25% replay holds every domain at 1.00 / 1.00 / 0.99 - the same protection Platypus gives its trunk.
- Recall claim only ('lives' one-hop recall per domain); consolidation not measured here.
- **DREAM REPLAY (23 Nov 15:36; JS_DREAM=0.25,0.1):** before each new domain the JEPA writes out its OWN recall of the earlier people ('<person> lives in <recalled city>.', one sentence a person: 167 then 334; recalled correctly 1.00) and replays only that - no stored sentences.
| dream replay | after domain 0 | after domain 1 | after domain 2 |
|---|---|---|---|
| 25% | 1.00 / 0.00 / 0.01 | **1.00 / 1.00** / 0.00 | **1.00 / 1.00 / 1.00** |
| 10% | 1.00 / 0.00 / 0.01 | **1.00 / 1.00** / 0.00 | **0.99 / 1.00 / 1.00** |
- **Replaying its own recall protects the keyed-memory JEPA as fully as replaying the real sentences (no replay: 0.03 / 0.30 / 1.00), and 10% is enough.** Platypus-style dreams carry over to a JEPA learner.
- Caveats: the dreams were written in the same wording the test asks in ('{a} lives in {b}.'), which favours them; and dreams are only as good as recall (here 1.00) - a wrong recall would be replayed as a false memory. Running: dreams in the OTHER wording ('{a} has a home in {b}.'), 10% and 5%.
- **CROSS-WORDING DREAMS (23 Nov 16:45; JS_DREAMW=1):** dreams written as '{a} has a home in {b}.', recall asked as '{a} lives in {b}.'. 10%: after domain 1 1.00 / 1.00, after domain 2 **0.99 / 1.00 / 1.00**; **5%: 0.99 / 1.00, then 0.99 / 1.00 / 1.00**. **The protection does not depend on the test's wording, and 5% dream replay is enough.** **Replicated on world 1 (LS_SEED 1; 23 Nov 18:47): 5% cross-wording dreams, after domain 1 0.99 / 1.00, after domain 2 0.95 / 0.98 / 1.00** (no replay on world 0: 0.03 / 0.30 / 1.00). Running: FALSE MEMORY - 10% of the first dream's recalls replaced by a wrong city; are they kept or corrected?
- **FALSE MEMORY (23 Nov 17:20; JS_DREAM=0.1 JS_DREAMBAD=0.1):** 21 of domain 0's 167 dreamed recalls replaced by a wrong city in the FIRST dream only; the next dream is the model's own recall again. The 21 planted people afterwards: **FALSE city 1.00, true city 0.00, after domain 1 and still after domain 2** (domain 0 recall 0.87 = everyone else kept). Domains 1 and 2 1.00.
- **A wrong memory replayed once is learned permanently and then re-dreamed as the model's own: self-replay protects whatever it replays, true or false.** Nothing in this world contradicts a 'lives' fact (no redundant view), so nothing can correct it - the same limit as retraction on real records. A JEPA protected by its own dreams needs Platypus's false-memory checks on what it dreams: verification before replay, and a contradiction trigger where redundant evidence exists (dense world: AUC 0.94-0.998; ensemble: false claims contradicted 1.00).

### kra43 VISIT 3 (GPU, seed 5; kra41 + arrow rows x2 + 16 real-text rows a step, no paraphrases; 23 Nov 17:10) - BEST VISIT 3 ON REAL TEXT
| REAL-TEXT TEST (n 150) | kra43 visit 3 | kra41 visit 3 (seed 4) | kra41 visit 4 |
|---|---|---|---|
| premises alone, hop 1 / hop 2 | **0.72 / 0.44** | 0.53 / 0.21 | 0.77 / 0.49 |
| bridge right (step 1) | **0.66** | 0.40 | 0.66 |
| C given the right bridge (step 2) | 0.16 | 0.07 | 0.23 |
| C right (baseline commonest C 0.07) | 0.11 | 0.04 | 0.16 |
| STEP BY STEP: hop 1 / C / valid | **0.72 / 0.33 / 0.33** (gold-labelled written conclusions true 7 / 7) | - | - |
| STEP BY STEP gated (keeps 75% of right taught) | answered 0.20, **right when answered 0.80** | - | - |
- Taught chains: bridge 0.96, C 0.95 (C given the bridge 0.99); taught step by step C 0.56, gated right 0.83. Premise rows 3,905 at visit 2 (16 a step), so the dose problem of kra42 is gone: test premises are learned a visit earlier than in kra41.
- **Step 1 is fixed a visit earlier (bridge 0.66 at visit 3) and step 2 is rising (0.07 -> 0.16 given the bridge, vs kra41 at the same visit).** The chain's C (0.11) is still limited by step 2; **asking each hop separately triples it (0.33) and the gate makes it reliable (0.80 right on the 20% it answers)** - the route J plans / trunk answers each step / gates check design works on real text.
- Route J (rule world) composite 0.99, null 0.01, never-read families pass the gate 0.03 (AUC 0.98). Rule world trunk chain: held back C 0.00 (14), pure families 0.00 (12). No Traceback.

### kra43 VISIT 4 (GPU, seed 5; 23 Nov 17:47) - BEST REAL-TEXT INFERENCE SO FAR
| REAL-TEXT TEST (n 150; bridge never in a taught chain) | kra43 visit 4 | kra43 visit 3 | kra41 visit 4 (seed 4) |
|---|---|---|---|
| premises alone, hop 1 / hop 2 | 0.88 / 0.68 | 0.72 / 0.44 | 0.77 / 0.49 |
| bridge right (step 1) | **0.75** | 0.66 | 0.66 |
| C given the right bridge (step 2) | **0.31** | 0.16 | 0.23 |
| **C right** (baseline commonest C 0.05) | **0.24** | 0.11 | 0.16 |
| C right on GOLD-TRUE chains (28) | **0.46** | 0.17 | - |
| ANY VALID: written conclusions gold-labelled true | 0.94 (18) | 0.78 (9) | - |
| **STEP BY STEP: hop 1 / C / valid** | **0.88 / 0.62 / 0.61** (gold-labelled true 20 / 20) | 0.72 / 0.33 / 0.33 | - |
| STEP BY STEP gated: answered / right when answered | 0.23 / **0.88** | 0.20 / 0.80 | - |
- **The written chain's answer is 0.24, five times the baseline (0.05) and 1.5x kra41's best; 0.46 on the chains whose links are gold-labelled containment. Asked step by step, the trunk answers 0.62 of never-composed chains right (every gold-labelled conclusion true, 20 / 20); gated, 0.88 right on the 23% it answers.**
- Both steps now improve together (step 1 0.75, step 2 0.31 given the bridge) - the step-1 frame plus the arrow rows at 16 rows a step is the real-text recipe. Taught chains 0.88. Route J composite 0.98, null 0.00. Rule world: held back C 0.23 (13; visit 3 0.00), planted 0.70.

### kra43 VISIT 5 (GPU, seed 5; 23 Nov 18:12)
- TEST (n 150): premises alone 0.69 / 0.63 (visit 4 0.88 / 0.68), bridge 0.67 (0.75), C given the bridge 0.23 (0.31), **C 0.16** (0.24; baseline 0.07), gold-true chains C 0.30 (23); ANY VALID written conclusions gold-labelled true 0.92 (13). **STEP BY STEP: hop 1 0.69, C 0.46, valid 0.45, gold-labelled true 0.94 (18); gated answered 0.20, right 0.63.** Taught chains C 0.90 (unchanged).
- **Not a regression of the model, most likely a change of test set:** the test list is rebuilt every visit from the chains found so far (848 -> 5,988 chains; first 150 by a hash of the head), so each visit's test set takes in chains found in the visit before, whose premises have been taught one visit only. The drop is in the PREMISES asked alone (hop 1 0.88 -> 0.69) while the taught chains hold (0.88 -> 0.90). kra45: freeze the test set at its first full build (and report by premise age) so visits compare like with like.
- Visits 3-5 together: bridge 0.66-0.75, C 0.11-0.24 (baseline 0.05-0.07), step by step C 0.33-0.62 with 0.94-1.00 of gold-labelled conclusions true.

### kra44 local check (23 Nov 18:15; 150 steps a domain, distant_10k, seed 4, 50-minute limit): PASS
- No Traceback through four domains (as kra41 / kra42's checks). The three readouts print at visits 2 and 3: OBJECTIVE COMPARISON (cos: composite 0.04 vs nce 0.03-0.04, AUC 0.93 both); ENSEMBLE CONTRADICTION (seeds agree 0.72 on read families and 0.71 on NEVER-READ ones; false claims contradicted 0.34 vs true 0.30, AUC 0.51 - **with the trunk's fixed name codes, refitting route J with other seeds gives no independent views: the seeds agree by construction**; the local LeCun test learned codes per seed); LATENT PROGRAM SEARCH: 'no composite relation with 5+ recorded examples' (the reader does not record composite facts under the rule's relation - the search needs another example source). Local route J is weak at this size (composite 0.03); the readouts are judged on the GPU.
- kra45 candidates from this: the ensemble by BAGGING (each seed fitted on a resample of the facts) and the program search on the planted / written composites.

### kra44 VISIT 2 (GPU; ran on SEED 5 - the notebook's SEED overrides the preset - so its real text repeats kra43; 23 Nov 19:18) - THE ROUTE J READOUTS ON THE GPU
- **NON-CONTRASTIVE route J (1 - cosine, no negatives) = InfoNCE:** first hop 1.00 / 1.00, composite 0.98 / 0.98, pure families 1.00 / 1.00, AUC right-vs-never-read 0.96 / 0.98.
- **ENSEMBLE CONTRADICTION (4 seeds):** read families: majority right 0.98, seeds agree 1.00; **FALSE claims CONTRADICTED 1.00, TRUE claims 0.02** (local LeCun test: 1.00 / 0.00). Never-read families: seeds agree 0.79, a claim called contradicted 0.41 (AUC agreement read vs never-read 0.80) - as the local check predicted, seeds on the trunk's fixed codes agree too often on unknowns; kra45 bags the facts.
- **LATENT PROGRAM SEARCH, no rule given:** 'is overseen by' -> found 'is supervised by o is supervised by' from 6 recorded examples (1.00; shuffled null 0.17; 272 programs) = the learned rule, targets 0.97 (70); 'is a grandchild of' -> 'is a child of o is a child of' from 11 examples (1.00; null 0.36) = the learned rule, targets 1.00 (49). (On the GPU the reader does record composites under the rule's relation; locally it did not.)
- **Three LeCun-direction results confirmed inside the model on the GPU:** non-contrastive latent prediction works as well as contrastive; an ensemble separates false claims from true ones (1.00 vs 0.02); latent search discovers the composition rules from a handful of examples. Open: unknown vs contradicted (0.41 of never-read called contradicted).
- kra45 moved to SEED 4 (notebook and preset) so it is the recipe's seed partner as well as the next readouts.

### kra45 local check (23 Nov 19:37; 150 steps a domain, distant_10k, seed 5, 50-minute limit): PASS
- No Traceback through three domains. BAGGED ensemble: the four route J fits now disagree (seeds agree 0.33 vs 0.72 unbagged) - independent views; PROGRAM SEARCH on the planted composites (fallback): 'is overseen by' found 'is supervised by o is supervised by' (0.83 on 6 examples, null 0.00) = the learned rule; RULE WORLD STEP BY STEP prints (no learned rule locally yet: plan 0.00 - judged on the GPU, where rules are learned by visit 2); OBJECTIVE COMPARISON as in kra44. Test-set freezing set to 150 chains (GPU: visit 2). kra45 moved to seed 4 (the recipe's seed partner).

### kra45 VISIT 2 (GPU, SEED 4; 23 Nov 20:03) - STEP BY STEP WITH A LEARNED PLAN GENERALISES
| RULE WORLD, visit 2 (learned rule plans, trunk answers each hop, gate) | n | plan from a learned rule | hop 1 | C right | GATED answered / right |
|---|---|---|---|---|---|
| planted composites | 18 | 1.00 | 0.72 | 0.50 | 0.56 / 0.70 |
| **HELD BACK, never planted, never written** | 14 | 1.00 | 0.71 | **0.64** | 0.71 / **0.80** |
| **PURE FAMILIES (no composite in the family)** | 16 | 1.00 | 0.69 | **0.56** | 0.62 / **0.90** |
- **The combined design works with a LEARNED plan: on composites the model never saw in any form, step by step answers 0.56-0.64 right (0.80-0.90 when the gate answers), where the trunk's own written chain answers 0.00 (kra43 visit 3, held back and pure) to 0.23 (visit 4).** Route J's learned rule supplies the plan; nothing about the composite is planted for these families.
- PROGRAM SEARCH replicates on the second seed: both rules found from 5 and 8 examples (1.00; nulls 0.20 / 0.12) = the learned rules; targets 1.00 / 1.00. NON-CONTRASTIVE route J = InfoNCE again (composite 1.00 / 1.00; AUC 0.94 / 0.98).
- **BAGGING over-corrects:** each bootstrap fit misses about a third of the facts, so the fits disagree on READ families too (majority right 0.56, agree 0.49) and nothing is contradicted (false claims 0.02). kra44's unbagged ensemble separates false from true (1.00 vs 0.02) but over-calls unknowns (0.41); route J's own MARGIN GATE passes only 0.04 of never-read families (AUC 0.98). kra46: CONTRADICTED = the unbagged seeds agree on another answer AND the margin gate passes; UNKNOWN = the gate fails.
- Real-text test set frozen at 225 chains at visit 2.

### kra45 VISIT 3 (GPU, SEED 4; 23 Nov 20:16)
| REAL-TEXT TEST (frozen set, first 150 of 225) | kra45 visit 3 (seed 4) | kra43 visit 3 (seed 5) |
|---|---|---|
| premises alone, hop 1 / hop 2 | 0.70 / 0.37 | 0.72 / 0.44 |
| bridge right (step 1) | 0.55 | 0.66 |
| C given the right bridge | 0.23 | 0.16 |
| C right (baseline 0.07) | **0.13** | 0.11 |
| gold-true chains: C right | 0.25 (24) | 0.17 (24) |
| STEP BY STEP: C / gated answered / right | **0.28** / 0.09 / **0.85** | 0.33 / 0.20 / 0.80 |
| written conclusions gold-labelled true | 1.00 (10), step by step 1.00 (11) | 0.78, 1.00 |
- **The real-text recipe replicates on the second seed at visit 3:** C 0.13 vs 0.11 (baseline 0.07), step by step 0.28 vs 0.33, gated precision 0.85 vs 0.80, every gold-labelled conclusion true.
- RULE WORLD STEP BY STEP, visit 3 (larger sets than visit 2): planted C 0.53, HELD BACK 0.37 (19), PURE 0.40 (20); **gated: right when answered 1.00 in all three** (answered 0.40 / 0.16 / 0.40). Visit 2 was 0.64 / 0.56 on 14 / 16; hop 1 fell (0.71 -> 0.53-0.60) as more families entered. The trunk's own chain: held back 0.18, pure 0.00.
- Route J composite 0.99, never-read families through the gate 0.01 (AUC 1.00); program search finds both rules again (13 / 12 examples, targets 0.99 / 1.00). Bagged ensemble: nothing contradicted (0.02), as at visit 2 - unbagged with the gate in kra46.
