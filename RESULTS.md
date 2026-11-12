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
control 0.00. Tiny n - the first sign that facts the model inferred itself, never stated in its text, consolidate into its weights.

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
