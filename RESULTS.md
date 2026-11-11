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

Repair replay brings forgetting back to the replay-only level (-0.156). The stochastic theta run's lower cost is not reproduced by fixed
phases and coincides with a 10-dream-step final visit (its allocation per visit was 5 / 0 / 1 / 3 / 10 against 5-6 normally): not
separable from where the random phases fell in 24-step visits; a full-run trio with bounded phases is queued. Spindle-style writing
fails; the beta brake changes nothing measurable.

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
