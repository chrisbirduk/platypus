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
