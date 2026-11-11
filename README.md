# Platypus

**A byte-level language model trained on ten text domains in turn, with a symbolic memory and sleep cycle beside it that extracts planted template facts, joins them, and teaches the model the results.**

> Copyright 2026 **CHRISTIAN BIRD**. Licensed under the [Apache License 2.0](#licence).

This is a personal project: one person's own research, done in spare time over two months on a single Colab GPU, with no institutional affiliation, funding or publication behind it. This README says what the code does and what has been measured, in plain terms. The full record of every build, mechanism, readout and result (136 KB, generated with the help of an AI model, with the caveats that implies) is kept at [`docs/README-full.md`](docs/README-full.md). The results log is [`RESULTS.md`](RESULTS.md).

Current build: **cg27a with interleaved distillation**: from 11 November the build's defaults include interleaved distillation of stored facts into the trunk (question rows writing the MLP output weights only, a question pool of up to 2000 facts), chunked probe evaluation and a GPU cache release after each sleep phase (the sleep peak fell from 70 to about 50 GB), and the Gaussian drift readout; older presets pin the previous defaults so they reproduce their runs. GPU presets: `configs/presets/cg27a.env` (full build, three passes - distillation has so far been run through pass 1 only, so passes 2-3 with it are untested); `configs/presets/cg27a_distillcap_pass1.env` (the reference distillation run, pass 1; seed 4, seed 5 via the notebook's `SEED` line); `configs/presets/cg27a_distillctl_pass1.env` (its control: no question rows trained); `configs/presets/cg27a_distillak_pass1.env` (soft anchor, with the composition and latent-chaining readouts). In the notebook, `SEED` overrides the preset's seed, and the code comes from the branch named on its `BRANCH` line.

---
## Note from author
Documentation here is LLM generated. This has proved problematic, and the LLM often got carried away - either making wild claims or underestimating the benchmarks which had been undertaken. 
The core memory was built, tested and benchmarked at a number of sizes (50M, 170M, 500M) and against several other baselines. In its initial form, it did well, but probably needs benchmarking again with all the extra bloat.
The original aim of the project was to create a continuous learning memory system, ideally for deployment with agents. This then expanded to understand if the dream system could be expanded to provide inference abilities. My gut feeling is that it can, but this would require more layers and additional features.  Taught facts are successfully recalled from the trunk alone (wordings never trained on) at 0.6-0.7 on GPU against 0.0-0.05 for never taught facts.
Investigation into using theta-like phase gate promising, but early.

The closed-loop approach also requires work. My gut feeling here is that should only be an encoder. An encoder would work within a "hippocampal" layer to identify similarity (soft features) and mix these with less predictable raw data, with the raw data component likely decaying faster. The trunk should be able to take the gist and raw data in parallel. Dreams would aid helping avoiding forgetting and inference building. 
Different dreaming would potentially occur during normal training, and gated depending on the state of the model.
Whilst not included within the results here, I had some partial success getting this to work - but I would need to test this at a scale which isn't available to me. Mixing of "concepts" is likely what is required to advance the model, but this cannot easily be achieved with a byte stream as SFA/JEPA decoding is not feasible back to byte form - ultimately, the decoder needs to go (and this creates new challenges).
Other thoughts here - there seems to be a struggle to ensure that memory traces are smaller than the original memory. IMHO this doesn't matter, instead the focus needs to be - can the traces convey all of the information required for modelling state change AND providing inference. Sure, memory traces should be as small as possible, but not so small that they prevent advanced function.
I have barely sufficient time to work on this project. The original aim was to understand if a memory system could be developed for agents which didn't decay into nonsense. I think the project demonstrates this is possible, and along with existing research points towards how this could be achieved. In the last few weeks, so many products have arrived in this area, that this work is less novel than when I started it.
The code is very messy, I apologise for this - it is largely LLM generated and only the core memory system was sufficiently reviewed and has since deteriorated.
I am not an AI engineer or AI researcher - this is a pet project.

## What it is, in one paragraph

A 171M-parameter language model (byte vocabulary, hashed 2/3/4-gram embeddings, one causal local-filter block, three gated delta-rule mixer blocks, one output head per domain) is trained on ten Pile domains one after another — Gutenberg, Wikipedia, ArXiv, PubMed, FreeLaw, GitHub, StackExchange, USPTO, DM Mathematics, EuroParl — three passes over the schedule, about 6.5 MB of real text per visit. Its forgetting is kept low by replaying its own generated text in each domain's style with stored teacher targets, by Fisher-whitened (K-FAC) updates, and by a small amount of private capacity per domain. After each visit a **sleep phase** runs over what was read. It stores every real sentence that names two capitalised words (tens of thousands per visit), pairs stored sentences by shared names and by a small JEPA head, keeps a pair only if each sentence makes the other cheaper for the model to predict, and replays the accepted splices into the weights under a brake. Alongside the real text, the harness plants a small set of template facts about invented people — `Xaquimon works with Zotarbel.`, six relations with a paraphrase each, 30 copies per fact per pass — as a measurement instrument. A hand-designed reader pulls these out, joins facts that share a name into two- and three-hop conclusions (`So A is connected to C.`), checks that the sources are still present, and teaches the conclusions to the model by replay and by closed-form MEMIT-style edits to one MLP matrix. A separate **fast store** — a hash table of logit corrections keyed by names found in the text — answers questions about these facts directly, and a protocol releases facts from the store once the weights answer them alone, deletes them after a grace period, and writes them back if the weights lose them. From 11 November the recorded facts are also **taught to the trunk while it reads** (interleaved distillation): four question rows about stored facts, in seven wordings the system writes itself (none of them a test wording), are added to each training step and update only the blocks' MLP output weights; a fixed fifth of the facts, chosen by a hash, is never taught and serves as the control. The reader only works on template-shaped facts, so every recall figure in this README is on the planted set or, for the distillation rows, on recorded facts in the same six relation phrases (38-60% of them planted); whether the real-text dreams do anything measurable is an open control.

**The model does not infer anything.** The reader and the join are symbolic. The model takes part in three places: it learns the real text, it judges proposed pairs (a pair passes if each sentence makes the other cheaper to predict by at least 2 bits, or by the 97th percentile of random pairs if that is higher), and it is taught the conclusions that survive. Tests on 11 November confirm this division of labour: facts taught into the trunk are recalled from it, but the trunk does not combine two of them into a conclusion it was never given, and neither does a JEPA-style predictor trained on the trunk's own name codes (see "What has been measured"). Composition happens in the symbolic sleep layer, which answers with its sources or abstains. The JEPA index's proven role is retrieval: finding the stored sentence a question needs, including in wordings never written anywhere. On branch `cg27a-distill` (not yet merged) that sentence can be put in front of the question and cited as the answer's source (`inf_jx_rag`).

## What it is not

- **Not a system that learns facts from real text.** Every recall, abstention, revision and consolidation figure in this repository is on planted template sentences. The dream path does read real text — splice cues and partners come from the stored real sentences, and about half the sleep replay rows are such splices (`inf_mix_dream 0.5`) — but no gain from them has yet been isolated from the rest of sleep. On 262 Re-DocRED Wikipedia paragraphs the reader's entity precision is 0.49 and its relation pairs are gold at the rate of chance (0.28 vs 0.29). The trunk shows no relational signal on those pairs either (AUC 0.45–0.50, local). See `tools/redocred_reader_test.py`.
- **Not a model that composes facts in its weights.** Given both premises of a chain and no written conclusion, two-hop recall from the weights is 0 of 11 locally and 0.17 on the GPU (0.95 with the premises in context <!-- not yet in RESULTS.md: add the source run or remove -->). The local figure is expected and says little: implicit two-hop composition appears only after long training well past memorisation (Wang et al. 2024), and the local harness is a 512-wide model trained 400 steps per visit; its 400-step ceiling test memorised without composing, too short to judge. The GPU figure is a weak-powered observation from ordinary training. The designed test — `CEILING LONG`, 2,000 steps with in- and out-of-distribution chains — has not yet returned a GPU result (it was the end-of-run readout that ran out of memory; it now runs last, in low-memory mode). Open-ended production of a held-out-wording two-hop answer from the weights alone is 0.00 in the configured build (0.04 in shadow experiments). Interleaved distillation does not change this: on the GPU (seeds 4 and 5) conclusions in the weights come from the closed-form edits (0.47-0.69 with them, 0.08-0.18 with them subtracted), and never-taught conclusions whose two premises were both taught are at 0.00-0.08 with the edits subtracted (6-14 questions). With every 4th planted chain never concluded, stored, replayed or edited (`inf_kgx_hold 4`), the trunk composes **0.00** of the held-out chains whose two facts it was taught, including those where it answers both premises (GPU, seed 4: 5-11 chains, 3-7 with both premises answered, in two runs). Bridge anchoring (after BRACE, NeurIPS 2026: one fixed attribute of the bridge entity added to both hops' question rows) is inconclusive: the planted bridge names occur only in their own chain, so the attribute had to be invented, and only 2 anchored chains became testable. A JEPA-style composer trained on the chains the system concluded answers the unconcluded ones at 0.00-0.29 on 1-11 chains (0.09 at 11; chance 0.05): it memorises its training chains.
- **Not continual learning without access to the data.** The "closed loop" (passes 2 and 3) still draws the real training batch at every step. The real-text cross-entropy is multiplied by zero; the batch is instead encoded by a frozen copy of the model into per-token key/value traces, decoded back to bytes, checked for consistency with the trace, and the decoded bytes train the student. The fast store reads the raw batch directly (`cd_store_text`). With `PLATYPUS_HINTS=1` (the default) 2–8% of the true bytes are also stored and forced. The traces for an 8-byte segment are per-token vectors of width 1088 across three blocks: far larger than the bytes they encode.
- **Not benchmarked against other methods.** Internal comparisons exist and are in [`docs/README-full.md`](docs/README-full.md): dream rehearsal on and off (local, October: forgetting +0.228 without it, +0.021 with it), private capacity on and off, plain against Fisher-protected replay at matched doses, the closed-loop bounds on the GPU (second pass learns nothing 1.739 BPC / verified replays only 1.474 / real text again 1.468), trunk replay against none on the small model, sleep against quiet rest, the oracle options on and off, and (local full runs, 11 November) dream replay against none with distillation off: forgetting at the end of the run -0.005 with replay against +0.071 without (final BPC 2.415 against 2.484), and at half width without replay +0.32 at the end of pass 1; and the system has run at two sizes, the five-domain local harness and the 171M GPU model. What is missing is a comparison with established methods — EWC alone, stored-example replay, per-episode adapters — on the current build and schedule, and any public benchmark result. Pass-1 headlines now have two GPU seeds (1 and 2, 9 November); the three-pass numbers have one. Local results differing by less than about 0.15 in recall or 0.017 in BPC are seed noise.
- **Not a trained hippocampus.** The fast store is a dictionary: a regex (or the "two rarest words" rule) finds a name, the name selects a bank of rows, and the rows add corrections to the output logits. "Exact recall" and "abstention on unknown names" are properties of a lookup table.

## What has been measured

All GPU figures: Colab, pass 1, seeds 1 and 2 where two values are given (seed 1 first), otherwise seed 1 unless the row says otherwise. "From the weights" means fast store off. "Held-out wording" means a question template that is never written anywhere by the system. Multiple choice is among 10 or 20 candidate names as stated.

| Measurement | Result | Conditions | What it does not show |
|---|---|---|---|
| Mean BPC after pass 1, weights only, domain known | 1.743 / 1.725 | GPU (store on: 1.744 / 1.725). From build `f10d4ef` the run ends with a `HEADLINE WEIGHTS ONLY (fast store off)` line | Pass 2–3 forgetting rose to +0.06–0.08 in earlier builds; no baseline run |
| Forgetting after pass 1 (mean rise in old-domain BPC) | +0.012 / +0.006 | GPU, domain known, self-contrast build (label-free: -0.002 / -0.006) | Measured with the per-domain private parts on; switching them off costs +0.14 BPC |
| Label-free BPC (domain inferred from the text) | 1.731 / 1.712 | GPU, self-contrast build; +0.013 over domain-known locally | Routing relies on a cap of one slot per domain and a "not the last slot" rule |
| Two-hop recall with the fast store, written wording | 0.97–0.99 | GPU, planted facts, 30 copies each | A lookup over regex-extracted triples |
| Two-hop from the weights, written wording, 20 candidates | 0.71 / 0.70 | GPU, end of pass 1, self-contrast build | Open-ended (no candidates): 0.54 / 0.45 |
| Two-hop from the weights, held-out wording, 20 candidates | 0.40 / 0.33 vs 0.21-0.22 without the write side | GPU, end of pass 1 | Open-ended held-out: 0.02 / 0.00 |
| Open-ended recall from the weights, fitted wording, edited conclusions only | 0.62 vs 0.20 without self-contrast | GPU, shadow edits, slot given, 74 conclusions | Shadow measurement, not the configured build; one seed |
| Temporal gradient: store-lesioned two-hop, old vs recent conclusions | 0.87–0.91 vs 0.51–0.59 (seed 1, three runs); 0.91 vs 0.53 (seed 2) | GPU. Seed 2 by exposure: never-edited conclusions 0.78 (older) vs 0.50 (recent); three or more sleeps in the edit 0.96 | Age still confounded with the number of replays (not yet split) |
| Release/reinstate protocol | 193 released, 10 restored | GPU, one seed, written wordings only | In pass 2 of an earlier build a third of released memories had to be restored |
| Verified replay, no hints | 0.87–0.99 of 8-byte segments rebuilt exactly | GPU (locally 0–1%) | Costs ~0.04 BPC per pass vs hinted; traces are larger than the text |
| JEPA dream selector vs random pairs | 20–40% of chosen pairs share a gold link vs ~1% random | Measured with benchmark-entity cues (now off) | Not yet shown to improve recall or forgetting over random dreams; the control run is pending |
| Reader on real prose (Re-DocRED) | entity precision 0.49, pair precision 0.28 (chance 0.29) | offline, 262 paragraphs | — |
| The trunk's state over a subject's name finds that subject's conclusion from a held-out question | 0.97 / 0.98 (ceiling 0.98) | GPU, readout; the edit's own key: 0.20–0.53 <!-- not yet in RESULTS.md --> | Finding the fact, not producing the answer |
| A ROME-style weight edit at the subject reaches the answer | No: equals no edit in every sleep (GPU and local); the best change at the name buys ~0.2–0.3 bits per answer byte <!-- not yet in RESULTS.md --> | Shadow readout | The trunk has no path from subject to answer; facts planted in one template are the likely cause, untested |
| Name-keyed memory with a learned answer-slot gate, no hint where the answer starts | first answer byte 0.07–0.25 (told the slot: 0.10–0.38); open-ended 0.06–0.09 in three of six sleeps | Local, two seeds, readout | Far below the fast store (0.97 in unseen wordings); parked as a readout |
| Facts taught by interleaved distillation, recalled from the weights, open-ended, in wordings never taught | taught 4+ visits: 0.69 / 0.71; 2-3 visits: 0.73 / 0.61; still being taught: 0.55 / 0.46; never-taught facts 0.06 / 0.01 | GPU, seeds 4 / 5, sleep 5, build `cg27a_distillcap` (pool of about 410 facts, MLP output weights only), over a fixed sample of every fact ever taught | Two seeds, pass 1 only; facts in six relation phrases, not real prose |
| The same build with no question rows trained (control) | fact recall from the weights 0.00-0.03 | GPU, seed 4, `cg27a_distillctl` | The recall above is the distillation's, not replay's or the edits' |
| Cost of the question rows | old domains' BPC at each sleep +0.018 to +0.023 against the control, steady (visited domain: none); local full runs +0.018 final BPC | GPU seed 4, sleeps 2-5 (fast store on); local seed 0 | End-of-run forgetting not yet measured on the GPU |
| Where the question rows write | 0.95 of a fact update lies in the 20-30 (of 512) MLP input directions that carry 90% of old-domain text (an update spread evenly: 0.05); on the GPU (8,704 units) 0.87-0.92 in 300-450 directions (evenly: 0.03-0.05) | Local and GPU readout (`inf_distill_ns_probe`) | Why the cost exists, why pulling those directions back erases the facts, and why a null-space projection would keep only 0.05 of the update |
| Tested on top of distillation and not kept | spaced rehearsal (GPU: equal recall); repair replay (GPU: erased the new facts, 0.31 -> 0.06); prompt distillation (the trunk does not read answers from its context: 0.47 -> 0.49 answer bytes with the source sentence); theta-rhythm dream gating (local full runs, none / stochastic / fixed phases: forgetting +0.014 / +0.008 / +0.011, final BPC 2.428 / 2.438 / 2.432); a soft anchor to the old domains' predictions (local, two seeds: taught recall 0.22 / 0.14 vs 0.11 / 0.08 at the same cost; GPU: old-domain drift 0.163 vs 0.186 but recall of facts taught 2-3 visits 0.49 vs 0.73 - a trade, not a gain); fact units, new MLP units per fact (local: about 2x selective, but no fact learning yet); hinted self-distillation writing only the MLP output weights (neutral); a beta-rhythm brake, spindle coupling, self-KL, an anchor-gradient projection | GPU and local, 11 November | Each one seed unless stated; details in `RESULTS.md` |
| Composition of two taught facts, held-out chains (never concluded, stored, replayed or edited) | trunk 0.00 (5-11 chains); JEPA-style composer 0.00-0.29 (0.09 at 11 chains, chance 0.05); two JEPA retrievals in a row 0.70 vs 0.58 with a random bridge (50 chains) | GPU, seed 4, `cg27a_distillbr2` and `cg27a_distillak`, sleeps 2-7 | Composition stays in the symbolic sleep layer |
| JEPA index as a memory key: the stored sentence for a question about A (sentences naming A, ranked by the JEPA key) | found 0.91-1.00 in the written wording, 0.77-0.85 in a held-out wording (the most recent sentence naming A: 0.55); one-hop answers among 20 candidates 0.72 -> 0.86 with that sentence in context (79 questions) | GPU, seed 4, sleeps 2-7, readout | Open-ended answering with the sentence, and its citation, is measured by `cg27a_rag` (branch `cg27a-distill`), not yet run on the GPU |

Earlier builds reported higher numbers measured with the domain label given, with summed byte scores that favour short names, with oracle options on, or on training text in the local harness. Those were corrected in `eval-fix` (8 November) and a second review (9 November); see [`docs/README-full.md`](docs/README-full.md#known-issues-and-their-fixes-8-november-2026-fixed-in-eval-fix-and-carried-into-cg27a).

## What is of merit

The parts most likely to be useful to someone else are infrastructure, not the memory claims:

- **`layers.py`**: a gated delta-rule mixer with five interchangeable solve paths — a Triton kernel (flash-linear-attention), a blocked WY solve with a hand-derived backward pass, a chunked state-passing form, a windowed form and a dense fallback — chosen per head dimension by a start-up benchmark. Memory state can be carried across windows as a dense matrix, as factored `(U, K)` write terms, or as fixed-size slot buffers. The mixer reads before it writes (`S_{t-1} k_t`, no separate query) and uses a diagonally preconditioned write key.
- **`protection.py`**: K-FAC Fisher factors per block, batched coupled Newton–Schulz inverse square roots, Muon via Gram Newton–Schulz, Gauss–Newton (VarPro) head refits, gradient-noise-scale estimation with Neyman allocation between replay and real rows.
- **`conduit.py`**: byte-by-byte generation under captured CUDA graphs with static buffers and a parameter-pointer signature check; incremental generation that is arithmetically identical to the full-prefix pass.
- **`ngrams.py`**: hashed n-gram tables with independent hash families, a lazy per-row Adam, and absolute-discount back-off count sketches.
- **`inference/closed_form.py`**: MEMIT-style least-squares edits with the sentence-frame direction removed from the keys, every edit re-solved against its current key each sleep, kept under a loss brake and a cumulative budget.

None of these is a research result; each is a careful implementation of 2017–2026 methods. The protocol-level ideas that are plausibly new — re-keying edits every sleep, release → ghost → reinstate timed by half-life regression, a replay signal gated on reproducing the encoder's own trace — are each tested on one or two seeds, on templates, without a baseline.

## Known defects

- The fast store is installed by replacing `Model.forward` and `Head.forward` at import time. It should be an explicit wrapper.
- The package is a 19k-line single-file program split into 51 modules: names are wired across modules by `_LATE` tables and `common._set_global`, module-level state is rebound at run time, and the config has ~640 fields of which most are off. The code carries dormant buffers and options from earlier designs (per-unit affinities, impedance gates, a differentiable meta-controller). Brakes are thresholds, not zero-tolerance: 0.015 BPC for closed-form edits, 0.03 for dream replay, 0.005 for consolidation.
- 145 `except Exception` handlers print and continue. A run-level `FAILURES:` count was added in `eval-fix`, and `scripts/check_undefined.py` (pyflakes minus `_LATE` names) now runs in CI after four undefined names were found behind off-by-default options.
- Tests check that modules import and link, that no name is undefined, and that scripts parse their fixtures; nothing tests model behaviour.
- Variable naming (`rf9n_`, `mo9n_`, `EN9_`) and 300+ lines over 200 characters make the sleep and edit code hard to review.

## Direction

The standing goal (8 November 2026) is to replace the trained parts of sleep with simple local rules: a slow-feature layer on the trunk in place of the trained JEPA selector, and eventually a learned reader in place of the hand-designed one, with the fast store kept to one job — making facts met far apart adjacent for replay. It is pursued as rungs, each with a pass criterion, recorded in [`RESULTS.md`](RESULTS.md).

- *Rung 0 — closed-form slow features vs the trained JEPA.* Ties on held-out-wording retrieval (0.15 vs 0.12), loses in the written wording (0.07–0.15 vs 0.38–0.42); names alone beat both (0.70–0.90). Local.
- *Rung 1 — slow features on the trunk.* Met by a closed-form SFA fit re-solved each sleep, which the latent judge and selector comparison already use (0.17 / 0.38 top-1 in unseen wordings). The online version — a small head trained by prediction, variance and decorrelation terms, plus three alternative rules — did not reach the closed-form solution in eight attempts and is parked. The per-sleep fit serves the purpose; the "online, local rule" part of the goal remains open.
- *Rung 2 — a latent judge (SFA cosine) for dream pairs.* Matches the byte judge on known-linked controls (AUC 0.92–1.00) at about 1 ms per pair against 20–40 ms, but the controls share a bridge name by construction; on real proposals its agreement with the byte judge is AUC 0.64–0.69. *Rung 2b*, a relational signal on real prose from any latent readout: negative.
- *Rung 3 — the name channel.* The lead described under Open questions.

Read strictly, the latent signals so far (JEPA, SFA, latent judge) match TF-IDF on names wherever a shared rare name exists and are at chance where none does.

## Open questions, in order

1. Does anything here beat EWC alone, stored-example replay or per-episode adapters on the same ten-domain schedule? Not yet run (the internal ablations under "Not benchmarked" compare the system's own parts, not other methods).
2. Do JEPA-chosen dreams beat random dreams on recall or forgetting? Control pending. (As a retrieval key the JEPA index is clearly useful; see the table.)
3. Is the temporal gradient consolidation or replay count? It holds on a second seed and among never-edited conclusions; replays per conclusion are not yet counted.
4. Can the weights answer in a wording they were never taught, open-ended, above 0.00? **For single recorded facts, yes, on two GPU seeds:** taught by interleaved distillation, 0.6-0.7 after a few visits in untrained wordings against 0.01-0.06 for never-taught facts. For conclusions, not yet (see 7). Earlier leads in `RESULTS.md`: a name-keyed memory (shadow, 0.15 with oracle answer positions; 0.06–0.09 locally with a learned gate). Ruled out: a ROME-style subject edit, and replaying facts in varied wrappers.
5. Can any reader extract relations from real prose above chance? Until it can, nothing downstream applies to real text.
6. Does the closed loop work with the training batch removed — traces from pass 1 only, no `cd_store_text`, no hints, store off for BPC? That experiment is not implemented.
7. Can the trunk compose two taught facts into a conclusion it was never taught? **No, at this scale:** 0.00 on held-out chains, including chains whose premises it answers; a JEPA-style composer does not generalise either. Bridge anchoring could not be tested properly on planted names. Composition is left to the symbolic layer.
8. Can facts be written where old domains do not read, to remove the +0.02 BPC cost? A null-space projection cannot work at this width (see above). Tested locally: fact units - new MLP units, one per new fact, keyed on its question context and the only weights the question rows train. With the inputs centred they are about 2x selective, but they did not learn the facts (taught recall 0.00 against 0.11); the cost stays.
9. Does the trunk answer better, open-ended, with the JEPA-retrieved sentence in front of the question, and is the cited source right? Among 20 candidates, yes (0.72 -> 0.86); open-ended is the `cg27a_rag` run.

## Getting started

Python 3.10+, PyTorch with CUDA, `datasets` (for `monology/pile-uncopyrighted`), `scipy`. Optional: `flash-linear-attention` for the Triton kernel; the code falls back to a pure PyTorch solve.

```bash
pip install torch datasets scipy
python -m platypus                     # from the repository root
```

In Colab use `colab/platypus_gpu.ipynb`, which runs `scripts/run_gpu.sh`. A three-pass run takes four to five hours on one large GPU. `scripts/run_local.sh` runs a five-domain CPU harness. Presets are in `configs/presets/`; the `PLATYPUS_*` switches are listed in `platypus/__main__.py`. Deployment, manifests and the release checklist are in [DEPLOY.md](DEPLOY.md).

Code layout, configuration options, log-line glossary, history and references: [`docs/README-full.md`](docs/README-full.md).

## A note on this documentation

`docs/README-full.md`, `RESULTS.md` and most comments were written with an AI model. Its literature citations — in particular the 2026 preprints named in the "Novelty and prior art" section — have not all been verified to exist and should be checked before being relied on or repeated. The percentage estimate of novelty given there is a model's guess, not a measurement. This README was also drafted with an AI model, from a reading of the code rather than of the earlier documentation, and reviewed by the author.

## Contributing

Issues and pull requests are welcome. The most useful contributions are the baselines and controls listed under Open questions, a behavioural test suite, and the cleanup items under Known defects.

## Citing

```bibtex
@software{bird2026platypus,
  author  = {Bird, Christian},
  title   = {Platypus: a continual-learning byte-level language model with a symbolic memory and sleep cycle},
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
