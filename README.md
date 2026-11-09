# Platypus

**A byte-level language model trained on ten text domains in turn, with a symbolic memory and sleep cycle beside it that extracts planted template facts, joins them, and teaches the model the results.**

> Copyright 2026 **CHRISTIAN BIRD**. Licensed under the [Apache License 2.0](#licence).

This is a one-person research project, two months old, run on a single Colab GPU. This README says what the code does and what has been measured, in plain terms. The full record of every build, mechanism, readout and result (136 KB, generated with the help of an AI model, with the caveats that implies) is kept at [`docs/README-full.md`](docs/README-full.md). The results log is [`RESULTS.md`](RESULTS.md).

Current build: **cg27a** (branch `cg27a`). GPU preset: `configs/presets/cg27a.env`.

---

## What it is, in one paragraph

A 171M-parameter language model (byte vocabulary, hashed 2/3/4-gram embeddings, one causal local-filter block, three gated delta-rule mixer blocks, one output head per domain) is trained on ten Pile domains one after another, three passes over the schedule. Its forgetting is kept low by replaying its own generated text with stored teacher targets, by Fisher-whitened (K-FAC) updates, and by a small amount of private capacity per domain. Into each domain's text, the harness plants short template sentences about invented people — `Xaquimon works with Zotarbel.` — six relation phrases, names built from syllables containing q, x or z, 30 copies of each fact per pass. After each training visit a **sleep phase** runs: a hand-designed reader pulls these sentences out of the stream, joins facts that share a name into two- and three-hop conclusions (`So A is connected to C.`), checks that the source sentences are still present, and then teaches the conclusions to the model by replay, by spliced "dream" windows chosen by a small JEPA head, and by closed-form MEMIT-style edits to one MLP matrix, each under a brake that rolls back any step that raises old-domain loss. A separate **fast store** — a hash table of logit corrections keyed by names found in the text — answers questions about these facts directly, and a protocol releases facts from the store once the weights answer them alone, deletes them after a grace period, and writes them back if the weights lose them.

**The model does not infer anything.** The reader and the join are symbolic. The model takes part in two places: it judges proposed links (a pair of sentences passes if each makes the other cheaper to predict by at least 2.2 bits), and it is taught the conclusions that survive.

## What it is not

- **Not a system that learns from real text.** Every recall, abstention, revision and consolidation figure in this repository is on planted template sentences. On 262 Re-DocRED Wikipedia paragraphs the reader's entity precision is 0.49 and its relation pairs are gold at the rate of chance (0.28 vs 0.29). The trunk shows no relational signal on those pairs either (AUC 0.45–0.50). See `tools/redocred_reader_test.py`.
- **Not a model that composes facts in its weights.** Given both premises of a chain and no written conclusion, two-hop recall from the weights is 0 of 11 locally and 0.17 on the GPU (0.95 with the premises in context). Open-ended production of a held-out-wording two-hop answer from the weights alone is 0.00 in the configured build (0.04 in shadow experiments).
- **Not continual learning without access to the data.** The "closed loop" (passes 2 and 3) still draws the real training batch at every step. The real-text cross-entropy is multiplied by zero; the batch is instead encoded by a frozen copy of the model into per-token key/value traces, decoded back to bytes, checked for consistency with the trace, and the decoded bytes train the student. The fast store reads the raw batch directly (`cd_store_text`). With `PLATYPUS_HINTS=1` (the default) 2–8% of the true bytes are also stored and forced. The traces for an 8-byte segment are per-token vectors of width 1088 across three blocks: far larger than the bytes they encode.
- **Not benchmarked.** There is no fine-tuning, EWC or replay baseline run under the same conditions, no public benchmark result, and no second GPU seed for any headline number. Local results differing by less than about 0.15 in recall or 0.017 in BPC are seed noise.
- **Not a trained hippocampus.** The fast store is a dictionary: a regex (or the "two rarest words" rule) finds a name, the name selects a bank of rows, and the rows add corrections to the output logits. "Exact recall" and "abstention on unknown names" are properties of a lookup table.

## What has been measured

All GPU figures: one seed (1), Colab, unless stated. "From the weights" means fast store off. "Held-out wording" means a question template that is never written anywhere by the system. Multiple choice is among 10 or 20 candidate names as stated.

| Measurement | Result | Conditions | What it does not show |
|---|---|---|---|
| Mean BPC after pass 1, weights only, domain known | 1.743 | GPU, seed 1 (store on: 1.744) | Pass 2–3 forgetting rose to +0.06–0.08 in earlier builds; no baseline run |
| Forgetting after pass 1 (mean rise in old-domain BPC) | +0.002 | GPU, seed 1, domain known | Measured with the per-domain private parts on; switching them off costs +0.14 BPC |
| Label-free BPC (domain inferred from the text) | 1.721 | GPU; +0.013 over domain-known locally | Routing relies on a cap of one slot per domain and a "not the last slot" rule |
| Two-hop recall with the fast store, written wording | 0.97–0.99 | GPU, planted facts, 30 copies each | A lookup over regex-extracted triples |
| Two-hop from the weights, written wording, 20 candidates | 0.53 | GPU, visit 9 | Open-ended (no candidates): 0.23 |
| Two-hop from the weights, held-out wording, 20 candidates | 0.29 (0.39 in one run) vs 0.21 without the write side | GPU, visit 9–10, one seed per arm | Open-ended held-out: 0.00 |
| Open-ended recall from the weights, fitted wording, edited conclusions only | 0.62 vs 0.20 without self-contrast | GPU, shadow edits, slot given, 74 conclusions | Shadow measurement, not the configured build; one seed |
| Temporal gradient: store-lesioned two-hop, old vs recent conclusions | 0.91 (n 47) vs 0.50 (n 20) | GPU, two near-identical runs of one seed | Age is confounded with number of replays and re-solves; the exposure control has not run |
| Release/reinstate protocol | 193 released, 10 restored | GPU, one seed, written wordings only | In pass 2 of an earlier build a third of released memories had to be restored |
| Verified replay, no hints | 0.87–0.99 of 8-byte segments rebuilt exactly | GPU (locally 0–1%) | Costs ~0.04 BPC per pass vs hinted; traces are larger than the text |
| JEPA dream selector vs random pairs | 20–40% of chosen pairs share a gold link vs ~1% random | Measured with benchmark-entity cues (now off) | Not yet shown to improve recall or forgetting over random dreams; the control run is pending |
| Reader on real prose (Re-DocRED) | entity precision 0.49, pair precision 0.28 (chance 0.29) | offline, 262 paragraphs | — |

Earlier builds reported higher numbers measured with the domain label given, with summed byte scores that favour short names, with oracle options on, or on training text in the local harness. Those were corrected in `eval-fix` (8 November) and a second review (9 November); see [`docs/README-full.md`](docs/README-full.md#known-issues-and-their-fixes-8-november-2026-fixed-in-eval-fix-and-carried-into-cg27a).

## What is of merit

The parts most likely to be useful to someone else are infrastructure, not the memory claims:

- **`layers.py`**: a gated delta-rule mixer with five interchangeable solve paths — a Triton kernel (flash-linear-attention), a blocked WY solve with a hand-derived backward pass, a chunked state-passing form, a windowed form and a dense fallback — chosen per head dimension by a start-up benchmark. Memory state can be carried across windows as a dense matrix, as factored `(U, K)` write terms, or as fixed-size slot buffers. The mixer reads before it writes (`S_{t-1} k_t`, no separate query) and uses a diagonally preconditioned write key.
- **`protection.py`**: K-FAC Fisher factors per block, batched coupled Newton–Schulz inverse square roots, Muon via Gram Newton–Schulz, Gauss–Newton (VarPro) head refits, gradient-noise-scale estimation with Neyman allocation between replay and real rows.
- **`conduit.py`**: byte-by-byte generation under captured CUDA graphs with static buffers and a parameter-pointer signature check; incremental generation that is arithmetically identical to the full-prefix pass.
- **`ngrams.py`**: hashed n-gram tables with independent hash families, a lazy per-row Adam, and absolute-discount back-off count sketches.
- **`inference/closed_form.py`**: MEMIT-style least-squares edits with the sentence-frame direction removed from the keys, every edit re-solved against its current key each sleep, kept under a loss brake and a cumulative budget.

None of these is a research result; each is a careful implementation of 2017–2026 methods. The protocol-level ideas that are plausibly new — re-keying edits every sleep, release → ghost → reinstate timed by half-life regression, a replay signal gated on reproducing the encoder's own trace — are each tested on one seed, on templates, without a control.

## Known defects

- The fast store is installed by replacing `Model.forward` and `Head.forward` at import time. It should be an explicit wrapper.
- The package is a 19k-line single-file program split into 51 modules: names are wired across modules by `_LATE` tables and `common._set_global`, module-level state is rebound at run time, and the config has ~640 fields of which most are off. The code carries dormant buffers and options from earlier designs (per-unit affinities, impedance gates, a differentiable meta-controller).
- 139 `except Exception` handlers print and continue. A run-level `FAILURES:` count was added in `eval-fix`, and `scripts/check_undefined.py` (pyflakes minus `_LATE` names) now runs in CI after four undefined names were found behind off-by-default options.
- Tests check that modules import and link; nothing tests behaviour.
- Variable naming (`rf9n_`, `mo9n_`, `EN9_`) and 300+ lines over 200 characters make the sleep and edit code hard to review.

## Open questions, in order

1. Does anything here beat a plain fine-tuning baseline, EWC, or stored-example replay on the same ten-domain schedule? Not yet run.
2. Do JEPA-chosen dreams beat random dreams on recall or forgetting? Control pending.
3. Does the temporal gradient survive an exposure-matched control and a second seed?
4. Can the weights answer in a wording they were never taught, open-ended, above 0.00? The name-keyed binding-memory experiment in `RESULTS.md` (0.15 in shadow with oracle answer positions) is the only lead.
5. Can any reader extract relations from real prose above chance? Until it can, nothing downstream applies to real text.
6. Does the closed loop work with the training batch removed — traces from pass 1 only, no `cd_store_text`, no hints, store off for BPC? That experiment is not implemented.

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
