# Platypus

**A language model that learns without forgetting and thinks in its sleep.**

Platypus is a 171-million-parameter byte-level language model that learns ten text domains one after another without forgetting earlier ones. After its first look at the data it learns only from what it can verify, and while it "sleeps" it links facts it met in different places into new conclusions. It needs no domain labels, no stored training data and no hand-written language rules. Its memory and sleep cycle also work when attached to a frozen GPT-2.

> Copyright 2026 **CHRISTIAN BIRD**. Licensed under the [Apache License 2.0](#licence).

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
- **Label-free routing.** The domain is never given. The model recognises returning domains by how well it predicts them.
- **Inference in sleep.** After each visit it links facts met in different places, replays salient facts (NREM-like), proposes links from memory and keeps only grounded ones (REM-like), revises beliefs and retracts conclusions that depended on superseded facts.
- **A fast memory that can abstain.** Conclusions live in an entity-banked, delta-rule logit corrector outside the weights. It is silent for entities it has never seen, so the model can say "I don't know", and it costs almost nothing on ordinary text.
- **An identifier memory for exact chains.** Verified facts are bound into a vector-symbolic memory, so multi-step chains need no round trip through text, and broken chains are declined.
- **Rules that pay for themselves.** Equivalences between relation phrases ("answers to" and "reports to") are discovered from the text and kept only if they compress held-out evidence (minimum description length), so paraphrased questions can be answered.
- **Works as a plug-in.** The store and sleep cycle have been attached to a frozen GPT-2 without changing a weight.

## How it works

```
             BY DAY                                         BY NIGHT
 real text ──► slow network ──► next-byte          dream rehearsal ─► verified replay ─► linking
 (one domain,   (171M params,    prediction ◄──┐                                        & revision
  no labels)     general skill)                 │                                           │
                                                └──── corrects ◄──── fast memory ◄──────────┘
                                                                  (facts by entity, silent when unsure)
```

The slow network learns general skill, and only from real text or verified replay. The fast memory holds facts. Sleep connects the two by doing inference offline.

**Architecture (GPU build).** 1024-byte windows; byte plus hashed 2/3/4-gram embeddings; block 0 is a causal local filter; blocks 1–3 are gated delta-rule mixers (8 heads, WY-form parallel solve); residual width 1088; one soft-capped output head per domain. About 171M trunk parameters plus 223M hashed n-gram parameters.

**Key principles**

| Decision | Principle |
|---|---|
| Dream rehearsal plus Fisher-whitened steps | Sequential Bayesian learning under a Laplace approximation: dreams supply the old likelihood, curvature supplies the precision. |
| Learn only from verified replay | Self-training is a fixed-point iteration that converges on its own errors. A truth gate removed 100 of 100 false attractors in a synthetic test. |
| Delta-rule fast memory | Writes are error-driven, so known facts cost nothing; unknown entities have no code, so abstention is exact; a confidence gate keeps the cost on ordinary text near zero. |
| Linking by specificity | Links are scored by pointwise mutual information, not raw probability, and high-fan-out bridges abstain. |
| Recency rule for belief revision | Exponentially decaying supports keep a constant ratio, so ratio tests stall. The most recent repeated statement wins, and retraction cascades through dependants. |
| Entities from surprise | A word becomes an entity when the model finds its completion surprising, measured against the model as it is now. |
| Rules by MDL | A rule is kept only if its two-part code saves bits on held-out evidence, judged against the going rate of the rules already paying their way. |

## Results

**Continual learning (full size, ten Pile domains).** With dream rehearsal and curvature-aware steps the model reached 1.17 bits per character (BPC) by its third pass, with forgetting of 0.03. In the strict closed loop (no real text after the first pass):

| Second pass | BPC | Forgetting |
|---|---|---|
| Learns nothing new (lower bound) | 1.739 | – |
| Learns only from verified replays | 1.474 | 0.049 |
| Reads real text again (upper bound) | 1.468 | 0.044 |

Verified replay recovered 98% of the gap, with no domain labels.

**Inference in sleep (five-domain harness, planted fact chains).** Two-step recall reached 0.91 with forgetting of 0.014, against 0.30 recall and 0.206 forgetting when conclusions were written into the weights. Store cost on ordinary text is 0.004–0.006 BPC. Belief revision took stale two-step answers from 40% to zero.

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
- Optional: `flash-linear-attention` for the fast Triton kernels. The script tries to install them and falls back to a pure PyTorch solve if they are unavailable.

**Run**

```bash
pip install torch datasets
python threshold_lean_stagec9_7cg20s_conduit.py
```

In Colab, upload the script, choose a GPU runtime and run it with `!python threshold_lean_stagec9_7cg20s_conduit.py`. If you need credentials (for example a Hugging Face token), keep them in Colab Secrets or environment variables. Never commit them to the repository.

A full run streams ten domains over several passes and takes several hours on a single modern GPU.

## Configuration

Options are set through the `LAB_OVR` environment variable as JSON, on top of the `Config` defaults. Anything you pass overrides the built-in run configuration.

```bash
LAB_OVR='{"n_passes": 3, "steps_per_task": 400, "inf_on": true, "inf_rules": 1, "seed": 1}' \
  python threshold_lean_stagec9_7cg20s_conduit.py
```

Useful switches:

| Option | Meaning |
|---|---|
| `n_passes`, `steps_per_task` | Passes over the domain sequence, and training steps per visit |
| `seed` | Random seed |
| `inf_on` | Sleep-time inference and the fast memory |
| `inf_mem_gate` | Confidence gate for applying fast-memory corrections |
| `inf_audit_rule`, `inf_audit_cascade` | Belief revision rule and cascading retraction |
| `inf_idmem`, `inf_id_dim` | Identifier memory and its dimension |
| `inf_rules`, `inf_para_frac` | Paraphrase rule layer, and the fraction of probes given a second wording |
| `pu_cons_brake` | Rolls back consolidation if it harms old domains by more than this many BPC |

The full list, with a change history, is in `NAME` and `LAB_OVR` near the end of the script.

## Reading the logs

- Per-visit **BPC** and **forgetting** for every domain seen so far
- **Routing**: which slot each visit was assigned to
- **Inference readouts**: one-, two- and three-hop recall from the weights and with the store; answer-or-abstain; sources; updated facts; store cost on ordinary text
- **IDENTIFIER MEMORY**, **PARAPHRASED** and **RULE LAYER** lines for the identifier memory and the paraphrase rules
- **ROLLED BACK** when the consolidation brake undoes a harmful consolidation step

## Project status

This is **research code**. It is bulky: it carries options, readouts and experiment paths from many iterations that are switched off in the current configuration. It needs a cleanup (dead options removed, modules split, tests added) and further optimisation (evaluation frequency, store writes, data loading and generation speed) before wider use.

Known limits:

- Inference has been tested on templated facts about invented people, not on natural text about real entities.
- The paraphrase layer handles relation wording; pronouns, descriptions and nested clauses are not handled yet.
- Word-level rule discovery with no relation list currently finds 33–40 of 58 planted chains, against 57 of 58 when relations are given. Work on this is ongoing.
- Inferred knowledge stays in the fast memory; moving it into the weights has not worked yet.
- There is no fine-tuning baseline and no benchmark result yet.

## Roadmap

1. **Learned reader.** Attach the memory and sleep cycle to a pretrained model that reads sentences into fact records, verified against their sources.
2. **References.** Resolve pronouns and descriptions as rules that must pay for themselves.
3. **Meaning on links.** Carry negation, hedging and time on each relation, so past and negated facts are not asserted as current.
4. **Real text and benchmarks.** MQuAKE for multi-step edits, two-hop question sets, and a fine-tuning baseline on the same facts.

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
