# Platypus

**A language model that learns without forgetting and thinks in its sleep.**

Platypus is a 171-million-parameter byte-level language model that learns ten text domains one after another without forgetting earlier ones. After its first look at the data it learns only from what it can verify, and while it "sleeps" it links facts it met in different places into new conclusions, checks them against their sources and keeps only the ones that hold up. It then dreams: a trained JEPA picks which memories to splice together, the model judges each dream by whether the two halves make each other easier to predict, and what passes is consolidated into its weights under a brake that protects old knowledge. It needs no domain labels, no stored training data and, in its default rule-free mode, no hand-written language rules: entities and relations are found from word statistics, and links are kept only if they pay for themselves in compression. Its memory and sleep cycle also work when attached to a frozen GPT-2.

> Copyright 2026 **CHRISTIAN BIRD**. Licensed under the [Apache License 2.0](#licence).

Current build: **cg26o**.

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
- [History](#history)
- [Contributing](#contributing)
- [Citing](#citing)
- [References](#references)
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
- **Cue repair.** At question time, a name the fast memory does not know (misspelt, truncated or lower-cased) is mapped to the stored name it most plausibly is: within two edits, strictly nearer than any other name, and never a common word. Ties abstain.
- **Dreams, judged by the model itself.** NREM-like dreams splice two stored sentences; REM-like dreams pair a recent memory (the cue) with a trace of older knowledge, chosen neither too close nor too far from the cue and different from each other. A dream is kept only if each half makes the other cheaper for the model to predict than random text does. A trained JEPA (LeJEPA objective, trained on the links the system verified, on what the dream judge accepts and on the measured outcome of earlier dreams) ranks the candidates from every source that shares a rare name with the cue, and one pair per cue is dreamed: 20–40% of these choices are about a gold-linked fact on Wikipedia-style text, against about 1% for random pairs.
- **Systems consolidation into the weights.** Verified conclusions and accepted dreams are replayed into the trunk, with the update steered away from what old domains rely on (their Fisher factors), a brake that rolls back any harmful sleep, and a dose that retries lower and climbs back when it can. Verified conclusions are also written straight into the weights in one least-squares step (closed-form, MEMIT-style edits), with the sentence frame removed from the keys, under their own brake and a cumulative budget.
- **Handing memories over to the weights.** Each sleep the model tests every tracked conclusion with the fast memory switched off. One the weights answer on their own, with a margin, at consecutive sleeps is released: erased from the fast memory and kept in replay. If the weights then keep answering it, the record is deleted after a grace period, so the weights are its only copy. If they stop, it is written back. Once a learned model of forgetting predicts better than a constant rate, it also releases the memories it predicts the weights are most likely to still hold at the next visit (either rule releases), and a deleted memory the weights lose again is reinstated: written back to the fast memory and to replay. Replay is spaced, aimed at what the weights are losing, and lighter for memories that fit what the model already knows (a schema): on the GPU such memories pass the test without the fast memory three to five times as often as misfits, and every release so far has been one of them.
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
| JEPA as the dream selector | A joint-embedding predictive model learns from links the system has verified itself, then predicts which memories belong together. It uses the LeJEPA objective (one embedding for both views, prediction to their centre, SIGReg towards an isotropic Gaussian), is also trained on what the dream judge accepts and on whether dreamed pairs later helped, and its ranking is averaged with the judge's, so it cannot win by construction. Its closed-form relative (spectral word codes from the platonic-representation result) seeds the splice proposals. |
| Release, then delete | A memory leaves the fast store only after the weights have answered it alone at consecutive sleeps, and its record is deleted only after a further grace period of passing tests. A memory that fails is written back. The store shrinks without losing what the weights cannot yet hold. |
| Forgetting as a model | Recall without the store, d visits after the last replay, is modelled as 2^(−d/h), with a half-life h learned from replays, their spacing and what the trunk already knows (half-life regression). It may time release only after it beats a constant rate on data it has not seen. |
| Spaced, targeted replay | Replay rows are capped per memory per sleep, counted only when the replay is kept, and weighted towards memories the weights failed and away from those that fit a schema; anchor windows follow the damage each old domain took. |
| Either rule releases | A calibrated model of forgetting may time release, but it does not switch off the fixed rule: on the GPU its half-lives (about 0.3 visits) never reached its 10-visit horizon, and releases had stopped. |
| Release by relative predicted recall | With half-lives of a fraction of a visit, an absolute bar on predicted recall never fires. The model of forgetting ranks memories instead: those in the top quarter of predicted recall at the next visit, at least twice the median and at least 0.2, are released; if the model is degenerate it stays idle. |
| Reinstate what fades | A deleted memory whose ghost fails the test without the fast memory at two sleeps running is written back, unless one of its sources has been superseded. Deletion can then be tried early without losing what the weights cannot hold. |
| Closed-form edits with centred keys | Writing a fact into one weight matrix in closed form is cheap, but its keys are mostly the sentence frame, so the edit also moves everything said in that frame. Removing the frame direction (about half the keys' energy) lets the edit act on what is specific to each fact. Each edit is kept only under a brake, and edits stop once their total cost reaches a budget. |
| Cue repair before abstaining | An exact, name-keyed memory is silent for a misspelt name. Mapping an unknown name to its unique nearest stored name, and abstaining on ties, keeps the memory exact while tolerating typos. |
| Relative replay by predicted forgetting | Replay weights follow each memory's predicted decay relative to the rest of the pool (×0.5 to ×2), since in absolute terms every memory is predicted to be lost and the weights were uniform. |
| Sleep keyed by the routed slot | Sleep, the sentence store and the brakes see only what routing decided, never the true domain label, so the whole cycle stays label-free. |
| Rest from replay | A random fifth of conclusions sits out each sleep's replay, so the model of forgetting sees what happens without it. |
| Dreams that share a name | Cross-domain REM traces that share no rare name with the cue are not dreamed: in cg26i, which dreamed them, a revisit damaged an old domain. |
| Name index first, JEPA second | Searching the whole store, the JEPA picks same-genre text. Restricting candidates to memories that share a rare name with the cue brings recall; the JEPA then ranks them, and is trained against those same name-sharing alternatives (hard negatives). |

## Results

**Refreshing older edits (cg26o, local five-domain harness, two seeds).** Each kept closed-form edit also re-fits the 8 least recently fitted earlier edited conclusions, so later edits and training refresh them instead of overwriting them:

| End of run | cg26o (seeds 1, 2) | cg26n (seeds 1, 2) |
|---|---|---|
| Two-hop from the weights alone | **0.49, 0.32** (mean 0.41) | 0.26, 0.15 (mean 0.21) |
| Older conclusions, weights alone (older vs recent half) | **0.73 vs 0.16, 0.31 vs 0.00** | 0.27 vs 0.21, 0.24 vs 0.06 |
| Two-hop with the fast store | 0.89, 0.97 | 0.94, 1.00 |
| Mean forgetting / final BPC | −0.009 / 2.078, −0.019 / 2.084 | −0.009 / 2.081, −0.020 / 2.094 |

The full cg26o build (refresh plus the edit decay diagnostic, seed 1): −0.009 / 2.077; two-hop from the weights 0.49, with the store 0.91; older vs recent 0.67 vs 0.16; asked by description 0.88; 31 REM conclusions, none wrong; run time +7% for the diagnostic. Refreshing the weakest conclusions instead of the oldest gave less (mean 0.35): the weakest are often ones a single edit cannot store.

The EDIT DECAY DIAGNOSTIC follows the output change each edit made on its own key and splits what is later lost exactly into ordinary training, later edits and key drift. Locally, 2–3 visits after an edit about 0.85–0.92 of the signal is left; ordinary training takes 0.05–0.12 and later edits 0.05–0.09, while key drift averages near zero (keys keep a cosine of about 0.95). On the GPU the same readout will show which of these drives the pass-2 decline.

**Capped weight edits and adaptive sleep (cg26n, local five-domain harness).** The local test of the full cg26n build, against the variants it combines:

| End of run | cg26n | cg26m + growth cap only | cg26m (uncapped edits) | cg26m without edits |
|---|---|---|---|---|
| Two-hop from the weights alone (typed wording) | 0.26 (**0.37**) | 0.26 (0.17) | 0.31 (0.26) | 0.00 (0.03) |
| Three-hop from the weights alone | **0.22** | 0.17 | 0.11 | 0.00 |
| Temporal gradient (weights alone, older vs recent) | 0.27 vs 0.21 | 0.27 vs 0.11 | 0.33 vs 0.05 | none |
| Size of the edited matrix (unedited 6.4) | 14.0 | 13.8 | 39.1 | – |
| Two-hop / three-hop with the fast store | 0.94 / 0.72 | 0.91 / 0.67 | 0.97 / 0.72 | 0.94 / 0.72 |
| Asked by description ("the one who ... B"), with the store | **0.88** | – | – | 0.62 |
| REM conclusions accepted (none wrong) | **34** | 27 | 24 | 25 |
| Mean forgetting / final BPC | −0.009 / 2.081 | −0.008 / 2.079 | −0.002 / 2.104 | −0.012 / 2.073 |
| Run time | **2975 s** | 3488 s | 3606 s | 3528 s |

The growth cap keeps most of the transfer into the weights at almost none of the BPC cost; holding back 16 conclusions to centre the keys on delays the first edit by a visit without losing recall; adaptive sleep finds more REM conclusions in less time. With edits on, conclusions made by REM pass the test without the fast memory more often than those made by NREM chaining (0.23 against 0.14 of tests; 6 against 4 released).

**Capped edits and adaptive sleep at full size (cg26n, ten Pile domains, seed 1).** The cg26n GPU run, pass 1 (to visit 10 of 30), against cg26m at the same visit:

| | cg26n | cg26m |
|---|---|---|
| Pass 1 | mean BPC 1.728, forgetting +0.001; mean BPC after pass 1 **1.727** | 1.733 / −0.002; 1.735 |
| Two-hop / three-hop from the weights alone, visit 10 | 0.48 / 0.28 | 0.55 / 0.34 |
| Two-hop / three-hop with the fast store, visit 10 | 0.92 / 0.32 | 0.94 / 0.32 |
| Edits | 7 kept, none rolled back (the first waits until 16 other conclusions exist, visit 4); edited matrix 38.5 → 68.5 | 6 kept by visit 10 after two roll-backs; 41.0 → 63.9 |
| Memories released / restored by visit 10 | 26 / **0** | 34 / 4 |
| REM conclusions accepted in pass 1 (none wrong) | **140** (1–5 cycles per sleep, ending at closure or falling yield) | 97 |
| NREM write steps per sleep | 40–125 (adaptive) | 200 (fixed) |
| Asked by description, with the store | 0.83–1.00 (0.88 at visit 10, n 24) | – |
| Sleep time, pass 1 | 3318 s (+39%) | 2382 s |

Pass 1 matches cg26m within seed noise, with a lower BPC after pass 1, no rolled-back edits and no restores; adaptive sleep spends its time on REM, which accepts 44% more conclusions with none wrong.

Pass 2 (visits 11–15):

| | cg26n | cg26m |
|---|---|---|
| Two-hop from the weights alone, visits 10 → 15 | 0.48 → 0.42, 0.45, 0.42, 0.35, 0.29 | 0.55 → 0.45, 0.45, 0.38, 0.32, 0.28 |
| Three-hop from the weights alone, visits 10 → 15 | 0.28 → 0.21 | 0.34 → 0.17 |
| Two-hop with the fast store | 0.95–0.98 | 0.97–0.98 |
| Released / restored by visit 15 | 49 / **11** | 56 / 24 |
| Edits | 5 more kept (12 in all), none rolled back; cumulative cost visited +0.013, old domains +0.004 (budget 0.05) | 5 more kept |

The weights-only decline is slower at first but reaches the same level by visit 15 (0.29 against 0.28); what improves is the handover: fewer than half as many restores (11 against 24), about a fifth of releases against nearly half. The growth cap never bound: the accumulated edits stayed at 0.6 times the rest of the matrix (58.8 against 94.8 at visit 14), because ordinary training grows the rest of the matrix too (the edited layer goes 38.5 → 106.7). So the remaining decline is not edit size; it is newer edits and training overwriting older ones, which is what the edit refresh (cg26o) targets.

**Closed-form edits at full size (cg26m, ten Pile domains, seed 1).** The cg26m GPU run, to visit 15 of 30:

| | cg26m | cg26l |
|---|---|---|
| Pass 1 | mean BPC 1.733, forgetting −0.002; mean BPC after pass 1 1.735 | 1.733 / +0.009; 1.724 |
| Two-hop from the weights alone | 0.03 at visit 1, **0.55 at visit 10**, then falling in pass 2 to 0.28 at visit 15 | 0.03–0.07 |
| Three-hop from the weights alone | 0.34 at visit 10, 0.17 at visit 15 | 0.02–0.04 |
| Two-hop / three-hop with the fast store | 0.97–0.98 / 0.68 at visit 15 | 0.98 / 0.72 |
| Edits | 11 kept at +0.0001 to +0.003 BPC each (old domains +0.0026 in all); the first two (visits 3–4) rolled back by the brake; edited matrix 41 → 113 | none |
| Memories released / restored by visit 15 | 65 / 24 (restores matched releases in pass 2) | 15 / 0 by visit 14 |
| Two-hop with a damaged subject name (cue repair) | 0.79–1.00, never the wrong subject | – |

The pass-2 decline coincides with the edited matrix growing past twice its first size; cg26n caps that growth and skips edits while too few conclusions exist to centre the keys on.

**Durable consolidation into the weights, and robust recall (cg26m, local five-domain harness).** The local test of the full cg26m build:

| | cg26m with closed-form edits (default) | cg26m without (`PLATYPUS_CF=0`) |
|---|---|---|
| Two-hop recall from the weights alone, end of run | **0.31** | 0.00 |
| Three-hop from the weights alone | 0.11 | 0.00 |
| Temporal gradient (weights alone, older vs recent conclusions) | **0.33 vs 0.05**: older memories depend less on the fast memory, as CLS theory predicts | none |
| Conclusions edited 4 or more visits earlier, weights alone | 0.29 (other wordings 0.29) | not edited |
| Two-hop with a damaged subject name (with cue repair) | 0.94–1.00 | 0.88–0.94 (0.00 before cue repair) |
| Mean forgetting / final BPC | −0.002 / 2.104 | −0.012 / 2.073 |

The edits cost little forgetting but slow later learning at this small width (the edited matrix grows about sixfold over six edits); the GPU model is 17 times wider at that layer. Cue repair never named the wrong subject in any visit.

**Releasing by predicted recall, and reinstating what fades (cg26l, ten Pile domains, seed 1).** The cg26l GPU run, to visit 14 of 30:

| | cg26l | cg26k |
|---|---|---|
| Pass 1 | mean BPC 1.733, forgetting +0.009 | 1.739 / +0.006 |
| Routing | 14 of 14 | 14 of 14 |
| Model of forgetting | beats a constant rate at visit 13 | visit 14 |
| Release by predicted recall | 13 and 12 memories marked at visits 13 and 14 (top quarter of the pool, bar 0.58–0.60) | 0 (the absolute 10-visit rule never fires) |
| Memories released by visit 14 | 15 (cg26k 6); right after release they answer without the fast memory half of the time, close to the 0.58 the model predicted | 6 |
| Deleted memories | both failed at visit 13, as in cg26k; one failed again at visit 14 and was **reinstated**, the other recovered | both faded and were lost |
| Old domains at visit 14 | 1.655 | 1.653 |
| Pass-2 Wikipedia revisit, old domains before sleep | 1.683 (the EuroParl guard holds) | 1.689 |
| Schema effect | every release (25) is a memory the trunk already predicts well; such memories pass the test without the fast memory 0.23 of the time, misfits 0.03 | same pattern |

**Changed facts are learned once they are seen again.** On the GPU (cg26e, a full pass 2), facts that change in pass 2 are answered correctly once their domain has been revisited: with the fast memory, updated one-hop and two-hop recall rise from 0.00 at visit 10 to 1.00 at visit 20, and the old answer is preferred 0.00 of the time; from the weights alone updated one-hop is 0.55, the same as ordinary facts (0.57). Readings taken mid-way through pass 2 looked stale only because most changed facts had not been revisited yet (the lab readout now scores them only after a revisit).

**Local tests since cg26k** (five-domain harness, seed noise about 0.017 BPC; mean forgetting / final BPC, cg26k −0.018 / 2.074):

| Test | Result |
|---|---|
| More JEPA training pairs (entity-sharing facts, weighted planted pairs) | 2.7× the pairs, held-out separation d′ 1.79 against 0.84 at the same sleep; −0.011 / 2.075. The selector is data-limited, as the analysis predicted; better selection did not move forgetting or BPC locally |
| Closed-form (MEMIT-style) consolidation, applied for real | two-hop from the weights 0.00 → 0.26, three-hop 0.11, but +0.013 / 2.119. A sweep traced the cost to keys dominated by the sentence template (87% of their energy in the directions ordinary text uses, at width 512); a null-space projection cut the cost 2.6× for 0.65× the recall |
| Platonic dream selection (dreams scored by whether their update moves linked passages towards a shared representation) | about 60% of dreams do; choosing by it gave a small, noisy edge in alignment that faded over four sleeps; −0.013 / 2.081 |
| REM analogy dreams in the JEPA space | analogies are found in the trunk's own space (0.58–0.77) and by shared words, not in the JEPA space (0.11–0.40); −0.008 / 2.067 |
| REM self-exam, LLM-JEPA term in replay | neutral (−0.014 / 2.078; −0.017 / 2.079) |

**A JEPA selector that learns, and schema-fast consolidation (cg26i–cg26k, ten Pile domains, seed 1).** The cg26k GPU run has finished pass 1:

| | Result |
|---|---|
| Pass 1 | mean BPC 1.739, forgetting +0.006 (cg26i 1.733 / +0.005, cg26h 1.735 / +0.008); routing 10 of 10 |
| Schema effect | at every sleep from visit 3, memories the trunk already predicts well pass the test without the fast memory 0.10–0.24 of the time, misfits 0.03–0.05; every release so far is such a memory (released after 3 visits) |
| JEPA selector, held-out separation d′ | rises from 1.25 to 1.78 as its training pairs grow from 114 to 391 (cg26i: flat at about 1.1); real-text held-out retrieval 0.33 (chance 0.03), against 0.12–0.16 in cg26i |
| JEPA against TF-IDF on names, choosing among sentences that share the cue's name | 0.41–0.57 against 0.78–0.85: names are still the stronger signal |
| Store handover | 3 released, 2 deleted; both deleted memories still answered from the weights 1 and 2–3 visits later |
| Two-hop recall with the fast memory | 0.94 at visit 10 |

From the cg26i run, which went further:

| | Result |
|---|---|
| Two-hop recall from the weights alone, pass 2 | 0.46 for conclusions of the visit (visit 12), 0.17–0.29 for recent ones: the best so far |
| Model of forgetting | beat a constant rate on unseen data at two sleeps running (visit 13), after which releases stopped: fixed in cg26k (either rule releases) |
| A revisit that hurt | the pass-2 Wikipedia visit raised the old-domain mean to 2.09 before consolidation (1.70 in earlier builds), mostly in EuroParl; sleep replay pulled it back, and by visit 13 the old domains were at 1.66. cg26k no longer dreams the cross-domain traces suspected of causing it |

A diagnostic on a saved local model shows why the JEPA was weak and what limits it now. Its head is linear, so its best possible result can be computed in closed form; the trained head matches it (held-out top-1 0.50–0.65 against 0.47–0.60, raw trunk codes 0.20–0.30), so training is no longer the limit. A simulation calibrated to the GPU curve puts it in the data-limited regime: about 390 linked pairs for an effective input of about 128 dimensions, where about 1,000–1,500 would be needed to pick partners reliably among 1,500 candidates.

Local five-domain harness (seed noise about 0.017 BPC): cg26i −0.017 / 2.082, cg26j −0.010 / 2.073, cg26k −0.018 / 2.074 (mean forgetting / final BPC).

**Handing memories to the weights (cg26g–cg26h).** First visits of the cg26h GPU run (ten Pile domains, seed 1, visits 0–3):

| | Result |
|---|---|
| JEPA held-out partner retrieval, permanent held-out set | 0.19 → 0.31 (chance 0.125, 8 pairs): the first figure after training that the old leak cannot have inflated |
| NREM dreams (`jepa-name`) related to a gold fact | 11 of 40 (0.275), 6 of them accepted, against 2 of 128 for random |
| Sequence dreams (a cue and the sentence stored next to it) | pass the test easily (12–14 accepted per sleep) but 0–1 of 31 are related: they are neighbouring lines of one document, such as bibliography entries. Capped at a quarter of the other dreams (2 per sleep) |
| Replay into the weights | kept at ×0.0033: visited domain +0.010, old domains +0.014 |
| REM fact inference | 5 accepted, none wrong; every answer backed by a stored conclusion is right |
| Routing | 3 of 3 |
| GPU memory | visit peak 52–53 GB, sleep peak 47–49 GB, about 11 GB after the cache is released |

Release, deletion and decay-model timing have not happened yet at this point of the run. Local five-domain harness (seed noise about 0.017 BPC):

| Build | Mean forgetting | Final BPC |
|---|---|---|
| cg26e (seed 2) | −0.011 | 2.087 |
| cg26g | −0.017 | 2.086 |
| cg26h | −0.018 | 2.081 to 2.086 |

On the local harness the permanent held-out set gives 0.06 → 0.25 (chance 0.125, 8 pairs).

**Dreams that find related memories (cg26b–cg26e, ten Pile domains, seed 1).** Relatedness is judged against Re-DocRED gold facts planted in the stream.

| | Result |
|---|---|
| NREM dreams (`jepa-name`) related to a gold fact or chain | 11 of 40 and 13 of 37 (ArXiv, PubMed), against 1–2 of 128 for random pairs |
| REM dreams (`cortex-jepa-name`) related | 12 of 30 at PubMed (40%), 7 accepted, against 3 of 128 for the earlier REM matcher and 1 of 128 for random |
| Dreams about the same entity, later domains | NREM 47–71%, REM 8 of 31, 14 of 16, 11 of 12, 7 of 11; random pairs never |
| JEPA held-out partner retrieval | first training sleep (the only clean measurement so far): all linked pairs 0.34–0.41 (chance 0.06), real-text pairs 0.10–0.20 (chance 0.10). Later sleeps reported 0.8–1.0 on real text, but those were inflated: the held-out split was re-drawn every sleep while the heads persisted, so most held-out pairs had been trained on before. Fixed in cg26g/h (a permanent held-out set): 0.19 → 0.31, chance 0.125, at the first cg26h sleep |
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
python threshold_lean_stagec9_7cg26o_conduit.py
```

In Colab, upload the script, choose a GPU runtime and run it with `!python threshold_lean_stagec9_7cg26o_conduit.py`. If you need credentials (for example a Hugging Face token), keep them in Colab Secrets or environment variables. Never commit them to the repository.

A full run streams ten domains over three passes and takes about four to five hours on a single modern GPU.

For recorded runs (a manifest of exactly what ran, a readout summary and a secret scan), the Colab notebook, a Docker image, presets and the release checklist, see [DEPLOY.md](DEPLOY.md).

## Configuration

Options are set through the `LAB_OVR` environment variable as JSON, on top of the `Config` defaults. Anything you pass overrides the built-in run configuration.

```bash
LAB_OVR='{"n_passes": 3, "steps_per_task": 400, "inf_on": true, "seed": 1}' \
  python threshold_lean_stagec9_7cg26o_conduit.py
```

Environment switches:

| Variable | Meaning |
|---|---|
| `PLATYPUS_RULEFREE=0` | Run the rules setup (relation list and name pattern) instead of the default rule-free mode |
| `PLATYPUS_ROUTEFIX=0` | Turn off the drift-corrected routing and use the original rule |
| `PLATYPUS_DREAMS` | `jepa` (default: dream only what the JEPA selector chose within the name index, NREM and REM), `both`, `nrem`, `rem`, `random` (control: random pairs), `noreplay` (control: no trunk replay, everything else identical), `0` (no dreams, no trunk replay) |
| `PLATYPUS_PASSES=1` | Run pass 1 only |
| `PLATYPUS_CF=0` | Turn off closed-form edits into the weights (on by default, capped and centred) |
| `PLATYPUS_SLEEP=fixed` | Use the fixed sleep schedule instead of adaptive sleep (adaptive by default from cg26n) |

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
| `inf_tgrad`, `inf_tgrad_n`, `inf_tgrad_track` | CLS readouts: temporal gradient and store handover, sample per age group, conclusions tracked |
| `inf_release`, `inf_release_margin`, `inf_schema_fast` | Store release after consecutive passing sleeps with the store off (3, or 2 for a schema fit), margin in nats; restore after 2 failures |
| `inf_delete`, `inf_delete_grace`, `inf_ghost_n` | Deletion of released conclusions after a grace period of passing sleeps; ghosts re-tested per sleep |
| `inf_hl`, `inf_release_hl`, `inf_release_horizon` | Decay model; release timed by it once calibrated; the predicted half-life (visits) needed |
| `inf_graph_window` | Sliding window of the link graph in visits (0 = off) |
| `inf_cons_spaced`, `inf_cons_cap_sleep`, `inf_cons_need`, `inf_cons_anchor_target` | Spaced replay and its per-sleep cap per conclusion; replay by need; anchors aimed at damaged domains |
| `inf_mix_seq`, `inf_mix_seq_cap`, `inf_mix_neardup`, `inf_rem_conflict` | Sequence dreams and their cap; near-duplicate splice filter; REM conflict check for single-valued relations |
| `inf_ledger`, `inf_mem_guard`, `inf_sleep_budget` | Memory ledger; lean sleeps after a memory peak above this share of the device; sleep time budget in seconds |
| `inf_jepa_probe`, `inf_jepa_geom`, `inf_jepa_next`, `inf_jepa_shadow`, `inf_jepa_consol`, `inf_jepa_every` | Latent readouts (no effect on training) and how often they run |
| `inf_bench`, `inf_ceiling`, `inf_ceiling2` | Re-DocRED real-text benchmark (`embedded` in the GPU file); composition ceiling tests |
| `inf_sleep_inferred` | Sleep, store and brakes keyed by the routed slot (on) |
| `inf_cons_rest`, `inf_cons_hlsched` | Share of conclusions resting from each sleep's replay (0.2); replay by predicted forgetting once the decay model is calibrated (2 = relative to the pool, the default) |
| `inf_release_pnext_q`, `inf_release_pnext_min` | Release by relative predicted recall: the top share of the pool (0.25) and the lowest predicted recall allowed (0.2) |
| `inf_ghost_reinstate` | A deleted memory whose ghost fails this many tests running is written back (2) |
| `inf_cue_repair`, `inf_cue_repair_d` | Cue repair at question time, and the edit distance allowed (2) |
| `inf_cf_apply`, `inf_cf_center`, `inf_cf_scale`, `inf_cf_brake`, `inf_cf_budget`, `inf_cf_n` | Closed-form edits: on, frame directions removed from the keys (1), strength (0.5), brake per edit (0.015 BPC), cumulative budget (0.05 BPC), conclusions per edit (32) |
| `inf_cf_maxgrow`, `inf_cf_min_others`, `inf_cf_decay` | Closed-form edits: accumulated edits kept within this multiple of the unedited matrix (1.5); live conclusions kept back from each edit to centre the keys on (16); share of the accumulated edits taken back each sleep (0, off) |
| `inf_cf_refresh`, `inf_cf_refresh_mode`, `inf_cf_diag` | Edit refresh: each kept edit re-fits this many earlier edited conclusions (8 in cg26o), the least recently fitted (`old`, default in cg26o) or the weakest of them (`weak`); the edit decay diagnostic (readout only, on in cg26o) |
| `inf_sleep_adapt`, `inf_sleep_cycles_max` | Adaptive sleep: NREM ends when the store writes converge, REM cycles end at closure or when the predicted next cycle adds less than one conclusion (on); most REM cycles per sleep (6) |
| `inf_jd_entpairs`, `inf_jd_plantw` | More JEPA training pairs: fact pairs sharing an entity (1000), planted pairs weighted against real text |
| `inf_upd_revisited` | Score changed facts only after the domain of the new fact has been revisited (1) |
| `inf_evict_clean`, `inf_schema_speed` | Evicted sentences leave the evidence record; schema-speed readout |
| `inf_jd_obj`, `inf_jd_lambda`, `inf_jd_sigreg_mode` | JEPA objective (`lejepa` or `hybrid`), SIGReg weight, SIGReg as LeJEPA defines it |
| `inf_jd_statm`, `inf_jd_es`, `inf_jd_es_tol`, `inf_jd_wd` | Running input statistics; early stopping and its tolerance; weight decay |
| `inf_jd_judge`, `inf_jd_judge_link`, `inf_jd_outcome`, `inf_jd_out_rel`, `inf_jd_negrand` | Training on the dream judge (positives need a shared rare name); training on measured outcomes, with relative labels; random negatives per pair |
| `inf_mix_jdfinal`, `inf_mix_jdfinal_k`, `inf_mix_jdfinal_fair`, `inf_mix_jdfinal_skip` | JEPA final selection over all sources; pairs dreamed per cue; ranking averaged with the judge; sources never dreamed |
| `inf_mix_jdwide`, `inf_mix_jdwide_bonus`, `inf_mix_jdwide_k` | JEPA choice among the whole candidate sample, the bonus for a shared rare name, picks per cue |
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
- **MEMORY LEDGER**: GPU and CPU tensor bytes by owner, with the change since the last sleep, and counts of every growing sleep structure
- **CLS | RELEASE** and **CLS | DELETED**: conclusions released, restored and deleted; ghosts of deleted conclusions still answered from the weights, by visits since deletion
- **CLS | RELEASE TIMING** and **DECAY MODEL**: which rule decides release, and how many memories were marked by predicted recall and the bar used; the fitted half-life and its out-of-sample score against a constant rate
- **CUE REPAIR**: two-hop with the subject name damaged, repair off and on, and how often the repair named the right subject, a wrong one, or abstained
- **CLOSED-FORM CONSOLIDATION APPLIED**: each edit KEPT or ROLLED BACK with its cost, recall from the weights before and after, the cumulative cost against the budget, the share of key energy removed by centring, and PERSISTENCE of earlier edits by age
- **CF DECAY DIAGNOSTIC** (cg26o): for conclusions edited 1, 2–3 and 4+ visits ago, the change along each edit's own signal since it was written, by cause (ordinary training, later edits, key drift), the key cosine and the share left
- **REFRESH (old)** at the end of a KEPT edit line: how many earlier edited conclusions were re-fitted, with their weights-only recall before and after
- **SLEEP CYCLES | ADAPTIVE**: REM cycles run and conclusions accepted per cycle, NREM write steps per chunk against the fixed schedule, and why the sleep stopped (closure, falling yield or the cycle cap)
- **CLS | SCHEMA SPEED BY ORIGIN**: store-off tests passed and releases for conclusions made by REM against those made by NREM chaining
- **CUE REPAIR ... described**: two-hop with the store when the subject is given by description ("the one who <relation> <B>") instead of by name
- **CLOSED-FORM ... skipped**: no edit this sleep, with the reason (fewer than 8 conclusions to edit, too few to centre the keys on, or the budget reached)
- **changed facts ... awaiting a revisit**: updated facts not yet scored because their domain has not been revisited
- **REINSTATED** (on the CLS | DELETED line): deleted memories written back after their ghosts failed, and those dropped because a source was superseded
- **SPACED REPLAY**, **REPLAY BY NEED**: rows per conclusion and kept replays; sampling weights by the last test and schema fit
- **MIXING | SEQUENCE CAP**: accepted sequence pairs and how many were dreamed
- **SLEEP KEYED BY SLOT**: the slot each sleep works on and the other slots it protects
- **REST**, **REPLAY BY PREDICTED FORGETTING**: conclusions resting from replay; whether replay follows the decay model yet
- **CLS | SCHEMA SPEED**: tests passed and releases for schema fits against misfits
- **JEPA DREAMS | TRAIN SETUP**: early stopping, SIGReg statistic, objective, judge- and outcome-trained pairs
- **MIXING | JEPA FINAL SELECTION**, **JEPA-WIDE**: accepted pairs by source and what was dreamed; the JEPA's picks among the whole sample

## Project status

This is **research code**. It is bulky: it carries options, readouts and experiment paths from many iterations that are switched off in the current configuration. It needs a cleanup (dead options removed, modules split, tests added) and further optimisation before wider use.

Known limits:

- Inference has been tested at scale on planted facts about invented people. On natural text, a first local benchmark (Re-DocRED Wikipedia paragraphs with gold relations) found that word-level linking cannot chain facts stated only once: two single reads never reach the linking threshold. Facts must recur, or links must be found another way (bridge-free mixing and latent "splice" dreams are under test).
- Natural prose is harder than templated sentences: about 0.77 two-hop recall in local tests.
- Pronouns, descriptions and nested clauses are not resolved, so chains through "he" or "she" are not found.
- Consolidation into the weights works at full size but slowly: replay at doses the brake accepts adds only a little two-hop recall per sleep, though what it adds survives later training. Protection against forgetting during replay is not yet reliable from one visit to the next, and the small local model shows no transfer at all.
- The JEPA selector now learns (cg26k), but it is data-limited: about 390 linked pairs for an effective input of about 128 dimensions. TF-IDF on names still picks the right partner among name-sharing sentences more often, and among the 1,500 candidates of `jepa-wide` the JEPA is right only about 7% of the time.
- Without weight edits the model of forgetting fits half-lives of only 0.2–0.5 visits, so cg26l releases by predicted recall relative to the pool. With the edits (cg26m, GPU) the fitted half-lives rose to 2–3.6 visits, but released memories began to fail again in pass 2: by visit 15, 24 of 65 had been restored.
- A pass-2 revisit in cg26i damaged one old domain (EuroParl) before sleep repaired it; with the guard (cg26k, cg26l) the same revisit leaves the old domains at 1.68–1.69.
- Dreams chosen by the JEPA are far more often related than random ones, but they are not yet shown to give more two-hop recall or less forgetting than random dreams; the control run (`PLATYPUS_DREAMS=random` beside the default) is pending.
- Closed-form edits reach two-hop 0.48–0.55 from the weights alone on the GPU (visit 10), but that falls to about 0.3 in pass 2 (cg26m 0.28 at visit 15; cg26n 0.29 at visit 15, 0.32 at visit 16). The growth cap of cg26n never bound, so edit size is not the cause; ordinary training and later edits overwrite older edits. cg26o refreshes older edits and measures the loss by cause.
- The weights store conclusions but do not compose facts: with both facts in the weights and no written conclusion, two-hop from the weights is 0 of 11 (local), and conclusions held back from writing stay at 0.00 after training on all the others. Asking in two steps (the bridge, then the answer) looks better but is a shortcut: a random wrong bridge scores the same, because the weights answer from the relation phrase alone.
- In the GPU file but off: a REM self-exam of the weights, REM analogy dreams (the local harness's 128-byte rows are too short to test them), an LLM-JEPA term in replay, platonic dream selection, principal-direction input and same-document negatives for the JEPA. Each was neutral or harmful in local tests.
- Completion beyond spelling is only partly shown: asked by description ("the one who <relation> <B>") the model answers 0.88 with cg26n locally and on the GPU (0.62 locally without edits; chance 0.10), but this is a readout on templated facts; completion from other partial context is not addressed, and the overlapping link memory was at chance on the GPU.
- At the replay dose the brake accepts (×0.0033), later sleeps no longer raise two-hop recall measurably, and transfer kept from earlier sleeps decays in some visits.
- The trunk's passage codes are strongly anisotropic; latent readouts and the JEPA head work around this, but the trunk itself is unchanged.
- Forgetting at full size rises in later passes, and routing confuses the most similar domains.
- There is no fine-tuning baseline and no benchmark result yet.

## Roadmap

1. **Dreams that beat random.** Run the JEPA dreams against random-dream and no-replay controls on the same build; show that related dreams give more lasting two-hop recall in the weights. Give the JEPA more linked pairs and a smaller input, so it can pick partners among the whole store.
2. **Free the fast store.** Release, deletion, ghosts, decay-model timing, release by predicted recall and reinstatement are in (cg26g–cg26l). Next, show that reinstated memories recover, tune how many memories are released at once, and hand over stored facts as well as conclusions.
3. **Faster consolidation.** Closed-form writes with centred keys lasted and gave the first temporal gradient locally (cg26m), and reached two-hop 0.55 from the weights alone on the GPU before decaying in pass 2. The growth cap (cg26n) did not stop the pass-2 decline; cg26o refreshes older edits and measures what removes them. Next, by what that shows: keep training out of the edited directions, solve new edits in the null space of earlier ones, or re-fit memories that fail.
4. **Real text.** Measure the precision of conclusions drawn from real text, and test on multi-hop question sets and MQuAKE for multi-step edits.
5. **Consolidating structure.** Move relation regularities (which relations chain, which are single-valued, paraphrase rules) into the weights, and show they become independent of the store over time.
6. **Stable domains at scale.** Protect slots from drift, to cut routing errors and later-pass forgetting; test learning rates that follow novelty and consolidation that stops itself at the brake.
7. **References and meaning.** Resolve pronouns and descriptions as rules that must pay for themselves; carry negation, hedging and time on each relation.
8. **Entity resolution.** Merge spelling variants only when the evidence supports it: rare typos of a common name, or variants that share context.

## History

Headline milestones; details are in each build's `NAME` and in the commit logs.

- **September 2026.** Started as a byte-level continual learner built around prototype orthogonality. A first "near-perfect" result turned out to be data leakage (small files tiled into both training and validation), so the project moved to streaming distinct Pile domains with clean splits, which it still uses.
- **Late September.** Routing by sequence-level byte statistics instead of per-token features; a learned controller that freezes important units.
- **Early October.** First GPU runs; hashed n-gram tables; the design recast as a complementary learning system: fast hashed tables (hippocampus), a slow protected core (cortex) and replay.
- **Mid October.** Dream rehearsal (local forgetting +0.228 → +0.021) and private per-domain capacity (about zero); backprop-free, closed-form consolidation of rules learned in context.
- **Late October.** The 171M "conduit" model: a strict closed loop in which, after the first pass, the model learns only from replays it can verify; label-free routing; best final BPC 1.465 at the time (cg14e2).
- **30 October.** Platypus learns to think in its sleep (cg20): offline inference over a sentence store, a fast memory that can abstain, belief revision with cascading retraction. Released under Apache 2.0.
- **31 October – 1 November.** Rule-free inference, three-hop chains, every two-step link scored in one pass.
- **2 November.** Dreams judged by mutual compression; Fisher-protected replay; the JEPA dream selector; memories released from the fast store, deleted, and timed by a model of forgetting (cg21–cg26h).
- **3 November.** The JEPA selector rebuilt on LeJEPA and trained on outcomes; CLS integration; release by predicted recall and ghost reinstatement, first seen on the GPU; cue repair; centred closed-form edits giving the first durable transfer into the weights and the first temporal gradient (cg26i–cg26m); on the GPU, two-hop 0.55 from the weights alone; capped edits and adaptive sleep, with sleep length set by the model's own statistics (cg26n).
- **4 November.** Older edits refreshed instead of overwritten, doubling two-hop recall from the weights alone locally, and a diagnostic that splits edit decay by cause (cg26o); on the GPU, cg26n halves restores of released memories. Tests that the weights do not compose facts on their own, and a sleep analogue of Cai et al. (2009): sleep solves about half the problems whose facts were just read, quiet rest none, and REM adds nothing over NREM while every premise stays in the store.

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

## References

- McClelland, McNaughton and O'Reilly (1995). [Why there are complementary learning systems in the hippocampus and neocortex](https://doi.org/10.1037/0033-295X.102.3.419). *Psychological Review* 102(3).
- Kumaran, Hassabis and McClelland (2016). [What learning systems do intelligent agents need? Complementary learning systems theory updated](https://doi.org/10.1016/j.tics.2016.05.004). *Trends in Cognitive Sciences* 20(7).
- Huh, Cheung, Wang and Isola (2024). [The platonic representation hypothesis](https://arxiv.org/abs/2405.07987). ICML 2024.
- Meng et al. (2023). [Mass-editing memory in a transformer](https://arxiv.org/abs/2210.07229) (MEMIT). ICLR 2023.
- Tse et al. (2007). [Schemas and memory consolidation](https://doi.org/10.1126/science.1135935). *Science* 316.
- Settles and Meeder (2016). [A trainable spaced repetition model for language learning](https://aclanthology.org/P16-1174/) (half-life regression). ACL 2016.
- Assran et al. (2023). [Self-supervised learning from images with a joint-embedding predictive architecture](https://arxiv.org/abs/2301.08243) (I-JEPA).
- Balestriero and LeCun (2025). [LeJEPA: provable and scalable self-supervised learning without the heuristics](https://arxiv.org/abs/2511.08544) (SIGReg); [code](https://github.com/rbalestr-lab/lejepa).
- Tan et al. (2022). [Revisiting DocRED: addressing the false negative problem in relation extraction](https://arxiv.org/abs/2205.12696) (Re-DocRED); [data](https://github.com/tonytan48/re-docred).
- [The Pile, uncopyrighted subset](https://huggingface.co/datasets/monology/pile-uncopyrighted) (training data).

## Licence

Copyright 2026 CHRISTIAN BIRD

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Add the full licence text as a `LICENSE` file in the repository root (GitHub offers the Apache 2.0 template when you create the file).
