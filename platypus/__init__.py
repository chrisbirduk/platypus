"""Platypus: a continual-learning byte-level language model with sleep inference, a complementary learning
system design and JEPA dream selection. Copyright Christian Bird, Apache 2.0.

The package is the GPU build split into modules; python -m platypus runs it (see __main__.py)."""
# =====================================================================================================================
# PLATYPUS: a language model that learns without forgetting and thinks in its sleep
# =====================================================================================================================
# Copyright 2026 CHRISTIAN BIRD
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# =====================================================================================================================
#
# WHAT THIS FILE IS
#   The full-size (GPU) build of Platypus, a byte-level continual-learning language model. It reads ten Pile domains
#   one after another over several passes and is evaluated on how much it keeps, how much it learns and what it can
#   infer. Build: cg26p. The run configuration and a full change history are in NAME and LAB_OVR near the end.
#
# WHAT IT DOES
#   * Continual learning without stored data: domains are learned in sequence; old domains are protected by
#     rehearsing the model's own dreams (stored self-generated windows with their top-32 predictions), curvature-aware
#     (Fisher-whitened) updates, and small per-domain capacity (heads, private n-gram rows, private units, FiLM codes).
#   * A strict closed loop: after the first pass, no real text. The model learns only from replays of its memory traces
#     that it can verify exactly against what it originally read (verified prefix credit, repair at the first wrong
#     byte, confidence gate, trace retirement).
#   * Label-free routing: the domain is never given; visits are assigned to slots by prediction quality.
#   * Inference in sleep: after every visit it links facts met in different places (bridge-centred chaining with hub
#     abstention), replays salient facts (NREM-like), proposes links from memory and keeps only grounded ones
#     (REM-like), revises beliefs (recency rule) and retracts dependent conclusions (cascading retraction).
#   * A fast store outside the weights: an entity-banked, drift-free-keyed delta-rule logit corrector, written to
#     convergence with do-no-harm anchors and applied only where it is confident, so it can say "I don't know".
#   * Readouts: probe recall from the weights and with the store, answer-or-abstain, sources, belief revision,
#     store cost on ordinary text, splice-and-appraise, and multi-hop question-time chaining.
#
# ARCHITECTURE
#   1024-byte windows; byte + hashed 2/3/4-gram embeddings (sparse); block 0 a causal local filter; blocks 1-3 gated
#   delta-rule mixers (8 heads, beta up to 2, preconditioned write key, WY-form parallel solve); residual width 1088;
#   one soft-capped head per domain. About 171M trunk parameters plus 223M hashed n-gram parameters.
#
# USAGE
#   Run as a script (e.g. in Colab with a GPU). Options are set through LAB_OVR (JSON) and the Config defaults.
#   Logs report per-visit BPC, forgetting since each visit, routing, and the inference readouts.
#
# STATUS
#   This is research code. It is bulky: it carries options, readouts and experiment paths from many iterations that are
#   switched off in this configuration, and it needs a cleanup (dead options removed, modules split, tests added) and
#   further optimisation (evaluation frequency, store writes, data loading and generation speed) before wider use.
# =====================================================================================================================

import importlib as _il, sys as _sys
MODULES = ('common', 'config', 'gating', 'protection', 'private', 'slots', 'layers', 'capacity', 'sampler', 'dream_bank', 'ngrams', 'model', 'meta', 'diagnostics', 'importance', 'data', 'heads', 'train', 'orth', 'conduit', 'inference.store', 'inference.units', 'inference.probes', 'inference.sentences', 'inference.links', 'inference.infer', 'inference.evaluate', 'inference.ledger', 'inference.decay', 'inference.closed_form', 'inference.supmem', 'inference.jindex', 'inference.sleep', 'inference.readouts', 'inference.selftaught', 'inference.mixing', 'inference.jepa', 'inference.rem', 'inference.qtime', 'inference.audit', 'inference.need', 'losses', 'runner', 'profiling', 'inference.gisttest')
_loaded = [_il.import_module('platypus.' + m_) for m_ in MODULES]
# link the names each module uses at run time from modules that load after it
for m_ in _loaded:
    for n_, p_ in getattr(m_, '_LATE', {}).items(): setattr(m_, n_, getattr(_sys.modules[p_], n_))
from platypus import common as _common, _order as _ord
_common._PKG_MODULES[:] = _loaded
_common._PKG_ORDER[:] = [(_sys.modules['platypus.' + m_], n_) for m_, n_ in _ord.ORDER]
from platypus import profiling as _profiling
_profiling._label_regions()
import os as _os
if _os.environ.get('PLATYPUS_FNTIME','0')!='0': _profiling._time_functions()
from platypus import presets
from platypus.config import Config, _CONFIG_DEFAULTS
from platypus.data import set_stream_source
from platypus.runner import run
