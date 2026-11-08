#!/usr/bin/env python3
"""Summarise a Platypus run log: the readouts that decide whether a build is kept.

Works on GPU logs and on the local harness logs, and on partial logs from a
run still in progress. Lines it does not recognise are ignored, so older
builds give whatever subset they print.

Usage:
    extract_readouts.py run.log            # JSON to stdout
    extract_readouts.py run.log --md       # Markdown summary
    extract_readouts.py a.log b.log --md   # side by side (e.g. old vs new code, same seed)
"""
import json
import re
import sys

NUM = r"(-?[0-9]*\.?[0-9]+|nan)"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def recall(line):
    """one-hop / two-hop / three-hop values from an eval line."""
    out = {}
    for key, pat in (("one_hop", r"\| one-hop " + NUM), ("two_hop", r"\| two-hop " + NUM),
                     ("typed_two_hop", r"typed two-hop " + NUM), ("three_hop", r"three-hop " + NUM),
                     # issue #24: the honest measures - unwritten wording, bridge among the candidates, no slot given; and no candidate list at all
                     ("held_out_two_hop", r"\| HELD-OUT two-hop \(picked a bridge [0-9.na]+\) " + NUM),
                     ("held_out_typed_two_hop", r"\| HELD-OUT typed two-hop \(picked a bridge [0-9.na]+\) " + NUM),
                     ("held_out_three_hop", r"\| HELD-OUT three-hop \(picked a bridge [0-9.na]+\) " + NUM),
                     ("open_one_hop", r"OPEN one-hop \(greedy, no candidates\) " + NUM),
                     ("open_held_out_two_hop", r"OPEN HELD-OUT two-hop \(greedy, no candidates\) " + NUM),
                     # cg27a: decoding restricted to the names in the store (exact)
                     ("known_name_one_hop", r"KNOWN-NAME one-hop \(decoding restricted[^)]*\) " + NUM),
                     ("known_name_two_hop", r"KNOWN-NAME two-hop \(decoding restricted[^)]*\) " + NUM),
                     ("known_name_held_out_two_hop", r"KNOWN-NAME HELD-OUT two-hop \(decoding restricted[^)]*\) " + NUM)):
        m = re.search(pat, line.replace("(chance 0.05): one-hop", "(chance 0.05): | one-hop"))
        if m:
            out[key] = f(m.group(1))
    return out


def parse(path):
    r = {"file": path, "name": None, "seed": None, "visits": {}, "routing": [], "passes": [],
         "final": {}, "temporal_gradient": None}
    visit = 0  # the eval visit most recently seen; sleep readouts after it belong to it
    pending = {}  # readouts printed in a sleep BEFORE that sleep's eval (release, adaptive cycles)

    def V():
        return r["visits"].setdefault(visit, {})

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = re.match(r"#{8,} (.*) \| seed (\d+) #{8,}", s)
            if m:
                r["name"], r["seed"] = m.group(1)[:160], int(m.group(2))
                continue
            m = re.search(r"\[routing\] visit (\d+):.*?true (\S+)\s*:?\s*(CORRECT|WRONG)", s)
            if m:
                r["routing"].append({"visit": int(m.group(1)), "domain": m.group(2), "ok": m.group(3) == "CORRECT"})
                continue
            m = re.search(r"\[inference eval \| visit (\d+)\] probe recall FROM THE WEIGHTS", s)
            if m:
                visit = int(m.group(1))
                V()["weights"] = recall(s)
                V().update(pending)
                pending.clear()
                continue
            m = re.search(r"\[inference eval \| visit (\d+) \| WITH THE FAST STORE\] probe recall", s)
            if m:
                visit = int(m.group(1))
                V()["store"] = recall(s)
                continue
            if "[inference eval | END OF RUN] probe recall FROM THE WEIGHTS" in s:
                r.setdefault("end_of_run", {})["weights"] = recall(s)
                continue
            if "[inference eval | END OF RUN | WITH THE FAST STORE] probe recall" in s:
                r.setdefault("end_of_run", {})["store"] = recall(s)
                continue
            if "[inference CLOSED-FORM CONSOLIDATION APPLIED]" in s:
                cf = {}
                if "] KEPT" in s:
                    m = re.search(r"KEPT at x([0-9.]+)", s)
                    cf["outcome"], cf["scale"] = "kept", f(m.group(1)) if m else None
                elif "ROLLED BACK" in s:
                    cf["outcome"] = "rolled_back"
                elif "skipped" in s:
                    cf["outcome"] = "skipped"
                m = re.search(r"\|dW\| " + NUM + r" vs \|W\| " + NUM, s)
                if m:
                    cf["dW"], cf["W"] = f(m.group(1)), f(m.group(2))
                m = re.search(r"cumulative cost of (\d+) kept edits \(rolled back (\d+)\): visited " + r"([+-][0-9.]+), old domains ([+-][0-9.]+)", s)
                if m:
                    cf.update(kept_total=int(m.group(1)), rolled_back_total=int(m.group(2)),
                              cum_cost_visited=f(m.group(3)), cum_cost_old=f(m.group(4)))
                m = re.search(r"edited 4\+ visit\(s\) ago \(n \d+ of \d+\): fitted wording " + NUM + r", other wordings " + NUM, s)
                if m:
                    cf["persist_4plus_fitted"], cf["persist_4plus_other"] = f(m.group(1)), f(m.group(2))
                V()["cf"] = cf
                continue
            if "[inference CUE REPAIR] two-hop" in s:
                cr = {}
                for key, lab in (("last_letter", "last letter dropped"), ("swapped", "inner letters swapped"), ("lower", "lower-cased")):
                    m = re.search(lab + r": off " + NUM + r" -> on " + NUM + r" \(repaired right (\d+), wrong (\d+)", s)
                    if m:
                        cr[key] = {"off": f(m.group(1)), "on": f(m.group(2)), "right": int(m.group(3)), "wrong": int(m.group(4))}
                V()["cue_repair"] = cr
                continue
            m = re.search(r"\[inference CLS \| RELEASE\] (\d+) conclusions released.*?(\d+) restored", s)
            if m:
                pending["released"], pending["restored"] = int(m.group(1)), int(m.group(2))
                continue
            m = re.search(r"\[inference CLS \| SCHEMA SPEED\].*?fit \(lower half\): n (\d+), store-off tests passed " + NUM +
                          r".*?misfit \(upper half\): n (\d+), store-off tests passed " + NUM, s)
            if m:
                V()["schema"] = {"fit_pass": f(m.group(2)), "misfit_pass": f(m.group(4))}
                continue
            m = re.search(r"\[inference DECAY MODEL\].*?half-life at the mean conclusion " + NUM, s)
            if m:
                V()["half_life_visits"] = f(m.group(1))
                continue
            m = re.search(r"\[inference SLEEP CYCLES \| ADAPTIVE\] (\d+) REM cycles, accepted per cycle \[([0-9, ]*)\].*?(\d+) write steps in all", s)
            if m:
                pending["sleep"] = {"rem_cycles": int(m.group(1)),
                                "rem_accepted": sum(int(x) for x in m.group(2).split(",") if x.strip()),
                                "write_steps": int(m.group(3))}
                continue
            m = re.search(r"PASS (\d+) SUMMARY: mean BPC " + NUM + r" \| mean forgetting since each domain's visit " + r"([+-]?[0-9.]+)", s)
            if m:
                r["passes"].append({"pass": int(m.group(1)), "mean_bpc": f(m.group(2)), "forgetting": f(m.group(3)), "domains": {}})
                continue
            m = re.match(r"(.+?)\s+end of pass " + NUM + r" \| post-train " + NUM + r" \| forgetting ([+-]?[0-9.]+)", s)
            if m and r["passes"]:
                r["passes"][-1]["domains"][m.group(1).strip()] = {"end": f(m.group(2)), "post": f(m.group(3)), "forgetting": f(m.group(4))}
                continue
            m = re.search(r"temporal gradient \| END OF RUN\].*?OLDER half " + NUM + r".*?RECENT half " + NUM, s)
            if m:
                r["temporal_gradient"] = {"older": f(m.group(1)), "recent": f(m.group(2))}
                continue
            m = re.search(r"MEAN FORGETTING: " + r"([+-]?[0-9.]+) \| MEAN FINAL BPC: " + NUM, s)
            if m:
                r["final"].update(forgetting=f(m.group(1)), bpc=f(m.group(2)))
                continue
            m = re.search(r"HEADLINE \(label-free.*?FORGETTING ([+-]?[0-9.]+) \| FINAL BPC " + NUM, s)
            if m:
                r["final"].update(label_free_forgetting=f(m.group(1)), label_free_bpc=f(m.group(2)))
                continue
            m = re.search(r"\[inferred heads: heads\] final BPC " + NUM + r" forget ([+-]?[0-9.]+)", s)
            if m:
                r["final"].setdefault("label_free_bpc", f(m.group(1))); r["final"].setdefault("label_free_forgetting", f(m.group(2)))
                continue
            m = re.search(r"\[label-free\] INFERRED - GIVEN(?: at the end)?: mean ([+-][0-9.]+) \(range ([+-][0-9.]+) to ([+-][0-9.]+)\)", s)
            if m:
                r.setdefault("label_free_delta", []).append({"end": "at the end" in s, "mean": f(m.group(1)), "min": f(m.group(2)), "max": f(m.group(3))})
                continue
            m = re.search(r"\[inference MISS LOG \| (.*?)\] (one-hop|two-hop) .*?from (the weights(?: \+ store)?)\): (\d+) greedy misses of (\d+) \| RANK-1 SHARE.*?summed ([0-9.]+), per byte ([0-9.]+).*?BEAM GAIN \(beam - greedy\) ([+-]?[0-9.na]+).*?SAME-RELATION COMPETITOR summed ([0-9.]+), per byte ([0-9.]+).*?SHORTER than the true name summed ([0-9.]+), per byte ([0-9.]+)", s)
            if m:
                r.setdefault("miss_log", []).append({"at": m.group(1), "kind": m.group(2), "from": m.group(3), "misses": int(m.group(4)), "n": int(m.group(5)),
                    "rank1_sum": f(m.group(6)), "rank1_byte": f(m.group(7)), "beam_gain": f(m.group(8)), "competitor_sum": f(m.group(9)),
                    "competitor_byte": f(m.group(10)), "shorter_sum": f(m.group(11)), "shorter_byte": f(m.group(12))})
                m2 = re.search(r"another chain's C \(any relation\) summed ([0-9.]+)", s)
                if m2: r["miss_log"][-1]["other_chain_c_sum"] = f(m2.group(1))
                continue
            m = re.search(r"\[inference MEMORY RESIDUAL \| (.*?)\] stored real sentences.*?\| ALL n (\d+): trunk alone ([0-9.]+) bits/byte \(names ([0-9.na]+)\), with the store ([0-9.]+) \(names ([0-9.na]+)\)", s)
            if m:
                r.setdefault("memory_residual", []).append({"at": m.group(1), "n": int(m.group(2)), "trunk": f(m.group(3)), "trunk_names": f(m.group(4)), "store": f(m.group(5)), "store_names": f(m.group(6))})
                continue
            m = re.search(r"\[inference MEMORY RESIDUAL \| (.*?)\] TRANSFER .*?all bytes ([+-][0-9.]+), name bytes ([+-][0-9.na]+)", s)
            if m:
                r.setdefault("memory_transfer", []).append({"at": m.group(1), "all": f(m.group(2)), "names": f(m.group(3))})
                continue
            m = re.search(r"\[GRADUATION \| (.*?)\] .*?mean rise (.*?) \| per domain", s)
            if m:
                r.setdefault("graduation", []).append({"at": m.group(1), **{k.strip(): f(v) for k, v in re.findall(r"([a-zA-Z -]+?) ([+-][0-9.]+)", m.group(2))}})
                continue
            m = re.search(r"\[inference CF COMPETITOR CONTRAST\].*?divergence byte ([0-9.na]+) before fitting, ([0-9.na]+) after", s)
            if m:
                r.setdefault("cf_contrast", []).append({"before": f(m.group(1)), "after": f(m.group(2))})
            m = re.match(r"FAILURES: (none|\d+)", s)
            if m:
                r["final"]["failures"] = 0 if m.group(1) == "none" else int(m.group(1))
                continue
            m = re.search(r"\[data\] train/validation overlap.*\| total (\d+)", s)
            if m:
                r["data_overlap"] = int(m.group(1))
                continue
            m = re.match(r"WALL (\d+)s", s)
            if m:
                r["final"]["wall_s"] = int(m.group(1))
    if pending:  # a sleep whose eval has not been printed yet (log still growing)
        r["visits"].setdefault(visit + 1, {}).update(pending)
    for p in r["passes"]:
        posts = [d["post"] for d in p["domains"].values() if d["post"] is not None]
        if posts:
            p["mean_post_train"] = round(sum(posts) / len(posts), 4)
    return r


def summary(r):
    vs = r["visits"]
    released = sum(v.get("released", 0) for v in vs.values())
    restored = sum(v.get("restored", 0) for v in vs.values())
    cf = [v["cf"] for v in vs.values() if "cf" in v]
    kept = sum(1 for c in cf if c.get("outcome") == "kept")
    rolled = sum(1 for c in cf if c.get("outcome") == "rolled_back")
    last = max(vs) if vs else None
    best2 = max((v.get("weights", {}).get("two_hop") or 0 for v in vs.values()), default=None)
    wrong = sum(x["wrong"] for v in vs.values() for x in v.get("cue_repair", {}).values())
    route = r["routing"]
    fin = r["final"] or {}
    return {
        # HEADLINES (issues #23, #24): slot inferred from the text; held-out wording with no slot given
        "HEADLINE_label_free_bpc": fin.get("label_free_bpc"),
        "HEADLINE_label_free_forgetting": fin.get("label_free_forgetting"),
        "HEADLINE_held_out_two_hop_weights_last": vs.get(last, {}).get("weights", {}).get("held_out_two_hop") if last else None,
        "HEADLINE_held_out_two_hop_store_last": vs.get(last, {}).get("store", {}).get("held_out_two_hop") if last else None,
        "open_held_out_two_hop_store_last": vs.get(last, {}).get("store", {}).get("open_held_out_two_hop") if last else None,
        "failures": fin.get("failures"),
        "label_free_minus_given": r.get("label_free_delta"),
        "miss_log_last": (r.get("miss_log") or [None])[-4:],
        "known_name_two_hop_store_last": vs.get(last, {}).get("store", {}).get("known_name_two_hop") if last else None,
        "memory_residual_last": (r.get("memory_residual") or [None])[-1],
        "memory_transfer_last": (r.get("memory_transfer") or [None])[-1],
        "graduation_last": (r.get("graduation") or [None])[-1],
        "cf_contrast_last": (r.get("cf_contrast") or [None])[-1],
        "data_overlap": r.get("data_overlap"),
        "last_visit": last,
        "routing_correct_where_printed": f"{sum(x['ok'] for x in route)} of {len(route)}" if route else None,
        "weights_two_hop_last": vs.get(last, {}).get("weights", {}).get("two_hop") if last else None,
        "weights_two_hop_best": best2,
        "store_two_hop_last": vs.get(last, {}).get("store", {}).get("two_hop") if last else None,
        "cf_kept": kept, "cf_rolled_back": rolled,
        "cf_matrix_norm_last": next((c.get("W") for c in reversed(cf) if c.get("W") is not None), None),
        "released_total": released, "restored_total": restored,
        "cue_repair_wrong_total": wrong,
        "end_of_run_weights": r.get("end_of_run", {}).get("weights"),
        "end_of_run_store": r.get("end_of_run", {}).get("store"),
        "temporal_gradient": r["temporal_gradient"],
        "passes": [{k: p.get(k) for k in ("pass", "mean_bpc", "forgetting", "mean_post_train")} for p in r["passes"]],
        "final": r["final"] or None,
    }


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 100 else f"{x:.1f}"
    if isinstance(x, dict):
        return ", ".join(f"{k} {fmt(v)}" for k, v in x.items())
    if isinstance(x, list):
        return "; ".join(fmt(i) for i in x)
    return str(x)


def markdown(runs):
    out = []
    S = [summary(r) for r in runs]
    head = "| Readout | " + " | ".join((r["name"] or r["file"])[:40] for r in runs) + " |"
    out += ["## Summary", "", head, "|---" * (len(runs) + 1) + "|"]
    for key in S[0]:
        out.append(f"| {key.replace('_', ' ')} | " + " | ".join(fmt(s[key]) for s in S) + " |")
    for r in runs:
        out += ["", f"## Per visit: {r['file']}", "",
                "| Visit | Weights 1h / 2h / 3h | Store 1h / 2h / 3h | Edit | Released | Schema fit / misfit | Half-life |",
                "|---|---|---|---|---|---|---|"]
        for v in sorted(r["visits"]):
            d = r["visits"][v]
            w, st = d.get("weights", {}), d.get("store", {})
            cf = d.get("cf", {})
            edit = cf.get("outcome", "-") + (f" x{cf['scale']}" if cf.get("scale") else "")
            sc = d.get("schema", {})
            out.append(f"| {v} | {fmt(w.get('one_hop'))} / {fmt(w.get('two_hop'))} / {fmt(w.get('three_hop'))} "
                       f"| {fmt(st.get('one_hop'))} / {fmt(st.get('two_hop'))} / {fmt(st.get('three_hop'))} "
                       f"| {edit} | {fmt(d.get('released'))} | {fmt(sc.get('fit_pass'))} / {fmt(sc.get('misfit_pass'))} "
                       f"| {fmt(d.get('half_life_visits'))} |")
    return "\n".join(out)


def main(argv):
    files = [a for a in argv[1:] if not a.startswith("--")]
    if not files:
        sys.exit(__doc__)
    runs = [parse(p) for p in files]
    if "--md" in argv:
        print(markdown(runs))
    else:
        print(json.dumps(runs if len(runs) > 1 else runs[0], indent=1))


if __name__ == "__main__":
    main(sys.argv)
