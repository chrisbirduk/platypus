# How the package was split from the single file

The `platypus` package is the single-file GPU build (cg26p) split into modules with no change in behaviour. These scripts did it and check it; rerun them to split another single-file build the same way (for example the `jepa` branch).

1. `stage0.py IN.py OUT.py`: small behaviour-preserving edits to the single file, so that it can be split safely.
   - The dead first `_cka` is dropped.
   - The first `bpc` and `ngp_step` get the names their later wrappers call (`_bpc_w0`, `_ngp_step_0`).
   - The module-level names rebound while running (`_INF_NAME`, `_INF_TRIP`, the profiling labels) go through `_set_global`.
   - The memory ledger reads `_pkg_globals()`.
   - `run()` asks the data module whether the Hugging Face source is in use.
2. `split.py STAGE0.py OUT_DIR`: writes `OUT_DIR/platypus/`.
   - Every top-level statement keeps its text verbatim. Modules are contiguous runs of the original file; `SPEC` holds the module names, their first names and their descriptions.
   - Imports are generated from the names each module uses. Names from a module that loads later go into the module's `_LATE` table.
   - It also writes `presets.py` and `__main__.py` from the old `if __name__ == "__main__":` block, `_order.py` (the original order of all module-level names) and `__init__.py` (load in order, link `_LATE`, label the profiling regions).
3. Checks:
   - `main_equiv.py ORIG.py PKG_DIR`: `python -m platypus` builds the same configuration and prints the same banner as the old main block, for every `PLATYPUS_*` switch and a custom `LAB_OVR`.
   - `harness.py SRC LOG short|safe`: runs a build locally on the five-domain stand-in data. `SRC` is a single file or a package directory; `LOCAL_HARNESS` names the folder with `threshold_sweep_local.py` and the data. It uses the options of the full-build local safety run; `short` is 2 passes of 24 steps.
   - `compare.py A.log B.log`: compares two logs with wall-clock readings masked.

`scripts/export_single_file.py` goes the other way: it writes the package back out as one script.
