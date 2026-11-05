"""Tests for the platypus package structure (no training): python -m unittest discover -s tests
They need torch and numpy; without them they are skipped."""
import ast
import importlib
import io
import os
import contextlib
import runpy
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    import torch  # noqa: F401
    import numpy  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch and numpy are needed to import the package")
class Package(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = importlib.import_module("platypus")

    def test_every_module_loads(self):
        for m in self.P.MODULES:
            self.assertIn("platypus." + m, sys.modules)

    def test_late_names_are_linked_to_their_definitions(self):
        for m in self.P.MODULES:
            mod = sys.modules["platypus." + m]
            for name, path in getattr(mod, "_LATE", {}).items():
                self.assertIs(getattr(mod, name), getattr(sys.modules[path], name), f"{m}.{name}")

    def test_imported_names_are_the_defining_objects(self):
        # a module-level name has one definition; every module that holds it holds the same object
        from platypus import _order
        owner = {n: sys.modules["platypus." + m] for m, n in _order.ORDER}
        for m in self.P.MODULES:
            mod = sys.modules["platypus." + m]
            for n, o in owner.items():
                if n in mod.__dict__ and mod is not o and not n.startswith("__"):
                    self.assertIs(mod.__dict__[n], o.__dict__[n], f"platypus.{m}.{n}")

    def test_set_global_reaches_every_holder(self):
        from platypus import common, inference
        from platypus.inference import units, sleep
        old = units._INF_NAME
        try:
            common._set_global("_INF_NAME", "probe")
            self.assertEqual(units._INF_NAME, "probe")
            self.assertEqual(sleep._INF_NAME, "probe")
        finally:
            common._set_global("_INF_NAME", old)
        self.assertTrue(inference)

    def test_main_builds_the_configuration_and_banner(self):
        from platypus import presets
        runner = sys.modules["platypus.runner"]
        got = {}

        def stub(cfg):
            got["cfg"] = cfg
            return {"bpc": 1.0, "forget": 0.0, "post": [1.0], "passes": [], "domains": ["a"], "timing": {}}
        saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("PLATYPUS_") or k == "LAB_OVR"}
        r0, runner.run = runner.run, stub
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(os.path.join(ROOT, "platypus", "__main__.py"), run_name="__main__")
        finally:
            runner.run = r0
            os.environ.update(saved)
        text = out.getvalue()
        self.assertIn(presets.BUILD_NAME, text)
        self.assertIn("| overrides " + presets.DEFAULT_LAB_OVR, text)
        self.assertEqual(got["cfg"].inf_mix_dream_src, "jepa-name,jepa-wide,cortex-jepa-name-fam,sequence")


class Export(unittest.TestCase):
    def test_single_file_export_compiles_and_keeps_every_name(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "single.py")
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_single_file.py"), out], check=True, capture_output=True)
            tree = ast.parse(open(out).read())
            top = set()
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    top.add(n.name)
                elif not isinstance(n, ast.If):
                    top.update(x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store))
            order = ast.literal_eval(ast.parse(open(os.path.join(ROOT, "platypus", "_order.py")).read()).body[1].value)
            self.assertEqual(sorted({n for m, n in order} - top - {"_nm"}), [])   # _nm: the profiling loop variable, now inside _label_regions

    def test_names_have_no_apostrophes(self):
        tree = ast.parse(open(os.path.join(ROOT, "platypus", "presets.py")).read())
        for n in tree.body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("BUILD_NAME", "BASE_NAME"):
                self.assertNotIn("'", n.value.value)


class MergeOverridesPackage(unittest.TestCase):
    def test_defaults_from_presets(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "merge_overrides.py"), os.path.join(ROOT, "platypus")],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn('"n_passes":3', out)


if __name__ == "__main__":
    unittest.main()
