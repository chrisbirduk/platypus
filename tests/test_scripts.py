"""Tests for the deployment scripts, run in CI: python -m unittest discover -s tests"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)
SCRIPTS = os.path.join(DEPLOY, "scripts")
FIX = os.path.join(DEPLOY, "fixtures")
sys.path.insert(0, SCRIPTS)

import extract_readouts  # noqa: E402
import scan_secrets  # noqa: E402


class ExtractReadouts(unittest.TestCase):
    def setUp(self):
        self.r = extract_readouts.parse(os.path.join(FIX, "gpu_log_cg26m_excerpt.txt"))
        self.s = extract_readouts.summary(self.r)

    def test_gpu_cg26m_headlines(self):
        # Values read by hand from the cg26m GPU log (to visit 11).
        self.assertEqual(self.s["weights_two_hop_best"], 0.55)
        self.assertEqual(self.r["visits"][10]["weights"]["two_hop"], 0.55)
        self.assertEqual(self.r["visits"][10]["store"]["two_hop"], 0.94)
        self.assertEqual(self.s["cf_kept"], 7)
        self.assertEqual(self.s["cf_rolled_back"], 2)
        self.assertEqual(self.s["released_total"], 43)
        self.assertEqual(self.s["cue_repair_wrong_total"], 0)
        self.assertEqual(self.r["passes"][0]["mean_bpc"], 1.733)
        self.assertEqual(self.r["passes"][0]["forgetting"], -0.002)

    def test_release_belongs_to_the_sleep_it_was_printed_in(self):
        # The release line is printed before that sleep's eval.
        self.assertEqual(self.r["visits"][9]["released"], 14)

    def test_local_final(self):
        r = extract_readouts.parse(os.path.join(FIX, "local_log_tail_excerpt.txt"))
        self.assertEqual(r["final"]["bpc"], 2.079)
        self.assertEqual(r["temporal_gradient"], {"older": 0.27, "recent": 0.11})

    def test_markdown_renders(self):
        md = extract_readouts.markdown([self.r])
        self.assertIn("## Summary", md)


class ScanSecrets(unittest.TestCase):
    def test_clean_fixtures(self):
        self.assertEqual(scan_secrets.main([FIX]), 0)

    def test_finds_a_planted_key(self):
        with tempfile.TemporaryDirectory() as d:
            fake = "sk-ant-" + "x" * 30  # not a real key
            open(os.path.join(d, "leak.txt"), "w").write(f"key = {fake}\n")
            self.assertEqual(scan_secrets.main([d]), 1)


class MergeOverrides(unittest.TestCase):
    def test_merge_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            build = os.path.join(d, "b.py")
            open(build, "w").write("import os\n_OVR=os.environ.get('LAB_OVR','{\"seed\": 1, \"a\": 2}')\n")
            ovr = os.path.join(d, "o.json")
            json.dump({"a": 3, "_note": "ignored", "b": 4}, open(ovr, "w"))
            out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "merge_overrides.py"), build, ovr],
                                 capture_output=True, text=True, check=True).stdout
            self.assertEqual(json.loads(out), {"seed": 1, "a": 3, "b": 4})


if __name__ == "__main__":
    unittest.main()


try:
    import pyflakes  # noqa: F401
    HAVE_PYFLAKES = True
except ImportError:
    HAVE_PYFLAKES = False


@unittest.skipUnless(HAVE_PYFLAKES, "pyflakes is needed")
class UndefinedNames(unittest.TestCase):
    def test_package_has_no_undefined_names(self):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "check_undefined.py"), os.path.join(DEPLOY, "platypus")],
                           capture_output=True, text=True, cwd=DEPLOY)
        self.assertEqual(r.returncode, 0, r.stdout)
