"""Checkpoint module state: names skipped by module, CUDA graph entries kept live, modules imported on demand.
(The whole-run check, a resumed run printing what the uninterrupted one prints, is tools/ckpt/equiv.py.)"""
import importlib
import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401
    import scipy  # noqa: F401  (inference.premises needs it)
    HAVE = True
except ImportError:
    HAVE = False

PREM = 'platypus.inference.premises'


class _O:
    pass


@unittest.skipUnless(HAVE, 'needs torch and scipy')
class CheckpointModuleState(unittest.TestCase):
    def setUp(self):
        import platypus  # noqa: F401
        from platypus import checkpoint, conduit
        self.ck, self.conduit = checkpoint, conduit
        self.prem = importlib.import_module(PREM)
        self.tmp = tempfile.TemporaryDirectory()
        self._env = os.environ.get('PLATYPUS_CKPT')
        os.environ['PLATYPUS_CKPT'] = self.tmp.name
        self.cdg = dict(conduit._CDG)
        self.cfg, self.model = _O(), _O()
        self.cfg.seed = 1; self.model.step = 7
        # ck_restore rebinds every module-level value to its unpickled copy (equal, not the same object): put the
        #   originals back afterwards, so the other tests still see one object per imported name
        self.bound = [(m_, dict(vars(m_))) for m_ in checkpoint._ck_modules()]

    def tearDown(self):
        for m_, d_ in self.bound:
            for k_, v_ in d_.items():
                if vars(m_).get(k_) is not v_: setattr(m_, k_, v_)
        sys.modules[PREM] = self.prem
        self.conduit._CDG.clear(); self.conduit._CDG.update(self.cdg)
        self.prem._PM.clear()
        if self._env is None: os.environ.pop('PLATYPUS_CKPT', None)
        else: os.environ['PLATYPUS_CKPT'] = self._env
        self.tmp.cleanup()

    def _save(self):
        ident = {'build': 'b', 'options': 'o', 'python': 'p', 'torch': 't'}
        self.ck.ck_save(0, 2, self.cfg, self.model, {}, {}, 0.0, ident)
        return ident

    def test_skips_are_per_module(self):
        self.assertFalse(self.ck._keep_live(PREM, '_PM'))                 # run state, despite sharing a name with no checkpoint name
        self.assertTrue(self.ck._keep_live('platypus.layers', '_FLA'))
        self.assertFalse(self.ck._keep_live('platypus.conduit', '_FLA'))   # the same name elsewhere is saved
        self.assertTrue(self.ck._keep_live('platypus.sampler', '_LATE'))   # '*': every module

    def test_premises_store_is_saved_and_restored(self):
        self.prem._PM.update(rti={('A', 'C'): {'judged': 1}}, ndv_now=3)
        ident = self._save()
        self.prem._PM.clear()
        p_ = self.ck.ck_load(self.cfg, self.model, ident)
        self.ck.ck_restore(p_, self.cfg, self.model)
        self.assertEqual(self.prem._PM, {'rti': {('A', 'C'): {'judged': 1}}, 'ndv_now': 3})

    def test_module_imported_on_demand_is_imported_on_resume(self):
        self.prem._PM['rtc_pend'] = [1, 2]
        ident = self._save()
        del sys.modules[PREM]                                              # as in a new process that has not imported it yet
        p_ = self.ck.ck_load(self.cfg, self.model, ident)
        self.ck.ck_restore(p_, self.cfg, self.model)
        self.assertIsNot(sys.modules[PREM], self.prem)
        self.assertEqual(sys.modules[PREM]._PM.get('rtc_pend'), [1, 2])

    def test_cuda_graph_entries_are_not_saved(self):
        cdg = self.conduit._CDG
        cdg['g'] = {'k': threading.Lock()}; cdg['pool'] = threading.Lock()   # stand-ins: neither can be pickled
        cdg['n_rep'] = 41
        ident = self._save()                                               # must not try to pickle the graph entries
        live_g = {}; cdg['g'] = live_g; cdg['pool'] = None; cdg['n_rep'] = 0
        p_ = self.ck.ck_load(self.cfg, self.model, ident)
        self.ck.ck_restore(p_, self.cfg, self.model)
        self.assertEqual(cdg['n_rep'], 41)                                 # counters come back
        self.assertIs(cdg['g'], live_g)                                    # graphs stay as the resumed run has them
        self.assertIsNone(cdg['pool'])


class CheckpointLocals(unittest.TestCase):
    """every run() local that is made before the visit loop and changed inside it must be in runner.CK_LOCALS,
    or a resumed run continues with its value as rebuilt at the start (post_w was missed this way)"""
    # not run state: model (restored with its attributes; _t2s is one), and comprehension or loop temporaries
    #   (names ending in '_', single letters, li)
    ALLOW = {'model', '_t2s', 'li', '_'}

    def test_carried_locals_are_saved(self):
        import ast
        t = ast.parse(open(os.path.join(ROOT, 'platypus', 'runner.py'), encoding='utf-8').read())
        ck = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CK_LOCALS')
        CK = set(ast.literal_eval(ck.value))
        run = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        loop = next(n for n in run.body if isinstance(n, ast.For) and getattr(getattr(n.iter, 'func', None), 'id', '') == 'enumerate'
                    and getattr(n.iter.args[0], 'id', '') == 'schedule')
        base = lambda x: base(x.value) if isinstance(x, (ast.Subscript, ast.Attribute)) else getattr(x, 'id', None)
        made = {x.id for n in run.body[:run.body.index(loop)] for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        changed = set()
        for x in ast.walk(loop):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store): changed.add(x.id)
            elif isinstance(x, (ast.Subscript, ast.Attribute)) and isinstance(x.ctx, ast.Store): changed.add(base(x.value))
            elif isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in (
                    'append', 'extend', 'update', 'add', 'pop', 'clear', 'setdefault', 'insert', 'remove', 'discard', 'popleft', 'appendleft', 'sort'):
                changed.add(base(x.func.value))
        missing = sorted(n for n in (made & changed) - CK - self.ALLOW if not n.endswith('_') and len(n) > 1)
        self.assertEqual(missing, [], 'add these to runner.CK_LOCALS (and to the resume block in run())')
        restored = {x.targets[0].id for x in ast.walk(run) if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)
                    and isinstance(x.value, ast.Subscript) and getattr(x.value.value, 'id', '') == 'L_'}
        self.assertEqual(sorted(CK - restored), [], 'saved in CK_LOCALS but not set from L_ in the resume block of run()')


if __name__ == '__main__':
    unittest.main()
