import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / 'run_local_regressions.py'


class LocalRegressionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_tests(self, names):
        return subprocess.run([sys.executable, str(RUNNER), '--root', str(self.root), '--files', *names], capture_output=True, text=True, encoding='utf-8')

    def test_files_use_separate_processes_and_offscreen_environment(self):
        for index in range(2):
            (self.root / f'test_pass_{index}.py').write_text(
                f"import os\nfrom pathlib import Path\ndef test_process():\n    assert os.environ['QT_QPA_PLATFORM'] == 'offscreen'\n    Path('pid-{index}.txt').write_text(str(os.getpid()))\n",
                encoding='utf-8')
        result = self.run_tests(['test_pass_0.py', 'test_pass_1.py'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotEqual((self.root / 'pid-0.txt').read_text(), (self.root / 'pid-1.txt').read_text())

    def test_failure_is_reported_and_causes_nonzero_exit(self):
        (self.root / 'test_fail.py').write_text('def test_failure():\n    assert False, "regression detected"\n', encoding='utf-8')
        result = self.run_tests(['test_fail.py'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('regression detected', result.stdout)
        self.assertIn('FAIL test_fail.py', result.stdout)

    def test_missing_file_stops_before_any_test_runs(self):
        (self.root / 'test_pass.py').write_text('from pathlib import Path\ndef test_run():\n    Path("ran").touch()\n', encoding='utf-8')
        result = self.run_tests(['test_pass.py', 'missing.py'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing.py', result.stderr)
        self.assertFalse((self.root / 'ran').exists())

    def test_korean_failure_survives_legacy_console_encoding_and_continues(self):
        (self.root / 'test_한국어.py').write_text('def test_failure():\n    assert False, "말풍선 확인 필요"\n', encoding='utf-8')
        (self.root / 'test_next.py').write_text('def test_next():\n    assert True\n', encoding='utf-8')
        result = subprocess.run(
            [sys.executable, str(RUNNER), '--root', str(self.root), '--files', 'test_한국어.py', 'test_next.py'],
            env=dict(os.environ, PYTHONIOENCODING='cp1252', PYTHONUTF8='0'),
            capture_output=True, text=True, encoding='utf-8',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('말풍선 확인 필요', result.stdout)
        self.assertIn('FAIL test_한국어.py', result.stdout)
        self.assertIn('PASS test_next.py', result.stdout)
        self.assertNotIn('UnicodeEncodeError', result.stderr)


if __name__ == '__main__':
    unittest.main()
