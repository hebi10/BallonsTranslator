import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

BUILDER = Path(__file__).resolve().parents[1] / 'build_local_package.py'


class SourcePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.output = Path(self.temp.name) / 'source.zip'

    def track(self, name, content='source'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        subprocess.run(['git', '-C', str(self.root), 'add', '--', name], check=True)
        return path

    def build(self):
        return subprocess.run([sys.executable, str(BUILDER), '--root', str(self.root), '--output', str(self.output)], capture_output=True, text=True)

    def test_packages_current_tracked_contents_and_excludes_private_runtime_data(self):
        source = self.track('ballontranslator/main.py', 'old')
        source.write_text('working change', encoding='utf-8')
        self.track('LICENSE')
        self.track('.env', 'secret')
        self.track('config/hebi-local.json', '{}')
        self.track('config/config.json', 'private configuration')
        self.track('data/models/model.bin', 'weights')
        self.track('id_ed25519', 'private key')
        (self.root / 'untracked-token.txt').write_text('secret', encoding='utf-8')
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertEqual(archive.read('BallonsTranslator-Local/ballontranslator/main.py'), b'working change')
            self.assertIn('BallonsTranslator-Local/config/hebi-local.json', names)
            self.assertFalse(any('secret' in archive.read(name).decode(errors='ignore') for name in names))
            self.assertNotIn('BallonsTranslator-Local/data/models/model.bin', names)
            self.assertNotIn('BallonsTranslator-Local/config/config.json', names)

    def test_refuses_to_overwrite_an_existing_package(self):
        self.track('LICENSE')
        self.output.write_bytes(b'existing result')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), b'existing result')

    def test_omits_deleted_tracked_files(self):
        self.track('LICENSE')
        self.track('deleted.py').unlink()
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.output) as archive:
            self.assertNotIn('BallonsTranslator-Local/deleted.py', archive.namelist())

    @unittest.skipIf(sys.platform == 'win32', 'Symlink creation requires special Windows privileges')
    def test_rejects_tracked_symlinks(self):
        outside = Path(self.temp.name) / 'private.txt'
        outside.write_text('secret', encoding='utf-8')
        (self.root / 'linked.txt').symlink_to(outside)
        subprocess.run(['git', '-C', str(self.root), 'add', 'linked.txt'], check=True)
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
