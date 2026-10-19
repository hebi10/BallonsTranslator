import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.local_preview import LocalPreviewCopyThread, prepare_preview_image
from ballontranslator.utils import shared


def make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 3), 'red').save(path)
    return path


def test_preview_copies_one_image_and_preserves_source_project(tmp_path):
    source = make_image(tmp_path / '원본' / '한 장.PNG')
    neighbors = {
        source: source.read_bytes(),
        source.parent / 'imgtrans_원본.json': b'{"existing": true}',
        source.parent / 'mask.png': b'existing mask',
        source.parent / 'result' / source.name: b'existing translation',
    }
    for path, data in neighbors.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    root = tmp_path / 'previews'

    folder = prepare_preview_image(source, root, threading.Event())

    assert folder.parent == root
    copies = list(folder.iterdir())
    assert len(copies) == 1
    assert copies[0].suffix.lower() == '.png'
    assert copies[0].read_bytes() == neighbors[source]
    assert all(path.read_bytes() == content for path, content in neighbors.items())
    second = prepare_preview_image(source, root, threading.Event())
    assert second != folder
    assert copies[0].read_bytes() == neighbors[source]


@pytest.mark.parametrize('kind', ['missing', 'directory', 'unsupported'])
def test_invalid_source_does_not_create_or_overwrite_files(tmp_path, kind):
    source = tmp_path / ('source.txt' if kind == 'unsupported' else 'source.png')
    if kind == 'directory':
        source.mkdir()
    elif kind == 'unsupported':
        source.write_text('preserve', encoding='utf-8')
    root = tmp_path / 'previews'
    with pytest.raises((ValueError, OSError)):
        prepare_preview_image(source, root, threading.Event())
    assert not root.exists()
    if kind == 'unsupported':
        assert source.read_text(encoding='utf-8') == 'preserve'


def test_already_cancelled_copy_does_not_create_folder(tmp_path):
    source = make_image(tmp_path / 'source.png')
    stop_event = threading.Event()
    stop_event.set()
    root = tmp_path / 'previews'
    with pytest.raises(RuntimeError, match='취소'):
        prepare_preview_image(source, root, stop_event)
    assert not root.exists()


def change_after_read(monkeypatch, source, action):
    original_open = Path.open

    class ChangingReader:
        def __init__(self, reader):
            self.reader = reader

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def fileno(self):
            return self.reader.fileno()

        def read(self, size):
            data = self.reader.read(size)
            if data:
                action()
            return data

    def patched_open(path, mode='r', *args, **kwargs):
        result = original_open(path, mode, *args, **kwargs)
        return ChangingReader(result) if path == source and mode == 'rb' else result

    monkeypatch.setattr(Path, 'open', patched_open)


def test_cancel_during_copy_removes_only_new_folder(tmp_path, monkeypatch):
    source = make_image(tmp_path / 'source.png')
    before = source.read_bytes()
    root = tmp_path / 'previews'
    existing = make_image(root / 'previous-preview' / 'page.png')
    stop_event = threading.Event()
    change_after_read(monkeypatch, source, stop_event.set)

    with pytest.raises(RuntimeError, match='취소'):
        prepare_preview_image(source, root, stop_event)

    monkeypatch.undo()
    assert list(root.iterdir()) == [existing.parent]
    assert source.read_bytes() == before
    assert existing.is_file()


def test_source_change_during_copy_discards_unstable_snapshot(tmp_path, monkeypatch):
    source = make_image(tmp_path / 'source.png')
    root = tmp_path / 'previews'
    original = source.read_bytes()
    change_after_read(monkeypatch, source, lambda: source.write_bytes(original + b'changed'))

    with pytest.raises(RuntimeError, match='변경'):
        prepare_preview_image(source, root, threading.Event())

    monkeypatch.undo()
    assert list(root.iterdir()) == []
    assert source.read_bytes() == original + b'changed'


def test_write_failure_removes_only_new_folder(tmp_path, monkeypatch):
    source = make_image(tmp_path / 'source.png')
    original = source.read_bytes()
    root = tmp_path / 'previews'
    existing = make_image(root / 'previous-preview' / 'page.png')
    original_open = Path.open

    def fail_destination(path, mode='r', *args, **kwargs):
        if mode == 'xb':
            raise OSError('disk full')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', fail_destination)
    with pytest.raises(OSError, match='disk full'):
        prepare_preview_image(source, root, threading.Event())
    assert list(root.iterdir()) == [existing.parent]
    assert source.read_bytes() == original


def test_stable_source_with_different_path_and_descriptor_ctime_can_be_copied(tmp_path, monkeypatch):
    source = make_image(tmp_path / 'source.png')
    original_fstat = os.fstat

    def descriptor_stat(fd):
        result = original_fstat(fd)
        # Windows Python 3.12 path-stat reports creation time, while fstat
        # reports metadata change time. Neither value changes during this copy.
        return SimpleNamespace(**{
            name: getattr(result, name) + (1 if name == 'st_ctime_ns' else 0)
            for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        })

    monkeypatch.setattr(os, 'fstat', descriptor_stat)
    folder = prepare_preview_image(source, tmp_path / 'previews', threading.Event())
    assert (folder / 'preview.png').read_bytes() == source.read_bytes()


def test_same_size_content_change_with_restored_mtime_is_rejected(tmp_path, monkeypatch):
    source = make_image(tmp_path / 'source.png')
    original = source.read_bytes()
    before = source.stat()
    root = tmp_path / 'previews'
    original_stat = Path.stat

    def path_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path != source:
            return result
        # Match Windows path-stat: creation time does not reveal this rewrite.
        return SimpleNamespace(**{
            name: before.st_ctime_ns if name == 'st_ctime_ns' else getattr(result, name)
            for name in ('st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        })

    def rewrite_source():
        source.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))

    monkeypatch.setattr(Path, 'stat', path_stat)
    change_after_read(monkeypatch, source, rewrite_source)
    with pytest.raises(RuntimeError, match='변경'):
        prepare_preview_image(source, root, threading.Event())
    monkeypatch.undo()
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert source.read_bytes() != original
    assert list(root.iterdir()) == []


def test_worker_runs_copy_and_uses_config_preview_root(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = make_image(tmp_path / '선택.png')
    monkeypatch.setattr(shared, 'CONFIG_PATH', str(tmp_path / 'config' / 'config.json'))
    worker = LocalPreviewCopyThread(source)
    completed, failed = QSignalSpy(worker.completed), QSignalSpy(worker.failed)
    root = tmp_path / 'config' / 'local-previews'
    assert not root.exists()

    worker.start()
    assert worker.wait(3000)
    app.processEvents()

    assert len(failed) == 0
    assert len(completed) == 1
    folder = Path(completed[0][0])
    assert folder.parent == root
    assert len(list(folder.iterdir())) == 1


def test_worker_reports_cancel_without_success(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = make_image(tmp_path / 'source.png')
    root = tmp_path / 'previews'
    worker = LocalPreviewCopyThread(source, root=root)
    worker.stop_event.set()
    completed, failed = QSignalSpy(worker.completed), QSignalSpy(worker.failed)
    worker.start()
    assert worker.wait(3000)
    app.processEvents()
    assert len(completed) == 0
    assert len(failed) == 1
    assert '취소' in failed[0][0]
    assert not root.exists()
