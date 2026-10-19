"""Prepare an isolated, one-image project without touching the source project."""

import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

from qtpy.QtCore import QObject, QThread, Signal

from ballontranslator.utils import shared
from ballontranslator.utils.io_utils import IMG_EXT


def prepare_preview_image(
    source: Union[str, os.PathLike],
    root: Union[str, os.PathLike],
    stop_event: threading.Event,
) -> Path:
    """Copy a stable source snapshot into a new, retained preview project.

    Only the folder allocated by this call is removed on failure or cancellation.
    Call from a worker thread: opening and copying large images can block.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     source = Path(directory) / 'page.png'
    ...     _ = source.write_bytes(b'image content')
    ...     folder = prepare_preview_image(source, Path(directory) / 'previews', threading.Event())
    ...     (folder / 'preview.png').read_bytes() == source.read_bytes()
    True
    """
    if stop_event.is_set():
        raise RuntimeError('미리보기 준비가 취소되었습니다.')
    source = Path(source).resolve(strict=True)
    extension = source.suffix.lower()
    if extension not in IMG_EXT:
        raise ValueError('미리보기에서 지원하지 않는 이미지 형식입니다.')
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('미리보기 원본은 이미지 파일이어야 합니다.')

    # Exclude access time, which reading the source can legitimately change.
    attributes = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    expected = tuple(getattr(before, name) for name in attributes)
    folder: Optional[Path] = None
    try:
        with source.open('rb') as reader:
            opened = os.fstat(reader.fileno())
            if tuple(getattr(opened, name) for name in attributes) != expected:
                raise RuntimeError('복사 중 원본 이미지가 변경되었습니다. 다시 시도하세요.')
            if stop_event.is_set():
                raise RuntimeError('미리보기 준비가 취소되었습니다.')
            root = Path(root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix='preview-', dir=root))
            # A fixed basename is portable even when the source name is not.
            with (folder / ('preview' + extension)).open('xb') as writer:
                while True:
                    if stop_event.is_set():
                        raise RuntimeError('미리보기 준비가 취소되었습니다.')
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
            for after in (os.fstat(reader.fileno()), source.stat()):
                if tuple(getattr(after, name) for name in attributes) != expected:
                    raise RuntimeError('복사 중 원본 이미지가 변경되었습니다. 다시 시도하세요.')
            if stop_event.is_set():
                raise RuntimeError('미리보기 준비가 취소되었습니다.')
        return folder
    except BaseException:
        if folder is not None:
            shutil.rmtree(folder)
        raise


class LocalPreviewCopyThread(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source_path: Union[str, os.PathLike],
        parent: Optional[QObject] = None,
        root: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.root = Path(root) if root is not None else Path(shared.CONFIG_PATH).parent / 'local-previews'
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            folder = prepare_preview_image(self.source_path, self.root, self.stop_event)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(str(folder))
