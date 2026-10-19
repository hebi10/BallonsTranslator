"""LocalWorkspace integration: snapshot saved results, then read/export them modally."""
from __future__ import annotations

from pathlib import Path
import threading

from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import QHBoxLayout, QProgressDialog, QWidget

from ballontranslator.utils import shared
from ballontranslator.utils.reader_export import ReaderHistory, load_book
from .result_reader import ReaderDialog, action_button, plain_label
from .image_merge_dialog import ImageMergeDialog


class BookLoadThread(QThread):
    def __init__(self, folder: Path, result_dir: Path, options: dict, parent=None) -> None:
        super().__init__(parent)
        self.folder, self.result_dir, self.options = folder, result_dir, options
        self.stop_event = threading.Event()
        self.result = None
        self.error = ''

    def run(self) -> None:
        try:
            self.result = load_book(self.folder, self.result_dir, stop=self.stop_event, **self.options)
        except Exception as exc:
            self.error = str(exc)


class ReaderTools(QWidget):
    def __init__(self, workspace, editor=None) -> None:
        super().__init__(workspace)
        self.workspace, self.editor = workspace, editor
        self._loader = None
        self._dialog = None
        self._progress = None
        self._operation = 'read'
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0)
        self.read_button = action_button('뷰어로 읽기', self.open_reader, self)
        self.merge_button = action_button('이미지 합치기 · 폰용 저장', self.open_merge, self)
        row.addWidget(self.read_button); row.addWidget(self.merge_button)
        row.addWidget(plain_label('선택한 폴더의 저장된 결과 사용 · 원본 유지', self), 1)
        workspace.table.itemSelectionChanged.connect(self.sync_state)
        self.sync_state()

    @property
    def busy(self) -> bool:
        return self._loader is not None or self._dialog is not None

    def sync_state(self) -> None:
        selected = 0 <= self.workspace.table.currentRow() < len(self.workspace.queue.jobs)
        enabled = selected and not self.workspace.busy
        self.read_button.setEnabled(enabled); self.merge_button.setEnabled(enabled)
        tip = '목록에서 이미지 폴더를 먼저 선택하세요.' if not selected else '저장된 결과만 사용합니다. 실행 중인 작업은 먼저 종료하세요.'
        self.read_button.setToolTip(tip); self.merge_button.setToolTip(tip)

    def open_reader(self) -> None:
        self._launch('read')

    def open_merge(self) -> None:
        self._launch('merge')

    def _launch(self, operation: str) -> None:
        if self.workspace.busy:
            return
        row = self.workspace.table.currentRow()
        if not 0 <= row < len(self.workspace.queue.jobs):
            self.workspace.status_label.setText('읽거나 내보낼 폴더를 먼저 선택하세요.'); return
        editor = self.editor
        if editor is not None and hasattr(editor, 'module_manager'):
            if editor._local_pipeline_busy() or editor.module_manager.regionReworkBusy() or editor.imsave_thread.isBusy():
                self.workspace.status_label.setText('번역·영역 재작업·이미지 저장이 끝난 뒤 열어 주세요.'); return
        job = self.workspace.queue.jobs[row]
        folder = Path(job['folder']).resolve()
        notices = []
        if job.get('state') != 'done' or job.get('resave'):
            notices.append('완료되지 않았거나 재저장이 필요한 폴더입니다. 이전 결과가 포함될 수 있습니다.')
        names = None
        if editor is not None and hasattr(editor, 'imgtrans_proj'):
            project = editor.imgtrans_proj
            if project.directory and Path(project.directory).resolve() == folder:
                names = tuple(project.pages)
                if editor.canvas.projstate_unsaved:
                    notices.append('편집기에 저장되지 않은 수정이 있습니다. 이 화면은 수정 전 저장된 파일만 사용합니다.')
        options = {'names': names, 'suffix': self.workspace.queue.format,
                   'review_pages': tuple(job.get('review_pages', ())), 'notices': tuple(notices)}
        self._operation = operation
        self._loader = BookLoadThread(folder, self.workspace.queue.output_directory(str(folder)), options, self)
        self._loader.finished.connect(self._loaded)
        self._progress = QProgressDialog('결과 파일과 페이지 순서를 확인하고 있습니다…', '취소', 0, 0, self.workspace)
        self._progress.setWindowTitle('저장된 결과 확인')
        self._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._progress.setMinimumDuration(0); self._progress.setAutoClose(False)
        self._progress.canceled.connect(self.cancel_load)
        self.workspace.set_controls()
        self._progress.show(); self._loader.start()

    def cancel_load(self) -> None:
        if self._loader is not None:
            self._loader.stop_event.set()
            self.workspace.status_label.setText('현재 파일 확인이 끝나는 대로 취소합니다.')

    def _loaded(self) -> None:
        worker = self._loader
        self._loader = None
        worker.wait()
        self._progress.hide(); self._progress.deleteLater(); self._progress = None
        worker.deleteLater()
        if worker.error or worker.stop_event.is_set() or worker.result is None:
            self.workspace.status_label.setText(worker.error or '결과 확인을 취소했습니다.')
            self.workspace.set_controls(); return
        book = worker.result
        if self._operation == 'read':
            history = ReaderHistory(Path(shared.CONFIG_PATH).with_name('local_reader.json'))
            dialog = ReaderDialog(book, history, self.workspace)
        else:
            dialog = ImageMergeDialog(book, self.workspace)
        self._dialog = dialog
        self.workspace.set_controls()
        try:
            dialog.exec_()
            requested_page = getattr(dialog, 'edit_requested_page', None)
            if getattr(dialog, 'history_save_error', ''):
                self.workspace.status_label.setText(dialog.history_save_error)
        finally:
            self._dialog = None
            dialog.deleteLater()
            self.workspace.set_controls()
        if requested_page is not None:
            self.open_editor_page(book.folder, requested_page)

    def open_editor_page(self, folder: Path, page: str) -> None:
        editor = self.editor
        if editor is None or not hasattr(editor, 'open_local_editor'):
            self.workspace.status_label.setText('이 실행 화면에는 편집기가 연결되어 있지 않습니다.'); return
        folder = Path(folder).resolve()
        # Save with the previous project's settings before activating this
        # folder, including when it is already open but another row was chosen.
        if not self.workspace._prepare_folder_action(str(folder)):
            return
        if not editor.imgtrans_proj.directory or Path(editor.imgtrans_proj.directory).resolve() != folder:
            if editor.open_local_editor(str(folder)) is False:
                return
        else:
            if hasattr(editor, 'on_local_settings_changed'):
                if editor.on_local_settings_changed(str(folder)) is False:
                    return
            editor._sync_local_source_language()
        if not editor.imgtrans_proj.directory or Path(editor.imgtrans_proj.directory).resolve() != folder:
            self.workspace.status_label.setText('현재 프로젝트 저장 상태를 확인한 뒤 다시 열어 주세요.'); return
        index = editor.imgtrans_proj.pagename2idx(page)
        if index < 0:
            self.workspace.status_label.setText('편집할 원본 페이지를 찾지 못했습니다.'); return
        editor.pageList.setCurrentRow(index)
        if editor.imgtrans_proj.current_img == page:
            self.workspace.hide(); editor.raise_(); editor.activateWindow()
        else:
            self.workspace.status_label.setText('페이지 전환이 보류되었습니다. 편집기의 저장 오류를 확인하세요.')
