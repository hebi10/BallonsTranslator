"""Local setup and a durable folder queue, using the existing editor pipeline."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

from qtpy.QtCore import Qt, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QMessageBox,
)

from ballontranslator.utils import shared
from ballontranslator.utils.logger import logger as LOGGER


STATES = {
    'pending': '대기', 'running': '번역 중', 'paused': '일시정지',
    'done': '완료', 'failed': '확인 필요',
}


class LocalJobQueue:
    """Persist folder states without writing during passive recovery.

    >>> LocalJobQueue.normalize_state('running')
    'paused'
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.jobs: list[dict] = []
        self.output = ''
        self.format = '.png'
        self.warning = ''
        self.writable = True
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('jobs'), list):
                raise ValueError('지원하지 않는 작업 목록입니다.')
        except (ValueError, OSError) as exc:
            self.writable = False
            self.warning = f'작업 목록을 읽지 못했습니다. 원본 파일은 보존됩니다: {self.path} ({exc})'
            LOGGER.warning(self.warning)
            return
        self.output = data.get('output', '') if isinstance(data.get('output', ''), str) else ''
        image_format = data.get('format')
        self.format = image_format if isinstance(image_format, str) and image_format in {'.png', '.jpg'} else '.png'
        seen = set()
        invalid = 0
        for row in data['jobs']:
            if not isinstance(row, dict) or not isinstance(row.get('folder'), str) or not row['folder'].strip():
                invalid += 1
                continue
            try:
                folder = str(Path(row['folder']).resolve())
            except (OSError, ValueError):
                invalid += 1
                continue
            key = os.path.normcase(folder)
            if key in seen:
                continue
            seen.add(key)
            state = self.normalize_state(row.get('state'))
            if not isinstance(row.get('state'), str) or row.get('state') not in STATES:
                invalid += 1
            review_pages = row.get('review_pages', [])
            if not isinstance(review_pages, list):
                review_pages = []
                invalid += 1
            valid_review_pages = []
            for page in review_pages:
                if not isinstance(page, str) or not page.strip():
                    invalid += 1
                elif page not in valid_review_pages:
                    valid_review_pages.append(page)
            self.jobs.append({
                'folder': folder, 'state': state,
                'done': self._count(row.get('done')), 'total': self._count(row.get('total')),
                'error': row.get('error', '') if isinstance(row.get('error', ''), str) else '',
                'restart': row.get('restart') is True,
                'resave': row.get('resave') is True or row.get('state') == 'running',
                'review_pages': valid_review_pages,
            })
        if invalid:
            self.warning = f'작업 목록에서 잘못된 항목 {invalid}개를 제외했습니다.'
            LOGGER.warning(self.warning)

    @staticmethod
    def normalize_state(state: str) -> str:
        return 'paused' if state == 'running' else state if isinstance(state, str) and state in STATES else 'pending'

    @staticmethod
    def _count(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def save(self) -> None:
        if not self.writable:
            raise ValueError(self.warning)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'jobs': self.jobs, 'output': self.output, 'format': self.format}
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def add(self, folders: list[str]) -> None:
        if not self.writable:
            raise ValueError(self.warning)
        seen = {os.path.normcase(job['folder']) for job in self.jobs}
        for folder in folders:
            path = Path(folder).resolve()
            if not path.is_dir():
                continue
            key = os.path.normcase(str(path))
            if key not in seen:
                self.jobs.append({'folder': str(path), 'state': 'pending', 'done': 0, 'total': 0, 'error': ''})
                seen.add(key)
        self.save()

    def output_directory(self, folder: str) -> Path:
        if not self.output:
            return Path(folder) / 'result'
        digest = hashlib.sha256(os.path.normcase(str(Path(folder).resolve())).encode('utf-8')).hexdigest()[:8]
        return Path(self.output) / f'{Path(folder).name}-{digest}'


class LocalSetupThread(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: str, parent=None) -> None:
        super().__init__(parent)
        self.operation = operation
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            from ballontranslator.utils.local_edition import local_readiness, prepare_local_models
            from ballontranslator.utils.local_ollama import OllamaClient
            client = OllamaClient()
            result = None
            if self.operation == 'status':
                checks = local_readiness()
                try:
                    status = client.status()
                    checks.append(('Ollama 실행', True, f"버전 {status.get('version', '')}"))
                    checks.append(('번역 모델', bool(status.get('installed')), str(status.get('model', ''))))
                except Exception as exc:
                    checks.append(('Ollama 실행', False, f'launch_local.bat로 실행하세요. {exc}'))
                result = checks
            elif self.operation == 'prepare':
                prepare_local_models(self.progress.emit, self.stop_event)
            elif self.operation == 'pull':
                client.pull(self.progress.emit, self.stop_event)
            elif self.operation == 'test':
                result = client.test_translation(stop_event=self.stop_event)
            else:
                raise ValueError('알 수 없는 준비 작업입니다.')
            if self.stop_event.is_set():
                self.failed.emit('작업을 중지했습니다. 준비 상태를 다시 확인하세요.')
            else:
                self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class LocalWorkspace(QDialog):
    run_requested = Signal(str)
    pause_requested = Signal()
    edit_requested = Signal(str)

    def __init__(self, parent=None, *, queue_path: Path = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('해비 만화 번역기 · 로컬 작업')
        self.setObjectName('LocalWorkspace')
        self.resize(1040, 820)
        self.setMinimumSize(780, 600)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAcceptDrops(True)
        self.worker: LocalSetupThread | None = None
        self.active_index: int | None = None
        self.running = False
        self.pause_pending = False
        self.queue = LocalJobQueue(queue_path or Path(shared.CONFIG_PATH).with_name('local_jobs.json'))
        self._last_operation = ''
        self._worker_result = None
        self._worker_error = ''
        self._build_ui()
        self.refresh_queue()
        self.status_label.setText(self.queue.warning or '준비 상태 확인 → 폴더 추가 → 번역 시작 순서로 진행하세요.')

    @property
    def busy(self) -> bool:
        return self.worker is not None or self.running

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(14)
        title = QLabel('로컬 만화 번역', self)
        font = title.font()
        font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        title.setFont(font)
        outer.addWidget(title)
        subtitle = QLabel('일본어 → 한국어  ·  CTD / manga_ocr / Qwen3.5 9B / LaMa', self)
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)
        setup = QFrame(self)
        setup.setObjectName('LocalSetupSurface')
        setup_layout = QVBoxLayout(setup)
        setup_layout.addWidget(QLabel('1  실행 준비', setup))
        self.checks = QTableWidget(0, 3, setup)
        self.checks.setHorizontalHeaderLabels(['항목', '상태', '필요한 작업'])
        self.checks.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.checks.setWordWrap(False)
        self.checks.verticalHeader().hide()
        self.checks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.checks.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.checks.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.checks.setMaximumHeight(170)
        setup_layout.addWidget(self.checks)
        buttons = QHBoxLayout()
        self.setup_buttons = []
        for label, operation in [('상태 확인', 'status'), ('OCR·복원 모델 준비', 'prepare'), ('번역 모델 다운로드', 'pull'), ('번역 실행 테스트', 'test')]:
            button = QPushButton(label, setup)
            button.setProperty('operation', operation)
            button.clicked.connect(self.on_setup_clicked)
            self.setup_buttons.append(button)
            buttons.addWidget(button)
        self.ollama_button = QPushButton('Ollama 설치 안내', setup)
        self.ollama_button.clicked.connect(self.open_ollama_download)
        buttons.addWidget(self.ollama_button)
        setup_layout.addLayout(buttons)
        hint = QLabel('번역 모델 다운로드 약 6.6GB. 설치 후 실행 테스트로 메모리와 실제 작동을 확인합니다.\n처음 설치에는 인터넷이 필요하며 번역은 PC 내부에서 처리합니다.', setup)
        hint.setWordWrap(True)
        setup_layout.addWidget(hint)
        outer.addWidget(setup)
        outer.addWidget(QLabel('2  번역 폴더 · 여러 폴더를 이 창에 끌어 놓을 수 있습니다.', self))
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(['이미지 폴더', '진행', '상태'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        outer.addWidget(self.table, 1)
        folder_buttons = QHBoxLayout()
        self.add_button = QPushButton('폴더 추가', self)
        self.add_button.clicked.connect(self.choose_folder)
        self.remove_button = QPushButton('목록에서 제거', self)
        self.remove_button.clicked.connect(self.remove_selected)
        self.edit_button = QPushButton('결과 검수·대사 수정', self)
        self.edit_button.clicked.connect(self.edit_selected)
        self.open_button = QPushButton('결과 폴더 열기', self)
        self.open_button.clicked.connect(self.open_result)
        for button in (self.add_button, self.remove_button, self.edit_button, self.open_button):
            folder_buttons.addWidget(button)
        self.restart_button = QPushButton('선택 폴더 새로 번역', self)
        self.restart_button.clicked.connect(self.restart_selected)
        folder_buttons.addWidget(self.restart_button)
        outer.addLayout(folder_buttons)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel('추가 저장 위치', self))
        self.output_edit = QLineEdit(self.queue.output, self)
        self.output_edit.setPlaceholderText('기본: 각 원본 폴더의 result 폴더')
        self.output_edit.editingFinished.connect(self.save_options)
        output_row.addWidget(self.output_edit, 1)
        self.output_button = QPushButton('선택', self)
        self.output_button.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_button)
        self.format_combo = QComboBox(self)
        self.format_combo.addItem('PNG', '.png')
        self.format_combo.addItem('JPEG', '.jpg')
        self.format_combo.setCurrentIndex(1 if self.queue.format == '.jpg' else 0)
        self.format_combo.currentIndexChanged.connect(self.save_options)
        output_row.addWidget(self.format_combo)
        outer.addLayout(output_row)
        self.progress = QProgressBar(self)
        self.progress.setMinimumHeight(24)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.start_button = QPushButton('번역 시작·이어서 실행', self)
        self.start_button.clicked.connect(self.start_queue)
        self.retry_button = QPushButton('실패한 폴더 재시도', self)
        self.retry_button.clicked.connect(self.retry_failed)
        self.pause_button = QPushButton('안전하게 일시정지', self)
        self.pause_button.clicked.connect(self.request_pause)
        self.cancel_button = QPushButton('설치 작업 중지', self)
        self.cancel_button.clicked.connect(self.cancel_setup)
        self.close_button = QPushButton('편집기로 돌아가기', self)
        self.close_button.clicked.connect(self.reject)
        for button in (self.start_button, self.retry_button, self.pause_button, self.cancel_button, self.close_button):
            actions.addWidget(button)
        outer.addLayout(actions)
        self.setStyleSheet('''
            QDialog#LocalWorkspace { background: #191919; color: #ededed; }
            QDialog#LocalWorkspace QLabel { color: #ededed; background: transparent; }
            QFrame#LocalSetupSurface { background: #252525; border: 1px solid #444444; }
            QDialog#LocalWorkspace QPushButton { color: #ededed; background: #333333; border: 1px solid #666666; padding: 8px; }
            QDialog#LocalWorkspace QPushButton:disabled { color: #888888; border: 1px solid #444444; }
            QDialog#LocalWorkspace QLineEdit, QDialog#LocalWorkspace QComboBox,
            QDialog#LocalWorkspace QTableWidget { background: #242424; color: #ededed; border: 1px solid #555555; selection-background-color: #505050; }
            QDialog#LocalWorkspace QHeaderView::section { background: #303030; color: #ededed; padding: 5px; border: 1px solid #444444; }
            QDialog#LocalWorkspace QProgressBar { color: #ededed; background: #303030; border: 1px solid #555555; text-align: center; }
            QDialog#LocalWorkspace QProgressBar::chunk { background: #666666; }
        ''')
        self.set_controls()

    def set_controls(self) -> None:
        idle = not self.busy
        for button in self.setup_buttons:
            button.setEnabled(idle)
        for widget in (self.add_button, self.remove_button, self.edit_button, self.output_edit,
                       self.output_button, self.format_combo, self.start_button, self.retry_button, self.restart_button):
            widget.setEnabled(idle and self.queue.writable)
        self.pause_button.setEnabled(self.running and not self.pause_pending)
        self.cancel_button.setEnabled(self.worker is not None and self._last_operation != 'status')
        self.close_button.setEnabled(idle)

    def on_setup_clicked(self) -> None:
        self.start_setup(self.sender().property('operation'))

    def start_setup(self, operation: str) -> None:
        if self.busy:
            return
        self._last_operation = operation
        self._worker_error = ''
        self._worker_result = None
        self.worker = LocalSetupThread(operation, self)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.completed.connect(self.on_setup_result)
        self.worker.failed.connect(self.on_setup_error)
        self.worker.finished.connect(self.on_setup_finished)
        self.status_label.setText('준비 상태 확인 중…' if operation == 'status' else '작업 중입니다. 큰 모델은 시간이 걸릴 수 있습니다.')
        self.progress.setRange(0, 0)
        self.set_controls()
        self.worker.start()

    def on_setup_result(self, result: object) -> None:
        self._worker_result = result

    def on_setup_error(self, message: str) -> None:
        self._worker_error = message

    def on_setup_finished(self) -> None:
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        worker.deleteLater()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if self._worker_error:
            self.status_label.setText(self._worker_error)
        elif self._last_operation == 'status':
            checks = self._worker_result or []
            self.checks.setRowCount(len(checks))
            for row, (name, ready, detail) in enumerate(checks):
                for column, value in enumerate((name, '준비됨' if ready else '설정 필요', detail)):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(str(detail))
                    self.checks.setItem(row, column, item)
            self.checks.resizeRowsToContents()
            self.checks.setFixedHeight(
                self.checks.horizontalHeader().height()
                + sum(self.checks.rowHeight(row) for row in range(self.checks.rowCount()))
                + 2 * self.checks.frameWidth()
            )
            pending = sum(not ready for _, ready, _ in checks)
            self.status_label.setText(f'남은 준비 항목 {pending}개. 준비 후 번역 실행 테스트를 진행하세요.' if pending else '설치 항목이 준비되었습니다. 번역 실행 테스트로 실제 작동을 확인하세요.')
        elif self._last_operation == 'test':
            self.status_label.setText(f'번역 실행 테스트 성공: {self._worker_result}')
        else:
            self.status_label.setText('준비 작업이 끝났습니다. 상태 확인 후 번역 실행 테스트를 진행하세요.')
        self.set_controls()

    def cancel_setup(self) -> None:
        if self.worker is not None:
            self.worker.stop_event.set()
            self.status_label.setText('현재 요청이 끝나는 대로 중지합니다. 완료될 때까지 창을 유지하세요.')
            self.cancel_button.setEnabled(False)

    def open_ollama_download(self) -> None:
        self.status_label.setText('Ollama 설치 후 앱을 닫고 launch_local.bat로 다시 실행하세요. 앱 전용 로컬 서버를 준비합니다.')
        QDesktopServices.openUrl(QUrl('https://ollama.com/download/windows'))

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '만화 이미지 폴더 선택')
        if folder:
            self.add_folders([folder])

    def add_folders(self, folders: list[str]) -> None:
        if self.busy:
            return
        try:
            self.queue.add(folders)
            self.refresh_queue()
        except (ValueError, OSError) as exc:
            self.status_label.setText(str(exc))

    def dragEnterEvent(self, event) -> None:
        if not self.busy and event.mimeData().hasUrls() and any(Path(url.toLocalFile()).is_dir() for url in event.mimeData().urls() if url.isLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self.add_folders([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def remove_selected(self) -> None:
        if self.busy:
            return
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.queue.jobs.pop(row)
        self.save_queue()
        self.refresh_queue()

    def refresh_queue(self) -> None:
        selected = self.table.currentRow()
        self.table.setRowCount(len(self.queue.jobs))
        for row, job in enumerate(self.queue.jobs):
            values = [job['folder'], f"{job['done']} / {job['total']}" if job['total'] else '—', STATES[job['state']]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(job['error'] or job['folder'])
                self.table.setItem(row, column, item)
        if 0 <= selected < len(self.queue.jobs):
            self.table.selectRow(selected)

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '번역 결과 추가 저장 위치')
        if folder:
            self.output_edit.setText(folder)
            self.save_options()

    def save_options(self) -> None:
        if self.busy:
            return
        self.queue.output = self.output_edit.text().strip()
        self.queue.format = self.format_combo.currentData()
        self.save_queue()

    def save_queue(self) -> bool:
        try:
            self.queue.save()
            return True
        except (OSError, ValueError) as exc:
            self.status_label.setText(f'작업 목록 저장 실패: {exc}')
            return False

    def start_queue(self) -> None:
        if self.busy or not self.queue.writable:
            return
        self.save_options()
        if not self.save_queue():
            return
        self.running = True
        self.pause_pending = False
        self.set_controls()
        self.start_next()

    def restart_selected(self) -> None:
        row = self.table.currentRow()
        if self.busy or not 0 <= row < len(self.queue.jobs):
            return
        reply = QMessageBox.question(
            self, '선택 폴더 새로 번역',
            '선택한 폴더의 대사 인식·번역·복원을 처음부터 다시 실행합니다. '
            '기존 번역과 수동 대사 수정은 새 결과로 바뀝니다. 원본 이미지는 유지됩니다. 계속하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.queue.jobs[row].update(state='pending', error='', restart=True)
            if self.save_queue():
                self.refresh_queue()
                self.status_label.setText('새로 번역하도록 대기 목록에 넣었습니다. 번역 시작을 누르면 실행합니다.')

    def retry_failed(self) -> None:
        if self.busy:
            return
        for job in self.queue.jobs:
            if job['state'] == 'failed':
                job['state'] = 'pending'
                job['error'] = ''
        if self.save_queue():
            self.start_queue()

    def start_next(self) -> None:
        if self.pause_pending:
            self.running = False
            self.set_controls()
            return
        self.active_index = next((index for index, job in enumerate(self.queue.jobs) if job['state'] in {'pending', 'paused'}), None)
        if self.active_index is None:
            self.running = False
            self.status_label.setText('대기 작업이 없습니다. 확인 필요 항목은 오류를 확인한 뒤 재시도하세요.')
            self.set_controls()
            return
        job = self.queue.jobs[self.active_index]
        job['state'] = 'running'
        job['error'] = ''
        if not self.save_queue():
            job['state'] = 'paused'
            self.running = False
            self.set_controls()
            return
        self.refresh_queue()
        self.table.selectRow(self.active_index)
        self.status_label.setText(f"번역 준비 중: {job['folder']}")
        self.run_requested.emit(job['folder'])

    def update_progress(self, done: int, total: int, detail: str) -> None:
        if self.active_index is None:
            return
        job = self.queue.jobs[self.active_index]
        changed = job['done'] != done or job['total'] != total
        job.update(done=done, total=total)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(detail)
        if changed:
            self.table.item(self.active_index, 1).setText(f'{done} / {total}')

    def finish_job(self, success: bool, error: str, done: int, total: int) -> None:
        if self.active_index is None:
            return
        # Completed inference bits do not imply completed output writes. A
        # pause may have cancelled pending rerenders of existing stale files.
        success = success and not self.pause_pending
        job = self.queue.jobs[self.active_index]
        review_pages = job.get('review_pages', [])
        if review_pages:
            success = False
            error = error or f"말풍선 배치 확인 필요: {', '.join(review_pages)}. 대사를 수정한 뒤 재시도하세요."
        state = 'done' if success else 'paused' if self.pause_pending and not review_pages else 'failed'
        job.update(state=state, error=error, done=done, total=total, resave=not success)
        self.active_index = None
        if not self.save_queue():
            self.pause_pending = True
        self.refresh_queue()
        self.status_label.setText(error or ('현재 작업을 안전하게 저장하고 일시정지했습니다.' if self.pause_pending else '폴더 번역이 완료되었습니다.'))
        self.start_next()
        if self.active_index is None and error:
            self.status_label.setText(error)

    def request_pause(self) -> None:
        if self.running:
            self.pause_pending = True
            self.status_label.setText('현재 단계가 종료되고 결과가 저장되면 일시정지합니다.')
            self.set_controls()
            self.pause_requested.emit()

    def edit_selected(self) -> None:
        row = self.table.currentRow()
        if not self.busy and 0 <= row < len(self.queue.jobs):
            self.edit_requested.emit(self.queue.jobs[row]['folder'])
            self.hide()

    def open_result(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.queue.jobs):
            path = self.queue.output_directory(self.queue.jobs[row]['folder'])
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                self.status_label.setText('아직 저장된 결과 폴더가 없습니다.')

    def reject(self) -> None:
        if self.busy:
            self.status_label.setText('진행 중인 작업을 먼저 중지하세요. 현재 단계와 저장이 끝나면 닫을 수 있습니다.')
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self.busy:
            event.ignore()
            self.status_label.setText('진행 중인 작업을 먼저 중지하세요. 현재 단계와 저장이 끝나면 닫을 수 있습니다.')
        else:
            super().closeEvent(event)
