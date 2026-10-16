"""Ordered image export UI with bounded previews and cancellable background writes."""
from __future__ import annotations

from pathlib import Path
import threading

from qtpy.QtCore import Qt, QThread, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QProgressBar, QSpinBox, QSplitter, QVBoxLayout, QWidget, QSizePolicy,
)

from ballontranslator.utils.reader_export import (
    Book, MergeOptions, OperationCancelled, export_plan, plan_export, render_group,
)
from .result_reader import action_button, plain_label, size_dialog


class RenderTask(QThread):
    progress = Signal(int, int)

    def __init__(self, *, group=None, plan=None, destination=None, parent=None) -> None:
        super().__init__(parent)
        self.group, self.plan, self.destination = group, plan, destination
        self.stop_event = threading.Event()
        self.result = None
        self.error = ''

    def run(self) -> None:
        try:
            if self.plan is not None:
                self.result = export_plan(self.plan, self.destination, self.stop_event, self.progress.emit)
            else:
                with render_group(self.group, preview_size=(720, 480), stop=self.stop_event) as image:
                    data = image.tobytes()
                    self.result = QImage(data, image.width, image.height, image.width * 3,
                                         QImage.Format.Format_RGB888).copy()
        except Exception as exc:
            self.error = str(exc)


class ImageMergeDialog(QDialog):
    def __init__(self, book: Book, parent=None) -> None:
        super().__init__(parent)
        self.book = book
        self.plan = None
        self.last_output = None
        self.preview_ready = False
        self._preview_worker = None
        self._export_worker = None
        self._preview_generation = 0
        self._pending_preview = False
        self._closing = False
        self._updating = True
        self.setWindowTitle(f'이미지 합치기 · {book.folder.name}')
        self.setObjectName('LocalImageMerge')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        size_dialog(self, 1100, 840)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16); root.setSpacing(10)
        title = plain_label('이미지 합치기 · 폰용 저장', self)
        font = title.font(); font.setPointSize(font.pointSize() + 3); font.setBold(True); title.setFont(font)
        root.addWidget(title)
        root.addWidget(plain_label('페이지 선택·순서 → 방향·크기 → 미리보기 → 별도 저장. 원본과 기존 결과는 변경하지 않습니다.', self))
        split = QSplitter(Qt.Orientation.Horizontal, self)
        left = QWidget(split); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(plain_label('1  포함 페이지 · 위에서 아래 순서', left))
        self.page_list = QListWidget(left)
        self.page_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.page_list.setMinimumWidth(180)
        for i, page in enumerate(book.pages):
            flag = ' · 오류' if page.issue else ' · 확인 필요' if page.warning else ''
            item = QListWidgetItem(f'{i + 1:03d}  {page.name}{flag}')
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(page.issue or page.warning or page.name)
            self.page_list.addItem(item)
        left_layout.addWidget(self.page_list, 1)
        selection = QHBoxLayout()
        self.all_button = action_button('모두 선택', self.select_all, left)
        self.none_button = action_button('선택 해제', self.select_none, left)
        selection.addWidget(self.all_button); selection.addWidget(self.none_button); left_layout.addLayout(selection)
        order = QHBoxLayout()
        self.up_button = action_button('위로', self.move_up, left)
        self.down_button = action_button('아래로', self.move_down, left)
        self.reverse_button = action_button('순서 반대로', self.reverse_order, left)
        for button in (self.up_button, self.down_button, self.reverse_button):
            order.addWidget(button)
        left_layout.addLayout(order)
        right = QWidget(split); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0, 0, 0, 0)
        options = QGroupBox('2  방향·크기', right); form = QFormLayout(options)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.direction_combo = QComboBox(options)
        self.direction_combo.addItem('세로 · 위에서 아래로', 'vertical')
        self.direction_combo.addItem('가로 · 왼쪽에서 오른쪽으로', 'horizontal')
        form.addRow('합치기 방향', self.direction_combo)
        size_row = QHBoxLayout()
        self.cross_spin = QSpinBox(options); self.cross_spin.setRange(120, 8000); self.cross_spin.setValue(1080); self.cross_spin.setSuffix(' px')
        self.cross_label = plain_label('기준 폭', options)
        size_row.addWidget(self.cross_spin); size_row.addWidget(plain_label('작은 원본은 확대하지 않음', options), 1)
        form.addRow(self.cross_label, size_row)
        self.gap_spin = QSpinBox(options); self.gap_spin.setRange(0, 200); self.gap_spin.setValue(0); self.gap_spin.setSuffix(' px')
        form.addRow('페이지 사이 간격', self.gap_spin)
        self.split_check = QCheckBox('길어지면 여러 파일로 나누기', options); self.split_check.setChecked(True)
        form.addRow(self.split_check)
        self.length_spin = QSpinBox(options); self.length_spin.setRange(500, 60000); self.length_spin.setValue(12000); self.length_spin.setSuffix(' px')
        self.length_label = plain_label('파일당 세로 길이', options); form.addRow(self.length_label, self.length_spin)
        format_row = QHBoxLayout()
        self.format_combo = QComboBox(options); self.format_combo.addItems(['PNG', 'JPEG'])
        self.quality_spin = QSpinBox(options); self.quality_spin.setRange(50, 100); self.quality_spin.setValue(92); self.quality_spin.setPrefix('품질 ')
        format_row.addWidget(self.format_combo); format_row.addWidget(self.quality_spin); form.addRow('이미지 형식', format_row)
        right_layout.addWidget(options)
        right_layout.addWidget(plain_label('3  파일별 미리보기 · 페이지 중간은 자르지 않습니다.', right))
        self.file_combo = QComboBox(right); right_layout.addWidget(self.file_combo)
        self.preview_label = plain_label('페이지를 선택하세요.', right)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setMinimumHeight(130)
        right_layout.addWidget(self.preview_label, 1)
        split.addWidget(left); split.addWidget(right); split.setSizes([290, 720]); split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        self.summary_label = plain_label('', self); root.addWidget(self.summary_label)
        self.warning_label = plain_label('', self); root.addWidget(self.warning_label)
        self.acknowledge = QCheckBox('주의 사항을 확인했습니다. 현재 저장된 결과 파일로 내보냅니다.', self)
        root.addWidget(self.acknowledge)
        output = QHBoxLayout(); output.addWidget(plain_label('저장 위치', self))
        self.output_edit = QLineEdit(str(book.folder / 'exports'), self); output.addWidget(self.output_edit, 1)
        self.output_button = action_button('폴더 선택', self.choose_output, self); output.addWidget(self.output_button)
        root.addLayout(output)
        self.progress = QProgressBar(self); self.progress.setVisible(False); root.addWidget(self.progress)
        self.status_label = plain_label('분할 기준은 앱 보호용입니다. 휴대폰에서의 표시 크기는 사용 앱에 따라 다릅니다.', self)
        root.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.open_button = action_button('저장 폴더 열기', self.open_output, self); self.open_button.setEnabled(False)
        self.cancel_button = action_button('저장 취소', self.cancel_export, self); self.cancel_button.setEnabled(False)
        self.export_button = action_button('별도 폴더로 저장', self.start_export, self)
        self.close_button = action_button('닫기', self.reject, self)
        actions.addWidget(self.open_button); actions.addStretch(); actions.addWidget(self.cancel_button)
        actions.addWidget(self.export_button); actions.addWidget(self.close_button); root.addLayout(actions)
        self._controls = [self.page_list, self.all_button, self.none_button, self.up_button, self.down_button,
                          self.reverse_button, self.direction_combo, self.cross_spin, self.gap_spin,
                          self.split_check, self.length_spin, self.format_combo, self.quality_spin,
                          self.acknowledge, self.output_edit, self.output_button, self.file_combo]
        for control in (self.direction_combo, self.cross_spin, self.gap_spin, self.length_spin,
                        self.format_combo, self.quality_spin, self.file_combo):
            control.setMinimumHeight(32)
        self._preview_delay = QTimer(self); self._preview_delay.setSingleShot(True); self._preview_delay.setInterval(120)
        self._preview_delay.timeout.connect(self._start_preview)
        self.page_list.itemChanged.connect(self.refresh_plan)
        self.direction_combo.currentIndexChanged.connect(self.direction_changed)
        for widget in (self.cross_spin, self.gap_spin, self.length_spin, self.quality_spin):
            widget.valueChanged.connect(self.refresh_plan)
        self.split_check.toggled.connect(self.refresh_plan)
        self.format_combo.currentIndexChanged.connect(self.refresh_plan)
        self.file_combo.currentIndexChanged.connect(self.schedule_preview)
        self.acknowledge.toggled.connect(self.sync_controls)
        self.output_edit.textChanged.connect(self.sync_controls)
        self._updating = False
        self.refresh_plan()

    @property
    def busy(self) -> bool:
        return self._preview_worker is not None or self._export_worker is not None

    def selected_pages(self) -> tuple:
        return tuple(self.book.pages[self.page_list.item(i).data(Qt.ItemDataRole.UserRole)]
                     for i in range(self.page_list.count())
                     if self.page_list.item(i).checkState() == Qt.CheckState.Checked)

    def _select(self, checked: bool) -> None:
        self._updating = True
        for i in range(self.page_list.count()):
            self.page_list.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._updating = False; self.refresh_plan()

    def select_all(self) -> None:
        self._select(True)

    def select_none(self) -> None:
        self._select(False)

    def _move(self, delta: int) -> None:
        row = self.page_list.currentRow()
        target = row + delta
        if 0 <= row < self.page_list.count() and 0 <= target < self.page_list.count():
            item = self.page_list.takeItem(row); self.page_list.insertItem(target, item)
            self.page_list.setCurrentRow(target); self.refresh_plan()

    def move_up(self) -> None:
        self._move(-1)

    def move_down(self) -> None:
        self._move(1)

    def reverse_order(self) -> None:
        items = [self.page_list.takeItem(0) for _ in range(self.page_list.count())]
        for item in reversed(items):
            self.page_list.addItem(item)
        self.refresh_plan()

    def direction_changed(self) -> None:
        self._updating = True
        vertical = self.direction_combo.currentData() == 'vertical'
        self.cross_label.setText('기준 폭' if vertical else '기준 높이')
        self.length_label.setText('파일당 세로 길이' if vertical else '파일당 가로 길이')
        self.split_check.setChecked(vertical)
        self._updating = False; self.refresh_plan()

    def refresh_plan(self) -> None:
        if self._updating or self._export_worker is not None or self._closing:
            return
        self._updating = True
        self._preview_generation += 1
        self.preview_ready = False
        self.plan = None
        index = max(0, self.file_combo.currentIndex())
        self.file_combo.clear()
        pages = self.selected_pages()
        notices = list(self.book.warnings)
        warned = sum(bool(p.warning) for p in pages)
        if warned:
            notices.append(f'선택한 페이지 중 {warned}장은 검수 필요 상태입니다. 이전 결과일 수 있습니다.')
        self.warning_label.setText('\n'.join(notices))
        self.warning_label.setVisible(bool(notices)); self.acknowledge.setVisible(bool(notices))
        self._needs_ack = bool(notices)
        try:
            opts = MergeOptions(self.direction_combo.currentData(), self.cross_spin.value(), self.gap_spin.value(),
                                self.split_check.isChecked(), self.length_spin.value(), self.format_combo.currentText(),
                                self.quality_spin.value())
            self.plan = plan_export(pages, opts)
            for i, group in enumerate(self.plan.groups):
                self.file_combo.addItem(f'{i + 1:03d} · {len(group.items)}페이지 · {group.size[0]:,} × {group.size[1]:,} px')
            self.file_combo.setCurrentIndex(min(index, len(self.plan.groups) - 1))
            self.summary_label.setText(f'선택 {len(pages)}페이지 → {len(self.plan.groups)}파일 · 실제 기준 크기 {self.plan.cross_size:,}px · 원본 비율 유지')
        except (ValueError, OSError) as exc:
            self.summary_label.setText(str(exc))
            self.preview_label.clear(); self.preview_label.setText('설정을 확인하면 미리보기가 표시됩니다.')
        finally:
            self._updating = False
        self.schedule_preview()
        self.sync_controls()

    def schedule_preview(self) -> None:
        if self._updating or self._closing:
            return
        self._preview_generation += 1
        self.preview_ready = False
        self._pending_preview = self.plan is not None
        if self._preview_worker is not None:
            self._preview_worker.stop_event.set()
        if self._pending_preview:
            self.preview_label.clear(); self.preview_label.setText('미리보기 생성 중…')
            self._preview_delay.start()

    def _start_preview(self) -> None:
        if self._closing or self._export_worker is not None or not self._pending_preview or self.plan is None:
            return
        if self._preview_worker is not None:
            return
        index = self.file_combo.currentIndex()
        if not 0 <= index < len(self.plan.groups):
            return
        self._pending_preview = False
        worker = RenderTask(group=self.plan.groups[index], parent=self)
        worker.generation = self._preview_generation
        self._preview_worker = worker
        worker.finished.connect(self._preview_finished)
        worker.start()

    def _preview_finished(self) -> None:
        worker = self._preview_worker
        worker.wait()
        self._preview_worker = None
        if not self._closing and worker.generation == self._preview_generation:
            if worker.error:
                self.preview_label.setText(worker.error)
            elif worker.result is not None:
                self._preview_image = worker.result
                self._show_preview()
                self.preview_ready = True
        worker.deleteLater()
        if self._closing:
            self._finish_close()
        elif self._pending_preview:
            self._preview_delay.start()

    def _show_preview(self) -> None:
        if getattr(self, '_preview_image', None) is not None:
            pixmap = QPixmap.fromImage(self._preview_image)
            self.preview_label.setPixmap(pixmap.scaled(max(1, self.preview_label.width() - 8),
                                                      max(1, self.preview_label.height() - 8),
                                                      Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, 'preview_label') and self.preview_ready:
            self._show_preview()

    def sync_controls(self) -> None:
        exporting = self._export_worker is not None
        for widget in self._controls:
            widget.setEnabled(not exporting and not self._closing)
        self.length_spin.setEnabled(not exporting and self.split_check.isChecked())
        self.quality_spin.setEnabled(not exporting and self.format_combo.currentText() == 'JPEG')
        allowed = not self._needs_ack or self.acknowledge.isChecked()
        self.export_button.setEnabled(not exporting and not self._closing and self.plan is not None and allowed and bool(self.output_edit.text().strip()))
        self.cancel_button.setEnabled(exporting and not self._closing)
        self.open_button.setEnabled(self.last_output is not None and not exporting)

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, '내보내기 저장 위치', self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def start_export(self) -> None:
        if self._export_worker is not None or self.plan is None or self._closing:
            return
        if self._needs_ack and not self.acknowledge.isChecked():
            self.status_label.setText('주의 사항을 확인한 뒤 저장된 결과 사용에 체크하세요.'); return
        destination = self.output_edit.text().strip()
        if not destination:
            self.status_label.setText('저장 위치를 선택하세요.'); return
        self._preview_delay.stop(); self._pending_preview = False
        if self._preview_worker is not None:
            self._preview_worker.stop_event.set()
        self.last_output = None
        worker = RenderTask(plan=self.plan, destination=Path(destination), parent=self)
        self._export_worker = worker
        worker.progress.connect(self._export_progress)
        worker.finished.connect(self._export_finished)
        self.progress.setRange(0, len(self.plan.groups)); self.progress.setValue(0); self.progress.setVisible(True)
        self.status_label.setText('별도 폴더에 저장 중입니다. 취소하면 이번 작업의 임시 결과만 제거합니다.')
        self.sync_controls(); worker.start()

    def _export_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total); self.progress.setValue(done)
        self.status_label.setText(f'{done} / {total} 파일 처리 완료 · 전체 저장 확인 중')

    def _export_finished(self) -> None:
        worker = self._export_worker
        worker.wait()
        self._export_worker = None
        if worker.error:
            self.status_label.setText(worker.error)
        else:
            self.last_output = worker.result
            self.status_label.setText(f'저장 완료 · {self.last_output}')
        worker.deleteLater()
        self.sync_controls()
        if self._closing:
            self._finish_close()
        elif not self.preview_ready:
            self.schedule_preview()

    def cancel_export(self) -> None:
        if self._export_worker is not None:
            self._export_worker.stop_event.set()
            self.cancel_button.setEnabled(False)
            self.status_label.setText('현재 이미지 처리 후 취소합니다. 기존 파일은 유지됩니다.')

    def open_output(self) -> None:
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def reject(self) -> None:
        self._closing = True
        self._preview_delay.stop(); self._pending_preview = False
        for worker in (self._preview_worker, self._export_worker):
            if worker is not None:
                worker.stop_event.set()
        self.status_label.setText('실행 중인 이미지 작업을 안전하게 종료하고 있습니다…')
        self.sync_controls()
        self._finish_close()

    def _finish_close(self) -> None:
        if self._closing and not self.busy:
            super().reject()

    def closeEvent(self, event) -> None:
        if not self._closing or self.busy:
            event.ignore(); self.reject()
        else:
            super().closeEvent(event)
