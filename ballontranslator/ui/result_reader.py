"""A read-only, virtualized reader of saved result images. No model dependencies."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import math
import threading
from typing import Optional

from qtpy.QtCore import Qt, QSize, QRect, QThread, QTimer, Signal
from qtpy.QtGui import QImage, QImageReader, QPainter, QPalette, QKeySequence
from qtpy.QtWidgets import (
    QApplication, QAbstractScrollArea, QComboBox, QDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)
from qtpy.QtWidgets import QShortcut

from ballontranslator.utils.reader_export import Book, Page, ReaderHistory, signature


def plain_label(text: str = '', parent=None) -> QLabel:
    label = QLabel(text, parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def action_button(text: str, slot, parent=None) -> QPushButton:
    button = QPushButton(text, parent)
    button.setMinimumHeight(40)
    button.clicked.connect(slot)
    return button


def size_dialog(dialog: QDialog, width: int, height: int) -> None:
    screen = QApplication.primaryScreen()
    available = screen.availableGeometry() if screen else QRect(0, 0, width + 32, height + 32)
    dialog.resize(min(width, available.width() - 32), min(height, available.height() - 48))
    dialog.setMinimumSize(min(720, available.width() - 32), min(500, available.height() - 48))


class PageLoader(QThread):
    decoded = Signal(int, int, object, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending = []
        self._stopping = False
        self._serial = 0

    def request(self, tasks: list) -> None:
        with self._condition:
            self._serial += 1
            self._pending = list(tasks)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify()

    def run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                row, epoch, page, target = self._pending.pop(0)
                serial = self._serial
            image, error = QImage(), ''
            try:
                if page.path is None or signature(page.path) != page.stamp:
                    raise ValueError('파일이 변경되었거나 없습니다. 뷰어를 다시 열어 주세요.')
                reader = QImageReader(str(page.path))
                reader.setAutoTransform(True)
                raw_size = reader.size()
                if (raw_size.height(), raw_size.width()) == (page.width, page.height) and page.width != page.height:
                    target = (target[1], target[0])
                reader.setScaledSize(QSize(*target))
                image = reader.read()
                if image.isNull():
                    raise ValueError(reader.errorString())
                if signature(page.path) != page.stamp:
                    raise ValueError('읽는 동안 파일이 변경되었습니다.')
            except Exception as exc:
                image, error = QImage(), str(exc)
            with self._condition:
                if self._stopping:
                    return
                current = serial == self._serial
            if current:
                self.decoded.emit(row, epoch, image, error)


class ReaderCanvas(QAbstractScrollArea):
    current_changed = Signal(int)
    CACHE_BYTES = 64 * 1024 * 1024

    def __init__(self, book: Book, parent=None) -> None:
        super().__init__(parent)
        self.book = book
        self.mode, self.fit, self.zoom = 'scroll', 'width', 100
        self.current_row = 0
        self.cache = OrderedDict()
        self.errors = {}
        self._epoch = 0
        self._rows, self._tops, self._sizes = [], [], []
        self._layout_busy = False
        self._closing = False
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.loader = PageLoader(self)
        self.loader.decoded.connect(self._decoded)
        self.loader.start()
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(35)
        self._load_timer.timeout.connect(self._queue_visible)
        self.reflow()

    def anchor(self) -> tuple[int, float]:
        if not self._rows:
            return self.current_row, 0.0
        y = self.verticalScrollBar().value()
        i = max(0, min(len(self._tops) - 1, bisect_right(self._tops, y + 1) - 1))
        return self._rows[i], min(1.0, max(0.0, (y - self._tops[i]) / max(1, self._sizes[i][1])))

    def configure(self, mode: str, fit: str, zoom: int) -> None:
        anchor = self.anchor()
        self.mode, self.fit, self.zoom = mode, fit, zoom
        self.current_row = anchor[0]
        self.reflow(anchor)

    def reflow(self, anchor: Optional[tuple[int, float]] = None) -> None:
        if self._layout_busy or self._closing:
            return
        anchor = anchor or self.anchor()
        self._layout_busy = True
        try:
            self._epoch += 1
            self.cache.clear()
            self.errors.clear()
            self._rows = list(range(len(self.book.pages))) if self.mode == 'scroll' else [self.current_row]
            self._tops, self._sizes = [], []
            y, maximum_width = 16, 0
            vw, vh = max(1, self.viewport().width() - 32), max(1, self.viewport().height() - 32)
            for row in self._rows:
                page = self.book.pages[row]
                ratio = vw / page.width
                if self.fit == 'page':
                    ratio = min(ratio, vh / page.height)
                ratio = min(ratio * self.zoom / 100, 60000 / max(page.width, page.height))
                w, h = max(1, round(page.width * ratio)), max(1, round(page.height * ratio))
                self._tops.append(y)
                self._sizes.append((w, h))
                maximum_width = max(maximum_width, w)
                y += h + 16
            self._content_width = max(self.viewport().width(), maximum_width + 32)
            self.verticalScrollBar().setPageStep(self.viewport().height())
            self.verticalScrollBar().setRange(0, max(0, y - self.viewport().height(),
                                                     self._tops[-1] if self.mode == 'scroll' else 0))
            self.horizontalScrollBar().setRange(0, max(0, self._content_width - self.viewport().width()))
            self.horizontalScrollBar().setPageStep(self.viewport().width())
            if anchor[0] in self._rows:
                i = self._rows.index(anchor[0])
                self.verticalScrollBar().setValue(round(self._tops[i] + anchor[1] * self._sizes[i][1]))
        finally:
            self._layout_busy = False
        self._load_timer.start()
        self.viewport().update()

    def go_to(self, row: int, offset: float = 0.0) -> None:
        row = max(0, min(len(self.book.pages) - 1, row))
        self.current_row = row
        if self.mode == 'page' and self._rows != [row]:
            self.reflow((row, offset))
        elif row in self._rows:
            i = self._rows.index(row)
            self.verticalScrollBar().setValue(round(self._tops[i] + offset * self._sizes[i][1]))
        self.current_changed.emit(row)
        self._load_timer.start()
        self.viewport().update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_load_timer'):
            self.reflow()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        self.viewport().update()
        if not self._layout_busy and not self._closing:
            row, _ = self.anchor()
            if row != self.current_row:
                self.current_row = row
                self.current_changed.emit(row)
            self._load_timer.start()

    def _visible(self, padding: int = 0) -> list[int]:
        if not self._rows:
            return []
        y = self.verticalScrollBar().value()
        begin = max(0, bisect_right(self._tops, y) - 1)
        end = min(len(self._rows), bisect_right(self._tops, y + self.viewport().height()) + 1)
        return list(range(max(0, begin - padding), min(len(self._rows), end + padding)))

    def _queue_visible(self) -> None:
        if self._closing:
            return
        visible = self._visible()
        indices = list(dict.fromkeys(visible + self._visible(1)))
        wanted = {self._rows[i] for i in indices}
        for row in list(self.cache):
            if row not in wanted:
                self.cache.pop(row)
        tasks = []
        for i in indices:
            row = self._rows[i]
            page = self.book.pages[row]
            if page.issue or row in self.cache or row in self.errors:
                continue
            w, h = self._sizes[i]
            ratio = min(1.0, w * self.devicePixelRatioF() / page.width,
                        2048 / page.width, 16384 / max(page.width, page.height),
                        math.sqrt(6_000_000 / (page.width * page.height)))
            target = (max(1, round(page.width * ratio)), max(1, round(page.height * ratio)))
            tasks.append((row, self._epoch, page, target))
        self.loader.request(tasks)

    def _decoded(self, row: int, epoch: int, image: QImage, error: str) -> None:
        if self._closing or epoch != self._epoch:
            return
        if error:
            self.errors[row] = error
        else:
            self.cache[row] = image
            while len(self.cache) > 1 and sum(im.sizeInBytes() for im in self.cache.values()) > self.CACHE_BYTES:
                self.cache.popitem(last=False)
        self.viewport().update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), self.palette().brush(QPalette.ColorRole.Dark))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        y_offset, x_offset = self.verticalScrollBar().value(), self.horizontalScrollBar().value()
        for i in self._visible():
            row = self._rows[i]
            page = self.book.pages[row]
            w, h = self._sizes[i]
            rect = QRect((self._content_width - w) // 2 - x_offset, self._tops[i] - y_offset, w, h)
            painter.fillRect(rect, Qt.GlobalColor.white)
            if row in self.cache:
                painter.drawImage(rect, self.cache[row])
            else:
                painter.setPen(Qt.GlobalColor.black)
                message = page.issue or self.errors.get(row) or '이미지 불러오는 중…'
                visible_rect = rect.intersected(self.viewport().rect()).adjusted(12, 12, -12, -12)
                painter.drawText(visible_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                                 f'{row + 1} / {len(self.book.pages)}\n{page.name}\n{message}')
        painter.end()

    def stop(self) -> None:
        self._closing = True
        self._load_timer.stop()
        self.loader.stop()


class ReaderDialog(QDialog):
    def __init__(self, book: Book, history: ReaderHistory, parent=None) -> None:
        super().__init__(parent)
        self.book, self.history = book, history
        self.current_row = 0
        self.edit_requested_page = None
        self.history_save_error = ''
        self._closing = False
        self._child_dialog = None
        self.setWindowTitle(f'결과 뷰어 · {book.folder.name}')
        self.setObjectName('LocalResultReader')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        size_dialog(self, 1160, 820)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)
        heading = QHBoxLayout()
        title = plain_label(f'{book.folder.name} · {len(book.pages)}페이지', self)
        font = title.font(); font.setPointSize(font.pointSize() + 3); font.setBold(True); title.setFont(font)
        heading.addWidget(title, 1)
        heading.addWidget(action_button('이미지 합치기', self.open_merge, self))
        heading.addWidget(action_button('전체 화면', self.toggle_fullscreen, self))
        root.addLayout(heading)
        self.notice = plain_label('', self)
        notes = list(book.warnings)
        count = sum(bool(p.issue or p.warning) for p in book.pages)
        if count:
            notes.append(f'확인이 필요한 페이지 {count}장. 목록의 표시를 확인하세요.')
        if history.warning:
            notes.append(history.warning)
        self.notice.setText('\n'.join(notes) or '저장된 결과를 읽습니다. 편집 내용은 저장 후 뷰어를 다시 열면 반영됩니다.')
        root.addWidget(self.notice)
        toolbar = QHBoxLayout()
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem('세로 이어 읽기', 'scroll'); self.mode_combo.addItem('한 장씩 보기', 'page')
        self.fit_combo = QComboBox(self)
        self.fit_combo.addItem('화면 폭 맞춤', 'width'); self.fit_combo.addItem('페이지 전체 맞춤', 'page')
        self.zoom_spin = QSpinBox(self); self.zoom_spin.setRange(25, 300); self.zoom_spin.setSingleStep(25)
        self.zoom_spin.setSuffix(' %'); self.zoom_spin.setValue(100); self.zoom_spin.setAccessibleName('화면 맞춤 기준 확대 비율')
        for widget in (self.mode_combo, self.fit_combo, self.zoom_spin):
            widget.setMinimumHeight(36); toolbar.addWidget(widget)
        toolbar.addStretch()
        self.page_spin = QSpinBox(self); self.page_spin.setRange(1, len(book.pages)); self.page_spin.setSuffix(' 페이지')
        self.page_spin.setAccessibleName('이동할 페이지'); self.page_spin.setMinimumHeight(36)
        toolbar.addWidget(self.page_spin)
        root.addLayout(toolbar)
        split = QSplitter(Qt.Orientation.Horizontal, self)
        self.page_list = QListWidget(split); self.page_list.setMinimumWidth(160)
        for i, page in enumerate(book.pages):
            tag = ' · 결과 없음/오류' if page.issue else ' · 확인 필요' if page.warning else ''
            item = QListWidgetItem(f'{i + 1:03d}  {page.name}{tag}')
            item.setToolTip(page.issue or page.warning or page.name)
            self.page_list.addItem(item)
        self.canvas = ReaderCanvas(book, split)
        split.addWidget(self.page_list); split.addWidget(self.canvas)
        split.setSizes([225, 850]); split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        footer = QHBoxLayout()
        self.previous_button = action_button('이전 페이지', self.previous_page, self)
        self.next_button = action_button('다음 페이지', self.next_page, self)
        footer.addWidget(self.previous_button); footer.addWidget(self.next_button)
        self.position_label = plain_label('', self); footer.addWidget(self.position_label, 1)
        self.edit_button = action_button('이 페이지 편집', self.edit_page, self)
        footer.addWidget(self.edit_button); footer.addWidget(action_button('닫기', self.reject, self))
        root.addLayout(footer)
        root.addWidget(plain_label('PgUp / PgDn 페이지 이동 · F11 전체 화면 · Esc 전체 화면 해제 / 닫기', self))
        self.mode_combo.currentIndexChanged.connect(self.configure_view)
        self.fit_combo.currentIndexChanged.connect(self.configure_view)
        self.zoom_spin.valueChanged.connect(self.configure_view)
        self.page_list.currentRowChanged.connect(self.jump_to)
        self.page_spin.valueChanged.connect(self.page_number_changed)
        self.canvas.current_changed.connect(self._current_changed)
        self.canvas.loader.finished.connect(self._finish_close)
        self._shortcuts = []
        for key, slot in (('PgUp', self.previous_page), ('PgDown', self.next_page), ('F11', self.toggle_fullscreen)):
            shortcut = QShortcut(QKeySequence(key), self); shortcut.activated.connect(slot); self._shortcuts.append(shortcut)
        self._restore_timer = QTimer(self); self._restore_timer.setSingleShot(True)
        self._restore_timer.timeout.connect(self.restore_position); self._restore_timer.start(0)
        self._current_changed(0)

    def restore_position(self) -> None:
        if self._closing:
            return
        record = self.history.get(str(self.book.folder))
        if record:
            self.mode_combo.setCurrentIndex(1 if record['mode'] == 'page' else 0)
            self.fit_combo.setCurrentIndex(1 if record['fit'] == 'page' else 0)
            self.zoom_spin.setValue(record['zoom'])
            row = next((i for i, p in enumerate(self.book.pages) if p.name == record['page']), 0)
            self.canvas.go_to(row, record['offset'])

    def configure_view(self) -> None:
        self.canvas.configure(self.mode_combo.currentData(), self.fit_combo.currentData(), self.zoom_spin.value())

    def page_number_changed(self, value: int) -> None:
        self.jump_to(value - 1)

    def jump_to(self, row: int) -> None:
        if row >= 0 and row != self.canvas.current_row:
            self.canvas.go_to(row)

    def _current_changed(self, row: int) -> None:
        self.current_row = row
        # Guard the two selectors so scrolling does not reset a page's reading offset.
        self.page_list.blockSignals(True); self.page_spin.blockSignals(True)
        self.page_list.setCurrentRow(row); self.page_spin.setValue(row + 1)
        self.page_list.blockSignals(False); self.page_spin.blockSignals(False)
        self.previous_button.setEnabled(row > 0); self.next_button.setEnabled(row < len(self.book.pages) - 1)
        self.position_label.setText(f'{row + 1} / {len(self.book.pages)}')
        self.edit_button.setEnabled((self.book.folder / self.book.pages[row].name).is_file())

    def next_page(self) -> None:
        self.canvas.go_to(self.current_row + 1)

    def previous_page(self) -> None:
        self.canvas.go_to(self.current_row - 1)

    def edit_page(self) -> None:
        self.edit_requested_page = self.book.pages[self.current_row].name
        self.reject()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def open_merge(self) -> None:
        from .image_merge_dialog import ImageMergeDialog
        dialog = ImageMergeDialog(self.book, self)
        self._child_dialog = dialog
        try:
            dialog.exec_()
        finally:
            self._child_dialog = None
            dialog.deleteLater()

    def reject(self) -> None:
        if self._child_dialog is not None:
            return
        if not self._closing:
            self._closing = True
            self._restore_timer.stop()
            row, offset = self.canvas.anchor()
            try:
                self.history.save_position(str(self.book.folder), self.book.pages[row].name, offset,
                                           self.canvas.mode, self.canvas.zoom, self.canvas.fit)
            except (OSError, ValueError) as exc:
                self.history_save_error = f'이어 읽기 위치를 저장하지 못했습니다: {exc}'
            self.canvas.stop()
            self.position_label.setText('이미지 읽기 종료 중…')
        if not self.canvas.loader.isRunning():
            self._finish_close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal(); event.accept(); return
        super().keyPressEvent(event)

    def _finish_close(self) -> None:
        if self._closing:
            self.canvas.loader.wait()
            super().reject()

    def closeEvent(self, event) -> None:
        if not self._closing or self.canvas.loader.isRunning():
            event.ignore(); self.reject()
        else:
            super().closeEvent(event)
