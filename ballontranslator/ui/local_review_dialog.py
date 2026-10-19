"""A reusable, nonmodal list of local translation issues requiring review."""

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ballontranslator.utils import shared
from ballontranslator.utils.local_review import ReviewItem
from .framelesswindow import DialogCloseButton, OutsideClickFramelessMixin


class LocalReviewDialog(OutsideClickFramelessMixin, QDialog):
    """Show evidenced issues and request explicit editor-owned actions.

    Signals use zero-based bubble indices; ``-1`` means the whole page.
    This cached dialog never mutates the project or blocks canvas interaction.

    >>> LocalReviewDialog.__name__
    'LocalReviewDialog'
    """

    open_requested = Signal(str, int)
    acknowledge_requested = Signal(object)
    language_requested = Signal(str, int, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName('LocalReviewDialog')
        self.setWindowTitle(self.tr('확인이 필요한 말풍선'))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(620, 380)
        self.resize(760, 440)
        self._items: list[ReviewItem] = []
        self._surface_style = ''

        outer = QVBoxLayout(self)
        outer.setContentsMargins(5, 5, 5, 5)
        self.surface = QFrame(self)
        self.surface.setObjectName('LocalReviewSurface')
        outer.addWidget(self.surface)
        layout = QVBoxLayout(self.surface)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(10)

        self.title_bar = QWidget(self.surface)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(QLabel(self.windowTitle(), self.title_bar))
        title_layout.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.hide)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.status_label = QLabel(self.surface)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.table = QTableWidget(0, 3, self.surface)
        self.table.setHorizontalHeaderLabels([self.tr('페이지'), self.tr('말풍선'), self.tr('확인 이유')])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._sync_selection)
        self.table.itemDoubleClicked.connect(self._open_selected)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.open_button = QPushButton(self.tr('해당 말풍선 열기'), self.surface)
        self.open_button.clicked.connect(self._open_selected)
        actions.addWidget(self.open_button)
        self.acknowledge_button = QPushButton(self.tr('확인 완료'), self.surface)
        self.acknowledge_button.clicked.connect(self._acknowledge_selected)
        actions.addWidget(self.acknowledge_button)
        actions.addStretch()
        layout.addLayout(actions)

        language_row = QHBoxLayout()
        language_row.addWidget(QLabel(self.tr('원문 언어'), self.surface))
        self.language_combo = QComboBox(self.surface)
        for language in ('Auto', '日本語', 'English'):
            self.language_combo.addItem(language, language)
        language_row.addWidget(self.language_combo)
        self.scope_combo = QComboBox(self.surface)
        self.scope_combo.addItem(self.tr('선택한 말풍선'), 'block')
        self.scope_combo.addItem(self.tr('선택한 페이지 전체'), 'page')
        language_row.addWidget(self.scope_combo)
        self.language_button = QPushButton(self.tr('언어 적용'), self.surface)
        self.language_button.clicked.connect(self._apply_language)
        language_row.addWidget(self.language_button)
        language_row.addStretch()
        layout.addLayout(language_row)
        note = QLabel(self.tr('확인 완료한 항목도 원문·번역·배치가 바뀌면 다시 표시됩니다.'), self.surface)
        note.setWordWrap(True)
        layout.addWidget(note)
        self._refresh_surface_style()
        self._sync_selection()
        self._update_status()

    def _refresh_surface_style(self) -> None:
        background = ','.join(map(str, shared.WIDGET_BACKGROUND_COLOR))
        border = ','.join(map(str, shared.BORDER_COLOR))
        stylesheet = (
            f'QFrame#LocalReviewSurface {{ background-color: rgb({background}); '
            f'border: 1px solid rgb({border}); border-radius: 10px; }}'
        )
        if stylesheet != self._surface_style:
            self.surface.setStyleSheet(stylesheet)
            self._surface_style = stylesheet

    def showEvent(self, event) -> None:
        self._refresh_surface_style()
        super().showEvent(event)
        # The stretch column receives its real width only when the dialog shows.
        self.table.resizeRowsToContents()

    def _preserve_on_outside_click(self) -> bool:
        # Review remains visible while its target bubble is edited on the canvas.
        return True

    def set_items(self, items: list[ReviewItem]) -> None:
        """Refresh changed rows while preserving the current issue when possible."""
        if self._items == items:
            return
        previous = self._selected_item()
        previous_row = self.table.currentRow()
        old_items = self._items
        self._items = [dict(item) for item in items]
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            if row < len(old_items) and old_items[row] == item:
                continue
            index = item['block_index']
            columns = [item['page'], self.tr('페이지 전체') if index is None else str(index + 1), item['message']]
            for column, value in enumerate(columns):
                cell = self.table.item(row, column)
                if cell is None:
                    cell = QTableWidgetItem()
                    self.table.setItem(row, column, cell)
                cell.setText(value)
                cell.setToolTip(value)
        if items:
            selected_row = next((row for row, item in enumerate(items) if previous and item['key'] == previous['key']), min(max(previous_row, 0), len(items) - 1))
            self.table.selectRow(selected_row)
        self.table.resizeRowsToContents()
        self._sync_selection()
        self._update_status()

    def _update_status(self) -> None:
        if self._items:
            self.status_label.setText(self.tr('확인이 필요한 항목: %1개').replace('%1', str(len(self._items))))
        else:
            self.status_label.setText(self.tr('현재 기록된 확인 항목이 없습니다.'))

    def _selected_item(self) -> Optional[ReviewItem]:
        row = self.table.currentRow()
        return self._items[row] if 0 <= row < len(self._items) else None

    def _sync_selection(self) -> None:
        item = self._selected_item()
        enabled = item is not None
        for widget in (self.open_button, self.acknowledge_button, self.language_combo, self.scope_combo, self.language_button):
            widget.setEnabled(enabled)
        page_scope = item is not None and item['block_index'] is None
        self.open_button.setText(self.tr('해당 페이지 열기') if page_scope else self.tr('해당 말풍선 열기'))
        self.scope_combo.model().item(0).setEnabled(not page_scope)
        self.scope_combo.setCurrentIndex(1 if page_scope else 0)

    def _open_selected(self, *_args) -> None:
        item = self._selected_item()
        if item is not None:
            index = item['block_index']
            self.open_requested.emit(item['page'], -1 if index is None else index)

    def _acknowledge_selected(self) -> None:
        item = self._selected_item()
        if item is not None:
            self.acknowledge_requested.emit(dict(item))

    def _apply_language(self) -> None:
        item = self._selected_item()
        if item is not None:
            index = item['block_index']
            if self.scope_combo.currentData() == 'page' or index is None:
                index = -1
            self.language_requested.emit(item['page'], index, self.language_combo.currentData())
