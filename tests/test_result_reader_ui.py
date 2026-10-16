"""Real Qt event-loop regressions; run separately for Qt5 and Qt6."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip('qtpy')
from qtpy.QtCore import Qt, QTimer, QCoreApplication, QEvent
from qtpy.QtGui import QFontDatabase, QFont, QFontMetrics
from qtpy.QtWidgets import QApplication

from ballontranslator.utils.reader_export import Book, inspect_page, ReaderHistory
from ballontranslator.ui.result_reader import ReaderDialog
from ballontranslator.ui.image_merge_dialog import ImageMergeDialog


@pytest.fixture(scope='module')
def app():
    application = QApplication.instance() or QApplication([])
    font_path = os.environ.get('READER_TEST_FONT')
    if font_path:
        font_id = QFontDatabase.addApplicationFont(font_path)
        assert font_id >= 0, 'Test font could not be loaded'
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
        application.setFont(QFont(family, 10))
        assert QFontMetrics(application.font()).inFontUcs4(ord('한'))
    return application


def until(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), 'Qt operation did not finish'


@pytest.fixture
def book(tmp_path):
    result = tmp_path / 'result'; result.mkdir()
    ps = []
    for i in range(4):
        path = result / f'{i+1:03d}.png'
        Image.new('RGB', (320, 480), ('red', 'blue', 'green', 'white')[i]).save(path)
        ps.append(inspect_page(path, path.name))
    return Book(tmp_path, result, tuple(ps))


def close_reader(app, dialog):
    dialog.reject()
    until(app, lambda: not dialog.isVisible() and not dialog.canvas.loader.isRunning())
    dialog.deleteLater(); app.processEvents()


def close_merge(app, dialog):
    dialog.reject()
    until(app, lambda: not dialog.isVisible() and not dialog.busy)
    dialog.deleteLater(); app.processEvents()


def test_reader_navigation_modes_and_async_close(app, book, tmp_path):
    d = ReaderDialog(book, ReaderHistory(tmp_path / 'history.json'))
    d.show(); until(app, lambda: bool(d.canvas.cache))
    d.jump_to(2)
    assert d.current_row == 2
    d.mode_combo.setCurrentIndex(1)
    assert d.canvas.mode == 'page'
    d.next_page()
    assert d.current_row == 3
    d.previous_page()
    assert d.current_row == 2
    d.zoom_spin.setValue(150)
    assert d.canvas.zoom == 150
    close_reader(app, d)
    assert ReaderHistory(tmp_path / 'history.json').get(str(book.folder))['page'] == '003.png'


def test_reader_edit_requests_exact_page(app, book, tmp_path):
    d = ReaderDialog(book, ReaderHistory(tmp_path / 'h.json'))
    d.show(); d.jump_to(1); d.edit_page()
    until(app, lambda: not d.isVisible())
    assert d.edit_requested_page == '002.png'
    assert not d.canvas.loader.isRunning()
    d.deleteLater(); app.processEvents()


def test_reader_restores_position(app, book, tmp_path):
    history = ReaderHistory(tmp_path / 'h.json')
    history.save_position(str(book.folder), '003.png', .3, 'scroll', 120, 'width')
    d = ReaderDialog(book, history); d.show()
    until(app, lambda: d.current_row == 2)
    assert d.zoom_spin.value() == 120
    close_reader(app, d)


def test_merge_selection_order_preview_and_png_export(app, book, tmp_path):
    d = ImageMergeDialog(book); d.show()
    until(app, lambda: d.preview_ready)
    assert d.plan is not None
    d.page_list.setCurrentRow(0); d.move_down()
    assert d.plan.groups[0].items[0].page.name == '002.png'
    d.format_combo.setCurrentIndex(1)
    d.output_edit.setText(str(tmp_path / 'out'))
    d.start_export()
    until(app, lambda: d.last_output is not None)
    with Image.open(next(d.last_output.glob('*.jpg'))) as image:
        assert image.format == 'JPEG'
    close_merge(app, d)


def test_warnings_require_explicit_export_acknowledgement(app, book):
    warned = Book(book.folder, book.result_dir, book.pages, ('저장되지 않은 수정이 있습니다.',))
    d = ImageMergeDialog(warned); d.show()
    assert not d.export_button.isEnabled()
    d.acknowledge.setChecked(True)
    until(app, lambda: d.export_button.isEnabled())
    close_merge(app, d)


def test_missing_results_remain_visible_and_block_export(app, book):
    from ballontranslator.utils.reader_export import Page
    bad = Book(book.folder, book.result_dir, book.pages + (Page('missing.png', None, issue='결과 없음'),))
    d = ImageMergeDialog(bad); d.show()
    assert d.page_list.count() == 5
    assert d.plan is None
    d.page_list.item(4).setCheckState(Qt.CheckState.Unchecked)
    assert d.plan is not None
    close_merge(app, d)


def test_empty_selection_is_explained(app, book):
    d = ImageMergeDialog(book); d.show(); d.select_none()
    assert not d.export_button.isEnabled()
    assert '선택' in d.summary_label.text()
    close_merge(app, d)


def test_close_during_preview_never_deletes_running_worker(app, book):
    d = ImageMergeDialog(book); d.show(); d.refresh_plan()
    close_merge(app, d)


def test_ui_screenshots_and_small_window(app, book, tmp_path):
    root = Path(os.environ.get('READER_SCREENSHOT_DIR', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    d = ReaderDialog(book, ReaderHistory(tmp_path / 'h.json'))
    try:
        d.resize(1000, 720); d.show(); until(app, lambda: bool(d.canvas.cache))
        assert d.grab().save(str(root / 'reader.png'))
        d.resize(760, 560); app.processEvents()
        assert d.grab().save(str(root / 'reader-small.png'))
        print('reader sizing', d.width(), d.layout().minimumSize().width(), app.font().toString())
        assert d.layout().minimumSize().width() <= d.width()
    finally:
        close_reader(app, d)
    m = ImageMergeDialog(book)
    try:
        m.resize(1050, 760); m.show()
        until(app, lambda: m.preview_ready)
        assert m.grab().save(str(root / 'image-export.png'))
        m.resize(800, 620); app.processEvents()
        assert m.grab().save(str(root / 'image-export-small.png'))
        print('export sizing', m.width(), m.layout().minimumSize().width())
        assert m.layout().minimumSize().width() <= m.width()
    finally:
        close_merge(app, m)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_workspace_has_guarded_reader_entry_points(app, tmp_path):
    from ballontranslator.ui.local_workspace import LocalWorkspace
    d = LocalWorkspace(queue_path=tmp_path / 'jobs.json')
    d.show()
    assert not d.reader_tools.read_button.isEnabled()
    folder = tmp_path / 'book'; folder.mkdir()
    d.add_folders([str(folder)]); d.table.selectRow(0); app.processEvents()
    assert d.reader_tools.read_button.isEnabled()
    d.running = True; d.set_controls()
    assert not d.reader_tools.read_button.isEnabled()
    d.running = False; d.set_controls(); d.close(); d.deleteLater(); app.processEvents()


def test_loader_cancel_keeps_folder_unchanged(app, book):
    from ballontranslator.ui.reader_tools import BookLoadThread
    worker = BookLoadThread(book.folder, book.result_dir, {})
    worker.stop_event.set(); worker.start()
    until(app, worker.isFinished); worker.wait()
    assert worker.result is None
    assert worker.error
    worker.deleteLater(); app.processEvents()


def test_fullscreen_close_shuts_down_loader(app, book, tmp_path):
    d = ReaderDialog(book, ReaderHistory(tmp_path / 'h.json'))
    d.show(); d.toggle_fullscreen(); app.processEvents()
    close_reader(app, d)
