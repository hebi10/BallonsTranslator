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


def test_workspace_owned_reader_and_small_export_are_readable(app, book, tmp_path):
    from ballontranslator.ui.local_workspace import LocalWorkspace
    from qtpy.QtGui import QPalette
    owner = LocalWorkspace(queue_path=tmp_path / 'jobs.json')
    owner.show(); app.processEvents()
    root = Path(os.environ.get('READER_SCREENSHOT_DIR', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    reader = ReaderDialog(book, ReaderHistory(tmp_path / 'h.json'), owner)
    try:
        reader.resize(1000, 720); reader.show()
        until(app, lambda: bool(reader.canvas.cache))
        background = reader.palette().color(QPalette.ColorRole.Window).lightness()
        foreground = reader.notice.palette().color(QPalette.ColorRole.WindowText).lightness()
        assert abs(background - foreground) >= 100, (background, foreground)
        assert reader.grab().save(str(root / 'reader-workspace.png'))
    finally:
        close_reader(app, reader)
    merge = ImageMergeDialog(book, owner)
    try:
        merge.resize(800, 620); merge.show()
        until(app, lambda: merge.preview_ready)
        assert merge.direction_combo.geometry().bottom() < merge.cross_spin.geometry().top()
        assert merge.cross_spin.geometry().bottom() < merge.gap_spin.geometry().top()
        assert merge.length_spin.geometry().bottom() < merge.format_combo.geometry().top()
        assert merge.export_button.isVisible()
        assert merge.grab().save(str(root / 'image-export-workspace-small.png'))
    finally:
        close_merge(app, merge)
        owner.close(); owner.deleteLater(); app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)



def test_workspace_export_settings_have_visible_text_contrast(app, book, tmp_path):
    from collections import Counter
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QColor, QPalette
    from ballontranslator.ui.local_workspace import LocalWorkspace
    owner = LocalWorkspace(queue_path=tmp_path / 'jobs.json')
    owner.show()
    app.processEvents()
    dialog = ImageMergeDialog(book, owner)
    try:
        dialog.resize(800, 620)
        dialog.show()
        until(app, lambda: dialog.preview_ready)
        picture = dialog.grab()
        image = picture.toImage()
        root = Path(os.environ.get('READER_SCREENSHOT_DIR', str(tmp_path)))
        root.mkdir(parents=True, exist_ok=True)
        assert picture.save(str(root / 'image-export-contrast.png'))
        label = dialog.cross_label
        origin = label.mapTo(dialog, QPoint(0, 0))
        scale = picture.devicePixelRatio()
        area = image.copy(round(origin.x() * scale), round(origin.y() * scale),
                          round(label.width() * scale), round(label.height() * scale))
        pixels = Counter(area.pixel(x, y) & 0xFFFFFF
                         for x in range(area.width()) for y in range(area.height()))
        background = QColor(pixels.most_common(1)[0][0])
        foreground = label.palette().color(QPalette.ColorRole.WindowText)
        def luminance(color):
            channels = [color.redF(), color.greenF(), color.blueF()]
            linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
            return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
        light, dark = sorted((luminance(background), luminance(foreground)), reverse=True)
        contrast = (light + 0.05) / (dark + 0.05)
        assert contrast >= 4.5, ('settings contrast', contrast, background.name(), foreground.name())
    finally:
        close_merge(app, dialog)
        owner.close()
        owner.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('same_project', [False, True])
@pytest.mark.parametrize('source_language', ['English', '日本語', 'Auto'])
def test_reader_edit_activates_saved_folder_after_preserving_current_settings(
    app, book, tmp_path, monkeypatch, same_project, source_language,
):
    import copy
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qtpy.QtWidgets import QListWidget, QWidget
    from ballontranslator.ui.local_workspace import LocalWorkspace
    from ballontranslator.utils.config import pcfg

    monkeypatch.setattr(pcfg, 'module', copy.deepcopy(pcfg.module))
    monkeypatch.setattr('ballontranslator.ui.local_workspace.save_config', Mock(return_value=True))
    events = []

    class Editor(QWidget):
        def __init__(self):
            super().__init__()
            directory = str(book.folder if same_project else tmp_path / 'previous')
            self.imgtrans_proj = SimpleNamespace(
                directory=directory, current_img='001.png', pagename2idx=Mock(return_value=1),
                local_settings={'source_language': '日本語', 'glossary_path': 'previous.json'},
            )
            self.pageList = QListWidget(self)
            self.pageList.addItems(['001.png', '002.png'])
            self.pageList.currentRowChanged.connect(self.page_selected)

        def conditional_save(self):
            events.append(('save', pcfg.module.translate_source, pcfg.module.local_work_name))
            return True

        def open_local_editor(self, folder):
            events.append(('open', pcfg.module.translate_source, pcfg.module.local_work_name))
            self.imgtrans_proj.directory = folder
            self.on_local_settings_changed(folder)
            self._sync_local_source_language()

        def on_local_settings_changed(self, folder):
            self.imgtrans_proj.local_settings = workspace.folder_settings(folder)

        def _sync_local_source_language(self):
            events.append(('sync', pcfg.module.translate_source, pcfg.module.local_work_name))

        def page_selected(self, row):
            self.imgtrans_proj.current_img = self.pageList.item(row).text()

    editor = Editor()
    workspace = LocalWorkspace(editor, queue_path=tmp_path / 'jobs.json')
    try:
        workspace.queue.add([str(book.folder)], source_language)
        job = workspace.queue.jobs[0]
        job.update(reading_direction='ltr', work_name='Target book',
                   glossary_path=str(tmp_path / 'target.json'), tone_notes='Polite dialogue')
        workspace.queue.save()
        pcfg.module.translate_source = 'English' if source_language != 'English' else '日本語'
        pcfg.module.local_work_name = 'Previous book'
        old_source = pcfg.module.translate_source

        workspace.reader_tools.open_editor_page(book.folder, '002.png')

        assert events[0] == ('save', old_source, 'Previous book')
        assert events[-1] == ('sync', source_language, 'Target book')
        assert [event[0] for event in events].count('open') == (0 if same_project else 1)
        assert pcfg.module.local_reading_direction == 'ltr'
        assert pcfg.module.llm_glossary_path == str(tmp_path / 'target.json')
        assert pcfg.module.local_tone_notes == 'Polite dialogue'
        from ballontranslator.utils.local_language import effective_source_language
        assert effective_source_language(editor.imgtrans_proj, '002.png') == source_language
        assert editor.imgtrans_proj.local_settings['glossary_path'] == str(tmp_path / 'target.json')
        assert editor.imgtrans_proj.local_settings['work_name'] == 'Target book'
        assert editor.imgtrans_proj.local_settings['tone_notes'] == 'Polite dialogue'
        assert editor.imgtrans_proj.current_img == '002.png'
    finally:
        workspace.deleteLater()
        editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_reader_edit_failed_presave_keeps_existing_project_and_settings(app, book, tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qtpy.QtWidgets import QWidget
    from ballontranslator.ui.local_workspace import LocalWorkspace
    from ballontranslator.utils.config import pcfg

    monkeypatch.setattr(pcfg, 'module', copy.deepcopy(pcfg.module))
    monkeypatch.setattr('ballontranslator.ui.local_workspace.save_config', Mock(return_value=True))
    editor = QWidget()
    editor.conditional_save = Mock(return_value=False)
    editor.open_local_editor = Mock()
    editor.imgtrans_proj = SimpleNamespace(directory=str(tmp_path / 'previous'), current_img='old.png')
    workspace = LocalWorkspace(editor, queue_path=tmp_path / 'jobs.json')
    try:
        workspace.queue.add([str(book.folder)], 'English')
        pcfg.module.translate_source = '日本語'
        workspace.reader_tools.open_editor_page(book.folder, '002.png')
        editor.open_local_editor.assert_not_called()
        assert editor.imgtrans_proj.directory == str(tmp_path / 'previous')
        assert editor.imgtrans_proj.current_img == 'old.png'
        assert pcfg.module.translate_source == '日本語'
        assert '저장' in workspace.status_label.text()
    finally:
        workspace.deleteLater()
        editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_closed_reader_releases_callbacks_and_worker_wrappers(app, book, tmp_path):
    import gc
    import weakref

    dialog = ReaderDialog(book, ReaderHistory(tmp_path / 'h.json'))
    dialog.show()
    until(app, lambda: bool(dialog.canvas.cache))
    references = [weakref.ref(dialog), weakref.ref(dialog.canvas), weakref.ref(dialog.canvas.loader)]
    close_reader(app, dialog)
    del dialog
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_reader_failed_project_open_does_not_navigate_partial_target(app, book, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qtpy.QtWidgets import QWidget
    from ballontranslator.ui.local_workspace import LocalWorkspace

    editor = QWidget()
    editor.imgtrans_proj = SimpleNamespace(directory=str(tmp_path / 'previous'), pagename2idx=Mock())

    def fail_after_path_changed(folder):
        editor.imgtrans_proj.directory = folder
        return False

    editor.open_local_editor = fail_after_path_changed
    workspace = LocalWorkspace(editor, queue_path=tmp_path / 'jobs.json')
    monkeypatch.setattr(workspace, '_prepare_folder_action', Mock(return_value=True))
    try:
        workspace.reader_tools.open_editor_page(book.folder, '002.png')
        editor.imgtrans_proj.pagename2idx.assert_not_called()
    finally:
        workspace.deleteLater()
        editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_reader_failed_same_project_settings_does_not_navigate(app, book, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qtpy.QtWidgets import QWidget
    from ballontranslator.ui.local_workspace import LocalWorkspace

    editor = QWidget()
    editor.imgtrans_proj = SimpleNamespace(directory=str(book.folder), pagename2idx=Mock())
    editor.open_local_editor = Mock()
    editor.on_local_settings_changed = Mock(return_value=False)
    editor._sync_local_source_language = Mock()
    workspace = LocalWorkspace(editor, queue_path=tmp_path / 'jobs.json')
    monkeypatch.setattr(workspace, '_prepare_folder_action', Mock(return_value=True))
    try:
        workspace.reader_tools.open_editor_page(book.folder, '002.png')
        editor.on_local_settings_changed.assert_called_once_with(str(book.folder))
        editor._sync_local_source_language.assert_not_called()
        editor.imgtrans_proj.pagename2idx.assert_not_called()
    finally:
        workspace.deleteLater()
        editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
