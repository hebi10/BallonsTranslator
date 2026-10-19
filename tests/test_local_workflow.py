"""Exercise local workflow boundaries with real projects, files, and Qt workers."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.local_review_dialog import LocalReviewDialog
from ballontranslator.ui.local_workspace import LocalWorkspace
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg, RunStatus
from ballontranslator.utils.local_review import collect_review_items
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock


class WorkflowWindow(QWidget):
    start_local_preview = MainWindow.start_local_preview
    _local_preview_copy_ready = MainWindow._local_preview_copy_ready
    _local_preview_copy_failed = MainWindow._local_preview_copy_failed
    _local_preview_copy_finished = MainWindow._local_preview_copy_finished
    _poll_local_preview = MainWindow._poll_local_preview
    poll_local_queue = MainWindow.poll_local_queue
    pause_local_queue = MainWindow.pause_local_queue
    on_run_imgtrans = MainWindow.on_run_imgtrans
    on_imgtrans_pipeline_finished = MainWindow.on_imgtrans_pipeline_finished
    _sync_local_source_language = MainWindow._sync_local_source_language
    _local_review_editable = MainWindow._local_review_editable
    _refresh_local_review = MainWindow._refresh_local_review
    set_local_review_language = MainWindow.set_local_review_language
    acknowledge_local_review = MainWindow.acknowledge_local_review
    open_local_review_item = MainWindow.open_local_review_item

    def _local_review_target_current(self, page, block_index):
        return MainWindow._local_review_target_current(self, page, block_index)

    def __init__(self, root: Path):
        super().__init__()
        self.imgtrans_proj = ProjImgTrans(str(root / 'original'))
        self.imgtrans_proj.set_current_img('01.png')
        self.module_manager = SimpleNamespace(
            translator=Mock(), regionReworkBusy=lambda: False,
            stopImgtransPipeline=Mock(), runImgtransPipeline=Mock(),
        )
        self.st_manager = Mock()
        self.bottomBar = Mock()
        self.bottomBar.textblockChecker.isChecked.return_value = False
        self.setTranslatorSelectionFromMetadata = Mock()
        self.conditional_save = Mock(return_value=True)
        self._local_pipeline_busy = Mock(return_value=False)
        self.imsave_thread = SimpleNamespace(isBusy=Mock(return_value=False), error_count=0)
        self.backup_blkstyles = []
        self._run_imgtrans_wo_textstyle_update = False
        self._local_timer = Mock()
        self._local_preview_active = False
        self._local_preview_worker = None
        self._local_review_identity = self.imgtrans_proj.load_identity
        self._local_review_dialog = LocalReviewDialog(self)
        self.show_local_review = Mock()
        self.pageList = Mock()
        self.local_workspace = LocalWorkspace(self, queue_path=root / 'queue.json')
        self.local_workspace.preview_requested.connect(self.start_local_preview)
        self.local_workspace.pause_requested.connect(self.pause_local_queue)

    def openDir(self, folder: str) -> bool:
        self.imgtrans_proj = ProjImgTrans(folder)
        self.imgtrans_proj.set_current_img_byidx(0)
        return True


def test_full_local_redetection_retains_manual_language_until_worker_can_match(workflow):
    host = workflow
    block = host.imgtrans_proj.pages['01.png'][0]
    block.source_language_override = '日本語'
    pcfg.module.keep_exist_textlines = False
    host.on_run_imgtrans(pages_to_process=['01.png'])
    assert host.imgtrans_proj.pages['01.png'] == [block]
    assert block.source_language_override == '日本語'
    assert host.imgtrans_proj._image_info['01.png']['finish_code'] == 0
    host.module_manager.runImgtransPipeline.assert_called_once()


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workflow(tmp_path, monkeypatch, app):
    for name in ('module', 'drawpanel', 'global_fontformat'):
        monkeypatch.setattr(pcfg, name, copy.deepcopy(getattr(pcfg, name)))
    for name in ('check_update_on_startup', 'display_lang', 'let_writing_mode_flag', 'let_family_flag', 'imgsave_ext'):
        monkeypatch.setattr(pcfg, name, getattr(pcfg, name))
    for name in ('enable_detect', 'enable_ocr', 'enable_translate', 'enable_inpaint'):
        setattr(pcfg.module, name, True)
    pcfg.module.update_finish_code()
    monkeypatch.setattr(shared, 'CONFIG_PATH', str(tmp_path / 'config' / 'config.json'))
    monkeypatch.setattr(shared, 'HEADLESS', False)
    monkeypatch.setattr(shared, 'args', SimpleNamespace(export_translation_txt=False, export_source_txt=False))
    monkeypatch.setattr('ballontranslator.ui.mainwindow.is_local_edition', lambda: True)
    monkeypatch.setattr('ballontranslator.ui.local_workspace.save_config', lambda: True)
    info = Mock()
    monkeypatch.setattr('ballontranslator.ui.mainwindow.create_info_dialog', info)
    original = tmp_path / 'original'
    original.mkdir()
    for name in ('01.png', '02.png'):
        Image.new('RGB', (32, 24), 'white').save(original / name)
    host = WorkflowWindow(tmp_path)
    host.info = info
    block = TextBlock(text=['Hello'], translation='직접 수정한 번역')
    block.language_review = '원문 언어 확인'
    block.source_language_override = 'English'
    host.imgtrans_proj.pages['01.png'] = [block]
    host.imgtrans_proj.set_page_progress('01.png', RunStatus.FIN_ALL)
    host.imgtrans_proj.save()
    host._refresh_local_review()
    workspace = host.local_workspace
    next_folder = tmp_path / 'next'
    next_folder.mkdir()
    Image.new('RGB', (8, 8), 'red').save(next_folder / 'next.png')
    workspace.queue.add([str(original), str(next_folder)])
    workspace.queue.jobs[0].update(source_language='English', work_name='Original work', tone_notes='반말')
    workspace.queue.jobs[1].update(source_language='日本語', work_name='Next work', tone_notes='존댓말')
    workspace.queue.save()
    workspace.refresh_queue()
    workspace.table.selectRow(0)
    yield host
    worker = host._local_preview_worker
    if worker is not None:
        worker.stop_event.set()
        worker.wait(3000)
        app.processEvents()
    host.local_workspace.running = False
    host.close()
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def begin_preview(host, monkeypatch, app):
    source = Path(host.imgtrans_proj.directory) / '01.png'
    monkeypatch.setattr('ballontranslator.ui.local_workspace.QFileDialog.getOpenFileName', lambda *args: (str(source), ''))
    host.local_workspace.preview_selected()
    worker = host._local_preview_worker
    assert worker is not None
    assert worker.wait(3000)
    app.processEvents()
    assert host._local_preview_worker is None
    host.poll_local_queue()
    return source


def finish_preview_page(host):
    page = next(iter(host.imgtrans_proj.pages))
    host.imgtrans_proj.set_page_progress(page, RunStatus.FIN_ALL)
    result = Path(host.imgtrans_proj.get_result_path(page))
    result.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 24), 'blue').save(result)
    host.on_imgtrans_pipeline_finished()


def project_state(project):
    return json.loads(json.dumps(project.to_dict(), cls=TextBlkEncoder))


def test_selected_preview_dispatches_one_isolated_page_with_folder_settings(workflow, monkeypatch, app):
    host = workflow
    original_project = host.imgtrans_proj
    original_json = Path(original_project.proj_path).read_bytes()
    queue_before = copy.deepcopy(host.local_workspace.queue.jobs)

    source = begin_preview(host, monkeypatch, app)

    assert Path(host.imgtrans_proj.directory).parent == Path(shared.CONFIG_PATH).parent / 'local-previews'
    pages = list(host.imgtrans_proj.pages)
    assert len(pages) == 1
    host.module_manager.runImgtransPipeline.assert_called_once_with(pages, render_only=False)
    assert Path(host.imgtrans_proj.directory, pages[0]).read_bytes() == source.read_bytes()
    assert Path(original_project.proj_path).read_bytes() == original_json
    assert original_project.pages['01.png'][0].translation == '직접 수정한 번역'
    assert original_project._image_info['01.png']['finish_code'] == RunStatus.FIN_ALL
    assert host.imgtrans_proj.local_settings['work_name'] == 'Original work'
    assert host.imgtrans_proj.local_settings['source_language'] == 'English'
    host.module_manager.translator.set_source.assert_called_with('English')
    assert host.local_workspace.queue.jobs == queue_before


def test_full_preview_enables_every_stage_despite_previous_editor_stage_choices(workflow, monkeypatch, app):
    for name in ('enable_detect', 'enable_ocr', 'enable_translate', 'enable_inpaint'):
        setattr(pcfg.module, name, False)
    pcfg.module.update_finish_code()

    begin_preview(workflow, monkeypatch, app)

    assert pcfg.module.finish_code == RunStatus.FIN_ALL
    assert all(getattr(pcfg.module, name) for name in ('enable_detect', 'enable_ocr', 'enable_translate', 'enable_inpaint'))
    finish_preview_page(workflow)
    workflow.poll_local_queue()
    assert pcfg.module.finish_code == 0
    assert not any(getattr(pcfg.module, name) for name in ('enable_detect', 'enable_ocr', 'enable_translate', 'enable_inpaint'))


def test_pipeline_completion_waits_for_pending_image_write(workflow, monkeypatch, app):
    host = workflow
    queue_before = copy.deepcopy(host.local_workspace.queue.jobs)
    begin_preview(host, monkeypatch, app)
    finish_preview_page(host)
    host.imsave_thread.isBusy.return_value = True

    host.poll_local_queue()

    assert host._local_preview_active
    assert host.local_workspace.running
    host.show_local_review.assert_not_called()
    host.imsave_thread.isBusy.return_value = False
    host.poll_local_queue()
    assert not host._local_preview_active
    assert not host.local_workspace.running
    assert '완료' in host.local_workspace.status_label.text()
    host.show_local_review.assert_called_once_with(host.imgtrans_proj.directory)
    assert host.local_workspace.queue.jobs == queue_before
    assert host.local_workspace.active_index is None


@pytest.mark.parametrize('outcome', ['cancel', 'copy_error', 'pipeline_error', 'save_error'])
def test_preview_cancellation_or_failure_never_advances_normal_queue(workflow, monkeypatch, app, outcome):
    host = workflow
    queue_before = copy.deepcopy(host.local_workspace.queue.jobs)
    if outcome == 'copy_error':
        host.start_local_preview(str(Path(host.imgtrans_proj.directory) / 'missing.png'))
        assert host._local_preview_worker.wait(3000)
        app.processEvents()
    else:
        begin_preview(host, monkeypatch, app)
    if outcome == 'cancel':
        host.local_workspace.request_pause()
    elif outcome == 'pipeline_error':
        host.imgtrans_proj.mark_stage_failed(next(iter(host.imgtrans_proj.pages)), 'ocr')
        host.on_imgtrans_pipeline_finished()
    elif outcome == 'save_error':
        finish_preview_page(host)
        host.imsave_thread.error_count += 1
    host.poll_local_queue()

    assert not host._local_preview_active
    assert not host.local_workspace.running
    assert host.local_workspace.active_index is None
    assert host.local_workspace.queue.jobs == queue_before
    host.show_local_review.assert_not_called()
    assert '전체 시험 완료' not in host.local_workspace.status_label.text()
    host.poll_local_queue()
    assert host.local_workspace.queue.jobs == queue_before


def test_paused_preview_preserves_completed_stage_metadata_without_advancing_queue(workflow, monkeypatch, app):
    host = workflow
    queue_before = copy.deepcopy(host.local_workspace.queue.jobs)
    begin_preview(host, monkeypatch, app)
    project = host.imgtrans_proj
    page = next(iter(project.pages))
    project.pages[page] = [TextBlock(text=['Recognized source'], translation='')]
    project.set_page_progress(page, RunStatus.FIN_DET | RunStatus.FIN_OCR)
    host._local_pipeline_busy.return_value = True
    host.local_workspace.request_pause()
    host.poll_local_queue()
    assert host._local_preview_active

    host._local_pipeline_busy.return_value = False
    host.poll_local_queue()

    restored = ProjImgTrans(project.directory)
    assert restored.pages[page][0].text == ['Recognized source']
    assert restored._image_info[page]['finish_code'] == RunStatus.FIN_DET | RunStatus.FIN_OCR
    assert not host._local_preview_active
    assert host.local_workspace.queue.jobs == queue_before
    assert host.local_workspace.active_index is None
    host.show_local_review.assert_not_called()
    assert '전체 시험 완료' not in host.local_workspace.status_label.text()
    host.poll_local_queue()
    assert host.local_workspace.queue.jobs == queue_before


def test_late_preview_completion_does_not_save_a_different_open_project(workflow, monkeypatch, app):
    host = workflow
    original_folder = host.imgtrans_proj.directory
    original_json = Path(host.imgtrans_proj.proj_path).read_bytes()
    begin_preview(host, monkeypatch, app)
    host.imgtrans_proj = ProjImgTrans(original_folder)
    host.imgtrans_proj.pages['01.png'][0].translation = '아직 저장하지 않은 수정'
    host.local_workspace.request_pause()

    host.poll_local_queue()

    assert Path(host.imgtrans_proj.proj_path).read_bytes() == original_json
    assert host.imgtrans_proj.pages['01.png'][0].translation == '아직 저장하지 않은 수정'
    assert not host._local_preview_active


@pytest.mark.parametrize('block_index,source', [(0, '日本語'), (-1, '日本語'), (-1, 'Auto')])
def test_review_language_change_persists_without_erasing_manual_work(workflow, block_index, source):
    host = workflow
    project = host.imgtrans_proj
    before_translation = project.pages['01.png'][0].translation
    before_progress = project._image_info['01.png']['finish_code']

    host.set_local_review_language('01.png', block_index, source)

    restored = ProjImgTrans(project.directory)
    block = restored.pages['01.png'][0]
    assert block.translation == before_translation
    assert restored._image_info['01.png']['finish_code'] == before_progress
    assert block.language_review == ''
    if block_index == -1:
        assert restored._image_info['01.png']['local_source_language'] == source
        assert block.source_language_override == ''
    else:
        assert block.source_language_override == source
    assert any(item['code'] == 'language_changed' for item in collect_review_items(restored))


@pytest.mark.parametrize('block_index', [0, -1])
def test_review_language_save_failure_restores_metadata_and_existing_project(workflow, monkeypatch, block_index):
    host = workflow
    project = host.imgtrans_proj
    before = project_state(project)
    before_json = Path(project.proj_path).read_bytes()
    monkeypatch.setattr('ballontranslator.utils.proj_imgtrans.os.replace', Mock(side_effect=OSError('disk full')))

    host.set_local_review_language('01.png', block_index, '日本語')

    assert project_state(project) == before
    assert Path(project.proj_path).read_bytes() == before_json
    assert 'disk full' in host.info.call_args.args[0]


def test_stale_review_actions_cannot_modify_or_navigate_reloaded_project(workflow):
    host = workflow
    item = collect_review_items(host.imgtrans_proj)[0]
    host.imgtrans_proj = ProjImgTrans(host.imgtrans_proj.directory)
    before = project_state(host.imgtrans_proj)
    before_json = Path(host.imgtrans_proj.proj_path).read_bytes()

    host.set_local_review_language('01.png', 0, '日本語')
    host.acknowledge_local_review(item)
    host.open_local_review_item('01.png', 0)

    assert project_state(host.imgtrans_proj) == before
    assert Path(host.imgtrans_proj.proj_path).read_bytes() == before_json
    host.pageList.setCurrentRow.assert_not_called()
    host.st_manager.updateTextBlkList.assert_not_called()


@pytest.mark.parametrize('change', ['geometry', 'inserted_block'])
@pytest.mark.parametrize('action', ['language', 'open'])
def test_stale_review_row_cannot_change_language_or_open_a_different_bubble(workflow, change, action):
    host = workflow
    project = host.imgtrans_proj
    if change == 'geometry':
        project.pages['01.png'][0].xyxy = [10, 10, 30, 20]
    else:
        project.pages['01.png'].insert(0, TextBlock(text=['Inserted later'], translation='새 말풍선'))
    before = project_state(project)
    before_json = Path(project.proj_path).read_bytes()

    if action == 'language':
        host.set_local_review_language('01.png', 0, '日本語')
    else:
        host.open_local_review_item('01.png', 0)

    assert project_state(project) == before
    assert Path(project.proj_path).read_bytes() == before_json
    host.pageList.setCurrentRow.assert_not_called()
