import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.local_workspace import LocalJobQueue, LocalSetupThread, LocalWorkspace


class LocalQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'queue.json'
        self.folder = Path(self.tmp.name) / 'comic'
        self.folder.mkdir()

    def test_queue_deduplicates_and_resumes_interrupted_job(self):
        queue = LocalJobQueue(self.path)
        queue.add([str(self.folder), str(self.folder)])
        self.assertEqual(len(queue.jobs), 1)
        queue.jobs[0]['state'] = 'running'
        queue.save()
        restored = LocalJobQueue(self.path)
        self.assertEqual(restored.jobs[0]['state'], 'paused')
        self.assertTrue(restored.jobs[0]['resave'])
        self.assertEqual(restored.jobs[0]['folder'], str(self.folder.resolve()))

    def test_corrupt_queue_is_preserved_until_explicit_recovery(self):
        self.path.write_text('{invalid', encoding='utf-8')
        queue = LocalJobQueue(self.path)
        self.assertTrue(queue.warning)
        with self.assertRaises(ValueError):
            queue.add([str(self.folder)])
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{invalid')

    def test_unknown_and_invalid_rows_are_discarded_without_passive_write(self):
        content = {'version': 1, 'jobs': [None, {'folder': 7}, {'folder': str(self.folder), 'state': 'future', 'done': -8}]}
        self.path.write_text(json.dumps(content), encoding='utf-8')
        queue = LocalJobQueue(self.path)
        self.assertEqual(len(queue.jobs), 1)
        self.assertEqual(queue.jobs[0]['state'], 'pending')
        self.assertEqual(queue.jobs[0]['done'], 0)
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), content)

    def test_failed_atomic_replace_preserves_existing_queue(self):
        queue = LocalJobQueue(self.path)
        queue.add([str(self.folder)])
        before = self.path.read_bytes()
        with patch('ballontranslator.ui.local_workspace.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                queue.save()
        self.assertEqual(self.path.read_bytes(), before)

    def test_malformed_optional_types_do_not_crash_load(self):
        self.path.write_text(json.dumps({'version': 1, 'format': [], 'jobs': [
            {'folder': str(self.folder), 'state': [], 'error': {}, 'total': True}
        ]}), encoding='utf-8')
        queue = LocalJobQueue(self.path)
        self.assertEqual(queue.format, '.png')
        self.assertEqual(queue.jobs[0]['state'], 'pending')
        self.assertEqual(queue.jobs[0]['total'], 0)


class LocalWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dialog = LocalWorkspace(queue_path=Path(self.tmp.name) / 'queue.json')
        self.addCleanup(self.dialog.deleteLater)

    def tearDown(self):
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_worker_failure_is_not_reported_as_ready(self):
        worker = LocalSetupThread('test', self.dialog)
        failures, results = [], []
        worker.failed.connect(failures.append)
        worker.completed.connect(results.append)
        with patch('ballontranslator.utils.local_ollama.OllamaClient.test_translation', side_effect=RuntimeError('offline')):
            worker.run()
        self.assertEqual(results, [])
        self.assertEqual(failures, ['offline'])

    def test_readiness_table_shows_all_rows_without_vertical_scrolling(self):
        self.dialog.worker = LocalSetupThread('status', self.dialog)
        self.dialog._last_operation = 'status'
        self.dialog._worker_result = [(f'항목 {index}', False, '설정 필요') for index in range(5)]
        self.dialog.on_setup_finished()
        self.dialog.show()
        self.app.processEvents()
        last_row = self.dialog.checks.visualItemRect(self.dialog.checks.item(4, 0))
        self.assertLessEqual(last_row.bottom(), self.dialog.checks.viewport().height())
        self.assertEqual(self.dialog.checks.verticalScrollBar().maximum(), 0)

    def test_dialog_cannot_close_while_worker_runs(self):
        blocker = threading.Event()
        with patch('ballontranslator.utils.local_ollama.OllamaClient.test_translation', side_effect=lambda **kwargs: blocker.wait(2)):
            self.dialog.start_setup('test')
            self.dialog.show()
            self.dialog.reject()
            self.assertTrue(self.dialog.isVisible())
            self.assertIsNotNone(self.dialog.worker)
            blocker.set()
            self.dialog.worker.wait(3000)
            self.app.processEvents()
        self.assertIsNone(self.dialog.worker)
        self.dialog.reject()
        self.assertFalse(self.dialog.isVisible())

    def test_save_failure_keeps_job_failed(self):
        folder = Path(self.tmp.name) / 'comic'
        folder.mkdir()
        self.dialog.queue.add([str(folder)])
        self.dialog.active_index = 0
        self.dialog.finish_job(False, '결과 저장 실패', 1, 1)
        self.assertEqual(self.dialog.queue.jobs[0]['state'], 'failed')
        self.assertIn('저장 실패', self.dialog.queue.jobs[0]['error'])
        self.assertTrue(self.dialog.queue.jobs[0]['resave'])

    def test_queue_waits_for_image_saves_and_detects_missing_output(self):
        from ballontranslator.ui.mainwindow import MainWindow
        folder = Path(self.tmp.name) / 'comic'
        folder.mkdir()
        self.dialog.queue.add([str(folder)])
        self.dialog.active_index = 0
        self.dialog.running = True
        self.dialog.refresh_queue()
        project = SimpleNamespace(
            pages={'page.png': []}, get_page_progress=Mock(return_value=True),
            failed_pages=Mock(return_value=[]), save=Mock(),
            get_result_path=Mock(return_value=str(folder / 'result' / 'page.png')),
        )
        owner = SimpleNamespace(
            local_workspace=self.dialog, imgtrans_proj=project,
            _local_pending_folder=None, _local_job_finished=True,
            _local_output_dir=folder / 'result', _local_save_error='',
            _local_save_errors_before=0, _local_pipeline_busy=Mock(return_value=False),
            imsave_thread=SimpleNamespace(isBusy=Mock(return_value=True), error_count=0),
        )
        MainWindow.poll_local_queue(owner)
        project.save.assert_not_called()
        self.assertEqual(self.dialog.active_index, 0)
        owner.imsave_thread.isBusy.return_value = False
        MainWindow.poll_local_queue(owner)
        project.save.assert_called_once()
        self.assertEqual(self.dialog.queue.jobs[0]['state'], 'failed')
        self.assertIn('저장되지', self.dialog.queue.jobs[0]['error'])

    def test_failed_image_write_is_recorded_and_remaining_jobs_are_saved(self):
        from ballontranslator.ui.io_thread import ImgSaveThread
        writer = ImgSaveThread(self.dialog)
        writer.job = writer._save_img
        writer.saveImg('failed.png', object())
        writer.saveImg('ok.png', object())
        written = []
        writer.img_writed.connect(written.append)
        with patch('ballontranslator.ui.io_thread.imwrite', side_effect=[OSError('disk full'), None]), patch('ballontranslator.ui.io_thread.create_error_dialog'):
            writer._save_img()
        self.assertEqual(writer.error_count, 1)
        self.assertEqual(len(written), 1)
        self.assertFalse(writer.isBusy())

    def test_late_save_submission_restarts_after_thread_finishes(self):
        from qtpy.QtTest import QTest
        from ballontranslator.ui.io_thread import ImgSaveThread
        writer = ImgSaveThread(self.dialog)
        drained, release = threading.Event(), threading.Event()
        original = writer._save_img
        first = True

        def delayed_exit():
            nonlocal first
            original()
            if first:
                first = False
                drained.set()
                release.wait(2)

        writer._save_img = delayed_exit
        with patch('ballontranslator.ui.io_thread.imwrite') as write:
            writer.saveImg('first.png', object())
            self.assertTrue(drained.wait(2))
            writer.saveImg('second.png', object())
            release.set()
            writer.wait(2000)
            for _ in range(20):
                self.app.processEvents()
                if not writer.isBusy():
                    break
                QTest.qWait(10)
            writer.wait(2000)
            self.assertEqual(write.call_count, 2)
            self.assertFalse(writer.isBusy())

    def test_result_save_writes_selected_output_with_identical_pixels(self):
        from qtpy.QtGui import QColor, QImage
        from ballontranslator.ui.io_thread import ImgSaveThread
        from ballontranslator.ui.mainwindow import MainWindow
        from ballontranslator.utils.config import pcfg
        from ballontranslator.utils.io_utils import imread
        import numpy as np
        default_dir = Path(self.tmp.name) / 'comic' / 'result'
        output_dir = Path(self.tmp.name) / 'selected'
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(QColor('red'))
        writer = ImgSaveThread(self.dialog)
        writer.job = writer._save_img
        project = SimpleNamespace(
            img_valid=True, current_img='page.png', result_dir=lambda: str(default_dir),
            get_result_path=lambda _: str(default_dir / 'page.png'), current_has_alpha=lambda: False,
        )
        canvas = Mock()
        canvas.render_result_img.return_value = image
        owner = SimpleNamespace(
            imgtrans_proj=project, canvas=canvas, imsave_thread=writer,
            _local_output_dir=output_dir, rightComicTransStackPanel=Mock(isHidden=Mock(return_value=False)),
            st_manager=Mock(txtblkShapeControl=Mock(isVisible=Mock(return_value=False))),
        )
        with patch.object(pcfg, 'imgtrans_textblock', False), patch.object(pcfg, 'imgsave_ext', '.png'):
            self.assertTrue(MainWindow.saveCurrentPage(owner, update_scene_text=False, save_proj=False))
            writer._save_img()
        self.assertTrue((output_dir / 'page.png').is_file())
        np.testing.assert_array_equal(imread(str(default_dir / 'page.png')), imread(str(output_dir / 'page.png')))

    def test_queue_does_not_open_next_folder_if_project_save_fails(self):
        from ballontranslator.ui.mainwindow import MainWindow
        canvas = Mock(projstate_unsaved=True)
        canvas.text_change_unsaved.return_value = False
        canvas.draw_change_unsaved.return_value = False
        owner = SimpleNamespace(
            canvas=canvas, opening_dir=False, _llm_context_dirty=True,
            _local_pipeline_busy=Mock(return_value=False),
            imsave_thread=SimpleNamespace(error_count=0), local_workspace=Mock(),
            imgtrans_proj=SimpleNamespace(save=Mock(side_effect=OSError('disk full'))),
            openDir=Mock(), _local_timer=Mock(),
        )
        owner.conditional_save = lambda: MainWindow.conditional_save(owner)
        MainWindow.start_local_folder(owner, '/new/folder')
        owner.openDir.assert_not_called()
        owner._local_timer.start.assert_not_called()
        canvas.setProjSaveState.assert_not_called()
        owner.local_workspace.finish_job.assert_called_once()
        self.assertIn('disk full', owner.local_workspace.finish_job.call_args.args[1])

    def test_queue_does_not_open_next_folder_if_previous_image_save_failed(self):
        from ballontranslator.ui.mainwindow import MainWindow
        owner = SimpleNamespace(
            local_workspace=Mock(active_index=0), _local_pending_folder='/next',
            imsave_thread=SimpleNamespace(isBusy=lambda: False, error_count=1),
            _local_presave_errors=0, _local_save_error='disk full', _begin_local_folder=Mock(),
        )
        MainWindow.poll_local_queue(owner)
        owner._begin_local_folder.assert_not_called()
        owner.local_workspace.finish_job.assert_called_once()

    def test_project_save_failure_keeps_editor_unsaved(self):
        from qtpy.QtGui import QImage
        from ballontranslator.ui.mainwindow import MainWindow
        from ballontranslator.utils.config import pcfg
        path = Path(self.tmp.name) / 'result'
        canvas = Mock()
        canvas.render_result_img.return_value = QImage(1, 1, QImage.Format.Format_RGB32)
        owner = SimpleNamespace(
            canvas=canvas, imsave_thread=Mock(),
            imgtrans_proj=SimpleNamespace(img_valid=True, current_img='page.png',
                save=Mock(side_effect=OSError('disk full')), result_dir=lambda: str(path),
                get_result_path=lambda _: str(path / 'page.png'), current_has_alpha=lambda: False),
            rightComicTransStackPanel=Mock(isHidden=Mock(return_value=False)),
            st_manager=Mock(txtblkShapeControl=Mock(isVisible=Mock(return_value=False))),
        )
        with patch.object(pcfg, 'imgtrans_textblock', False):
            self.assertFalse(MainWindow.saveCurrentPage(owner, update_scene_text=False, save_proj=True))
        canvas.setProjSaveState.assert_called_once_with(True)
        canvas.update_saved_undostep.assert_not_called()

    def test_paused_rerender_keeps_resave_and_replaces_remaining_stale_output(self):
        from ballontranslator.ui.mainwindow import MainWindow
        folder = Path(self.tmp.name) / 'comic'
        result = folder / 'result'
        result.mkdir(parents=True)
        pages = ['first.png', 'second.png']
        for page in pages:
            (result / page).write_text('stale', encoding='utf-8')
        self.dialog.queue.add([str(folder)])
        self.dialog.queue.jobs[0]['resave'] = True
        self.dialog.start_queue()
        project = SimpleNamespace(
            pages=dict.fromkeys(pages, []), is_empty=False,
            get_page_progress=Mock(return_value=True), failed_pages=Mock(return_value=[]),
            get_result_path=lambda page: str(result / page), save=Mock(),
        )
        page_list = SimpleNamespace(current=0)
        page_list.setCurrentRow = lambda index: setattr(page_list, 'current', index)
        rendered = []

        def save_result(*args):
            page = pages[page_list.current]
            (result / page).write_text('updated', encoding='utf-8')
            rendered.append(page)
            return True

        owner = SimpleNamespace(
            local_workspace=self.dialog, imgtrans_proj=project,
            imsave_thread=SimpleNamespace(error_count=0, isBusy=lambda: False),
            openDir=Mock(return_value=True), pageList=page_list,
            saveCurrentPage=save_result, _local_pending_folder=None,
            _local_job_finished=False, _local_pipeline_busy=lambda: False,
            module_manager=Mock(), _local_timer=Mock(),
        )
        owner._resume_local_pipeline = lambda: MainWindow._resume_local_pipeline(owner)
        MainWindow._begin_local_folder(owner, str(folder))
        MainWindow.poll_local_queue(owner)
        self.assertEqual(rendered, ['first.png'])
        self.dialog.request_pause()
        MainWindow.pause_local_queue(owner)
        MainWindow.poll_local_queue(owner)
        job = self.dialog.queue.jobs[0]
        self.assertEqual(job['state'], 'paused')
        self.assertTrue(job['resave'])
        self.assertEqual((result / 'second.png').read_text(encoding='utf-8'), 'stale')
        restored = LocalJobQueue(self.dialog.queue.path)
        self.assertTrue(restored.jobs[0]['resave'])

        self.dialog.start_queue()
        owner._local_job_finished = False
        MainWindow._begin_local_folder(owner, str(folder))
        for _ in range(4):
            MainWindow.poll_local_queue(owner)
        self.assertEqual((result / 'second.png').read_text(encoding='utf-8'), 'updated')
        self.assertEqual(self.dialog.queue.jobs[0]['state'], 'done')
        self.assertFalse(self.dialog.queue.jobs[0]['resave'])


if __name__ == '__main__':
    unittest.main()
