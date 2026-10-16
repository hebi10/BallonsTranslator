import copy
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.custom_widget import ImgtransProgressMessageBox
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class RegionLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.project = ProjImgTrans()
        self.project.current_img = '01.png'
        self.project.pages = {'01.png': []}
        self.project.img_array = np.full((20, 30, 3), 200, np.uint8)
        self.project.inpainted_array = self.project.img_array.copy()
        self.project.mask_array = np.zeros((20, 30), np.uint8)
        self.window = QWidget()
        self.config = ConfigPanel(self.window)
        self.progress = ImgtransProgressMessageBox(self.window)
        self.manager = ModuleManager(self.project)
        self.manager.setupThread(self.config, self.progress, self.window)
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        self.manager.imgtrans_thread.wait(3000)
        self.app.processEvents()
        self.manager.deleteLater()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def request(self):
        snapshot = copy.deepcopy(self.project)
        return SimpleNamespace(rect=(2, 3, 15, 16), image=snapshot.img_array,
                               inpainted=snapshot.inpainted_array, mask=snapshot.mask_array,
                               page_key='01.png', project=snapshot, replaced_indices=(), dilate=0)

    def assert_feature(self):
        self.assertTrue(hasattr(self.manager, 'runRegionRework'))

    def wait_until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            self.assertLess(time.monotonic(), deadline)
            QTest.qWait(5)

    def test_changed_source_during_preparation_discards_request(self):
        self.assert_feature()
        failed = QSignalSpy(self.manager.region_rework_failed)
        with patch.object(self.manager, '_prepare_modules_then'):
            self.assertTrue(self.manager.runRegionRework(self.request()))
        self.project.inpainted_array[5, 6] = 20
        self.manager._start_region_rework()
        self.assertEqual(len(failed), 1)
        self.assertFalse(self.manager.regionReworkBusy())
        self.assertFalse(self.manager.imgtrans_thread.isRunning())

    def test_second_request_and_page_pipeline_are_refused_while_preparing(self):
        self.assert_feature()
        with patch.object(self.manager, '_prepare_modules_then') as prepare:
            self.assertTrue(self.manager.runRegionRework(self.request()))
            self.assertFalse(self.manager.runRegionRework(self.request()))
            self.manager.runImgtransPipeline()
            self.assertEqual(prepare.call_count, 1)

    def test_page_change_discards_pending_work(self):
        self.assert_feature()
        failed = QSignalSpy(self.manager.region_rework_failed)
        with patch.object(self.manager, '_prepare_modules_then'):
            self.assertTrue(self.manager.runRegionRework(self.request()))
        self.project.current_img = '02.png'
        self.manager.handle_page_changed()
        self.assertFalse(self.manager.regionReworkBusy())
        self.assertEqual(len(failed), 1)

    def test_late_worker_result_after_same_array_edit_is_rejected(self):
        self.assert_feature()
        started = threading.Event()
        completed = QSignalSpy(self.manager.region_rework_finished)
        failed = QSignalSpy(self.manager.region_rework_failed)
        full_page = QSignalSpy(self.manager.imgtrans_pipeline_finished)

        def worker(*args, **kwargs):
            started.set()
            if not self.release.wait(3):
                raise TimeoutError('Test worker not released')
            return SimpleNamespace(rect=(2, 3, 15, 16))

        with patch('ballontranslator.ui.region_rework.rework_region', side_effect=worker), \
                patch('ballontranslator.utils.local_edition.is_local_edition', return_value=False), \
                patch.object(self.manager, '_prepare_modules_then', side_effect=lambda _req, success, **kw: success()):
            self.assertTrue(self.manager.runRegionRework(self.request()))
            self.wait_until(started.is_set)
            self.project.mask_array[4, 4] = 255
            self.release.set()
            self.wait_until(lambda: not self.manager.regionReworkBusy())
        self.assertEqual(len(completed), 0)
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(full_page), 0)

    def test_preparation_callback_before_native_thread_exit_resumes_once(self):
        self.assert_feature()
        completed = QSignalSpy(self.manager.region_rework_finished)
        with patch.object(self.manager, '_prepare_modules_then'):
            self.assertTrue(self.manager.runRegionRework(self.request()))
        with patch.object(self.manager, '_module_preparation_running', return_value=True):
            self.manager._start_region_rework()
        self.assertFalse(self.manager.imgtrans_thread.isRunning())
        with patch('ballontranslator.ui.region_rework.rework_region', return_value=SimpleNamespace(rect=(2, 3, 15, 16))), \
                patch('ballontranslator.utils.local_edition.is_local_edition', return_value=False):
            self.manager.ocr_thread.finished.emit()
            self.wait_until(lambda: not self.manager.regionReworkBusy())
        self.assertEqual(len(completed), 1)

    def test_stop_during_native_work_discards_late_success(self):
        started = threading.Event()
        completed = QSignalSpy(self.manager.region_rework_finished)
        failed = QSignalSpy(self.manager.region_rework_failed)

        def worker(*args, **kwargs):
            started.set()
            if not self.release.wait(3):
                raise TimeoutError('Test worker not released')
            return SimpleNamespace(rect=(2, 3, 15, 16))

        with patch('ballontranslator.ui.region_rework.rework_region', side_effect=worker), \
                patch('ballontranslator.utils.local_edition.is_local_edition', return_value=False), \
                patch.object(self.manager, '_prepare_modules_then', side_effect=lambda _req, success, **kw: success()):
            self.assertTrue(self.manager.runRegionRework(self.request()))
            self.wait_until(started.is_set)
            self.manager.stopImgtransPipeline()
            self.assertTrue(self.manager.imgtrans_thread.isRunning())
            self.release.set()
            self.wait_until(lambda: not self.manager.regionReworkBusy())
        self.assertEqual(len(completed), 0)
        self.assertEqual(len(failed), 1)

    def test_cancel_between_prepare_callback_and_native_exit_prevents_start(self):
        for cancel in (self.manager.stopImgtransPipeline, self.manager.cancelModulePreparation):
            with self.subTest(cancel=cancel.__name__):
                completed = QSignalSpy(self.manager.region_rework_finished)
                failed = QSignalSpy(self.manager.region_rework_failed)
                with patch.object(self.manager, '_prepare_modules_then'):
                    self.assertTrue(self.manager.runRegionRework(self.request()))
                with patch.object(self.manager, '_module_preparation_running', return_value=True):
                    self.manager._start_region_rework()
                    cancel()
                with patch.object(self.manager.imgtrans_thread, 'runRegionRework') as start:
                    self.manager.ocr_thread.finished.emit()
                    self.app.processEvents()
                    start.assert_not_called()
                self.assertFalse(self.manager.regionReworkBusy())
                self.assertEqual(len(completed), 0)
                self.assertEqual(len(failed), 1)


if __name__ == '__main__':
    unittest.main()
