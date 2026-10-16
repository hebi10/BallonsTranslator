import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qtpy.QtGui import QTextCursor
from qtpy.QtTest import QTest

from tests import test_canvas_save_state as save_fixture
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.region_rework import RegionReworkResult
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


class RegionReworkWindowTests(unittest.TestCase):
    setUpClass = save_fixture.CanvasSaveStateTests.__dict__['setUpClass']
    tearDown = save_fixture.CanvasSaveStateTests.tearDown
    _save_page = save_fixture.CanvasSaveStateTests._save_page

    def setUp(self) -> None:
        save_fixture.CanvasSaveStateTests.setUp(self)
        self.project.inpainted_array = self.project.img_array.copy()
        self.project.mask_array = np.zeros(self.project.img_array.shape[:2], np.uint8)
        self.host.imgtrans_proj = self.project
        self.host.st_manager = self.manager
        self.host.textPanel = self.panel
        self.host.module_manager = Mock()
        self.host.module_manager.regionReworkBusy.return_value = False
        self.host.module_manager.runRegionRework.return_value = True
        self.host.imsave_thread = Mock()
        self.host.imsave_thread.isBusy.return_value = False
        self.host.local_workspace = None
        self.host._local_pipeline_busy = Mock(return_value=False)
        self.host.drawingPanel = Mock()
        self.host.drawingPanel.rectPanel.dilate_slider.value.return_value = 2
        self.host.bottomBar = Mock()
        self.host.setTextEditMode = Mock()
        self.host.llmContextEditor = Mock()
        self.host.global_search_widget = Mock()
        self.host._region_rework_pending = None
        self.host._region_rework_revision = 0
        for name in ('on_region_rework_requested', 'on_region_rework_finished',
                     'on_region_rework_failed', '_region_scene_fingerprint',
                     '_region_history_changed', '_prepare_region_blocks'):
            setattr(self.host, name, types.MethodType(getattr(MainWindow, name), self.host))
        self.canvas.textstack_changed.connect(self.host._region_history_changed)
        self.canvas.draw_undo_stack.indexChanged.connect(self.host._region_history_changed)
        info = patch('ballontranslator.ui.mainwindow.create_info_dialog')
        self.info = info.start()
        self.addCleanup(info.stop)
        local = patch('ballontranslator.ui.mainwindow.is_local_edition', return_value=True)
        local.start()
        self.addCleanup(local.stop)

    def _request(self, rect=(0, 0, 330, 160)):
        self.host.on_region_rework_requested(rect)
        return self.host.module_manager.runRegionRework.call_args.args[0]

    @staticmethod
    def _block() -> TextBlock:
        return TextBlock([40, 30, 150, 100], _bounding_rect=[40, 30, 110, 70],
                         translation='new dialogue', text=['new source'],
                         fontformat=FontFormat(font_family='DejaVu Sans', font_size=16))

    def _result(self, request) -> RegionReworkResult:
        x1, y1, x2, y2 = request.rect
        return RegionReworkResult(request.rect, [self._block()],
                                  np.full((y2-y1, x2-x1, 3), 37, np.uint8),
                                  np.full((y2-y1, x2-x1), 255, np.uint8))

    def test_partial_source_or_render_intersection_rejects_before_work(self) -> None:
        for rect in ((100, 0, 330, 160), (0, 0, 250, 160)):
            with self.subTest(rect=rect):
                self.host.on_region_rework_requested(rect)
                self.host.module_manager.runRegionRework.assert_not_called()
        self.item.blk.xyxy = [350, 150, 380, 190]
        self.host.on_region_rework_requested((0, 0, 330, 160))
        self.host.module_manager.runRegionRework.assert_not_called()

    def test_request_snapshots_original_pixels_and_live_editor_values(self) -> None:
        edit = self.manager.pairwidget_list[0].e_trans
        edit.setFocus()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, ' latest')
        self.project.inpainted_array[:] = 123
        request = self._request((330, 160, -50, -20))
        self.assertEqual(request.rect, (0, 0, 330, 160))
        self.assertEqual(request.replaced_indices, (0,))
        self.assertEqual(request.project.pages['01.png'][0].translation, 'Hello\nworld latest')
        self.assertIsNot(request.project.pages, self.project.pages)
        self.assertIsNot(request.project.pages['01.png'][0], self.item.blk)
        self.assertTrue(np.all(request.image == 255))
        self.assertTrue(np.all(request.inpainted == 123))
        self.project.img_array[:] = 19
        self.assertTrue(np.all(request.image == 255))

    def test_scene_change_and_undone_history_discard_completed_result(self) -> None:
        request = self._request()
        edit = self.manager.pairwidget_list[0].e_trans
        edit.setFocus()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, ' change')
        self.canvas.undo_textedit()
        before = self.project.inpainted_array.copy()
        self.host.on_region_rework_finished(self._result(request))
        self.assertEqual(self.manager.textblk_item_list, [self.item])
        np.testing.assert_array_equal(self.project.inpainted_array, before)
        self.assertIsNone(self.host._region_rework_pending)

    def test_success_is_one_undo_and_keeps_unselected_items(self) -> None:
        outside = self.manager.addTextBlock(TextBlock(
            [350, 160, 395, 195], _bounding_rect=[350, 160, 45, 35],
            translation='Z', text=['other'],
            fontformat=FontFormat(font_family='DejaVu Sans', font_size=8)))
        request = self._request()
        old_image = self.project.inpainted_array.copy()
        old_html = outside.toHtml()
        def layout(item, text):
            item.setPlainText(text)
            return True
        with patch.object(self.manager, 'layout_textblk', side_effect=layout):
            self.host.on_region_rework_finished(self._result(request))
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertEqual(self.manager.textblk_item_list[0].toPlainText(), 'new dialogue')
        self.assertIs(self.manager.textblk_item_list[1], outside)
        self.assertEqual(outside.toHtml(), old_html)
        self.assertTrue(np.all(self.project.inpainted_array[:160, :330] == 37))
        self.canvas.undo_textedit()
        self.assertEqual(self.manager.textblk_item_list, [self.item, outside])
        np.testing.assert_array_equal(self.project.inpainted_array, old_image)

    def test_unfitted_background_caption_applies_with_review_notice(self) -> None:
        request = self._request()
        def layout(item, text):
            item.setPlainText(text)
            return False
        with patch.object(self.manager, 'layout_textblk', side_effect=layout):
            self.host.on_region_rework_finished(self._result(request))
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertIn('직접 확인', self.info.call_args.args[0])

    def test_actual_korean_layout_keeps_short_background_caption(self) -> None:
        request = self._request()
        with patch.object(pcfg.module, 'translate_target', '한국어'):
            self.host.on_region_rework_finished(self._result(request))
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertEqual(self.manager.textblk_item_list[0].toPlainText(), 'new dialogue')
        self.assertIn('직접 확인', self.info.call_args.args[0])

    def test_actual_korean_layout_rejects_long_background_caption(self) -> None:
        request = self._request()
        result = self._result(request)
        result.blocks[0].translation = 'long dialogue ' * 100
        with patch.object(pcfg.module, 'translate_target', '한국어'):
            self.host.on_region_rework_finished(result)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        self.assertEqual(self.manager.textblk_item_list, [self.item])
        self.assertTrue(np.all(self.project.inpainted_array == 255))

    def test_untracked_source_edit_and_mismatched_rect_discard_results(self) -> None:
        request = self._request()
        self.item.blk.xyxy = [1, 1, 250, 100]
        self.host.on_region_rework_finished(self._result(request))
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        request = self._request()
        result = self._result(request)
        result.rect = (1, 0, 330, 160)
        self.host.on_region_rework_finished(result)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_layout_failure_or_overflow_leaves_entire_job_unchanged(self) -> None:
        for outcome in (False, True):
            with self.subTest(outcome=outcome):
                request = self._request()
                result = self._result(request)
                before = self.project.inpainted_array.copy()
                def layout(item, text):
                    item.setPlainText(text)
                    if outcome:
                        item.setPos(500, 500)
                    else:
                        raise ValueError('layout failed')
                    return outcome
                with patch.object(self.manager, 'layout_textblk', side_effect=layout):
                    self.host.on_region_rework_finished(result)
                self.assertEqual(self.canvas.text_undo_stack.count(), 0)
                self.assertEqual(self.manager.textblk_item_list, [self.item])
                np.testing.assert_array_equal(self.project.inpainted_array, before)

    def test_synchronous_prepare_failure_clears_pending_state(self) -> None:
        def reject(request):
            self.host.on_region_rework_failed('준비가 취소되었습니다.')
            return True
        self.host.module_manager.runRegionRework.side_effect = reject
        self._request()
        self.assertIsNone(self.host._region_rework_pending)


if __name__ == '__main__':
    unittest.main()
