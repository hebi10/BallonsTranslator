import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawing_commands import StrokeItemUndoCommand
from ballontranslator.ui.io_thread import ImgSaveThread
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.editing.manager import (
    RegionReworkCommand, SceneTextManager, TextPanel,
)
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class RegionReworkUndoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Image.new('RGB', (400, 200), 'white').save(
            os.path.join(self.directory.name, '01.png')
        )
        self.project = ProjImgTrans(self.directory.name)
        self.project.set_current_img('01.png')
        self.project.inpainted_array = np.full((200, 400, 3), 255, np.uint8)
        self.project.mask_array = np.zeros((200, 400), np.uint8)
        self.host = QWidget()
        self.canvas = Canvas(self.host)
        self.canvas.editor_index = 1
        self.canvas.imgtrans_proj = self.project
        canvas_patch = patch.object(SW, 'canvas', self.canvas)
        canvas_patch.start()
        self.addCleanup(canvas_patch.stop)
        self.addCleanup(setattr, C, 'active_format', C.active_format)
        with patch.object(shared, 'register_view_widget', create=True):
            self.panel = TextPanel(self.app, self.host)
        self.panel.formatpanel.global_format = FontFormat()
        self.manager = SceneTextManager(
            self.app, self.host, self.canvas, self.panel, parent=self.host,
        )
        self.originals = [
            self.manager.addTextBlock(self._block(f'old {index}', index * 70))
            for index in range(5)
        ]
        self.pairs = list(self.manager.pairwidget_list)
        self.manager.updateTextBlkList()
        self.block_list = self.project.current_block_list()
        self.canvas.search_widget.pairwidget_list = self.manager.pairwidget_list
        self.canvas.search_widget.textblk_item_list = self.manager.textblk_item_list
        self.manager.new_textblk.connect(self.canvas.search_widget.on_new_textblk)
        layout = QHBoxLayout(self.host)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.host.show()
        self.host.activateWindow()
        self.app.processEvents()
        self.canvas.update_saved_undostep()
        self.canvas.setProjSaveState(False)

    def tearDown(self) -> None:
        self.canvas.search_widget.clearSearchResult()
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.clearSceneTextitems()
        self.host.close()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    @staticmethod
    def _block(text: str, x: int = 80) -> TextBlock:
        return TextBlock(
            [x, 20, x + 60, 100], _bounding_rect=[x, 20, 60, 80],
            translation=text, text=[f'source {text}'],
            fontformat=FontFormat(font_family='DejaVu Sans', font_size=16),
        )

    def _command(self) -> RegionReworkCommand:
        return RegionReworkCommand(
            self.manager, [self.originals[3], self.originals[1]],
            [self._block('new one'), self._block('new two', 160)],
            (70, 10, 280, 130), np.full((120, 210, 3), 37, np.uint8),
            np.full((120, 210), 255, np.uint8),
        )

    def _assert_indexes(self) -> None:
        for index, (item, pair) in enumerate(zip(
            self.manager.textblk_item_list, self.manager.pairwidget_list,
        )):
            self.assertEqual(item.idx, index)
            self.assertEqual(pair.idx, index)
            self.assertEqual(pair.e_trans.idx, index)
        self.assertIs(self.project.current_block_list(), self.block_list)
        self.assertEqual(
            [id(block) for block in self.block_list],
            [id(item.blk) for item in self.manager.textblk_item_list],
        )

    def test_roi_replacement_preserves_outside_items_and_undo_restores_instances(self) -> None:
        before_image = self.project.inpainted_array.copy()
        before_mask = self.project.mask_array.copy()
        outside_html = [self.originals[i].toHtml() for i in (0, 2, 4)]
        self.canvas.push_text_command(self._command())
        after_items = list(self.manager.textblk_item_list)
        after_pairs = list(self.manager.pairwidget_list)
        self.assertEqual([item.toPlainText() for item in after_items],
                         ['old 0', 'new one', 'new two', 'old 2', 'old 4'])
        for old_index, new_index in ((0, 0), (2, 3), (4, 4)):
            self.assertIs(after_items[new_index], self.originals[old_index])
            self.assertIs(after_pairs[new_index], self.pairs[old_index])
        self.assertEqual(outside_html, [after_items[i].toHtml() for i in (0, 3, 4)])
        outside = np.ones((200, 400), bool)
        outside[10:130, 70:280] = False
        np.testing.assert_array_equal(self.project.inpainted_array[outside], before_image[outside])
        np.testing.assert_array_equal(self.project.mask_array[outside], before_mask[outside])
        self.assertTrue(np.all(self.project.inpainted_array[10:130, 70:280] == 37))
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self._assert_indexes()
        self.canvas.undo_textedit()
        self.assertEqual(self.manager.textblk_item_list, self.originals)
        self.assertEqual(self.manager.pairwidget_list, self.pairs)
        np.testing.assert_array_equal(self.project.inpainted_array, before_image)
        np.testing.assert_array_equal(self.project.mask_array, before_mask)
        self._assert_indexes()
        self.canvas.redo_textedit()
        self.assertEqual(self.manager.textblk_item_list, after_items)
        self.assertEqual(self.manager.pairwidget_list, after_pairs)
        self._assert_indexes()

    def test_prior_and_subsequent_text_edits_keep_their_original_undo_owners(self) -> None:
        edit = self.pairs[1].e_trans
        edit.setFocus()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, ' appended')
        self.assertEqual(self.originals[1].toPlainText(), 'old 1 appended')
        prior_count = self.canvas.text_undo_stack.count()
        self.canvas.push_text_command(self._command())
        replacement = self.manager.textblk_item_list[1]
        new_edit = self.manager.pairwidget_list[1].e_trans
        new_edit.setFocus()
        new_edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(new_edit, ' changed')
        self.assertEqual(replacement.toPlainText(), 'new one changed')
        self.canvas.undo_textedit()
        self.assertEqual(replacement.toPlainText(), 'new one')
        self.canvas.undo_textedit()
        self.assertIs(self.manager.textblk_item_list[1], self.originals[1])
        self.assertEqual(edit.toPlainText(), 'old 1 appended')
        self.assertEqual(self.canvas.text_undo_stack.index(), prior_count)
        self.canvas.undo_textedit()
        self.assertEqual(self.originals[1].toPlainText(), 'old 1')
        self.canvas.redo_textedit()
        self.canvas.redo_textedit()
        self.canvas.redo_textedit()
        self.assertIs(self.manager.textblk_item_list[1], replacement)
        self.assertEqual(replacement.toPlainText(), 'new one changed')

    def test_saved_pixels_and_blocks_round_trip_and_replay_remains_dirty(self) -> None:
        self.canvas.push_text_command(self._command())
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)
        self.manager.updateTextBlkList()
        self.project.save()
        self.project.save_mask('01.png', self.project.mask_array)
        self.project.save_inpainted('01.png', self.project.inpainted_array)
        reopened = ProjImgTrans(self.directory.name)
        reopened.set_current_img('01.png')
        self.assertEqual([block.translation for block in reopened.current_block_list()],
                         ['old 0', 'new one', 'new two', 'old 2', 'old 4'])
        np.testing.assert_array_equal(reopened.inpainted_array, self.project.inpainted_array)
        np.testing.assert_array_equal(reopened.mask_array, self.project.mask_array)
        self.canvas.update_saved_undostep()
        self.canvas.setProjSaveState(False)
        self.canvas.undo_textedit()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)
        self.canvas.clear_draw_stack()
        self.canvas.redo_textedit()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)

    def test_search_never_keeps_detached_editors(self) -> None:
        search = self.canvas.search_widget
        search.search_editor.setPlainText('old')
        search.show()
        search.page_search()
        self.assertIn(self.pairs[1].e_trans, search.search_rstedit_list)
        self.canvas.push_text_command(self._command())
        self.assertNotIn(self.pairs[1].e_trans, search.search_rstedit_list)
        self.assertNotIn(self.pairs[3].e_trans, search.search_rstedit_list)
        self.assertEqual(search.counter_sum, 3)
        self.canvas.undo_textedit()
        self.assertIn(self.pairs[1].e_trans, search.search_rstedit_list)
        self.assertEqual(search.counter_sum, 5)

    def test_pending_edits_resolve_before_capture_and_replay_skips_layout(self) -> None:
        def finish_pending() -> None:
            self.project.inpainted_array[10:130, 70:280] = 99
        with patch.object(self.manager.formatpanel,
                          'resolve_text_transform_edits_for_history_change',
                          side_effect=finish_pending):
            command = self._command()
        self.manager.auto_textlayout_flag = True
        with patch.object(self.manager, 'layout_textblk', side_effect=AssertionError('layout replay')):
            self.canvas.push_text_command(command)
            self.canvas.undo_textedit()
            self.assertTrue(np.all(self.project.inpainted_array[10:130, 70:280] == 99))
            self.canvas.redo_textedit()

    def test_empty_or_invalid_requests_do_not_change_scene_or_history(self) -> None:
        for rect, image, mask in (
            ((0, 0, 0, 10), np.zeros((10, 0, 3), np.uint8), np.zeros((10, 0), np.uint8)),
            ((0, 0, 10, 10), np.zeros((11, 10, 3), np.uint8), np.zeros((10, 10), np.uint8)),
        ):
            with self.subTest(rect=rect), self.assertRaises(ValueError):
                RegionReworkCommand(self.manager, [self.originals[1]],
                                    [self._block('new')], rect, image, mask)
        with self.assertRaises(ValueError):
            RegionReworkCommand(self.manager, [], [], (0, 0, 10, 10),
                                np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))
        self.assertEqual(self.manager.textblk_item_list, self.originals)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_newly_detected_region_appends_without_dirtying_unchanged_pixels(self) -> None:
        self.canvas.push_text_command(RegionReworkCommand(
            self.manager, [], [self._block('found')], (0, 0, 10, 10),
            np.full((10, 10, 3), 255, np.uint8), np.zeros((10, 10), np.uint8),
        ))
        self.assertEqual(self.manager.textblk_item_list[:-1], self.originals)
        self.assertEqual(self.manager.textblk_item_list[-1].toPlainText(), 'found')
        self.assertFalse(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.text_change_unsaved())
        self._assert_indexes()
        self.canvas.undo_textedit()
        self.assertEqual(self.manager.textblk_item_list, self.originals)
        self.assertFalse(self.canvas.projstate_unsaved)
        self._assert_indexes()

    def test_merging_noncontiguous_blocks_restores_original_order(self) -> None:
        self.canvas.push_text_command(RegionReworkCommand(
            self.manager, [self.originals[3], self.originals[1]],
            [self._block('combined')], (70, 10, 280, 130),
            np.full((120, 210, 3), 37, np.uint8), np.full((120, 210), 255, np.uint8),
        ))
        self.assertEqual([item.toPlainText() for item in self.manager.textblk_item_list],
                         ['old 0', 'combined', 'old 2', 'old 4'])
        self._assert_indexes()
        self.canvas.undo_textedit()
        self.assertEqual(self.manager.textblk_item_list, self.originals)
        self.assertEqual(self.manager.pairwidget_list, self.pairs)
        self._assert_indexes()

    def _add_crossing_stroke(self) -> StrokeItemUndoCommand:
        self.canvas.updateCanvas()
        image = QImage(280, 10, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor('red'))
        command = StrokeItemUndoCommand(self.canvas.drawingLayer, (40, 110, 320, 120), image)
        self.canvas.push_draw_command(command)
        return command

    def test_region_replay_clears_crossing_stroke_only_inside_and_keeps_later_strokes(self) -> None:
        stroke = self._add_crossing_stroke()
        before = pixmap2ndarray(self.canvas.render_result_img())
        np.testing.assert_array_equal(before[115, 100, :3], (255, 0, 0))
        self.canvas.push_text_command(self._command())
        after = pixmap2ndarray(self.canvas.render_result_img())
        np.testing.assert_array_equal(after[115, 100, :3], (37, 37, 37))
        outside = np.ones((200, 400), bool)
        outside[10:130, 70:280] = False
        np.testing.assert_array_equal(after[outside], before[outside])
        self.assertIs(self.canvas.draw_undo_stack.command(0), stroke)
        self.canvas.undo_textedit()
        np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), before)
        self.canvas.redo_textedit()
        np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), after)

        image = QImage(10, 10, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor('blue'))
        later = StrokeItemUndoCommand(self.canvas.drawingLayer, (100, 110, 110, 120), image)
        self.canvas.push_draw_command(later)
        for redo in (False, True):
            if redo:
                self.canvas.redo_textedit()
            else:
                self.canvas.undo_textedit()
            rendered = pixmap2ndarray(self.canvas.render_result_img())
            np.testing.assert_array_equal(rendered[115, 105, :3], (0, 0, 255))
            np.testing.assert_array_equal(rendered[115, 50, :3], (255, 0, 0))
        self.canvas.editor_index = 0
        self.canvas.undo()
        np.testing.assert_array_equal(
            pixmap2ndarray(self.canvas.render_result_img())[115, 105, :3], (37, 37, 37),
        )

    def test_drawing_only_region_change_is_dirty_and_saved_without_old_stroke(self) -> None:
        self._add_crossing_stroke()
        self.canvas.update_saved_undostep()
        self.canvas.setProjSaveState(False)
        self.canvas.push_text_command(RegionReworkCommand(
            self.manager, [self.originals[1]], [self._block('new')],
            (70, 10, 280, 130), np.full((120, 210, 3), 255, np.uint8),
            np.zeros((120, 210), np.uint8),
        ))
        self.assertTrue(self.canvas.draw_change_unsaved())
        writer = ImgSaveThread()
        writer.job = writer._save_img
        owner = SimpleNamespace(
            canvas=self.canvas, imgtrans_proj=self.project, st_manager=self.manager,
            rightComicTransStackPanel=self.host, imsave_thread=writer,
        )
        with patch.object(C.pcfg, 'imgtrans_textblock', False):
            self.assertTrue(MainWindow.saveCurrentPage(owner, check_local_layout=False))
        writer._save_img()
        reopened = ProjImgTrans(self.directory.name)
        reopened.set_current_img('01.png')
        np.testing.assert_array_equal(reopened.inpainted_array[115, 100], (255, 255, 255))
        np.testing.assert_array_equal(reopened.inpainted_array[115, 50], (255, 0, 0))
        self.assertFalse(self.canvas.projstate_unsaved)
        self.canvas.undo_textedit()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)

    def test_earlier_drawing_undo_redo_stays_below_active_region_rework(self) -> None:
        stroke = self._add_crossing_stroke()
        self.canvas.push_text_command(self._command())
        after = pixmap2ndarray(self.canvas.render_result_img())
        self.canvas.editor_index = 0
        self.canvas.undo()
        self.assertNotIn(stroke.key, self.canvas.drawingLayer.qimg_dict)
        self.canvas.redo()
        np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), after)
        self.canvas.undo_textedit()
        restored = pixmap2ndarray(self.canvas.render_result_img())
        np.testing.assert_array_equal(restored[115, 100, :3], (255, 0, 0))
        self.canvas.redo_textedit()
        np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), after)

    def test_page_clear_resets_remembered_drawing_composition_order(self) -> None:
        self.canvas.updateCanvas()
        layer = self.canvas.drawingLayer
        mode = QPainter.CompositionMode.CompositionMode_SourceOver
        red = QImage(10, 10, QImage.Format.Format_ARGB32_Premultiplied)
        red.fill(QColor('red'))
        blue = QImage(red)
        blue.fill(QColor('blue'))
        layer.addQImage(100, 110, red, mode, 'first')
        layer.addQImage(100, 110, blue, mode, 'second')
        layer.removeQImage('first')
        layer.addQImage(100, 110, red, mode, 'first')
        np.testing.assert_array_equal(
            pixmap2ndarray(self.canvas.render_result_img())[115, 105, :3], (0, 0, 255),
        )
        layer.clearAllDrawings()
        self.assertFalse(layer.drawed())
        layer.addQImage(100, 110, blue, mode, 'second')
        layer.addQImage(100, 110, red, mode, 'first')
        np.testing.assert_array_equal(
            pixmap2ndarray(self.canvas.render_result_img())[115, 105, :3], (255, 0, 0),
        )


if __name__ == '__main__':
    unittest.main()
