import copy
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QRectF
from qtpy.QtGui import QFont, QFontInfo, QFontMetricsF, QTextBlockFormat, QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager, TextPanel
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import group_output, TextAlignment
from ballontranslator.utils.text_effects import StrokeEffect, TextEffectStack


class KoreanBalloonLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.font_family = '맑은 고딕' if sys.platform == 'win32' else 'DejaVu Sans'
        self.center = (150, 150)
        self.image = np.full((300, 300, 3), 255, np.uint8)
        cv2.ellipse(self.image, (150, 150), (90, 120), 0, 0, 360, (0, 0, 0), 3)
        Image.fromarray(self.image).save(os.path.join(self.directory.name, 'page.png'))
        self.project = ProjImgTrans(self.directory.name)
        self.project.set_current_img('page.png')
        self.host = QWidget()
        self.canvas = Canvas(self.host)
        self.canvas.editor_index = 1
        self.canvas.imgtrans_proj = self.project
        self.addCleanup(setattr, C, 'active_format', C.active_format)
        self.addCleanup(setattr, C.pcfg, 'module', copy.deepcopy(C.pcfg.module))
        self.addCleanup(setattr, C.pcfg, 'let_autolayout_flag', C.pcfg.let_autolayout_flag)
        self.addCleanup(setattr, C.pcfg, 'let_fntsize_flag', C.pcfg.let_fntsize_flag)
        C.pcfg.module.translate_source = '日本語'
        C.pcfg.module.translate_target = '한국어'
        C.pcfg.let_autolayout_flag = True
        C.pcfg.let_fntsize_flag = 0
        canvas_patch = patch.object(SW, 'canvas', self.canvas)
        canvas_patch.start()
        self.addCleanup(canvas_patch.stop)
        with patch.object(shared, 'register_view_widget', create=True):
            self.panel = TextPanel(self.app, self.host)
        self.panel.formatpanel.global_format = FontFormat()
        self.manager = SceneTextManager(self.app, self.host, self.canvas, self.panel, parent=self.host)
        self.manager.auto_textlayout_flag = True

    def tearDown(self):
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.clearSceneTextitems()
        self.host.close()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def _add_dialogue(self, text, *, alignment=None, angle=0, small_source=False, font_size=None):
        cx, cy = self.center
        half_height = 15 if small_source else 65
        lines = np.array([[[cx - 15, cy - half_height], [cx + 15, cy - half_height], [cx + 15, cy + half_height], [cx - 15, cy + half_height]]])
        height, width = self.project.img_array.shape[:2]
        block = group_output([], lines, width, height, np.full((height, width), 255, np.uint8))[0]
        MainWindow.postprocess_translations(None, [block])
        if alignment is not None:
            block.alignment = alignment
        block.angle = angle
        if font_size is not None:
            block.font_size = font_size
        block.font_family = self.font_family
        block.translation = text
        self.project.pages['page.png'] = [block]
        return self.manager.addTextBlock(block)

    def test_long_dialogue_fits_both_width_and_height_without_losing_text(self):
        source = '한국어 대사를 길게 번역해도 말풍선 안에 자연스럽게 들어가야 합니다. 글자가 넘치지 않도록 확인합니다.'
        # Success needs a genuinely large enough balloon for the platform's
        # chosen font. Measure that premise with ordinary Qt document layout,
        # independently of our fitter; DejaVu fallback differs on Windows.
        font = QFont(self.font_family)
        font.setPointSizeF(30 * 72 / shared.LDPI * .6)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 115)
        metrics = QFontMetricsF(font)
        reference = QTextDocument()
        reference.setDefaultFont(font)
        reference.setDocumentMargin(0)
        reference.setPlainText(source)
        reference.setTextWidth(math.ceil(metrics.horizontalAdvance('한국어 대사를 길게')))
        cursor = QTextCursor(reference)
        cursor.select(QTextCursor.SelectionType.Document)
        block_format = QTextBlockFormat()
        proportional = QTextBlockFormat.LineHeightTypes.ProportionalHeight
        block_format.setLineHeight(120, int(getattr(proportional, 'value', proportional)))
        cursor.mergeBlockFormat(block_format)
        needed = reference.size()
        axes = (math.ceil(needed.width() * .85 + 8), max(85, math.ceil(needed.height() * .85 + 8)))
        self.center = (axes[0] + 24, axes[1] + 24)
        image = np.full((self.center[1] * 2, self.center[0] * 2, 3), 255, np.uint8)
        cv2.ellipse(image, self.center, axes, 0, 0, 360, (0, 0, 0), 3)
        self.project.img_array = image
        balloon = QRectF(24, 24, axes[0] * 2, axes[1] * 2)
        item = self._add_dialogue(source)
        diagnostic = f'font={QFontInfo(item.font()).family()!r}, size={item.font().pointSizeF()}, reference={needed.width()}x{needed.height()}, balloon={balloon}'
        rect = item.mapRectToParent(item.boundingRect())
        self.assertTrue(balloon.contains(rect), diagnostic)
        self.assertEqual(''.join(item.toPlainText().split()), ''.join(source.split()))
        self.assertEqual(self.manager.auto_layout_failed, [], diagnostic)
        self.assertTrue(self.manager.korean_layout_fits(item), diagnostic)
        self.assertLessEqual(item.font().pointSizeF(), 30 * 72 / shared.LDPI)

    def test_short_dialogue_stays_centered_and_readable(self):
        item = self._add_dialogue('너 지금 뭐 하는 거야?')
        rect = item.absBoundingRect(qrect=True)
        self.assertLess(abs(rect.center().x() - 150), 2)
        self.assertLess(abs(rect.center().y() - 150), 2)
        self.assertGreaterEqual(item.font().pointSizeF(), 8)
        self.assertTrue(self.manager.korean_layout_fits(item))

    def test_unfit_dialogue_is_preserved_and_marked_for_review(self):
        source = '말풍선에 넣기에는 너무 긴 문장입니다. ' * 100
        item = self._add_dialogue(source)
        self.assertEqual(item.toPlainText(), source)
        self.assertEqual(self.manager.auto_layout_failed, [0])
        self.assertFalse(self.manager.korean_layout_fits(item))

    def test_validator_rejects_text_moved_outside_without_mutating_it(self):
        item = self._add_dialogue('안녕하세요.')
        item.setPos(-200, -200)
        before = (item.toPlainText(), item.pos(), item.font().pointSizeF())
        self.assertFalse(self.manager.korean_layout_fits(item))
        self.assertEqual((item.toPlainText(), item.pos(), item.font().pointSizeF()), before)

    def test_missing_balloon_preserves_text_for_review(self):
        self.project.img_array[:] = 0
        source = '말풍선 영역을 확인해 주세요.'
        item = self._add_dialogue(source)
        self.assertEqual(item.toPlainText(), source)
        self.assertEqual(self.manager.auto_layout_failed, [0])

    def test_fixed_size_is_not_silently_shrunk(self):
        C.pcfg.let_fntsize_flag = 1
        source = '고정 크기로는 말풍선에 들어가지 않는 긴 대사입니다. ' * 5
        item = self._add_dialogue(source)
        self.assertAlmostEqual(item.blk.font_size, 30)
        self.assertEqual(item.toPlainText(), source)
        self.assertEqual(self.manager.auto_layout_failed, [0])

    def test_explicit_blank_line_and_authored_alignment_survive(self):
        source = '처음입니다.\n\n다음입니다.'
        item = self._add_dialogue(source, alignment=TextAlignment.Right)
        self.assertIn('\n\n', item.toPlainText())
        self.assertEqual(item.blk.alignment, TextAlignment.Right)
        self.assertTrue(self.manager.korean_layout_fits(item))

    def test_rotated_source_requires_review_without_changing_angle(self):
        source = '기울어진 대사입니다.'
        item = self._add_dialogue(source, angle=20)
        self.assertEqual(item.toPlainText(), source)
        self.assertEqual(item.blk.angle, 20)
        self.assertEqual(self.manager.auto_layout_failed, [0])
        self.assertFalse(self.manager.korean_layout_fits(item))

    def test_plain_white_background_is_not_a_detected_balloon(self):
        self.project.img_array[:] = 255
        item = self._add_dialogue('말풍선이 없습니다.')
        self.assertEqual(self.manager.auto_layout_failed, [0])
        self.assertFalse(self.manager.korean_layout_fits(item))

    def test_text_on_white_background_without_outline_requires_review(self):
        self.project.img_array[:] = 255
        cv2.putText(self.project.img_array, 'TEXT', (130, 150), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)
        item = self._add_dialogue('말풍선 없는 이미지입니다.')
        self.assertEqual(self.manager.auto_layout_failed, [0])
        self.assertFalse(self.manager.korean_layout_fits(item))

    def test_long_unbroken_word_is_split_without_dropping_graphemes(self):
        source = '정말정말정말정말정말정말정말정말정말정말정말정말!'
        item = self._add_dialogue(source)
        self.assertEqual(item.toPlainText().replace('\n', ''), source)
        self.assertTrue(self.manager.korean_layout_fits(item))

    def test_validator_includes_visible_outline_beyond_logical_box(self):
        item = self._add_dialogue('안녕하세요.')
        fmt = item.get_fontformat()
        fmt.text_effects = TextEffectStack(effects=(StrokeEffect(width=3.0),))
        item.set_fontformat(fmt)
        self.assertFalse(self.manager.korean_layout_fits(item))

    def test_validation_uses_image_coordinates_at_nondefault_zoom(self):
        item = self._add_dialogue('화면을 확대해도 같습니다.')
        self.canvas._set_scene_scale(1.75)
        self.assertTrue(self.manager.korean_layout_fits(item))

    def test_small_original_dialogue_finds_its_larger_enclosing_balloon(self):
        item = self._add_dialogue('응.', small_source=True)
        self.assertEqual(self.manager.auto_layout_failed, [])
        self.assertTrue(self.manager.korean_layout_fits(item))

    def test_readability_floor_is_relative_to_original_letter_size(self):
        source = '이 대사는 원래 글자보다 지나치게 작게 줄이지 않습니다.'
        item = self._add_dialogue(source, font_size=100)
        self.assertEqual(self.manager.auto_layout_failed, [0])
        self.assertEqual(item.toPlainText(), source)

