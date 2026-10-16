import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QRectF
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class RegionSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.draw = copy.deepcopy(pcfg.drawpanel)
        self.canvas = Canvas()
        self.canvas.editor_index = 0
        self.project = ProjImgTrans()
        self.project.img_array = np.full((120, 160, 3), 255, np.uint8)
        self.project.inpainted_array = self.project.img_array.copy()
        self.project.mask_array = np.zeros((120, 160), np.uint8)
        self.canvas.imgtrans_proj = self.project
        self.panel = DrawingPanel(self.canvas)

    def tearDown(self):
        self.panel.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.draw

    def test_selection_waits_for_explicit_rework_and_keeps_edge_pixels(self):
        self.assertTrue(hasattr(self.panel, 'begin_region_rework'))
        self.panel.begin_region_rework()
        submitted = QSignalSpy(self.panel.region_rework_requested)
        with patch.object(self.panel, 'inpaintRect', side_effect=AssertionError('Unexpected inference')):
            self.panel.on_end_create_rect(QRectF(100, 80, 90, 60), 0)
        self.assertEqual(len(submitted), 0)
        self.assertTrue(self.panel.rectPanel.rework_btn.isEnabled())
        self.panel.rectPanel.rework_btn.click()
        self.assertEqual(list(submitted[0][0]), [100, 80, 160, 120])
        self.assertFalse(self.panel.rectPanel.rework_btn.isEnabled())

    def test_cancel_and_page_change_clear_selection(self):
        self.assertTrue(hasattr(self.panel, 'begin_region_rework'))
        self.panel.begin_region_rework()
        submitted = QSignalSpy(self.panel.region_rework_requested)
        self.panel.on_end_create_rect(QRectF(10, 20, 50, 60), 0)
        self.panel.rectPanel.delete_btn.click()
        self.panel.rectPanel.rework_btn.click()
        self.assertEqual(len(submitted), 0)
        self.panel.on_end_create_rect(QRectF(10, 20, 50, 60), 0)
        self.panel.handle_page_changed()
        self.panel.rectPanel.rework_btn.click()
        self.assertEqual(len(submitted), 0)

    def test_right_drag_and_tiny_region_do_not_restore_or_submit(self):
        self.assertTrue(hasattr(self.panel, 'begin_region_rework'))
        self.panel.begin_region_rework()
        submitted = QSignalSpy(self.panel.region_rework_requested)
        before = self.project.inpainted_array.copy()
        self.panel.on_end_create_rect(QRectF(1, 1, 50, 40), 1)
        self.panel.on_end_create_rect(QRectF(1, 1, 1, 1), 0)
        self.panel.rectPanel.rework_btn.click()
        self.assertEqual(len(submitted), 0)
        np.testing.assert_array_equal(self.project.inpainted_array, before)

    def test_selection_guide_is_excluded_from_export_and_remains_available(self):
        self.canvas.baseLayer.setRect(0, 0, 160, 120)
        self.canvas.setSceneRect(0, 0, 160, 120)
        self.canvas.updateLayers()
        self.panel.begin_region_rework()
        self.panel.on_end_create_rect(QRectF(10, 20, 50, 60), 0)
        self.assertTrue(self.panel.inpaint_mask_item.isVisible())
        rendered = pixmap2ndarray(self.canvas.render_result_img(), keep_alpha=False)
        np.testing.assert_array_equal(rendered, self.project.inpainted_array)
        self.assertTrue(self.panel.inpaint_mask_item.isVisible())
        self.assertTrue(self.panel.rectPanel.rework_btn.isEnabled())


if __name__ == '__main__':
    unittest.main()
