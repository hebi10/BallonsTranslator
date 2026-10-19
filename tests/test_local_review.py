import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.local_review_dialog import LocalReviewDialog
from ballontranslator.utils.config import RunStatus
from ballontranslator.utils.local_review import (
    acknowledge_review_item, clear_review_issues, collect_review_items,
    normalize_local_review, record_review_issue,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


def make_project(*blocks, progress=RunStatus.FIN_OCR):
    project = ProjImgTrans()
    project.pages = {'01.png': list(blocks)}
    project._image_info = {'01.png': {'finish_code': progress}}
    project.local_review = normalize_local_review(None)
    return project


class LocalReviewTests(unittest.TestCase):
    def test_untouched_detected_bubbles_are_not_prematurely_flagged(self):
        project = make_project(
            TextBlock(xyxy=[10, 20, 30, 40], det_model='test'),
            TextBlock(text=['Hello'], det_model='test'),
            progress=RunStatus.FIN_DET,
        )
        self.assertEqual(collect_review_items(project), [])

    def test_processed_empty_ocr_and_missing_translation_have_correct_ids(self):
        blocks = [
            TextBlock(det_model='test'),
            TextBlock(text=['Hello']),
            TextBlock(text=['Done'], translation='완료'),
            TextBlock(),
        ]
        project = make_project(*blocks)
        issues = collect_review_items(project)
        self.assertEqual([(item['page'], item['block_index'], item['code']) for item in issues], [
            ('01.png', 0, 'ocr_empty'), ('01.png', 1, 'translation_missing'),
        ])
        self.assertEqual(collect_review_items(project), issues)

    def test_ocr_engine_evidence_and_failed_stages_create_review(self):
        block = TextBlock()
        block.ocr_engine = 'local'
        project = make_project(block, progress=0)
        self.assertEqual(collect_review_items(project)[0]['code'], 'ocr_empty')
        block.ocr_engine = ''
        block.det_model = 'test'
        project._image_info['01.png']['pipeline_failed_stage'] = 'ocr'
        self.assertEqual(collect_review_items(project)[0]['code'], 'ocr_empty')
        block.text = ['Hello']
        project._image_info['01.png']['pipeline_failed_stage'] = 'translate'
        self.assertEqual(collect_review_items(project)[0]['code'], 'translation_missing')

    def test_persisted_language_reason_is_observed_without_fake_inference(self):
        block = TextBlock(text=['123'], translation='123')
        block.language_review = '원문 언어를 확인하세요.'
        project = make_project(block, progress=0)
        issues = collect_review_items(project)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['message'], block.language_review)
        self.assertEqual(issues[0]['code'], 'language_review')
        block.language_review = ''
        self.assertEqual(collect_review_items(project), [])

    def test_acknowledgments_are_invalidated_by_source_translation_and_geometry(self):
        for field, value in (
            ('text', ['Changed']), ('translation', '변경됨'),
            ('xyxy', [1, 2, 60, 70]), ('angle', 25),
            ('_bounding_rect', [1, 2, 80, 90]),
        ):
            with self.subTest(field=field):
                block = TextBlock(text=['Hello'], translation='안녕')
                block.language_review = '원문 언어를 확인하세요.'
                project = make_project(block)
                item = collect_review_items(project)[0]
                self.assertTrue(acknowledge_review_item(project, item))
                self.assertEqual(collect_review_items(project), [])
                setattr(block, field, value)
                new_items = collect_review_items(project)
                self.assertEqual(len(new_items), 1)
                self.assertNotEqual(new_items[0]['key'], item['key'])

    def test_acknowledgment_rejects_stale_selection(self):
        block = TextBlock(text=['Hello'])
        project = make_project(block)
        item = collect_review_items(project)[0]
        block.text = ['Changed']
        self.assertFalse(acknowledge_review_item(project, item))
        self.assertEqual(len(collect_review_items(project)), 1)

    def test_page_issues_track_page_contents_and_rework_revision(self):
        project = make_project(TextBlock(text=['Hi'], translation='안녕'))
        self.assertEqual(collect_review_items(project), [])
        record_review_issue(project, '01.png', 'background_review', '복원한 배경을 직접 확인하세요.', revision='rework-1')
        item = collect_review_items(project)[0]
        self.assertIsNone(item['block_index'])
        self.assertTrue(acknowledge_review_item(project, item))
        self.assertEqual(collect_review_items(project), [])
        record_review_issue(project, '01.png', 'background_review', item['message'], revision='rework-2')
        next_item = collect_review_items(project)[0]
        self.assertNotEqual(next_item['key'], item['key'])
        self.assertTrue(acknowledge_review_item(project, next_item))
        project.pages['01.png'][0].translation = '안녕하세요'
        self.assertEqual(len(collect_review_items(project)), 1)

    def test_page_and_bubble_identifiers_remain_distinct(self):
        project = make_project(TextBlock(text=['Hello']), TextBlock(text=['Hello']))
        project.pages['02.png'] = [TextBlock(text=['Hello'])]
        project._image_info['02.png'] = {'finish_code': RunStatus.FIN_OCR}
        issues = collect_review_items(project)
        self.assertEqual(len({item['key'] for item in issues}), 3)
        self.assertEqual([(item['page'], item['block_index']) for item in issues], [('01.png', 0), ('01.png', 1), ('02.png', 0)])

    def test_clear_recorded_issues_preserves_other_issues_and_acknowledgments(self):
        project = make_project(TextBlock(text=['Hello']))
        acknowledge_review_item(project, collect_review_items(project)[0])
        acknowledged = list(project.local_review['acknowledged'])
        record_review_issue(project, '01.png', 'layout_hold', '배치를 확인하세요.')
        record_review_issue(project, '01.png', 'background_review', '배경을 확인하세요.')
        self.assertTrue(clear_review_issues(project, '01.png', 'layout_hold'))
        self.assertEqual([item['code'] for item in collect_review_items(project)], ['background_review'])
        self.assertEqual(project.local_review['acknowledged'], acknowledged)
        self.assertTrue(clear_review_issues(project, '01.png'))
        self.assertFalse(clear_review_issues(project, '01.png'))

    def test_recording_runtime_issues_validates_targets(self):
        project = make_project(TextBlock())
        for page, index in [('missing.png', None), ('01.png', -1), ('01.png', 1), ('01.png', True)]:
            with self.subTest(page=page, index=index), self.assertRaises(ValueError):
                record_review_issue(project, page, 'layout_hold', '확인', index)

    def test_loading_invalid_optional_metadata_preserves_valid_parts(self):
        with patch('ballontranslator.utils.local_review.LOGGER.warning') as warning:
            state = normalize_local_review({
                'pages': {'01.png': [
                    {'code': 'layout_hold', 'message': '확인', 'revision': 42, 'future': True},
                    {'code': 'layout_hold', 'message': '확인', 'block_index': True},
                    None,
                ], '02.png': False},
                'acknowledged': ['valid', None, 'valid', 2], 'future': 'unknown',
            })
        self.assertGreater(warning.call_count, 0)
        self.assertEqual(state, {
            'pages': {'01.png': [{'code': 'layout_hold', 'message': '확인', 'block_index': None, 'revision': ''}]},
            'acknowledged': ['valid'],
        })


class LocalReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.dialog = LocalReviewDialog(self.parent)
        self.items = [{
            'page': '02.png', 'block_index': 4, 'code': 'language_review',
            'message': '원문 언어를 확인하세요.', 'key': 'bubble-key',
        }, {
            'page': '03.png', 'block_index': None, 'code': 'layout_hold',
            'message': '배치를 확인하세요.', 'key': 'page-key',
        }]

    def tearDown(self):
        self.dialog.hide()
        self.parent.deleteLater()
        self.app.processEvents()

    def test_actions_emit_exact_ids_and_never_remove_rows_silently(self):
        opened = QSignalSpy(self.dialog.open_requested)
        acknowledged = QSignalSpy(self.dialog.acknowledge_requested)
        language = QSignalSpy(self.dialog.language_requested)
        self.dialog.set_items(self.items)
        self.dialog.open_button.click()
        self.dialog.acknowledge_button.click()
        self.dialog.language_combo.setCurrentText('English')
        self.dialog.language_button.click()
        self.assertEqual(list(opened[0]), ['02.png', 4])
        self.assertEqual(acknowledged[0][0], self.items[0])
        self.assertEqual(list(language[0]), ['02.png', 4, 'English'])
        self.assertEqual(self.dialog.table.rowCount(), 2)

    def test_language_can_explicitly_apply_to_page_from_bubble_or_page_row(self):
        language = QSignalSpy(self.dialog.language_requested)
        opened = QSignalSpy(self.dialog.open_requested)
        self.dialog.set_items(self.items)
        self.dialog.scope_combo.setCurrentIndex(1)
        self.dialog.language_combo.setCurrentText('日本語')
        self.dialog.language_button.click()
        self.dialog.table.selectRow(1)
        self.dialog.language_button.click()
        self.dialog.open_button.click()
        self.assertEqual(list(language[0]), ['02.png', -1, '日本語'])
        self.assertEqual(list(language[1]), ['03.png', -1, '日本語'])
        self.assertEqual(list(opened[0]), ['03.png', -1])

    def test_refresh_preserves_selection_and_empty_list_disables_actions(self):
        self.dialog.set_items(self.items)
        self.dialog.table.selectRow(1)
        self.dialog.set_items([self.items[1]])
        opened = QSignalSpy(self.dialog.open_requested)
        self.dialog.open_button.click()
        self.assertEqual(list(opened[0]), ['03.png', -1])
        self.dialog.set_items([])
        self.assertFalse(self.dialog.open_button.isEnabled())
        self.assertFalse(self.dialog.acknowledge_button.isEnabled())
        self.assertFalse(self.dialog.language_button.isEnabled())

    def test_dialog_is_nonmodal_reusable_and_allows_editor_interaction(self):
        self.dialog.show()
        self.app.processEvents()
        self.assertEqual(self.dialog.windowModality(), Qt.WindowModality.NonModal)
        self.assertIsNone(QApplication.activeModalWidget())
        self.assertTrue(self.dialog._preserve_on_outside_click())
        self.dialog.hide()
        self.assertFalse(self.dialog._outside_click_filter_installed)
        self.dialog.show()
        self.assertTrue(self.dialog.isVisible())


if __name__ == '__main__':
    unittest.main()
