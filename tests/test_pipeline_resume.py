import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ballontranslator.ui import module_manager
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.utils.config import RunStatus, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils import local_edition, local_ollama


class Translator:
    lang_target = '한국어'
    low_vram_mode = False

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def is_computational_intensive(self):
        return True

    def delay(self):
        return 0

    def translate_textblk_lst(self, blocks, **kwargs):
        self.calls.append(kwargs['page_key'])
        if self.fail:
            raise RuntimeError('backend unavailable')
        for block in blocks:
            block.translation = 'new translation'

    def on_page_translation_finished(self, project, page_key):
        pass


class PipelineResumeTest(unittest.TestCase):
    def setUp(self):
        self.old_stages = [pcfg.module.stage_enabled(i) for i in range(4)]
        self.addCleanup(self.restore_stages)
        for index in range(4):
            pcfg.module.set_stage_enabled(index, True)
        self.error_patch = patch.object(module_manager, 'create_error_dialog')
        self.error_patch.start()
        self.addCleanup(self.error_patch.stop)
        self.image = np.zeros((8, 8, 3), np.uint8)
        self.mask = np.ones((8, 8), np.uint8)
        self.events = []

    def restore_stages(self):
        for index, enabled in enumerate(self.old_stages):
            pcfg.module.set_stage_enabled(index, enabled)

    def project(self, progress=0):
        project = ProjImgTrans()
        project.pages = {'one.png': [TextBlock(
            xyxy=[0, 0, 8, 8], text=['edited source'],
            translation='manually corrected translation',
        )]}
        project._image_info = {'one.png': {'finish_code': progress}}
        project.read_img = lambda _: self.image
        project.load_mask_by_imgname = lambda _: self.mask
        project.save_mask = lambda *_: None
        project.save_inpainted = lambda *_: self.events.append('saved inpaint')
        return project

    def pipeline(self, project, translator=None, *, inpaint_fails=False):
        def detect(*_):
            self.events.append('detect')
            return self.mask, [TextBlock(xyxy=[0, 0, 8, 8], text=['detected'])]

        def ocr(_image, blocks):
            self.events.append('ocr')
            for block in blocks:
                block.text = ['new OCR']

        def inpaint(*_):
            self.events.append('inpaint')
            if inpaint_fails:
                raise RuntimeError('inpaint unavailable')
            return self.image

        translate_thread = module_manager.TranslateThread()
        translate_thread.module = translate_thread.translator = translator or Translator()
        thread = module_manager.ImgtransThread(
            SimpleNamespace(textdetector=SimpleNamespace(detect=detect)),
            SimpleNamespace(ocr=SimpleNamespace(run_ocr=ocr)),
            translate_thread,
            SimpleNamespace(inpainter=SimpleNamespace(inpaint=inpaint)),
        )
        thread.imgtrans_proj = project
        thread.process_idx_to_page_idx = {}
        self.addCleanup(thread.deleteLater)
        self.addCleanup(translate_thread.deleteLater)
        return thread

    def test_resume_reuses_completed_steps_and_preserves_manual_text(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_OCR | RunStatus.FIN_TRANSLATE)
        block = project.pages['one.png'][0]
        translator = Translator()
        thread = self.pipeline(project, translator)
        thread.continue_mode = True

        thread._imgtrans_pipeline()

        self.assertEqual(self.events, ['inpaint', 'saved inpaint'])
        self.assertIs(project.pages['one.png'][0], block)
        self.assertEqual(block.get_text(), 'edited source')
        self.assertEqual(block.translation, 'manually corrected translation')
        self.assertEqual(translator.calls, [])
        self.assertTrue(project.get_page_progress('one.png'))
        self.assertEqual((thread.detect_counter, thread.ocr_counter, thread.translate_counter, thread.inpaint_counter), (1, 1, 1, 1))

    def test_resumed_ocr_invalidates_translation_and_changed_mask_result(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_TRANSLATE | RunStatus.FIN_INPAINT)
        translator = Translator()
        thread = self.pipeline(project, translator)
        thread.continue_mode = True

        thread._imgtrans_pipeline()

        self.assertEqual(self.events, ['ocr', 'inpaint', 'saved inpaint'])
        self.assertEqual(translator.calls, ['one.png'])
        self.assertEqual(project.pages['one.png'][0].translation, 'new translation')

    def test_failed_serial_translation_is_not_counted_or_followed_by_inpainting(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_OCR)
        thread = self.pipeline(project, Translator(fail=True))
        thread.continue_mode = True

        thread._imgtrans_pipeline()

        self.assertTrue(thread.isStopRequested())
        self.assertEqual(thread.translate_counter, 0)
        self.assertEqual(self.events, [])
        self.assertFalse(project.get_page_progress('one.png'))
        self.assertEqual(project.failed_pages(), ['one.png'])

    def test_failed_parallel_translation_keeps_the_next_page_queued(self):
        project = self.project()
        thread = self.pipeline(project, Translator(fail=True)).translate_thread
        thread.imgtrans_proj = project
        thread.num_process_pages = 2
        thread.pipeline_stop_event = threading.Event()
        thread.pipeline_pagekey_queue = ['one.png', 'two.png']

        thread._run_translate_pipeline()

        self.assertTrue(thread.pipeline_stop_event.is_set())
        self.assertEqual(thread.finished_counter, 0)
        self.assertEqual(thread.pipeline_pagekey_queue, ['two.png'])

    def test_failed_inpaint_retains_completed_translation_for_retry(self):
        progress = RunStatus.FIN_DET | RunStatus.FIN_OCR | RunStatus.FIN_TRANSLATE
        project = self.project(progress)
        thread = self.pipeline(project, inpaint_fails=True)
        thread.continue_mode = True

        thread._imgtrans_pipeline()

        self.assertTrue(thread.isStopRequested())
        self.assertEqual(project._image_info['one.png']['finish_code'], progress)
        self.assertEqual(thread.inpaint_counter, 0)
        self.assertEqual(project.failed_pages(), ['one.png'])
        self.events.clear()
        thread.inpaint_thread.inpainter.inpaint = lambda *_: self.image
        thread.clearStopRequest()
        thread._imgtrans_pipeline()
        self.assertEqual(self.events, ['saved inpaint'])
        self.assertEqual(project.failed_pages(), [])
        self.assertEqual(project.pages['one.png'][0].translation, 'manually corrected translation')

    def test_low_vram_resume_keeps_completed_translation_after_model_unload(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_OCR | RunStatus.FIN_TRANSLATE)
        translator = Translator()
        translator.low_vram_mode = True
        thread = self.pipeline(project, translator)
        thread.continue_mode = True
        with patch.object(module_manager, 'unload_modules'):
            thread._imgtrans_pipeline()
        self.assertEqual(translator.calls, [])
        self.assertEqual(project.pages['one.png'][0].translation, 'manually corrected translation')
        self.assertTrue(project.get_page_progress('one.png'))

    def test_parallel_resume_reuses_completed_manual_translation(self):
        project = self.project(RunStatus.FIN_TRANSLATE)
        translator = Translator()
        thread = self.pipeline(project, translator).translate_thread
        thread.continue_mode = True
        self.assertTrue(thread._translate_page(project, 'one.png'))
        self.assertEqual(translator.calls, [])
        self.assertEqual(project.pages['one.png'][0].translation, 'manually corrected translation')

    def test_failed_page_metadata_roundtrips_and_clears_after_success(self):
        project = self.project(RunStatus.FIN_DET)
        project.mark_stage_failed('one.png', 'ocr')
        restored = ProjImgTrans()
        restored.directory = '/unused'
        with patch('ballontranslator.utils.proj_imgtrans.find_all_imgs', return_value=['one.png']), patch.object(restored, 'set_current_img'):
            restored.load_from_dict(project.to_dict())
        self.assertEqual(restored.failed_pages(), ['one.png'])
        restored.update_page_progress('one.png', RunStatus.FIN_OCR)
        self.assertEqual(restored.failed_pages(), [])

    def test_invalid_optional_failure_metadata_does_not_block_loading(self):
        for invalid in (False, [], {}, 'unknown', 42):
            with self.subTest(value=invalid):
                project = ProjImgTrans()
                project.directory = '/unused'
                raw = {'pages': {'one.png': []}, 'image_info': {'one.png': {
                    'finish_code': RunStatus.FIN_DET,
                    'pipeline_failed_stage': invalid,
                }}}
                with patch('ballontranslator.utils.proj_imgtrans.find_all_imgs', return_value=['one.png']), patch.object(project, 'set_current_img'):
                    project.load_from_dict(raw)
                self.assertNotIn('pipeline_failed_stage', project._image_info['one.png'])
                self.assertEqual(project._image_info['one.png']['finish_code'], RunStatus.FIN_DET)

    def test_user_cancellation_leaves_page_pending_instead_of_failed(self):
        class CancelledTranslator(Translator):
            def translate_textblk_lst(self, blocks, **kwargs):
                raise LLMRequestStopped()

        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_OCR)
        thread = self.pipeline(project, CancelledTranslator())
        thread.continue_mode = True
        thread._imgtrans_pipeline()
        self.assertTrue(thread.isStopRequested())
        self.assertEqual(project.failed_pages(), [])

    def test_stop_during_model_preparation_cannot_start_pending_pipeline(self):
        manager = module_manager.ModuleManager(self.project())
        self.addCleanup(manager.deleteLater)
        pipeline = self.pipeline(manager.imgtrans_proj)
        manager.imgtrans_thread = pipeline
        calls = []
        manager._preparing_thread = SimpleNamespace(
            requestCancelModuleInit=lambda: calls.append('cancel model'),
        )
        manager._pending_prepare_success = lambda: calls.append('started pipeline')
        manager._pending_prepare_failure = lambda: calls.append('finished')
        manager.stopImgtransPipeline()
        manager._continue_pending_prepare()
        self.assertEqual(calls, ['cancel model', 'finished'])

    def test_unexpected_pipeline_failure_emits_stopped_completion(self):
        thread = self.pipeline(self.project())
        stopped = []
        thread.pipeline_stopped.connect(lambda: stopped.append(True))

        def unreadable_image():
            raise OSError('image no longer readable')

        thread.job = unreadable_image
        thread.run()
        self.assertTrue(thread.isStopRequested())
        self.assertEqual(stopped, [True])

    def test_local_image_model_unloads_ollama_on_worker_before_constructor(self):
        events = []
        main_thread = threading.get_ident()

        class ImageModel:
            def __init__(self, **_kwargs):
                events.append(('construct', threading.get_ident()))

            def load_model(self):
                events.append(('load', threading.get_ident()))

        thread = module_manager.ModuleThread('ocr', None)
        self.addCleanup(thread.deleteLater)
        thread.job = lambda: thread._set_module('manga_ocr')
        with patch.object(local_edition, '_active', True), patch.object(
            local_ollama.OllamaClient, 'unload', create=True,
            side_effect=lambda **_: events.append(('unload', threading.get_ident())),
        ), patch.object(thread, '_prepare_module_class', return_value=ImageModel):
            thread.start()
            self.assertTrue(thread.wait(3000))
        self.assertTrue(thread.last_set_success, thread.last_error)
        self.assertEqual([name for name, _ in events], ['unload', 'construct', 'load'])
        self.assertTrue(all(ident != main_thread for _, ident in events))

    def test_local_pipeline_unloads_once_before_image_work_across_pages(self):
        project = self.project()
        project.pages['two.png'] = [TextBlock(xyxy=[0, 0, 8, 8], text=['second'])]
        project._image_info['two.png'] = {'finish_code': 0}
        translator = Translator()
        translator.low_vram_mode = True
        thread = self.pipeline(project, translator)
        with patch.object(local_edition, '_active', True), patch.object(
            local_ollama.OllamaClient, 'unload', create=True,
            side_effect=lambda **_: self.events.append('unload Ollama'),
        ), patch.object(module_manager, 'unload_modules'):
            thread._imgtrans_pipeline()
        self.assertEqual(self.events[0], 'unload Ollama')
        self.assertEqual(self.events.count('unload Ollama'), 1)
        self.assertEqual(translator.calls, ['one.png', 'two.png'])

    def test_translation_only_resume_skips_image_preparation_and_ollama_unload(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_OCR | RunStatus.FIN_INPAINT)
        manager = module_manager.ModuleManager(project)
        self.addCleanup(manager.deleteLater)
        prepared = []
        with patch.object(manager, 'terminateRunningThread'), patch.object(
            manager, '_prepare_modules_then', side_effect=lambda required, *_args, **_kwargs: prepared.extend(required),
        ):
            manager.runImgtransPipeline(['one.png'], continue_mode=True)
        self.assertEqual(prepared, [('translator', pcfg.module.translator)])
        translator = Translator()
        translator.low_vram_mode = True
        thread = self.pipeline(project, translator)
        thread.continue_mode = True
        with patch.object(local_edition, '_active', True), patch.object(
            local_ollama.OllamaClient, 'unload', create=True,
            side_effect=AssertionError('No image work must keep the translation model resident'),
        ), patch.object(module_manager, 'unload_modules'):
            thread._imgtrans_pipeline()
        self.assertEqual(translator.calls, ['one.png'])

    def test_unload_failure_prevents_image_model_preparation(self):
        thread = module_manager.ModuleThread('ocr', SimpleNamespace(get_spec=lambda _: None))
        self.addCleanup(thread.deleteLater)
        with patch.object(local_edition, '_active', True), patch.object(
            local_ollama.OllamaClient, 'unload', side_effect=RuntimeError('release failed'),
        ), patch.object(thread, '_prepare_module_class', side_effect=AssertionError('must not load')):
            thread._set_module('manga_ocr')
        self.assertFalse(thread.last_set_success)
        self.assertEqual(str(thread.last_error), 'release failed')
        self.assertIsNone(thread.module)

    def test_resume_prepares_dependencies_invalidated_by_pending_ocr(self):
        project = self.project(RunStatus.FIN_DET | RunStatus.FIN_TRANSLATE | RunStatus.FIN_INPAINT)
        manager = module_manager.ModuleManager(project)
        self.addCleanup(manager.deleteLater)
        prepared = []
        with patch.object(manager, 'terminateRunningThread'), patch.object(
            manager, '_prepare_modules_then', side_effect=lambda required, *_args, **_kwargs: prepared.extend(required),
        ):
            manager.runImgtransPipeline(['one.png'], continue_mode=True)
            self.assertEqual([kind for kind, _ in prepared], ['ocr', 'translator', 'inpainter'])
            prepared.clear()
            manager.runImgtransPipeline(['one.png'], continue_mode=False)
            self.assertEqual([kind for kind, _ in prepared], ['textdetector', 'ocr', 'translator', 'inpainter'])
            prepared.clear()
            manager.runImgtransPipeline([], continue_mode=False)
            self.assertEqual([kind for kind, _ in prepared], ['textdetector', 'ocr', 'translator', 'inpainter'])


if __name__ == '__main__':
    unittest.main()
