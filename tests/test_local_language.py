"""Offline routing, identity and override regressions without model weights."""

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ballontranslator.modules.ocr import OCR
from ballontranslator.modules.ocr.ocr_local_auto import LocalAutoOCR
from ballontranslator.modules.exceptions import LLMRequestStopped, ModuleRunError
from ballontranslator.utils.config import ModuleConfig, pcfg
from ballontranslator.utils.local_language import (
    effective_source_language, inherit_source_overrides, order_local_blocks,
    recognize_source_language, run_local_ocr,
)
from ballontranslator.utils.textblock import TextBlock, sort_regions


@pytest.fixture
def engines(monkeypatch):
    events = []
    outputs = {'mit48px': {}, 'manga_ocr': {}}
    active = set()

    class Engine:
        def __init__(self, key):
            self.name = key
            self.loaded = False

        def all_model_loaded(self):
            return self.loaded

        def load_model(self):
            assert not active, 'OCR models must not overlap in memory'
            self.loaded = True
            active.add(self.name)
            events.append(('load', self.name))

        def unload_model(self, empty_cache=False):
            active.discard(self.name)
            self.loaded = False
            events.append(('unload', self.name))

        def _ocr_blk_list(self, image, blocks, **kwargs):
            events.append(('read', self.name, tuple(block.label for block in blocks)))
            for block in blocks:
                value = outputs[self.name][block.label]
                if isinstance(value, BaseException):
                    raise value
                block.text = [value]
                if self.name == 'mit48px':
                    block.fontformat.frgb = [17, 27, 37]
                    block.bg_colors = [47, 57, 67]

    monkeypatch.setattr(OCR, 'resolve_module', lambda key: lambda **kw: Engine(key))
    monkeypatch.setattr('ballontranslator.utils.local_edition._model_files_present', lambda spec: True)
    monkeypatch.setattr('ballontranslator.modules.prepare_local_files._missing_module_requirements', lambda spec: [])
    monkeypatch.setattr(pcfg, 'ocr_sublist', [])
    monkeypatch.setattr(pcfg.module, 'ocr_text_postprocess', 'none')
    return outputs, events, active


def block(label, hint='unknown', **kwargs):
    return TextBlock([0, 0, 8, 8], language=hint, label=label, **kwargs)


def test_mixed_page_batches_local_engines_and_preserves_identities(engines):
    output, events, active = engines
    output['mit48px']['en'] = 'Hello there!'
    output['manga_ocr']['ja'] = 'こんにちは'
    blocks = [block('ja', 'ja'), block('en', 'eng')]
    identities = [id(item) for item in blocks]
    result = LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), blocks)
    assert result is blocks
    assert [id(item) for item in result] == identities
    assert [item.source_language for item in result] == ['日本語', 'English']
    assert [item.ocr_engine for item in result] == ['manga_ocr', 'mit48px']
    assert all(not item.language_review for item in result)
    assert not active
    assert [event for event in events if event[0] == 'read'] == [
        ('read', 'mit48px', ('en',)), ('read', 'manga_ocr', ('ja',)),
    ]


def test_contradictory_hint_is_verified_with_other_local_engine(engines):
    output, _, _ = engines
    output['mit48px']['a'] = 'こんにちは'
    output['manga_ocr']['a'] = 'こんにちは！'
    item = block('a', 'eng')
    LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), [item])
    assert item.language == 'eng'  # Detector metadata stays independent.
    assert item.source_language == '日本語'
    assert item.ocr_engine == 'manga_ocr'
    assert item.language_review


@pytest.mark.parametrize('text', ['123', '!?', 'Hi', '東京', '안녕하세요'])
def test_ambiguous_or_unsupported_text_is_flagged(text):
    language, reason = recognize_source_language(text)
    assert language == 'unknown'
    assert reason


def test_manual_block_override_wins_over_page_mode_without_second_guessing(engines):
    output, events, _ = engines
    output['mit48px']['en'] = 'Hi'
    item = block('en', 'ja', source_language_override='English')
    LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), [item], source_language='日本語')
    assert item.source_language == item.source_language_override == 'English'
    assert item.ocr_engine == 'mit48px'
    assert item.language_review
    assert not any(event[0] == 'load' and event[1] == 'manga_ocr' for event in events)


def test_failure_restores_existing_text_and_language_metadata_and_unloads(engines):
    output, _, active = engines
    output['mit48px']['en'] = 'Hello there!'
    output['manga_ocr']['ja'] = RuntimeError('broken model')
    blocks = [block('en', 'eng', text=['old English'], source_language='unknown', language_review='old review'),
              block('ja', 'ja', text=['古い'], ocr_engine='old-engine')]
    before = [copy.deepcopy(item.to_dict()) for item in blocks]
    with pytest.raises(ModuleRunError, match='broken model'):
        LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), blocks)
    assert [item.to_dict() for item in blocks] == before
    assert not active


def test_cancel_before_inference_preserves_original_state(engines):
    _, events, _ = engines
    router = LocalAutoOCR()
    stop = threading.Event()
    stop.set()
    router.set_stop_event(stop)
    item = block('en', 'eng', text=['saved'])
    with pytest.raises(LLMRequestStopped):
        router.run_ocr(np.zeros((10, 10, 3), np.uint8), [item])
    assert item.get_text() == 'saved'
    assert not events


def test_page_project_and_application_source_precedence():
    project = SimpleNamespace(local_settings={'source_language': 'English'},
                              _image_info={'p': {'local_source_language': '日本語'}})
    assert effective_source_language(project, 'p', 'Auto') == '日本語'
    assert effective_source_language(project, 'q', 'Auto') == 'English'
    assert effective_source_language(None, 'q', 'Auto') == 'Auto'


def test_local_worker_uses_page_and_block_overrides_without_reordering(engines, monkeypatch):
    output, _, _ = engines
    output['manga_ocr']['ja'] = 'こんにちは'
    output['mit48px']['en'] = 'Hello there!'
    monkeypatch.setattr('ballontranslator.utils.local_edition._active', True)
    project = SimpleNamespace(local_settings={'source_language': 'English'},
                              _image_info={'p': {'local_source_language': '日本語'}})
    items = [block('ja'), block('en', source_language_override='English')]
    ids = [id(item) for item in items]
    run_local_ocr(LocalAutoOCR(), np.zeros((10, 10, 3), np.uint8), items, project, 'p')
    assert [id(item) for item in items] == ids
    assert [item.source_language for item in items] == ['日本語', 'English']


def test_direction_override_applies_only_to_new_detection_order(monkeypatch):
    left = TextBlock([0, 0, 10, 20])
    right = TextBlock([40, 0, 50, 20])
    items = [right, left]
    project = SimpleNamespace(local_settings={'reading_direction': 'ltr'})
    assert order_local_blocks(items, project) == [left, right]
    assert items == [right, left]
    project.local_settings['reading_direction'] = 'rtl'
    assert order_local_blocks(items, project) == [right, left]
    project.local_settings['reading_direction'] = 'auto'
    assert order_local_blocks(items, project) == sort_regions(items, japanese_vertical_overlap=True)


def test_replacement_geometry_retains_manual_override():
    original = TextBlock([10, 8, 16, 16], source_language_override='English')
    replacement = TextBlock([2, 2, 8, 10])
    inherit_source_overrides([replacement], [original], (8, 6))
    assert replacement.source_language_override == 'English'
    assert original.source_language_override == 'English'


def test_new_metadata_roundtrips_and_malformed_optional_values_warn(caplog):
    item = TextBlock(source_language='English', source_language_override='日本語',
                     ocr_engine='manga_ocr', language_review='확인')
    loaded = TextBlock(**item.to_dict(deep_copy=True))
    assert loaded.source_language == 'English'
    assert loaded.source_language_override == '日本語'
    assert loaded.ocr_engine == 'manga_ocr'
    assert loaded.language_review == '확인'
    invalid = TextBlock(source_language=[], source_language_override=123,
                        ocr_engine=None, language_review={})
    assert (invalid.source_language, invalid.source_language_override,
            invalid.ocr_engine, invalid.language_review) == ('unknown', '', '', '')
    config = ModuleConfig(local_reading_direction='sideways', local_work_name=[], local_tone_notes=None)
    assert (config.local_reading_direction, config.local_work_name, config.local_tone_notes) == ('auto', '', '')


def test_prepared_auto_router_completes_real_module_queue_without_eager_weights(monkeypatch):
    from ballontranslator.ui.module_manager import ModuleManager, OCRThread
    from ballontranslator.utils.proj_imgtrans import ProjImgTrans
    manager = ModuleManager(ProjImgTrans())
    try:
        router = LocalAutoOCR()
        router.load_model()
        manager.ocr_thread = OCRThread()
        manager.textdetect_thread = manager.translate_thread = manager.inpaint_thread = None
        manager.ocr_thread.module = router
        completed = []
        monkeypatch.setattr(manager, '_setter_for_module_key',
                            lambda key: lambda name: pytest.fail('prepared router must not repeat setup'))
        manager._begin_prepare_queue([('ocr', 'local_auto')], lambda: completed.append(True))
        assert completed == [True]
        assert router.all_model_loaded()
        assert router.engine is None
    finally:
        manager.ocr_thread.deleteLater()
        manager.deleteLater()


def test_rejected_mit_retry_restores_first_engine_format(engines):
    output, _, _ = engines
    output['manga_ocr']['a'] = '東京'
    output['mit48px']['a'] = '123'
    item = block('a', 'ja')
    item.fontformat.frgb = [3, 4, 5]
    item.bg_colors = [7, 8, 9]
    format_object = item.fontformat
    original = copy.deepcopy(vars(format_object))
    LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), [item])
    assert item.get_text() == '東京'
    assert item.ocr_engine == 'manga_ocr'
    assert item.fontformat is format_object
    assert vars(item.fontformat) == original


def test_failed_second_engine_restores_manual_format_in_place(engines):
    output, _, _ = engines
    output['mit48px']['en'] = 'Hello there!'
    output['manga_ocr']['ja'] = LLMRequestStopped('cancelled')
    english, japanese = block('en', 'eng'), block('ja', 'ja')
    english.fontformat.frgb = [1, 2, 3]
    english.bg_colors = [4, 5, 6]
    format_object = english.fontformat
    original = copy.deepcopy(vars(format_object))
    with pytest.raises(LLMRequestStopped):
        LocalAutoOCR().run_ocr(np.zeros((10, 10, 3), np.uint8), [english, japanese])
    assert english.fontformat is format_object
    assert vars(english.fontformat) == original


def test_small_manual_region_cannot_force_language_on_a_large_replacement():
    original = TextBlock([10, 10, 20, 20], source_language_override='English')
    replacement = TextBlock([0, 0, 100, 100])
    inherit_source_overrides([replacement], [original])
    assert replacement.source_language_override == ''
    assert replacement.language_review
