"""Lazy offline English/Japanese OCR with a bounded one-engine lifecycle."""

from __future__ import annotations

from copy import deepcopy
from threading import Event

import numpy as np

from .base import OCRBase, OCR, register_OCR, DEVICE_SELECTOR, TextBlock
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.local_language import OCR_ENGINES, SOURCE_LANGUAGES, recognize_source_language


@register_OCR('local_auto')
class LocalAutoOCR(OCRBase):
    """A prepared router stays ready while its temporary engine is unloaded.

    >>> LocalAutoOCR._load_model_keys is None
    True
    """

    # Metadata must remain literal/SafeEval-compatible; child specs are prepared
    # by the local setup workflow, never downloaded during recognition.
    dependencies = []
    params = {'device': DEVICE_SELECTOR(not_supported=['privateuseone']), 'description': 'Local English / Japanese OCR'}
    _load_model_keys = None

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.engine: OCRBase | None = None
        self._stop_event: Event | None = None

    def unload_model(self, empty_cache: bool = False) -> bool:
        engine = getattr(self, 'engine', None)
        if engine is None:
            return False
        try:
            engine.unload_model(empty_cache=empty_cache)
        finally:
            self.engine = None
        return True

    def set_stop_event(self, stop_event: Event | None) -> None:
        self._stop_event = stop_event

    def _check_stopped(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise LLMRequestStopped('원문 인식을 취소했습니다.')

    def _recognize_group(self, image: np.ndarray, blocks: list[TextBlock], key: str,
                         **kwargs) -> None:
        self._check_stopped()
        try:
            if self.engine is None or self.engine.name != key:
                self.unload_model(empty_cache=True)
                from ballontranslator.utils.local_edition import _model_files_present
                from ballontranslator.modules.prepare_local_files import _missing_module_requirements
                spec = OCR.get_spec(key)
                if spec is None or _missing_module_requirements(spec) or not _model_files_present(spec):
                    raise RuntimeError(f'{key} 로컬 OCR 준비가 필요합니다. 모델 준비 화면에서 설치해 주세요.')
                params = deepcopy(pcfg.module.ocr_params.get(key) or {})
                params['device'] = self.get_param_value('device')
                self.engine = OCR.resolve_module(key)(**params)
            if not self.engine.all_model_loaded():
                self.engine.load_model()
            for block in blocks:
                block.text = []
            if key == 'manga_ocr':
                # The generic block crop loop permits cancellation between
                # balloons; MIT owns batching across its detected text lines.
                for block in blocks:
                    self._check_stopped()
                    self.engine._ocr_blk_list(image, [block], **kwargs)
            else:
                self.engine._ocr_blk_list(image, blocks, **kwargs)
            self._check_stopped()
        finally:
            self.unload_model(empty_cache=True)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: list[TextBlock],
                      source_language: str = 'Auto', **kwargs) -> list[TextBlock]:
        snapshots = [(block, deepcopy(block.text), block.source_language,
                      block.ocr_engine, block.language_review, deepcopy(vars(block.fontformat)))
                     for block in blk_list]
        original_formats = {id(item[0]): item[5] for item in snapshots}
        selections = {}
        groups = {'mit48px': [], 'manga_ocr': []}
        try:
            for block in blk_list:
                override = block.source_language_override
                forced = override if override in SOURCE_LANGUAGES else source_language
                hint = {'eng': 'English', 'ja': '日本語'}.get(block.language, 'unknown')
                selected = forced if forced in SOURCE_LANGUAGES else hint
                key = OCR_ENGINES.get(selected, 'mit48px')
                selections[id(block)] = (forced, hint, key)
                groups[key].append(block)
            for key, group in groups.items():
                if group:
                    self._recognize_group(img, group, key, **kwargs)
                    for block in group:
                        block.ocr_engine = key

            retries = {'mit48px': [], 'manga_ocr': []}
            first_results = {}
            for block in blk_list:
                forced, hint, key = selections[id(block)]
                language, reason = recognize_source_language(block.get_text())
                if forced not in SOURCE_LANGUAGES:
                    other = OCR_ENGINES.get(language)
                    # A neutral/contradictory detector hint needs a second OCR
                    # opinion; a short/kanji-only result stays reviewable even
                    # if the alternative produces plausible text.
                    if (other and other != key) or language == 'unknown':
                        other = other or ('manga_ocr' if key == 'mit48px' else 'mit48px')
                        first_results[id(block)] = (deepcopy(block.text), language, reason, key,
                                                    deepcopy(vars(block.fontformat)))
                        retries[other].append(block)
                block.source_language = forced if forced in SOURCE_LANGUAGES else language
                block.language_review = reason
                if forced in SOURCE_LANGUAGES and language in SOURCE_LANGUAGES and language != forced:
                    block.language_review = '수동 언어 설정과 인식한 문자가 다릅니다. 원문을 확인해 주세요.'
                elif hint in SOURCE_LANGUAGES and language in SOURCE_LANGUAGES and hint != language:
                    block.language_review = '검출 언어와 OCR 결과가 달라 원문 확인이 필요합니다.'
            for key, group in retries.items():
                if not group:
                    continue
                # Rejected retries must not leak MIT color updates, and manga
                # results must not inherit colors from an abandoned MIT pass.
                for block in group:
                    vars(block.fontformat).clear()
                    vars(block.fontformat).update(deepcopy(original_formats[id(block)]))
                self._recognize_group(img, group, key, **kwargs)
                for block in group:
                    original, language, reason, original_key, original_format = first_results[id(block)]
                    new_language, new_reason = recognize_source_language(block.get_text())
                    expected = 'English' if key == 'mit48px' else '日本語'
                    if new_language == expected and (language == new_language or language == 'unknown'):
                        block.source_language = new_language
                        block.ocr_engine = key
                        block.language_review = (reason or new_reason or
                                                 '검출 언어와 OCR 결과가 달라 원문 확인이 필요합니다.')
                    else:
                        block.text = original
                        vars(block.fontformat).clear()
                        vars(block.fontformat).update(original_format)
                        block.source_language = language
                        block.ocr_engine = original_key
                        block.language_review = reason or '두 OCR 결과가 일치하지 않아 원문 확인이 필요합니다.'
            for block, _text, _language, _engine, prior_review, _format in snapshots:
                if prior_review.startswith('재검출한 영역'):
                    block.language_review = prior_review
            self._check_stopped()
            return blk_list
        except Exception:
            for block, text, language, engine, review, original_format in snapshots:
                vars(block.fontformat).clear()
                vars(block.fontformat).update(original_format)
                block.text, block.source_language = text, language
                block.ocr_engine, block.language_review = engine, review
            raise
        finally:
            self.unload_model(empty_cache=True)

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        height, width = img.shape[:2]
        block = TextBlock([0, 0, width, height], lines=[[[0, 0], [width, 0], [width, height], [0, height]]])
        self._ocr_blk_list(img, [block], **kwargs)
        return block.get_text()
