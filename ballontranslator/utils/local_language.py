"""Offline language decisions and context shared by local worker pipelines."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ballontranslator.modules.ocr.base import OCRBase
    from .textblock import TextBlock
    from .proj_imgtrans import ProjImgTrans

SOURCE_MODES = ('Auto', 'English', '日本語')
SOURCE_LANGUAGES = ('English', '日本語')
OCR_ENGINES = {'English': 'mit48px', '日本語': 'manga_ocr'}


def effective_source_language(project: ProjImgTrans | None, page_key: str | None,
                              fallback: str = 'Auto') -> str:
    """Resolve page, project and application preferences in that order.

    >>> effective_source_language(None, None, 'English')
    'English'
    """
    settings = getattr(project, 'local_settings', {}) or {}
    mode = settings.get('source_language', fallback)
    info = getattr(project, '_image_info', {}) or {}
    page = info.get(page_key, {}) if page_key is not None else {}
    mode = page.get('local_source_language', mode)
    return mode if mode in SOURCE_MODES else 'Auto'


def local_reading_direction(project: ProjImgTrans | None, fallback: str = 'auto') -> str:
    settings = getattr(project, 'local_settings', {}) or {}
    value = settings.get('reading_direction', fallback)
    return value if value in ('auto', 'rtl', 'ltr') else 'auto'


def order_local_blocks(blocks: list[TextBlock], project: ProjImgTrans | None,
                       source_language: str = 'Auto') -> list[TextBlock]:
    """Order newly detected blocks only; callers preserve selected-block indices."""
    from .config import pcfg
    from .textblock import sort_regions
    direction = local_reading_direction(project, pcfg.module.local_reading_direction)
    if direction == 'auto':
        # Language and panel reading direction are independent. Preserve the
        # existing orientation heuristic until the user selects a direction.
        rtl = None
    else:
        rtl = direction == 'rtl'
    return sort_regions(blocks, right_to_left=rtl, japanese_vertical_overlap=True)


def recognize_source_language(text: str) -> tuple[str, str]:
    """Conservatively inspect scripts, without claiming calibrated confidence.

    >>> recognize_source_language('Hello there!')
    ('English', '')
    >>> recognize_source_language('123')[0]
    'unknown'
    """
    text = unicodedata.normalize('NFKC', text)
    kana = sum('\u3041' <= ch <= '\u3096' or '\u30a1' <= ch <= '\u30fa' for ch in text)
    kanji = sum('\u3400' <= ch <= '\u9fff' or '\uf900' <= ch <= '\ufaff' for ch in text)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    unsupported = [ch for ch in text if ch.isalpha() and not (
        ch.isascii() or '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u9fff'
        or '\uf900' <= ch <= '\ufaff')]
    if unsupported:
        return 'unknown', '지원 언어(영어·일본어) 외의 문자가 있어 원문을 확인해 주세요.'
    if kana:
        if kana + kanji < 2:
            return 'unknown', '원문이 너무 짧아 언어를 확정하지 못했습니다.'
        return '日本語', '한 말풍선에 영어와 일본어가 섞여 있어 확인이 필요합니다.' if latin else ''
    if kanji:
        return 'unknown', '한자만 있는 원문은 일본어인지 확정할 수 없습니다.'
    if latin >= 4:
        return 'English', ''
    if latin:
        return 'unknown', '원문이 너무 짧아 언어를 확정하지 못했습니다.'
    return 'unknown', ('숫자·기호만 있어 언어를 확정하지 못했습니다.' if text.strip()
                       else '인식한 원문이 비어 있습니다.')


def inherit_source_overrides(blocks: list[TextBlock], previous: list[TextBlock],
                             offset: tuple[int, int] = (0, 0)) -> None:
    """Retain a manual choice when replacement geometry matches unambiguously."""
    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        dx, dy = offset
        x1, x2, y1, y2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
        area = max(0, x2 - x1) * max(0, y2 - y1)
        matches = set()
        overlaps_manual = False
        for old in previous:
            choice = old.source_language_override
            if choice not in SOURCE_LANGUAGES:
                continue
            ox1, oy1, ox2, oy2 = old.xyxy
            overlap = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            overlaps_manual = overlaps_manual or overlap > 0
            # A split inside an explicitly chosen bubble can inherit its mode;
            # a small old bubble must not force a newly merged large region.
            if area > 0 and overlap >= area * 0.6:
                matches.add(choice)
        if len(matches) == 1:
            block.source_language_override = matches.pop()
        elif matches:
            block.language_review = '재검출한 영역에 서로 다른 수동 언어 설정이 있어 확인이 필요합니다.'
        elif overlaps_manual:
            block.language_review = '재검출한 영역과 이전 수동 언어 영역이 충분히 겹치지 않아 확인이 필요합니다.'


def run_local_ocr(ocr: OCRBase, image: np.ndarray, blocks: list[TextBlock],
                  project: ProjImgTrans | None, page_key: str | None, *, stop_event=None, **kwargs):
    """Run the local router on the existing worker, including manual overrides."""
    from .local_edition import is_local_edition
    if (not is_local_edition()
            or getattr(ocr, 'name', '') not in ('local_auto', 'manga_ocr', 'mit48px')):
        return ocr.run_ocr(image, blocks, **kwargs)
    from .config import pcfg
    from ballontranslator.modules.ocr.ocr_local_auto import LocalAutoOCR
    mode = effective_source_language(project, page_key, pcfg.module.translate_source)
    if isinstance(ocr, LocalAutoOCR):
        ocr.set_stop_event(stop_event)
        return ocr.run_ocr(image, blocks, source_language=mode, **kwargs)
    # A manually selected engine may already be loaded by ModuleThread. Borrow
    # it for its first group, then release it before any other engine is loaded.
    router = LocalAutoOCR()
    router.engine = ocr
    router.set_stop_event(stop_event)
    return router.run_ocr(image, blocks, source_language=mode, **kwargs)
