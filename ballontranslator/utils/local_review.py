"""Observable local-pipeline issues and explicit, content-bound review decisions."""

import hashlib
import json
from typing import TYPE_CHECKING, Optional, TypedDict

from .config import RunStatus
from .logger import logger as LOGGER

if TYPE_CHECKING:
    from .proj_imgtrans import ProjImgTrans
    from .textblock import TextBlock


class ReviewItem(TypedDict):
    page: str
    block_index: Optional[int]
    code: str
    message: str
    key: str


def normalize_local_review(value: object) -> dict:
    """Load optional review metadata, retaining valid portions of older projects.

    >>> normalize_local_review(None)
    {'pages': {}, 'acknowledged': []}
    """
    result = {'pages': {}, 'acknowledged': []}
    if value is None:
        return result
    if not isinstance(value, dict):
        LOGGER.warning('Ignoring invalid local_review metadata; expected an object.')
        return result
    unknown = set(value) - {'pages', 'acknowledged'}
    if unknown:
        LOGGER.warning('Ignoring unknown local_review fields: %s.', sorted(map(str, unknown)))
    pages = value.get('pages', {})
    if not isinstance(pages, dict):
        LOGGER.warning('Ignoring invalid local_review.pages; expected an object.')
        pages = {}
    for page, entries in pages.items():
        if not isinstance(page, str) or not page or not isinstance(entries, list):
            LOGGER.warning('Ignoring invalid local review page entry %r.', page)
            continue
        valid_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                LOGGER.warning('Ignoring invalid local review issue for page %s.', page)
                continue
            code, message = entry.get('code'), entry.get('message')
            block_index = entry.get('block_index')
            if (
                not isinstance(code, str) or not code.strip()
                or not isinstance(message, str) or not message.strip()
                or (block_index is not None and (
                    type(block_index) is not int or block_index < 0
                ))
            ):
                LOGGER.warning('Ignoring invalid local review issue for page %s.', page)
                continue
            unknown = set(entry) - {'code', 'message', 'block_index', 'revision'}
            if unknown:
                LOGGER.warning('Ignoring unknown local review issue fields for %s: %s.', page, sorted(map(str, unknown)))
            revision = entry.get('revision', '')
            if not isinstance(revision, str):
                LOGGER.warning('Ignoring invalid local review issue revision for %s.', page)
                revision = ''
            valid_entries.append({
                'code': code, 'message': message,
                'block_index': block_index, 'revision': revision,
            })
        if valid_entries:
            result['pages'][page] = valid_entries
    acknowledged = value.get('acknowledged', [])
    if not isinstance(acknowledged, list):
        LOGGER.warning('Ignoring invalid local_review.acknowledged; expected an array.')
        acknowledged = []
    seen = set()
    for key in acknowledged:
        if not isinstance(key, str) or not key:
            LOGGER.warning('Ignoring invalid local review acknowledgment.')
        elif key not in seen:
            result['acknowledged'].append(key)
            seen.add(key)
    return result


def _json_value(value: object) -> object:
    # Committed geometry may contain numpy arrays/scalars after an edit.
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'to_serializable_dict'):
        return value.to_serializable_dict()
    raise TypeError(f'Unsupported review fingerprint value: {type(value).__name__}')


def _block_state(block: 'TextBlock') -> dict:
    return {
        name: getattr(block, name, None)
        for name in (
            'text', 'translation', 'rich_text', 'xyxy', '_bounding_rect',
            'angle', 'lines', 'fontformat', 'source_language_override',
        )
    }


def collect_review_items(project: 'ProjImgTrans') -> list[ReviewItem]:
    """Return unresolved, evidenced issues in page and bubble order.

    Acknowledgments apply to the current text, translation and geometry only.
    Background checks are explicit recorded requests, never guessed from text.

    >>> from types import SimpleNamespace
    >>> collect_review_items(SimpleNamespace(pages={}, local_review={}))
    []
    """
    review = getattr(project, 'local_review', {})
    extra_pages = review.get('pages', {})
    acknowledged = set(review.get('acknowledged', []))
    result = []
    seen = set()
    for page, blocks in project.pages.items():
        states = [_block_state(block) for block in blocks]
        info = getattr(project, '_image_info', {}).get(page, {})
        progress = info.get('finish_code', 0)
        failed_stage = info.get('pipeline_failed_stage')
        ocr_processed = bool(progress & (RunStatus.FIN_OCR | RunStatus.FIN_TRANSLATE))
        translation_attempted = failed_stage == 'translate'
        issues = []
        for index, block in enumerate(blocks):
            reason = getattr(block, 'language_review', '')
            if reason:
                issues.append((index, 'language_review', str(reason), ''))
            source = block.get_text().strip()
            has_ocr_evidence = bool(getattr(block, 'ocr_engine', ''))
            processed = ocr_processed or has_ocr_evidence
            detected = bool(getattr(block, 'det_model', '')) or len(block.lines) > 0 or has_ocr_evidence
            if detected and not source and (processed or failed_stage == 'ocr'):
                issues.append((index, 'ocr_empty', '감지된 말풍선의 원문이 비어 있습니다. OCR 결과를 확인하세요.', ''))
            elif source and not block.translation.strip() and (processed or translation_attempted):
                issues.append((index, 'translation_missing', '번역이 비어 있습니다. 원문과 번역 결과를 확인하세요.', ''))
        for entry in extra_pages.get(page, []):
            index = entry.get('block_index')
            # A deleted bubble must not request an out-of-range editor target.
            if index is not None and not 0 <= index < len(blocks):
                continue
            issues.append((index, entry['code'], entry['message'], entry.get('revision', '')))
        for index, code, message, revision in issues:
            fingerprint = [page, index, code, message, revision, states if index is None else states[index]]
            payload = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, default=_json_value)
            key = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            if key not in acknowledged and key not in seen:
                result.append({'page': page, 'block_index': index, 'code': code, 'message': message, 'key': key})
                seen.add(key)
    return result


def acknowledge_review_item(project: 'ProjImgTrans', item: ReviewItem) -> bool:
    """Acknowledge an unchanged visible issue; reject a stale dialog selection."""
    key = item.get('key')
    if not key or not any(current['key'] == key for current in collect_review_items(project)):
        return False
    state = project.local_review
    state.setdefault('acknowledged', []).append(key)
    return True


def record_review_issue(
    project: 'ProjImgTrans', page: str, code: str, message: str,
    block_index: Optional[int] = None, revision: str = '',
) -> None:
    """Record an observed or manual issue; callers own save and dirty state.

    Use a new revision for repeated image rework with unchanged text/geometry.
    """
    if page not in project.pages:
        raise ValueError('Review page is not in the project.')
    if block_index is not None and (
        type(block_index) is not int or not 0 <= block_index < len(project.pages[page])
    ):
        raise ValueError('Review bubble index is out of range.')
    if not isinstance(code, str) or not code.strip() or not isinstance(message, str) or not message.strip():
        raise ValueError('Review code and message must be non-empty strings.')
    if not isinstance(revision, str):
        raise TypeError('Review revision must be a string.')
    state = getattr(project, 'local_review', None)
    if state is None:
        state = project.local_review = normalize_local_review(None)
    entries = state.setdefault('pages', {}).setdefault(page, [])
    entry = {'code': code, 'message': message, 'block_index': block_index, 'revision': revision}
    for index, previous in enumerate(entries):
        if previous['code'] == code and previous.get('block_index') == block_index:
            entries[index] = entry
            return
    entries.append(entry)


def clear_review_issues(project: 'ProjImgTrans', page: str, code: Optional[str] = None) -> bool:
    """Remove recorded issues after their owner fixes them, preserving other decisions."""
    pages = getattr(project, 'local_review', {}).get('pages', {})
    entries = pages.get(page, [])
    retained = [entry for entry in entries if code is not None and entry['code'] != code]
    if len(retained) == len(entries):
        return False
    if retained:
        pages[page] = retained
    else:
        pages.pop(page, None)
    return True
