"""Detached inference for an explicitly selected page rectangle."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from threading import Event
from typing import Callable, TYPE_CHECKING

import cv2
import numpy as np

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.local_language import inherit_source_overrides, order_local_blocks, run_local_ocr
from ballontranslator.utils.local_edition import is_local_edition

if TYPE_CHECKING:
    from ballontranslator.modules.inpaint.base import InpainterBase
    from ballontranslator.modules.ocr.base import OCRBase
    from ballontranslator.modules.textdetector.base import TextDetectorBase
    from ballontranslator.modules.translators.base import BaseTranslator


@dataclass
class RegionReworkRequest:
    rect: tuple[int, int, int, int]
    image: np.ndarray
    inpainted: np.ndarray
    mask: np.ndarray
    project: ProjImgTrans
    page_key: str
    replaced_indices: tuple[int, ...]
    dilate: int = 0


@dataclass
class RegionReworkResult:
    rect: tuple[int, int, int, int]
    blocks: list[TextBlock]
    inpainted: np.ndarray
    mask: np.ndarray


def rework_region(
    request: RegionReworkRequest,
    detector: TextDetectorBase,
    ocr: OCRBase,
    translator: BaseTranslator,
    inpainter: InpainterBase,
    stop_event: Event,
    progress: Callable[[str, int], None],
    before_translate: Callable[[], None] | None = None,
) -> RegionReworkResult:
    """Compute a reversible ROI replacement without publishing project edits.

    The caller owns snapshots, worker scheduling, source validation and commit.
    Image models finish before the callback releases them for local translation.

    >>> RegionReworkResult.__annotations__['blocks']
    'list[TextBlock]'
    """
    def check_stopped() -> None:
        if stop_event.is_set():
            raise LLMRequestStopped('영역 재작업을 취소했습니다.')

    def stage_progress(stage: str, value: int) -> None:
        progress(stage, value)
        check_stopped()

    check_stopped()
    image = request.image
    if (image.ndim != 3 or image.shape[2] not in (3, 4)
            or request.inpainted.shape != image.shape
            or request.mask.shape != image.shape[:2]
            or any(array.dtype != np.uint8 for array in (image, request.inpainted, request.mask))):
        raise ValueError('영역 재작업 이미지 또는 마스크 형식이 올바르지 않습니다.')
    height, width = image.shape[:2]
    x1, y1, x2, y2 = request.rect
    if (not all(isinstance(value, (int, np.integer)) for value in request.rect)
            or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height):
        raise ValueError('이미지 안에서 유효한 영역을 선택해 주세요.')
    page = request.project.pages.get(request.page_key)
    if (page is None or len(set(request.replaced_indices)) != len(request.replaced_indices)
            or any(index < 0 or index >= len(page) for index in request.replaced_indices)):
        raise ValueError('재작업 대상 페이지 또는 말풍선이 변경되었습니다.')
    if not isinstance(request.dilate, int) or not 0 <= request.dilate <= 100:
        raise ValueError('텍스트 제거 범위 설정이 올바르지 않습니다.')
    for module in (ocr, inpainter, translator):
        if hasattr(module, 'set_stop_event'):
            module.set_stop_event(stop_event)

    stage_progress('detect', 0)
    detected_mask, blocks = detector.detect(image[y1:y2, x1:x2].copy())
    stage_progress('detect', 100)
    if not blocks:
        raise ValueError('선택 영역에서 대사를 찾지 못했습니다. 영역을 조정해 주세요.')
    if detected_mask is None or detected_mask.shape != (y2 - y1, x2 - x1):
        raise ValueError('선택 영역의 텍스트 마스크를 생성하지 못했습니다.')
    blocks = copy.deepcopy(blocks)
    if is_local_edition():
        inherit_source_overrides(blocks, [page[index] for index in request.replaced_indices], (x1, y1))
        blocks = order_local_blocks(blocks, request.project)
    stage_progress('ocr', 0)
    recognized = run_local_ocr(
        ocr, image[y1:y2, x1:x2].copy(), blocks, request.project, request.page_key,
        stop_event=stop_event,
    )
    if isinstance(recognized, list):
        blocks = recognized
    stage_progress('ocr', 100)
    if not blocks or any(not block.get_text().strip() for block in blocks):
        raise ValueError('선택 영역의 대사를 읽지 못했습니다. 기존 결과를 유지합니다.')
    for block in blocks:
        bx1, by1, bx2, by2 = block.xyxy
        if not (0 <= bx1 < bx2 <= x2 - x1 and 0 <= by1 < by2 <= y2 - y1):
            raise ValueError('인식한 대사가 선택 영역을 벗어났습니다. 영역을 넓혀 주세요.')
        lines = np.asarray(block.lines)
        if lines.size and (lines.ndim != 3 or lines.shape[1:] != (4, 2)
                or not np.isfinite(lines).all()
                or np.any(lines < 0)
                or np.any(lines[..., 0] > x2 - x1)
                or np.any(lines[..., 1] > y2 - y1)):
            raise ValueError('인식한 글자 위치가 선택 영역을 벗어났습니다.')
        if block._bounding_rect is not None:
            bx, by, bw, bh = block._bounding_rect
            if not (0 <= bx < bx + bw <= x2 - x1 and 0 <= by < by + bh <= y2 - y1):
                raise ValueError('인식한 글상자가 선택 영역을 벗어났습니다.')
        block.adjust_pos(x1, y1)
        if lines.size:
            block.lines = (lines + np.array([x1, y1])).tolist()

    new_mask = np.where(detected_mask > 0, 255, 0).astype(np.uint8)
    if request.dilate:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * request.dilate + 1, 2 * request.dilate + 1),
        )
        new_mask = cv2.dilate(new_mask, kernel)
    merged_mask = np.bitwise_or(new_mask, request.mask[y1:y2, x1:x2])
    if not np.any(merged_mask):
        raise ValueError('선택 영역에서 제거할 원문을 찾지 못했습니다.')

    padding = max(16, round(max(x2 - x1, y2 - y1) / 4))
    cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
    cx2, cy2 = min(width, x2 + padding), min(height, y2 + padding)
    context_image = request.inpainted[cy1:cy2, cx1:cx2].copy()
    rx1, ry1, rx2, ry2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    context_image[ry1:ry2, rx1:rx2] = image[y1:y2, x1:x2]
    context_mask = np.zeros(context_image.shape[:2], np.uint8)
    context_mask[ry1:ry2, rx1:rx2] = merged_mask
    stage_progress('inpaint', 0)
    painted = inpainter.inpaint(
        context_image, context_mask,
        textblock_list=None, check_need_inpaint=False,
    )
    stage_progress('inpaint', 100)
    if not isinstance(painted, np.ndarray) or painted.shape != context_image.shape or painted.dtype != image.dtype:
        raise ValueError('배경 복원 결과의 이미지 형식이 올바르지 않습니다.')
    output_image = request.inpainted[y1:y2, x1:x2].copy()
    selected = merged_mask > 0
    output_image[selected] = painted[ry1:ry2, rx1:rx2][selected]
    if image.shape[2] == 4:
        output_image[:, :, 3] = request.inpainted[y1:y2, x1:x2, 3]

    project = copy.copy(request.project)
    project.pages = copy.deepcopy(request.project.pages)
    project._image_info = copy.deepcopy(request.project._image_info)
    project._llm_compact_memory = copy.deepcopy(request.project._llm_compact_memory)
    replaced = set(request.replaced_indices)
    insertion = min(replaced) if replaced else len(page)
    context_page = project.pages[request.page_key]
    project.pages[request.page_key] = (
        [block for index, block in enumerate(context_page[:insertion]) if index not in replaced]
        + blocks
        + [block for index, block in enumerate(context_page[insertion:], start=insertion) if index not in replaced]
    )
    check_stopped()
    if before_translate is not None:
        before_translate()
    stage_progress('translate', 0)
    translator.translate_textblk_lst(
        blocks, project=project, page_key=request.page_key, full_page=False,
    )
    stage_progress('translate', 100)
    if any(not str(block.translation or '').strip() for block in blocks):
        raise ValueError('일부 대사의 번역 결과가 비어 있습니다. 기존 결과를 유지합니다.')
    return RegionReworkResult(request.rect, blocks, output_image, merged_mask)
