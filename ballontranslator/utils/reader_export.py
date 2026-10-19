"""Read saved pages and export composites without loading models or mutating projects."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Callable, Optional, Sequence
import uuid
import warnings

from PIL import Image, ImageOps

MAX_SOURCE_PIXELS = 40_000_000
MAX_OUTPUT_PIXELS = 32_000_000
MAX_OUTPUT_SIDE = 60_000
RESULT_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


class OperationCancelled(RuntimeError):
    pass


def check_cancel(stop: Optional[threading.Event]) -> None:
    if stop is not None and stop.is_set():
        raise OperationCancelled('작업을 취소했습니다. 원본과 기존 결과는 유지됩니다.')


def signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_mtime_ns, info.st_size


@dataclass(frozen=True)
class Page:
    name: str
    path: Optional[Path]
    width: int = 1000
    height: int = 1400
    stamp: tuple[int, int] = (0, 0)
    issue: str = ''
    warning: str = ''


@dataclass(frozen=True)
class Book:
    folder: Path
    result_dir: Path
    pages: tuple[Page, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MergeOptions:
    direction: str = 'vertical'
    cross_size: int = 1080
    gap: int = 0
    split: bool = False
    max_length: int = 12000
    image_format: str = 'PNG'
    quality: int = 92


@dataclass(frozen=True)
class Placement:
    page: Page
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Group:
    size: tuple[int, int]
    items: tuple[Placement, ...]


@dataclass(frozen=True)
class ExportPlan:
    groups: tuple[Group, ...]
    options: MergeOptions
    cross_size: int


def inspect_page(path: Path, name: str, warning: str = '') -> Page:
    """Read headers only, retaining unreadable pages as visible placeholders."""
    path = Path(path)
    try:
        before = signature(path)
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                w, h = image.size
                if min(w, h) < 1 or w * h > MAX_SOURCE_PIXELS:
                    raise ValueError('입력 이미지가 4,000만 픽셀 제한을 초과합니다.')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('여러 프레임 이미지는 단일 이미지로 저장한 뒤 사용하세요.')
                if image.getexif().get(274, 1) in (5, 6, 7, 8):
                    w, h = h, w
            with Image.open(path) as verified:
                verified.verify()
        if signature(path) != before:
            raise ValueError('파일 확인 중 내용이 변경되었습니다. 다시 열어 주세요.')
        return Page(name, path, w, h, before, warning=warning)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        return Page(name, path, issue=f'결과를 읽지 못했습니다: {exc}', warning=warning)


def _leaf(name: object) -> bool:
    return isinstance(name, str) and bool(name) and name not in ('.', '..') and '/' not in name and '\\' not in name and ':' not in name


def _natural_key(name: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r'(\d+)', name))


def load_book(folder: Path, result_dir: Path, *, names: Optional[Sequence[str]] = None,
              suffix: str = '.png', review_pages: Sequence[str] = (),
              notices: Sequence[str] = (), stop: Optional[threading.Event] = None) -> Book:
    """Respect project order; missing results never become original images.

    >>> _natural_key('2.png') < _natural_key('10.png')
    True
    """
    folder, result_dir = Path(folder).resolve(), Path(result_dir).resolve()
    messages = list(notices)
    if not folder.is_dir():
        raise ValueError('원본 프로젝트 폴더를 찾지 못했습니다.')
    if not result_dir.is_dir():
        raise ValueError('저장된 결과 폴더가 없습니다. 번역 결과를 먼저 저장하세요.')
    if names is None:
        manifest = folder / f'imgtrans_{folder.name}.json'
        if manifest.exists():
            try:
                if manifest.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError('프로젝트 파일이 읽기 한도를 초과합니다.')
                data = json.loads(manifest.read_text(encoding='utf-8'))
                if not isinstance(data, dict) or not isinstance(data.get('pages'), dict):
                    raise ValueError('페이지 목록이 없습니다.')
                names = tuple(data['pages'])
            except (OSError, ValueError) as exc:
                raise ValueError(f'프로젝트 목록을 읽지 못했습니다. 파일은 변경하지 않습니다: {exc}') from exc
        else:
            names = sorted((p.name for p in result_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES), key=_natural_key)
            messages.append('프로젝트 목록이 없어 결과 파일의 숫자 순서로 표시합니다. 누락 여부를 확인하세요.')
    if not names:
        raise ValueError('표시할 결과 페이지가 없습니다.')
    if len(names) > 10000:
        raise ValueError('한 번에 10,000페이지까지 읽을 수 있습니다. 폴더를 나눠 주세요.')
    stems = Counter(Path(n).stem.casefold() for n in names if _leaf(n))
    by_stem = {}
    for path in result_dir.iterdir():
        if path.is_file() and path.suffix.lower() in RESULT_SUFFIXES:
            by_stem.setdefault(path.stem.casefold(), []).append(path)
    review = set(review_pages)
    output = []
    for name in names:
        check_cancel(stop)
        if not _leaf(name):
            output.append(Page(str(name), None, issue='지원하지 않는 페이지 경로입니다.'))
            continue
        if stems[Path(name).stem.casefold()] > 1:
            output.append(Page(name, None, issue='같은 기본 파일명의 원본이 여러 개여서 결과를 구분할 수 없습니다.'))
            continue
        candidates = by_stem.get(Path(name).stem.casefold(), [])
        preferred = [p for p in candidates if p.suffix.lower() == suffix.lower()]
        chosen = preferred if preferred else candidates
        note = '말풍선 배치 검수가 필요합니다. 이전 저장 결과일 수 있습니다.' if name in review else ''
        if len(chosen) != 1:
            issue = '저장된 결과가 없습니다.' if not chosen else '결과 파일이 여러 형식으로 중복됩니다. 사용할 결과를 정리하세요.'
            output.append(Page(name, None, issue=issue, warning=note))
            continue
        if len(candidates) > 1:
            note = (note + ' ' if note else '') + f'여러 결과 형식 중 현재 설정({suffix}) 파일을 사용합니다.'
        path = chosen[0]
        if not path.resolve().is_relative_to(result_dir):
            output.append(Page(name, None, issue='결과 폴더 밖을 가리키는 파일은 열지 않습니다.'))
            continue
        output.append(inspect_page(path, name, note))
    check_cancel(stop)
    return Book(folder, result_dir, tuple(output), tuple(messages))


def plan_export(pages: Sequence[Page], options: MergeOptions) -> ExportPlan:
    """Lay out whole pages before any canvas allocation or filesystem mutation."""
    if options.direction not in ('vertical', 'horizontal') or options.image_format not in ('PNG', 'JPEG'):
        raise ValueError('합치기 방향 또는 이미지 형식을 확인하세요.')
    for value, low, high in ((options.cross_size, 1, MAX_OUTPUT_SIDE), (options.gap, 0, 1000),
                             (options.max_length, 1, MAX_OUTPUT_SIDE), (options.quality, 1, 100)):
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError('출력 크기·간격·품질 설정이 올바르지 않습니다.')
    if not pages:
        raise ValueError('내보낼 페이지를 한 장 이상 선택하세요.')
    for page in pages:
        if page.issue or page.path is None or min(page.width, page.height) < 1:
            raise ValueError(f'{page.name}: {page.issue or "유효한 결과 파일이 없습니다."}')
    vertical = options.direction == 'vertical'
    cross = min(options.cross_size, min(p.width if vertical else p.height for p in pages))
    groups, items, length = [], [], 0

    def finish() -> None:
        if items:
            size = (cross, length) if vertical else (length, cross)
            if max(size) > MAX_OUTPUT_SIDE or size[0] * size[1] > MAX_OUTPUT_PIXELS:
                raise ValueError('한 파일의 크기가 앱 보호 한도를 초과합니다. 기준 크기를 줄이거나 분할 저장을 켜세요.')
            groups.append(Group(size, tuple(items)))

    for page in pages:
        along = max(1, round((page.height if vertical else page.width) * cross /
                             (page.width if vertical else page.height)))
        if options.split and along > options.max_length:
            raise ValueError(f'{page.name}: 한 페이지가 분할 길이보다 큽니다. 기준 크기를 줄이거나 분할 길이를 늘리세요. 페이지 중간은 자르지 않습니다.')
        next_length = length + (options.gap if items else 0) + along
        if options.split and items and (next_length > options.max_length or cross * next_length > MAX_OUTPUT_PIXELS):
            finish()
            items, length = [], 0
        start = length + (options.gap if items else 0)
        size = (cross, along) if vertical else (along, cross)
        box = (0, start, size[0], size[1]) if vertical else (start, 0, size[0], size[1])
        items.append(Placement(page, box))
        length = start + along
    finish()
    return ExportPlan(tuple(groups), options, cross)


def _verified_image(page: Page) -> Image.Image:
    if page.path is None or signature(page.path) != page.stamp:
        raise ValueError(f'{page.name}: 미리보기 이후 파일이 변경되었습니다. 창을 닫고 다시 열어 주세요.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(page.path) as source:
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError(f'{page.name}: 입력 이미지 크기 제한을 초과합니다.')
            image = ImageOps.exif_transpose(source)
            image.load()
            if image.size != (page.width, page.height) or signature(page.path) != page.stamp:
                image.close()
                raise ValueError(f'{page.name}: 이미지 내용이 변경되었습니다.')
            return image.copy() if image is source else image


def render_group(group: Group, *, preview_size: Optional[tuple[int, int]] = None,
                 stop: Optional[threading.Event] = None) -> Image.Image:
    check_cancel(stop)
    scale = min(1.0, preview_size[0] / group.size[0], preview_size[1] / group.size[1]) if preview_size else 1.0
    size = tuple(max(1, round(n * scale)) for n in group.size)
    if max(size) > MAX_OUTPUT_SIDE or size[0] * size[1] > MAX_OUTPUT_PIXELS:
        raise ValueError('이미지 크기가 앱 보호 한도를 초과합니다.')
    canvas = Image.new('RGB', size, 'white')
    try:
        for item in group.items:
            check_cancel(stop)
            x, y, w, h = item.box
            target = (max(1, round(w * scale)), max(1, round(h * scale)))
            with _verified_image(item.page) as source:
                rgba = source.convert('RGBA')
                try:
                    resized = rgba.resize(target, Image.Resampling.LANCZOS)
                    try:
                        canvas.paste(resized, (round(x * scale), round(y * scale)), resized)
                    finally:
                        resized.close()
                finally:
                    rgba.close()
            check_cancel(stop)
        return canvas
    except BaseException:
        canvas.close()
        raise


def export_plan(plan: ExportPlan, destination: Path, stop: Optional[threading.Event] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Publish a new batch folder only after every page was rendered successfully."""
    check_cancel(stop)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bt-export-', dir=destination))
    try:
        extension = '.png' if plan.options.image_format == 'PNG' else '.jpg'
        for i, group in enumerate(plan.groups, 1):
            check_cancel(stop)
            with render_group(group, stop=stop) as image:
                output = temporary / f'{i:03d}{extension}'
                with output.open('xb') as stream:
                    image.save(stream, format=plan.options.image_format, quality=plan.options.quality)
                    stream.flush()
                    os.fsync(stream.fileno())
            if progress is not None:
                progress(i, len(plan.groups))
        check_cancel(stop)
        # Re-check the whole snapshot, including pages rendered before later files.
        for group in plan.groups:
            for item in group.items:
                if signature(item.page.path) != item.page.stamp:
                    raise ValueError(f'{item.page.name}: 내보내기 중 파일이 변경되었습니다. 다시 열어 주세요.')
        check_cancel(stop)
        final = destination / f'읽기용_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
        if final.exists():
            raise FileExistsError('같은 이름의 결과 폴더가 있습니다. 다시 시도하세요.')
        temporary.rename(final)
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class ReaderHistory:
    """Persist only reading positions; malformed existing files remain untouched."""
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records = {}
        self.warning = ''
        if self.path.exists():
            try:
                if self.path.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError('기록 크기 제한 초과')
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('books'), dict):
                    raise ValueError('알 수 없는 형식')
                self.records = data['books']
            except (OSError, ValueError) as exc:
                self.warning = f'이어 읽기 기록을 읽지 못했습니다. 기존 파일은 보존됩니다: {exc}'

    @staticmethod
    def _key(book: str) -> str:
        return hashlib.sha256(os.path.normcase(str(Path(book).resolve())).encode('utf-8')).hexdigest()

    def get(self, book: str) -> dict:
        item = self.records.get(self._key(book), {})
        if not isinstance(item, dict) or not isinstance(item.get('page'), str):
            return {}
        try:
            offset, zoom = float(item.get('offset', 0)), int(item.get('zoom', 100))
            if not math.isfinite(offset):
                return {}
            return {'page': item['page'], 'offset': min(1, max(0, offset)),
                    'zoom': min(300, max(25, zoom)),
                    'mode': item.get('mode') if item.get('mode') in ('scroll', 'page') else 'scroll',
                    'fit': item.get('fit') if item.get('fit') in ('width', 'page') else 'width'}
        except (TypeError, ValueError, OverflowError):
            return {}

    def save_position(self, book: str, page: str, offset: float, mode: str, zoom: int, fit: str) -> None:
        if self.warning:
            raise ValueError(self.warning)
        if not math.isfinite(offset):
            raise ValueError('읽기 위치가 올바르지 않습니다.')
        record = {'page': page, 'offset': min(1, max(0, offset)), 'mode': mode,
                  'zoom': min(300, max(25, zoom)), 'fit': fit}
        records = dict(self.records)
        records[self._key(book)] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='reader-', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump({'version': 1, 'books': records}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            self.records = records
        finally:
            if os.path.exists(name):
                os.unlink(name)
