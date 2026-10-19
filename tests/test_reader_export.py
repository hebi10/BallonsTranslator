import importlib.util
import json
import threading
from pathlib import Path

import pytest
from PIL import Image


def service():
    assert importlib.util.find_spec('ballontranslator.utils.reader_export') is not None, 'Reader/export service has not been implemented'
    from ballontranslator.utils import reader_export
    return reader_export


def picture(path, size=(100, 150), color='red', mode='RGB'):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)
    return path


def pages(tmp_path, sizes=((100, 150), (100, 150))):
    m = service()
    return tuple(m.inspect_page(picture(tmp_path / f'{i}.png', size, color), f'{i}.png')
                 for i, (size, color) in enumerate(zip(sizes, ['red', 'blue', 'green', 'black'])))


def test_feature_exists():
    service()


def test_vertical_pixels_and_no_enlargement(tmp_path):
    m = service()
    ps = pages(tmp_path, ((100, 150), (200, 300)))
    plan = m.plan_export(ps, m.MergeOptions(cross_size=1080, gap=5))
    assert plan.groups[0].size == (100, 305)
    image = m.render_group(plan.groups[0])
    assert image.getpixel((20, 20)) == (255, 0, 0)
    assert image.getpixel((20, 151)) == (255, 255, 255)
    assert image.getpixel((20, 200)) == (0, 0, 255)


def test_horizontal_order_is_not_mirroring(tmp_path):
    m = service()
    ps = pages(tmp_path, ((100, 150), (200, 300)))
    image = m.render_group(m.plan_export(tuple(reversed(ps)), m.MergeOptions(direction='horizontal', cross_size=150)).groups[0])
    assert image.size == (200, 150)
    assert image.getpixel((20, 20)) == (0, 0, 255)
    assert image.getpixel((150, 20)) == (255, 0, 0)


def test_split_only_at_page_boundaries(tmp_path):
    m = service()
    plan = m.plan_export(pages(tmp_path, ((100, 150),) * 3), m.MergeOptions(cross_size=100, split=True, max_length=305, gap=5))
    assert [g.size for g in plan.groups] == [(100, 305), (100, 150)]
    assert [len(g.items) for g in plan.groups] == [2, 1]


def test_oversize_single_page_is_not_cut(tmp_path):
    m = service()
    with pytest.raises(ValueError, match='페이지'):
        m.plan_export(pages(tmp_path), m.MergeOptions(split=True, max_length=100))


@pytest.mark.parametrize('kwargs', [{'direction': 'diagonal'}, {'cross_size': 0}, {'gap': -1}, {'max_length': 0}, {'quality': 101}, {'image_format': 'EXE'}])
def test_invalid_options_fail_without_files(tmp_path, kwargs):
    m = service()
    with pytest.raises(ValueError):
        m.plan_export(pages(tmp_path), m.MergeOptions(**kwargs))


def test_empty_selection_is_error():
    m = service()
    with pytest.raises(ValueError):
        m.plan_export((), m.MergeOptions())


def test_canvas_budget_checked_before_allocation(tmp_path):
    m = service()
    page = m.Page('tall.png', tmp_path / 'tall.png', 10000, 60000, (0, 0))
    with pytest.raises(ValueError, match='분할|크기'):
        m.plan_export((page,) * 5, m.MergeOptions(cross_size=10000))


def test_snapshot_change_rejects_export(tmp_path):
    m = service()
    ps = pages(tmp_path)
    plan = m.plan_export(ps, m.MergeOptions())
    picture(ps[0].path, (90, 90), 'black')
    with pytest.raises(ValueError, match='변경'):
        m.export_plan(plan, tmp_path / 'out')
    assert not list((tmp_path / 'out').glob('*'))


@pytest.mark.parametrize('fmt', ['PNG', 'JPEG'])
def test_export_is_new_folder_and_preserves_sources(tmp_path, fmt):
    m = service()
    ps = pages(tmp_path)
    before = [p.path.read_bytes() for p in ps]
    plan = m.plan_export(ps, m.MergeOptions(image_format=fmt))
    out = m.export_plan(plan, tmp_path / 'out')
    second = m.export_plan(plan, tmp_path / 'out')
    assert out != second
    files = list(out.iterdir())
    assert len(files) == 1
    with Image.open(files[0]) as im:
        assert im.size == (100, 300)
        assert im.format == fmt
    assert [p.path.read_bytes() for p in ps] == before


def test_cancel_cleans_only_owned_staging(tmp_path):
    m = service()
    ps = pages(tmp_path)
    plan = m.plan_export(ps, m.MergeOptions(split=True, max_length=150))
    dest = tmp_path / 'out'
    dest.mkdir()
    keep = dest / 'keep.txt'
    keep.write_text('keep', encoding='utf-8')
    stop = threading.Event()
    def progress(done, total):
        stop.set()
    with pytest.raises(m.OperationCancelled):
        m.export_plan(plan, dest, stop, progress)
    assert sorted(p.name for p in dest.iterdir()) == ['keep.txt']


def test_preview_is_bounded(tmp_path):
    m = service()
    group = m.plan_export(pages(tmp_path), m.MergeOptions()).groups[0]
    assert m.render_group(group, preview_size=(80, 80)).size == (27, 80)


def test_rgba_flattens_on_white(tmp_path):
    m = service()
    p = m.inspect_page(picture(tmp_path / 'alpha.png', (10, 10), (0, 0, 0, 0), 'RGBA'), 'alpha.png')
    image = m.render_group(m.plan_export((p,), m.MergeOptions()).groups[0])
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_respected(tmp_path):
    m = service()
    path = tmp_path / 'rotate.jpg'
    exif = Image.Exif(); exif[274] = 6
    Image.new('RGB', (20, 10)).save(path, exif=exif)
    page = m.inspect_page(path, 'rotate.jpg')
    assert (page.width, page.height) == (10, 20)
    assert m.render_group(m.plan_export((page,), m.MergeOptions()).groups[0]).size == (10, 20)


def test_manifest_order_and_missing_page_preserved(tmp_path):
    m = service()
    result = tmp_path / 'result'
    picture(result / '10.png'); picture(result / '2.png')
    path = tmp_path / f'imgtrans_{tmp_path.name}.json'
    path.write_text(json.dumps({'pages': {'10.jpg': [], 'missing.jpg': [], '2.jpg': []}}), encoding='utf-8')
    book = m.load_book(tmp_path, result)
    assert [p.name for p in book.pages] == ['10.jpg', 'missing.jpg', '2.jpg']
    assert book.pages[1].issue
    with pytest.raises(ValueError, match='missing'):
        m.plan_export(book.pages, m.MergeOptions())


def test_result_directory_fallback_uses_natural_sort(tmp_path):
    m = service()
    result = tmp_path / 'result'
    picture(result / '10.png'); picture(result / '2.png'); picture(result / '1.png')
    book = m.load_book(tmp_path, result)
    assert [p.name for p in book.pages] == ['1.png', '2.png', '10.png']
    assert book.warnings


def test_corrupt_manifest_is_not_silently_replaced(tmp_path):
    m = service()
    path = tmp_path / f'imgtrans_{tmp_path.name}.json'
    path.write_text('{broken', encoding='utf-8')
    (tmp_path / 'result').mkdir()
    with pytest.raises(ValueError, match='프로젝트'):
        m.load_book(tmp_path, tmp_path / 'result')
    assert path.read_text() == '{broken'


def test_duplicate_stems_and_traversal_are_not_opened(tmp_path):
    m = service()
    picture(tmp_path / 'result' / 'a.png')
    book = m.load_book(tmp_path, tmp_path / 'result', names=('a.jpg', 'a.png', '../outside.png'))
    assert all(p.issue for p in book.pages)


def test_corrupt_image_is_visible_error(tmp_path):
    m = service()
    bad = tmp_path / 'bad.png'; bad.write_bytes(b'not an image')
    assert m.inspect_page(bad, 'bad.png').issue


def test_history_keeps_last_page_and_position(tmp_path):
    m = service()
    store = m.ReaderHistory(tmp_path / 'reader.json')
    store.save_position('book', '003.png', 0.42, 'scroll', 125, 'width')
    assert m.ReaderHistory(tmp_path / 'reader.json').get('book')['page'] == '003.png'
    assert m.ReaderHistory(tmp_path / 'reader.json').get('book')['offset'] == 0.42


def test_corrupt_history_is_read_only_and_preserved(tmp_path):
    m = service()
    p = tmp_path / 'reader.json'; p.write_text('bad', encoding='utf-8')
    store = m.ReaderHistory(p)
    assert store.warning
    with pytest.raises(ValueError):
        store.save_position('book', 'a.png', 0, 'scroll', 100, 'width')
    assert p.read_text() == 'bad'


def test_reader_and_merge_modules_exist():
    assert importlib.util.find_spec('ballontranslator.ui.result_reader') is not None
    assert importlib.util.find_spec('ballontranslator.ui.image_merge_dialog') is not None


def test_changed_earlier_page_during_later_export_rolls_back(tmp_path):
    m = service(); ps = pages(tmp_path)
    plan = m.plan_export(ps, m.MergeOptions(split=True, max_length=150))
    def change(done, total):
        if done == 2:
            picture(ps[0].path, (80, 80), 'black')
    with pytest.raises(ValueError, match='변경'):
        m.export_plan(plan, tmp_path / 'out', progress=change)
    assert not list((tmp_path / 'out').iterdir())


def test_mixed_saved_formats_get_a_visible_notice(tmp_path):
    m = service()
    picture(tmp_path / 'result' / 'one.png')
    picture(tmp_path / 'result' / 'one.jpg')
    book = m.load_book(tmp_path, tmp_path / 'result', names=('one.png',))
    assert book.pages[0].warning
