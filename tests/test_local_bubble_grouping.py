"""Synthetic CTD postprocessing cases; no detector weights or network needed."""

from collections import Counter
from itertools import permutations

import cv2
import numpy as np
import pytest

from ballontranslator.utils.textblock import group_output, sort_regions


def line(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def page():
    return np.full((480, 320, 3), 255, np.uint8)


def group(lines, canvas, **kwargs):
    kwargs.setdefault('local_bubble_grouping', True)
    return group_output([], np.asarray(lines, dtype=np.int32), 320, 480,
                        canvas=canvas, **kwargs)


def line_inventory(blocks):
    return Counter(tuple(sorted(tuple(point) for point in polygon))
                   for block in blocks for polygon in block.lines)


@pytest.mark.parametrize('gap', [0, 4, 12])
def test_distinct_closed_balloons_stay_separate_even_with_overlapping_boxes(gap):
    canvas = page()
    cv2.rectangle(canvas, (25, 20), (99, 240), (0, 0, 0), 3)
    cv2.rectangle(canvas, (99 + gap, 20), (173 + gap, 240), (0, 0, 0), 3)
    lines = [line(73, 55, 103, 205), line(95 + gap, 55, 125 + gap, 205)]

    blocks = group(lines, canvas)

    assert len(blocks) == 2
    assert line_inventory(blocks) == line_inventory(group(lines, None))


def test_same_balloon_columns_merge_and_read_from_right_to_left():
    canvas = page()
    cv2.ellipse(canvas, (125, 140), (90, 130), 0, 0, 360, (0, 0, 0), 3)
    left, right = line(83, 55, 113, 205), line(121, 55, 151, 205)
    # Glyph-like ink remains inside the same surrounding white component.
    for x in (93, 131):
        for y in (80, 115, 150, 185):
            cv2.rectangle(canvas, (x, y), (x + 7, y + 12), (0, 0, 0), -1)

    blocks = group([left, right], canvas)

    assert len(blocks) == 1
    assert blocks[0].lines == [right, left]


@pytest.mark.parametrize('boundary', ['blank', 'open', 'missing'])
def test_unknown_boundaries_keep_existing_geometry_fallback(boundary):
    canvas = page()
    if boundary == 'open':
        for x1, x2 in ((25, 99), (103, 177)):
            cv2.rectangle(canvas, (x1, 20), (x2, 240), (0, 0, 0), 3)
            cv2.rectangle(canvas, (x1 + 15, 16), (x1 + 25, 24), (255, 255, 255), -1)
    elif boundary == 'missing':
        canvas = None

    blocks = group([line(65, 55, 95, 205), line(109, 55, 139, 205)], canvas)

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 2


def test_horizontal_dialogue_in_distinct_closed_balloons_stays_separate():
    canvas = page()
    cv2.rectangle(canvas, (20, 20), (240, 109), (0, 0, 0), 3)
    cv2.rectangle(canvas, (20, 109), (240, 198), (0, 0, 0), 3)

    blocks = group([line(55, 85, 205, 115), line(55, 107, 205, 137)], canvas)

    assert len(blocks) == 2
    assert sum(len(block.lines) for block in blocks) == 2


def test_unknown_line_does_not_bridge_two_known_closed_balloons():
    canvas = page()
    cv2.rectangle(canvas, (45, 20), (100, 240), (0, 0, 0), 3)
    cv2.rectangle(canvas, (125, 20), (185, 240), (0, 0, 0), 3)
    left = line(65, 55, 95, 205)
    unknown = line(92, 55, 122, 205)
    right = line(130, 55, 160, 205)

    blocks = group([left, unknown, right], canvas)

    assert len(blocks) == 2
    assert sum(len(block.lines) for block in blocks) == 3
    assert not any(left in block.lines and right in block.lines for block in blocks)


def test_overlapping_incompatible_font_sizes_preserve_both_detected_lines():
    broad = line(90, 20, 140, 240)
    narrow = line(130, 90, 140, 170)

    blocks = group([broad, narrow], None)

    assert len(blocks) == 2
    assert line_inventory(blocks) == Counter({tuple(sorted(map(tuple, broad))): 1,
                                             tuple(sorted(map(tuple, narrow))): 1})


def test_intersecting_lines_with_different_directions_do_not_merge():
    descending = [[80, 30], [100, 30], [175, 180], [155, 180]]
    ascending = [[160, 30], [180, 30], [105, 180], [85, 180]]

    blocks = group([descending, ascending], None)

    assert len(blocks) == 2
    assert sum(len(block.lines) for block in blocks) == 2


def test_tall_vertical_overlap_reads_right_to_left_without_moving_lower_row():
    left = line(40, 20, 65, 170)
    right = line(180, 100, 205, 360)
    below = line(50, 390, 75, 450)

    blocks = group([left, below, right], None)

    assert [block.lines for block in blocks] == [[right], [left], [below]]


def test_nonlocal_grouping_keeps_legacy_merging_and_order():
    canvas = page()
    cv2.rectangle(canvas, (25, 20), (99, 240), (0, 0, 0), 3)
    cv2.rectangle(canvas, (99, 20), (173, 240), (0, 0, 0), 3)
    assert len(group([line(73, 55, 103, 205), line(95, 55, 125, 205)], canvas,
                     local_bubble_grouping=False)) == 1

    left, right = line(40, 20, 65, 170), line(180, 100, 205, 360)
    assert [block.lines for block in group([right, left], None,
                                           local_bubble_grouping=False)] == [[left], [right]]


def test_generic_grouping_does_not_implicitly_enable_local_policy(monkeypatch):
    from ballontranslator.utils import local_edition
    monkeypatch.setattr(local_edition, '_active', True)
    broad, narrow = line(90, 20, 140, 240), line(130, 90, 140, 170)

    blocks = group_output([], np.asarray([broad, narrow]), 320, 480)

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 2


@pytest.mark.parametrize('color', [150, (170, 180, 150)])
def test_dark_gray_and_colored_balloons_fall_back_when_white_evidence_is_unknown(color):
    shape = (480, 320) if isinstance(color, int) else (480, 320, 3)
    canvas = np.full(shape, color, np.uint8)
    cv2.rectangle(canvas, (25, 20), (99, 240), 0, 3)
    cv2.rectangle(canvas, (103, 20), (177, 240), 0, 3)

    blocks = group([line(65, 55, 95, 205), line(109, 55, 139, 205)], canvas)

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 2


def test_narrow_closed_balloons_remain_separate():
    canvas = page()
    cv2.rectangle(canvas, (40, 20), (80, 240), (0, 0, 0), 3)
    cv2.rectangle(canvas, (80, 20), (120, 240), (0, 0, 0), 3)

    blocks = group([line(52, 55, 77, 205), line(83, 55, 108, 205)], canvas)

    assert len(blocks) == 2
    assert sum(len(block.lines) for block in blocks) == 2


def test_ink_crossing_a_balloon_does_not_invent_separate_enclosures():
    canvas = page()
    cv2.rectangle(canvas, (35, 10), (215, 270), (0, 0, 0), 3)
    # A glyph/ink stroke breaks the bright interior into two components, but
    # neither component encloses a complete detected text column.
    cv2.line(canvas, (35, 140), (215, 140), (0, 0, 0), 4)

    blocks = group([line(83, 55, 113, 205), line(121, 55, 151, 205)], canvas)

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 2


def test_boundary_scan_runs_once_per_page_and_keeps_source_pixels(monkeypatch):
    canvas = page()
    cv2.rectangle(canvas, (20, 20), (250, 260), (0, 0, 0), 3)
    original = canvas.copy()
    connected_components = cv2.connectedComponentsWithStats
    scans = []

    def count_scans(image, *args, **kwargs):
        scans.append(image.shape)
        return connected_components(image, *args, **kwargs)

    monkeypatch.setattr(cv2, 'connectedComponentsWithStats', count_scans)
    lines = [line(x, 55, x + 25, 205) for x in (45, 80, 115, 150, 185)]

    blocks = group(lines, canvas)

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 5
    assert scans == [canvas.shape[:2]]
    np.testing.assert_array_equal(canvas, original)


def test_local_order_retains_block_objects_text_and_labels():
    blocks = group([line(40, 20, 65, 170), line(180, 100, 205, 360)], None,
                   local_bubble_grouping=False)
    left, right = blocks
    left.text, left.label = ['left dialogue'], 'left-id'
    right.text, right.label = ['right dialogue'], 'right-id'

    ordered = sort_regions(blocks, right_to_left=True, japanese_vertical_overlap=True)

    assert ordered[0] is right
    assert ordered[1] is left
    assert [(block.text, block.label) for block in ordered] == [
        (['right dialogue'], 'right-id'), (['left dialogue'], 'left-id')]
    assert blocks[0] is left and blocks[1] is right


def test_explicit_left_to_right_order_keeps_precedence_over_vertical_overlap():
    blocks = group([line(40, 20, 65, 170), line(180, 100, 205, 360)], None,
                   local_bubble_grouping=False)

    ordered = sort_regions(blocks, right_to_left=False, japanese_vertical_overlap=True)

    assert ordered[0] is blocks[0]
    assert ordered[1] is blocks[1]


@pytest.mark.parametrize('order', list(permutations(range(3))))
@pytest.mark.parametrize('bottom_rect', [(200, 150, 225, 250), (130, 150, 140, 220), (130, 150, 200, 180)])
def test_overlap_chain_does_not_move_a_lower_row_before_a_disjoint_upper_row(order, bottom_rect):
    top = line(20, 0, 45, 100)
    middle = line(110, 60, 135, 190)
    bottom = line(*bottom_rect)
    lines = [top, middle, bottom]

    blocks = group([lines[index] for index in order], None)

    ordered_lines = [block.lines[0] for block in blocks]
    assert len(blocks) == 3
    assert ordered_lines.index(top) < ordered_lines.index(bottom)
    assert sorted(map(str, ordered_lines)) == sorted(map(str, lines))
