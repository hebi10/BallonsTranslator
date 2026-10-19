"""Region inference contracts with synthetic models and no network or weights."""

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.ui.region_rework import RegionReworkRequest, rework_region
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


def make_request(channels=3):
    project = ProjImgTrans()
    project.current_img = 'page.png'
    project.pages['page.png'] = [
        TextBlock([24, 0, 30, 4], text=['前'], translation='앞 대사'),
        TextBlock([10, 8, 16, 16], text=['古い'], translation='기존 대사'),
        TextBlock([0, 22, 6, 27], text=['後'], translation='뒤 대사'),
    ]
    project._image_info = {'page.png': {'sentinel': ['untouched']}}
    image = np.arange(28 * 32 * channels, dtype=np.uint8).reshape(28, 32, channels)
    inpainted = np.full_like(image, 27)
    mask = np.zeros((28, 32), np.uint8)
    mask[8, 10] = 255
    mask[0, 0] = 255
    return RegionReworkRequest((8, 6, 20, 18), image, inpainted, mask,
                               project, 'page.png', (1,))


class Models:
    def __init__(self):
        self.calls = []
        self.ocr_input = None
        self.inpaint_input = None
        self.context = None
        self.detect_blocks = [TextBlock(
            [2, 2, 8, 10], lines=[[[2, 2], [8, 2], [8, 10], [2, 10]]],
            _bounding_rect=[2, 2, 6, 8], src_is_vertical=True,
        )]
        self.detector = SimpleNamespace(detect=self.detect)
        self.ocr = SimpleNamespace(run_ocr=self.read)
        self.inpainter = SimpleNamespace(inpaint=self.paint)
        self.translator = SimpleNamespace(translate_textblk_lst=self.translate)

    def detect(self, image, project=None):
        self.calls.append('detect')
        self.detect_input = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[4, 4] = 255
        return mask, copy.deepcopy(self.detect_blocks)

    def read(self, image, blocks):
        self.calls.append('ocr')
        self.ocr_input = image.copy()
        self.ocr_boxes = [list(block.xyxy) for block in blocks]
        for block in blocks:
            block.text = ['新しい']
        return blocks

    def paint(self, image, mask, *, textblock_list=None, check_need_inpaint=False):
        self.calls.append('inpaint')
        self.inpaint_input = (image.copy(), mask.copy(), textblock_list, check_need_inpaint)
        return np.full_like(image, 199)

    def translate(self, blocks, *, project, page_key, full_page=False):
        self.calls.append('translate')
        self.context = project
        self.translation_inputs = blocks
        self.full_page = full_page
        for block in blocks:
            block.translation = '새 대사'

    def run(self, request, **kwargs):
        return rework_region(request, self.detector, self.ocr, self.translator,
                             self.inpainter, kwargs.pop('stop_event', threading.Event()),
                             kwargs.pop('progress', lambda _stage, _value: None), **kwargs)


def test_original_crop_and_full_page_geometry_preserve_context_neighbors():
    request = make_request()
    models = Models()
    before = copy.deepcopy(request.project.pages)
    result = models.run(request, before_translate=lambda: models.calls.append('unload'))

    np.testing.assert_array_equal(models.detect_input, request.image[6:18, 8:20])
    np.testing.assert_array_equal(models.ocr_input, request.image[6:18, 8:20])
    assert models.ocr_boxes == [[2, 2, 8, 10]]
    assert result.blocks[0].xyxy == [10, 8, 16, 16]
    assert result.blocks[0]._bounding_rect == [10, 8, 6, 8]
    assert result.blocks[0].lines == [[[10, 8], [16, 8], [16, 16], [10, 16]]]
    assert models.calls == ['detect', 'ocr', 'inpaint', 'unload', 'translate']
    assert not models.full_page
    page = models.context.pages['page.png']
    assert page[1] is result.blocks[0] is models.translation_inputs[0]
    assert [block.get_text() for block in page] == ['前', '新しい', '後']
    assert [block.translation for block in page] == ['앞 대사', '새 대사', '뒤 대사']
    assert [block.to_dict() for block in request.project.pages['page.png']] == [
        block.to_dict() for block in before['page.png']
    ]


def test_old_mask_and_new_mask_change_only_masked_roi_pixels():
    request = make_request()
    models = Models()
    originals = [array.copy() for array in (request.image, request.inpainted, request.mask)]
    result = models.run(request)
    expected_mask = np.zeros((12, 12), np.uint8)
    expected_mask[2, 2] = expected_mask[4, 4] = 255
    np.testing.assert_array_equal(result.mask, expected_mask)
    expected = request.inpainted[6:18, 8:20].copy()
    expected[expected_mask > 0] = 199
    np.testing.assert_array_equal(result.inpainted, expected)
    assert result.rect == request.rect
    image, mask, blocks, quick_fill = models.inpaint_input
    assert image.shape[0] > 12 and image.shape[1] > 12
    expected_context = request.inpainted.copy()
    expected_context[6:18, 8:20] = request.image[6:18, 8:20]
    np.testing.assert_array_equal(image, expected_context)
    assert np.count_nonzero(mask) == 2
    assert mask[0, 0] == 0
    assert blocks is None and quick_fill is False
    for original, current in zip(originals, (request.image, request.inpainted, request.mask)):
        np.testing.assert_array_equal(original, current)


def test_current_alpha_is_preserved_even_when_model_changes_it():
    request = make_request(4)
    request.inpainted[:, :, 3] = np.arange(28 * 32, dtype=np.uint8).reshape(28, 32)
    result = Models().run(request)
    np.testing.assert_array_equal(result.inpainted[:, :, 3], request.inpainted[6:18, 8:20, 3])


def test_only_new_detection_mask_is_dilated():
    request = make_request()
    request.mask[8, 10] = 0
    request.mask[7, 9] = 255
    request.dilate = 1
    result = Models().run(request)
    assert result.mask[1, 1] == 255
    assert result.mask[0, 1] == result.mask[1, 0] == 0
    assert result.mask[4, 3] == result.mask[4, 5] == 255


def test_existing_dilation_slider_maximum_is_supported():
    request = make_request()
    request.dilate = 100
    result = Models().run(request)
    assert result.mask.shape == (12, 12)
    assert np.all(result.mask == 255)


@pytest.mark.parametrize('stage', ['detect', 'ocr', 'inpaint', 'translate'])
def test_cancel_after_each_stage_never_returns_partial_result(stage):
    request = make_request()
    models = Models()
    event = threading.Event()

    def progress(current, percent):
        if current == stage and percent == 100:
            event.set()

    with pytest.raises(LLMRequestStopped):
        models.run(request, stop_event=event, progress=progress)
    assert models.calls[-1] == stage
    assert request.project.pages['page.png'][1].translation == '기존 대사'
    np.testing.assert_array_equal(request.inpainted, 27)


@pytest.mark.parametrize('empty_stage', ['detect', 'ocr', 'translate'])
def test_empty_stage_output_keeps_original_request(empty_stage):
    request = make_request()
    models = Models()
    if empty_stage == 'detect':
        models.detect_blocks = []
    elif empty_stage == 'ocr':
        models.ocr.run_ocr = lambda _image, blocks: blocks
    else:
        models.translator.translate_textblk_lst = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        models.run(request)
    assert request.project.pages['page.png'][1].translation == '기존 대사'
    np.testing.assert_array_equal(request.inpainted, 27)


def test_model_failure_does_not_publish_partial_text_or_pixels():
    request = make_request()
    models = Models()

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic failure')

    models.translator.translate_textblk_lst = fail
    with pytest.raises(RuntimeError, match='synthetic failure'):
        models.run(request)
    assert request.project.pages['page.png'][1].get_text() == '古い'
    np.testing.assert_array_equal(request.inpainted, 27)


def test_project_metadata_and_neighbor_mutation_is_isolated_from_request():
    request = make_request()
    models = Models()
    translate = models.translator.translate_textblk_lst

    def mutate(blocks, **kwargs):
        translate(blocks, **kwargs)
        kwargs['project']._image_info['page.png']['sentinel'].clear()
        kwargs['project'].pages['page.png'][0].translation = 'changed by model'

    models.translator.translate_textblk_lst = mutate
    models.run(request)
    assert request.project._image_info['page.png']['sentinel'] == ['untouched']
    assert request.project.pages['page.png'][0].translation == '앞 대사'


@pytest.mark.parametrize('rect', [(-1, 0, 5, 5), (0, 0, 33, 28), (8, 6, 8, 18)])
def test_invalid_rectangle_rejected_before_models_run(rect):
    request = make_request()
    request.rect = rect
    models = Models()
    with pytest.raises(ValueError):
        models.run(request)
    assert models.calls == []


def test_detected_geometry_outside_selection_is_rejected_without_inpainting():
    request = make_request()
    models = Models()
    models.detect_blocks[0].xyxy = [-3, 2, 8, 10]
    with pytest.raises(ValueError):
        models.run(request)
    assert 'inpaint' not in models.calls


def test_detector_cannot_modify_original_snapshot_or_ocr_input():
    request = make_request()
    models = Models()
    original = request.image.copy()
    detect = models.detector.detect

    def changing_detector(image):
        result = detect(image)
        image[:] = 0
        return result

    models.detector.detect = changing_detector
    models.run(request)
    np.testing.assert_array_equal(request.image, original)
    np.testing.assert_array_equal(models.ocr_input, original[6:18, 8:20])


def test_cancel_before_start_never_runs_models():
    models = Models()
    event = threading.Event()
    event.set()
    with pytest.raises(LLMRequestStopped):
        models.run(make_request(), stop_event=event)
    assert models.calls == []


def test_cancel_while_unloading_never_starts_translation():
    models = Models()
    event = threading.Event()
    with pytest.raises(LLMRequestStopped):
        models.run(make_request(), stop_event=event, before_translate=event.set)
    assert models.calls == ['detect', 'ocr', 'inpaint']


def test_no_replaced_blocks_append_without_reordering_neighbors():
    request = make_request()
    request.replaced_indices = ()
    models = Models()
    result = models.run(request)
    page = models.context.pages['page.png']
    assert [block.get_text() for block in page[:-1]] == ['前', '古い', '後']
    assert page[-1] is result.blocks[0]


def test_inpaint_result_wrong_shape_rejected_before_translation():
    models = Models()
    models.inpainter.inpaint = lambda *args, **kwargs: np.zeros((1, 1, 3), np.uint8)
    with pytest.raises(ValueError):
        models.run(make_request())
    assert 'translate' not in models.calls


def test_local_region_inherits_manual_override_and_applies_explicit_direction(monkeypatch):
    request = make_request()
    request.project.local_settings = {'source_language': 'Auto', 'reading_direction': 'rtl'}
    request.project.pages['page.png'][1].source_language_override = 'English'
    models = Models()
    models.detect_blocks.append(TextBlock([9, 2, 11, 10], lines=[[[9, 2], [11, 2], [11, 10], [9, 10]]]))
    monkeypatch.setattr('ballontranslator.ui.region_rework.is_local_edition', lambda: True)
    captured = []
    def route(ocr, image, blocks, project, page_key, **kwargs):
        captured.append((page_key, [item.source_language_override for item in blocks], [item.xyxy[0] for item in blocks]))
        return ocr.run_ocr(image, blocks)
    monkeypatch.setattr('ballontranslator.ui.region_rework.run_local_ocr', route)
    result = models.run(request)
    assert captured == [('page.png', ['', 'English'], [9, 2])]
    assert [item.source_language_override for item in result.blocks] == ['', 'English']
    assert request.project.pages['page.png'][1].source_language_override == 'English'
