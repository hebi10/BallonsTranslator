import json
from pathlib import Path

import numpy as np
import pytest

from ballontranslator.utils.io_utils import imwrite
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


def make_project(tmp_path):
    imwrite(str(tmp_path / '01.png'), np.full((30, 40, 3), 255, np.uint8))
    return ProjImgTrans(str(tmp_path))


def test_local_settings_and_review_round_trip(tmp_path):
    project = make_project(tmp_path)
    project.local_settings = {'source_language': 'Auto', 'reading_direction': 'rtl',
                              'work_name': '작품', 'glossary_path': '', 'tone_notes': '이름 유지'}
    project.local_review = {'pages': {}, 'acknowledged': ['abc']}
    project._image_info['01.png']['local_source_language'] = 'English'
    project.save()
    restored = ProjImgTrans(str(tmp_path))
    assert restored.local_settings == project.local_settings
    assert restored.local_review == project.local_review
    assert restored._image_info['01.png']['local_source_language'] == 'English'


def test_bad_optional_settings_preserve_project_and_file(tmp_path):
    project = make_project(tmp_path)
    payload = project.to_dict()
    payload['local_settings'] = {'source_language': [], 'reading_direction': 'rtl',
                                  'tone_notes': 4, 'work_name': '정상 작품', 'future': True}
    payload['local_review'] = []
    payload['image_info']['01.png']['local_source_language'] = []
    path = Path(project.proj_path)
    path.write_text(json.dumps(payload), encoding='utf-8')
    before = path.read_bytes()
    restored = ProjImgTrans(str(tmp_path))
    assert restored.local_settings == {'reading_direction': 'rtl', 'work_name': '정상 작품'}
    assert 'local_source_language' not in restored._image_info['01.png']
    assert list(restored.pages) == ['01.png']
    assert path.read_bytes() == before


def test_loading_another_project_resets_work_and_review(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    project = make_project(first)
    project.local_settings = {'work_name': 'first'}
    project.local_review = {'pages': {}, 'acknowledged': ['old']}
    other = make_project(second)
    project.load(str(second))
    assert project.local_settings == {}
    assert project.local_review == {'pages': {}, 'acknowledged': []}
    assert project.current_img == other.current_img


def test_failed_load_keeps_previous_project_and_corrupt_target(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    project = make_project(first)
    target = make_project(second)
    path = Path(target.proj_path)
    path.write_bytes(b'{invalid')
    identity = project.load_identity
    with pytest.raises(Exception):
        project.load(str(second))
    assert project.directory == str(first)
    assert project.load_identity is identity
    assert project.current_img == '01.png'
    project.save()
    assert path.read_bytes() == b'{invalid'
