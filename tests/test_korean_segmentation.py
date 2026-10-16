import pytest

from ballontranslator.utils.text_processing import is_cjk, seg_text


def test_korean_short_words_keep_their_own_break_opportunities():
    words, delimiter = seg_text('난 네 곁에 늘 있을 거야.', '한국어')

    assert words == ['난 ', '네 ', '곁에 ', '늘 ', '있을 ', '거야.']
    assert delimiter == ''


@pytest.mark.parametrize('text', [
    '',
    '   ',
    '  나는   너를\t기다렸어.  ',
    '이름은 A.B야!정말?',
    '「정말이야?」 괜찮아…',
    '가자.\n\n다음은 너야.',
    ' 첫째\r\n 둘째\r셋째\n',
    '둘\u3000사이\u00a0약속',
    '한글과 English 123, emoji 👩🏽‍💻도 그대로.',
])
def test_korean_segmentation_preserves_all_source_characters(text):
    words, delimiter = seg_text(text, '한국어')

    assert delimiter.join(words) == text


def test_korean_hard_breaks_are_separate_tokens():
    words, delimiter = seg_text('첫째\n\n 둘째\r\n셋째', '한국어')

    assert words == ['첫째', '\n', '\n', ' 둘째', '\r\n', '셋째']
    assert delimiter == ''


@pytest.mark.parametrize(('text', 'expected'), [
    ('그는 “ 가자 ” 라고 말했다.', ['그는 ', '“ 가자 ” ', '라고 ', '말했다.']),
    ('여기 ( 정말 ) 맞아 ?', ['여기 ', '( 정말 ) ', '맞아 ?']),
    ('잠깐 「 『 기다려 』 」', ['잠깐 ', '「 『 기다려 』 」']),
])
def test_korean_isolated_punctuation_stays_with_its_word(text, expected):
    words, _ = seg_text(text, '한국어')

    assert words == expected
    assert ''.join(words) == text


def test_korean_punctuation_does_not_swallow_explicit_line_breaks():
    words, _ = seg_text('“\n안녕\n”', '한국어')

    assert words == ['“', '\n', '안녕', '\n', '”']


def test_long_korean_word_is_preserved_for_width_based_layout():
    text = '띄어쓰기가없는아주긴한국어대사입니다.'

    assert seg_text(text, '한국어') == ([text], '')


def test_korean_keeps_existing_non_cjk_layout_routing():
    assert not is_cjk('한국어')


def test_english_segmentation_is_unchanged():
    assert seg_text('I am a hero.', 'English') == (['I am a', 'hero.'], ' ')


def test_japanese_segmentation_is_unchanged():
    assert seg_text('こんにちは。\n次へ', '日本語') == (
        list('こんにちは。次へ'), '',
    )
