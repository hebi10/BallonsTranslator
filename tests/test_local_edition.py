from unittest.mock import patch
import json
import subprocess
import sys

from ballontranslator.utils import local_edition
from ballontranslator.utils.config import ProgramConfig


def test_profile_is_fixed_local_and_does_not_mutate_existing_config_file(tmp_path):
    old = tmp_path / 'config.json'
    old.write_text('{"keep":"user settings"}', encoding='utf-8')
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    assert (config.module.textdetector, config.module.ocr, config.module.inpainter,
            config.module.translator) == ('ctd', 'local_auto', 'lama_large_512px', 'LLMTranslator')
    assert config.module.translate_target == '한국어'
    assert len(config.module.llm_profiles) == 1
    profile = config.module.llm_profiles[0]
    assert profile.model == local_edition.MODEL_TAG
    assert profile.base_url == 'http://127.0.0.1:11435/v1'
    assert profile.support_vision and not profile.support_image
    assert config.module.local_visual_context and config.module.llm_translate_vision
    assert not config.check_update_on_startup
    assert old.read_text(encoding='utf-8') == '{"keep":"user settings"}'


def test_profile_keeps_user_glossary_and_prompt():
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    config.module.llm_glossary_path = 'my-terms.tsv'
    config.module.llm_profiles[0].prompt += ' Character A uses polite Korean.'
    local_edition.apply_local_profile(config)
    assert config.module.llm_glossary_path == 'my-terms.tsv'
    assert 'Character A' in config.module.llm_profiles[0].prompt


def test_local_visual_context_preference_survives_config_reload_and_profile_refresh():
    from ballontranslator.utils.config import ModuleConfig
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    config.module.local_visual_context = False
    config.module = ModuleConfig(**config.module.get_saving_params())
    local_edition.apply_local_profile(config)
    assert not config.module.local_visual_context
    assert not config.module.llm_translate_vision
    assert config.module.llm_profiles[0].support_vision


def test_old_or_malformed_visual_context_setting_uses_safe_default():
    from ballontranslator.utils.config import ModuleConfig
    assert ModuleConfig().local_visual_context is True
    with patch('ballontranslator.utils.config.LOGGER.warning') as warning:
        restored = ModuleConfig(local_visual_context='false')
    assert restored.local_visual_context is True
    warning.assert_called()


def test_local_catalog_contains_only_recommended_modules():
    assert local_edition.module_allowed('translator', 'LLMTranslator')
    assert not local_edition.module_allowed('translator', 'google')
    assert local_edition.module_allowed('ocr', 'mit48px')
    assert not local_edition.module_allowed('ocr', 'google_lens')
    assert not local_edition.module_allowed('inpainter', 'LLMInpaint')


def test_readiness_is_passive_and_reports_missing_files():
    with patch('ballontranslator.utils.local_edition._model_files_present', return_value=False):
        items = local_edition.local_readiness()
    assert len(items) == 4
    assert all(isinstance(item[1], bool) for item in items)
    assert not any(item[1] for item in items[:3])


def test_active_runtime_exposes_only_fixed_catalog():
    result = subprocess.run([sys.executable, '-c', '''
import json
from ballontranslator.utils.local_edition import activate_local_edition
activate_local_edition()
from ballontranslator.modules import GET_VALID_TRANSLATORS, GET_VALID_OCR, GET_VALID_INPAINTERS, TRANSLATORS
print(json.dumps([GET_VALID_TRANSLATORS(), GET_VALID_OCR(), GET_VALID_INPAINTERS(), TRANSLATORS.get_spec('LLMTranslator').dependencies]))
'''], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [
        ['LLMTranslator'], ['local_auto', 'manga_ocr', 'mit48px'], ['lama_large_512px'], []]


def test_local_requester_uses_native_client_without_openai():
    from ballontranslator.modules.llm_chat import LLMChatRequester
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    requester = LLMChatRequester()
    with patch.object(local_edition, '_active', True), patch(
        'ballontranslator.utils.local_ollama.OllamaClient.chat',
        return_value={'content': '번역', 'finish_reason': 'stop', 'usage': {}},
    ) as native, patch.object(requester, '_initialize_client', side_effect=AssertionError('cloud client')):
        result = requester.request_chat_completion(config.module.llm_profiles[0], {
            'model': local_edition.MODEL_TAG, 'messages': [],
        })
    assert result.content == '번역'
    native.assert_called_once()


def test_local_translator_separates_image_and_language_model_memory():
    from ballontranslator.modules.translators.trans_llm import LLMTranslator
    translator = LLMTranslator('日本語', '한국어', raise_unsupported_lang=False)
    with patch.object(local_edition, '_active', True):
        assert translator.low_vram_mode


def test_local_edition_rejects_image_generation_before_selecting_a_profile():
    import pytest
    from ballontranslator.ui.text_engine.effects.image_generation import (
        ImageGenerationBackendUnavailable, create_image_generation_backend,
    )
    from ballontranslator.utils.text_effects import ImageGenerationRecipe
    with patch.object(local_edition, '_active', True), patch(
        'ballontranslator.ui.text_engine.effects.image_generation.profile_by_id',
        side_effect=AssertionError('remote profile lookup'),
    ), pytest.raises(ImageGenerationBackendUnavailable, match='로컬'):
        create_image_generation_backend(ImageGenerationRecipe())



def test_english_profile_selects_mit_ocr_and_reuses_qwen_model():
    from ballontranslator.utils.config import ModuleConfig
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    config.module.translate_source = 'English'
    local_edition.apply_local_profile(config)
    assert config.module.translate_source == 'English'
    assert config.module.translate_target == '한국어'
    assert config.module.ocr == 'mit48px'
    assert config.module.llm_profiles[0].model == local_edition.MODEL_TAG
    assert 'English comic dialogue' in config.module.llm_profiles[0].prompt
    config.module = ModuleConfig(**config.module.get_saving_params())
    local_edition.apply_local_profile(config)
    assert config.module.translate_source == 'English'
    assert config.module.ocr == 'mit48px'
    config.module.translate_source = '日本語'
    local_edition.apply_local_profile(config)
    assert config.module.ocr == 'manga_ocr'
    assert 'Japanese comic dialogue' in config.module.llm_profiles[0].prompt


def test_invalid_source_falls_back_to_auto_and_custom_prompt_is_retained():
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    config.module.llm_profiles[0].prompt = 'Custom character glossary instructions.'
    config.module.translate_source = 'English'
    local_edition.apply_local_profile(config)
    assert config.module.llm_profiles[0].prompt == 'Custom character glossary instructions.'
    config.module.translate_source = 'invalid'
    local_edition.apply_local_profile(config)
    assert config.module.translate_source == 'Auto'
    assert config.module.ocr == 'local_auto'


def test_readiness_checks_only_selected_ocr(monkeypatch):
    from ballontranslator.utils.config import pcfg
    monkeypatch.setattr(pcfg.module, 'translate_source', 'English')
    checked = []
    def fake_present(spec):
        checked.append(spec.key)
        return False
    with patch('ballontranslator.utils.local_edition._model_files_present', side_effect=fake_present):
        rows = local_edition.local_readiness()
    assert len(rows) == 4
    assert '영어 인식' in rows[1][0]
    assert 'mit48px' in checked
    assert 'manga_ocr' not in checked
    with patch('ballontranslator.utils.local_edition._model_files_present', return_value=False):
        rows_ja = local_edition.local_readiness('日本語')
    assert '일본어 인식' in rows_ja[1][0]


def test_auto_prepares_and_checks_both_local_ocr_models(monkeypatch):
    from threading import Event
    import ballontranslator.modules.prepare_local_files as preparation
    checked = []
    prepared = []
    monkeypatch.setattr(preparation, '_missing_module_requirements', lambda spec: [])
    monkeypatch.setattr(local_edition, '_model_files_present', lambda spec: checked.append(spec.key) or True)
    monkeypatch.setattr(preparation, 'ensure_module_files', lambda spec, **kwargs: prepared.append(spec.key) or True)
    rows = local_edition.local_readiness('Auto')
    assert len(rows) == 5
    assert {'manga_ocr', 'mit48px'}.issubset(checked)
    local_edition.prepare_local_models(lambda message: None, Event(), 'Auto')
    assert {'manga_ocr', 'mit48px'}.issubset(prepared)
    assert len(prepared) == len(set(prepared))


def test_explicit_manual_japanese_survives_initial_profile_activation():
    config = ProgramConfig()
    local_edition.apply_local_profile(config, '日本語')
    assert config.module.translate_source == '日本語'
    assert config.module.ocr == 'manga_ocr'
    local_edition.apply_local_profile(config)
    assert config.module.translate_source == '日本語'


def test_auto_registry_is_lazy_and_does_not_import_ocr_weights():
    result = subprocess.run([sys.executable, '-c', """
import sys
from ballontranslator.modules import OCR
spec = OCR.get_spec('local_auto')
assert spec is not None and spec.metadata_warnings == []
assert 'ballontranslator.modules.ocr.ocr_local_auto' not in sys.modules
assert 'ballontranslator.modules.ocr.ocr_manga' not in sys.modules
assert 'ballontranslator.modules.ocr.ocr_mit' not in sys.modules
"""], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
