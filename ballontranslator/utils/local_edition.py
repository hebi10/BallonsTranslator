"""Fixed local pipeline policy; importing this module performs no IO."""

import os
import sys
import threading
from pathlib import Path
from typing import Callable, List, Tuple

from .local_ollama import MODEL_TAG

_active = False
LOCAL_PROFILE_ID = 'hebi-local-qwen'
LOCAL_SOURCE_OCR = {'日本語': 'manga_ocr', 'English': 'mit48px'}
LOCAL_MODULES = {'textdetector': 'ctd', 'ocr': 'manga_ocr',
                 'inpainter': 'lama_large_512px', 'translator': 'LLMTranslator'}
LOCAL_PROMPT = (
    'Translate Japanese comic dialogue into natural Korean. Preserve every line, meaning, '
    'emotion and speaker intent. Use the provided glossary consistently for names and honorifics. '
    'Use earlier dialogue only as context; do not invent missing facts or identify a speaker '
    'without evidence. Keep established polite or casual speech consistent. '
    'Keep dialogue concise enough for speech bubbles without omitting meaning. '
    'Return exactly the requested JSON, with one translation per input id and no explanations.'
)
ENGLISH_PROMPT = LOCAL_PROMPT.replace('Japanese comic dialogue', 'English comic dialogue')


def local_modules_for_source(source_language: str) -> dict:
    """Choose the offline OCR for the selected language."""
    modules = LOCAL_MODULES.copy()
    modules['ocr'] = LOCAL_SOURCE_OCR.get(source_language, LOCAL_SOURCE_OCR['日本語'])
    return modules


def is_local_edition() -> bool:
    return _active


def activate_local_edition() -> None:
    global _active
    _active = True
    os.environ['OLLAMA_NO_CLOUD'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def module_allowed(module_type: str, key: str) -> bool:
    if module_type == 'ocr':
        return key in LOCAL_SOURCE_OCR.values()
    return LOCAL_MODULES.get(module_type) == key


def apply_local_profile(config) -> None:
    """Constrain this edition's config without touching previous on-disk settings.

    >>> LOCAL_MODULES['ocr']
    'manga_ocr'
    """
    from .llm_profiles import LLMProfile, THINKING_DISABLED
    module = config.module
    source_language = module.translate_source if module.translate_source in LOCAL_SOURCE_OCR else '日本語'
    previous = next((p for p in module.llm_profiles if p.id == LOCAL_PROFILE_ID), None)
    default_prompt = ENGLISH_PROMPT if source_language == 'English' else LOCAL_PROMPT
    custom_prompt = previous.prompt if previous and previous.prompt not in (LOCAL_PROMPT, ENGLISH_PROMPT) else None
    profile = LLMProfile(
        id=LOCAL_PROFILE_ID, name='로컬 번역 · Qwen3.5 9B', backend='openai',
        base_url='http://127.0.0.1:11435/v1', require_api_key=False,
        model=MODEL_TAG, model_options=[MODEL_TAG], support_text=True,
        support_vision=True, support_image=False, thinking_level=THINKING_DISABLED,
        prompt=custom_prompt or default_prompt,
        max_tokens=2048, temperature=0.2, json_schema_response_format=True,
    )
    for kind, key in local_modules_for_source(source_language).items():
        setattr(module, kind, key)
    module.llm_profiles = [profile]
    module.translator_llm_id = LOCAL_PROFILE_ID
    module.ocr_llm_id = module.inpaint_llm_id = ''
    module.translate_source, module.translate_target = source_language, '한국어'
    module.translate_context = 'page'
    module.llm_translate_context = 'history'
    module.llm_prior_context_token_budget = 2048
    module.llm_translate_vision = module.local_visual_context
    module.llm_translate_summary_memory = False
    module.ocr_font_detect = False
    module.empty_runcache = True
    config.check_update_on_startup = False
    config.drawpanel.inpainter = 'lama_large_512px'
    config.drawpanel.inpaint_llm_id = ''
    config.display_lang = 'ko_KR'
    if previous is None:
        config.global_fontformat.vertical = False
        config.let_writing_mode_flag = 1
        if sys.platform == 'win32':
            config.global_fontformat.font_family = '맑은 고딕'
            config.let_family_flag = 1
    module.update_finish_code()


def _model_files_present(spec) -> bool:
    from . import shared
    for group in spec.download_file_list or []:
        names = group.get('save_files') or group.get('files', [])
        names = [names] if isinstance(names, str) else names
        for name in names:
            path = Path(shared.PROGRAM_PATH) / group.get('save_dir', '') / name
            if not path.is_file() or path.stat().st_size == 0:
                return False
    return True


def local_readiness(source_language: str = None) -> List[Tuple[str, bool, str]]:
    """Check only the selected OCR and shared offline models, without downloads."""
    from ballontranslator.modules import MODULETYPE_TO_REGISTRIES
    from ballontranslator.modules.prepare_local_files import _missing_module_requirements
    from .config import pcfg
    selected = source_language or pcfg.module.translate_source
    modules = local_modules_for_source(selected)
    ocr_label = '영어 인식 · MIT 48px' if selected == 'English' else '일본어 인식 · manga_ocr'
    labels = {'textdetector': '대사 검출 · CTD', 'ocr': ocr_label,
              'inpainter': '원문 제거 · LaMa', 'translator': '번역 연결 구성'}
    result = []
    for kind, key in modules.items():
        spec = MODULETYPE_TO_REGISTRIES[kind].get_spec(key)
        if spec is None:
            result.append((labels[kind], False, '필수 모듈이 없습니다. 프로그램 파일을 다시 확인해 주세요.'))
            continue
        missing = _missing_module_requirements(spec)
        files_ready = _model_files_present(spec)
        detail = '설치됨 · 실행 시 모델 검사'
        if missing:
            detail = '필수 실행 구성 설치 필요'
        elif not files_ready:
            detail = '모델 파일 다운로드 필요'
        result.append((labels[kind], not missing and files_ready, detail))
    return result


def prepare_local_models(progress: Callable[[str], None], stop_event: threading.Event,
                         source_language: str = None) -> None:
    from ballontranslator.modules import MODULETYPE_TO_REGISTRIES
    from .config import pcfg
    from ballontranslator.modules.prepare_local_files import ensure_module_files, _missing_module_requirements
    from .py_package_manager import PyPackageManager
    from .local_ollama import LocalOperationCancelled
    modules = local_modules_for_source(source_language or pcfg.module.translate_source)
    specs = [MODULETYPE_TO_REGISTRIES[kind].get_spec(key) for kind, key in modules.items()]
    if any(spec is None for spec in specs):
        raise RuntimeError('필수 모듈이 누락되었습니다. 프로그램을 다시 설치해 주세요.')
    missing = list(dict.fromkeys(item.requirement for spec in specs for item in _missing_module_requirements(spec)))
    if stop_event.is_set():
        raise LocalOperationCancelled('설치가 중단되었습니다.')
    if missing:
        progress('실행 구성 설치 중 · 중단 요청은 현재 설치 단계가 끝나면 적용됩니다.')
        manager = PyPackageManager()
        result = manager.install(missing, progress_callback=lambda p: progress(str(p.get('message') or p.get('event', '설치 중'))))
        if not result.ok:
            raise RuntimeError('실행 구성 설치에 실패했습니다. 설치 로그를 확인한 뒤 다시 시도해 주세요.')
    for spec in specs:
        if stop_event.is_set():
            raise LocalOperationCancelled('설치가 중단되었습니다.')
        progress(f'{spec.key} 파일 확인 및 다운로드')
        if not ensure_module_files(spec, progress_callback=lambda p: progress(str(p.get('message') or p.get('event', '파일 준비 중'))), cancel_event=stop_event):
            raise RuntimeError(f'{spec.key} 파일 준비에 실패했습니다. 다시 설치하면 재검사합니다.')
    progress('이미지 처리 모델 준비 완료')
