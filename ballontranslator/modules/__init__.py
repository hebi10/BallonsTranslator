from .ocr import OCR, OCRBase
from .textdetector import TEXTDETECTORS, TextDetectorBase
from .translators import TRANSLATORS, BaseTranslator
from .inpaint import INPAINTERS, InpainterBase
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET, LOGGER, merge_config_module_params, \
    init_module_registries

def _available_modules(kind, registry):
    from ballontranslator.utils.local_edition import is_local_edition, module_allowed
    return [key for key in registry.module_dict if not is_local_edition() or module_allowed(kind, key)]


GET_VALID_TEXTDETECTORS = lambda: _available_modules('textdetector', TEXTDETECTORS)
GET_VALID_TRANSLATORS = lambda: _available_modules('translator', TRANSLATORS)
GET_VALID_INPAINTERS = lambda: _available_modules('inpainter', INPAINTERS)
GET_VALID_OCR = lambda: _available_modules('ocr', OCR)


MODULETYPE_TO_REGISTRIES = {
    'textdetector': TEXTDETECTORS,
    'ocr': OCR,
    'inpainter': INPAINTERS,
    'translator': TRANSLATORS
}

# Registers static ModuleSpec entries only; concrete module files stay unimported.
init_module_registries()

# TODO: use manga-image-translator as backend...
