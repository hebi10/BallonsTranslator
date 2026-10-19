"""Optional local-edition settings stored with a comic project."""

from .logger import logger as LOGGER


def normalize_local_settings(value: object) -> dict:
    """Keep valid work settings when loading older or partially malformed files.

    >>> normalize_local_settings({'reading_direction': 'rtl'})
    {'reading_direction': 'rtl'}
    """
    if not isinstance(value, dict):
        LOGGER.warning('Ignoring invalid local project settings.')
        return {}
    choices = {'source_language': ('Auto', '日本語', 'English'),
               'reading_direction': ('auto', 'rtl', 'ltr')}
    text_fields = {'work_name', 'glossary_path', 'tone_notes'}
    result = {}
    for key, item in value.items():
        valid = isinstance(item, str) and (
            key in text_fields or key in choices and item in choices[key])
        if valid:
            result[key] = item
        else:
            LOGGER.warning('Ignoring invalid local project setting: %s', key)
    return result
