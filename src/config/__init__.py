"""
配置模块
"""

from .constants import (
    DEFAULT_SOURCES,
    AI_KEYWORDS,
    EXCLUDE_KEYWORDS,
    MIN_TITLE_LENGTH,
    MAX_TITLE_LENGTH,
    REQUEST_TIMEOUT,
    REQUEST_HEADERS,
    NEWS_SELECTORS,
    EMAIL_SUBJECT_TEMPLATE,
    DEFAULT_CONFIG,
    AI_PROMPTS,
    COLOR_THEME
)
from .settings import (
    EmailConfig,
    SettingsConfig,
    AIConfig,
    FetcherConfig,
    OutputConfig,
    AppConfig,
    ConfigManager,
    get_config_manager
)

__all__ = [
    'DEFAULT_SOURCES',
    'AI_KEYWORDS',
    'EXCLUDE_KEYWORDS',
    'MIN_TITLE_LENGTH',
    'MAX_TITLE_LENGTH',
    'REQUEST_TIMEOUT',
    'REQUEST_HEADERS',
    'NEWS_SELECTORS',
    'EMAIL_SUBJECT_TEMPLATE',
    'DEFAULT_CONFIG',
    'AI_PROMPTS',
    'COLOR_THEME',
    'EmailConfig',
    'SettingsConfig',
    'AIConfig',
    'FetcherConfig',
    'OutputConfig',
    'AppConfig',
    'ConfigManager',
    'get_config_manager'
]
