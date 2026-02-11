"""
邮件模块
"""

from .templates import EmailTemplate, EditorialNewsTemplate, MinimalNewsTemplate, ModernNewsTemplate, TextEmailTemplate
from .sender import EmailSender, EmailPreviewGenerator

__all__ = [
    'EmailTemplate',
    'EditorialNewsTemplate',
    'MinimalNewsTemplate',
    'ModernNewsTemplate',
    'TextEmailTemplate',
    'EmailSender',
    'EmailPreviewGenerator'
]
