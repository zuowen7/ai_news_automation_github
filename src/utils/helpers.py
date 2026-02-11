"""
辅助工具模块
提供各种辅助函数
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse


def generate_hash(content: str) -> str:
    """生成内容的哈希值用于去重"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def is_ai_related(text: str, keywords: Optional[List[str]] = None) -> bool:
    """
    判断文本是否与AI相关

    Args:
        text: 要判断的文本
        keywords: AI关键词列表，如果为None则使用默认关键词

    Returns:
        是否与AI相关
    """
    if not text:
        return False

    text_lower = text.lower()

    default_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'transformer', 'gpt', 'chatgpt', 'claude', 'gemini',
        'llm', 'large language model', 'computer vision', 'nlp', 'natural language',
        '大模型', '语言模型', '生成式ai', '人工智能', '机器学习', '深度学习',
        '神经网络', 'transformer', 'gpt', 'chatgpt', '文心一言', '通义千问',
        '悟道', 'llm', '多模态', '自动驾驶', '强化学习', 'diffusion',
        'stable diffusion', 'midjourney', 'openai', 'anthropic', 'huggingface',
        'pytorch', 'tensorflow', 'agent', 'rag', 'fine-tuning'
    ]

    keywords_to_check = keywords if keywords is not None else default_keywords

    for keyword in keywords_to_check:
        if keyword.lower() in text_lower:
            return True
    return False


def is_valid_email(email: str) -> bool:
    """验证邮箱格式"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def make_absolute_url(base_url: str, relative_url: str) -> str:
    """
    将相对URL转换为绝对URL

    Args:
        base_url: 基础URL
        relative_url: 相对URL

    Returns:
        绝对URL
    """
    if not relative_url:
        return ""
    if relative_url.startswith('http'):
        return relative_url
    return urljoin(base_url, relative_url)


def extract_domain(url: str) -> str:
    """从URL中提取域名"""
    parsed = urlparse(url)
    return parsed.netloc


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 要截断的文本
        max_length: 最大长度
        suffix: 截断后添加的后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_date(date_string: str, input_format: str = "%Y-%m-%d",
                output_format: str = "%Y年%m月%d日") -> str:
    """
    格式化日期字符串

    Args:
        date_string: 输入日期字符串
        input_format: 输入格式
        output_format: 输出格式

    Returns:
        格式化后的日期字符串
    """
    try:
        date_obj = datetime.strptime(date_string, input_format)
        return date_obj.strftime(output_format)
    except ValueError:
        return date_string


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    将列表分块

    Args:
        lst: 要分块的列表
        chunk_size: 每块大小

    Returns:
        分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
