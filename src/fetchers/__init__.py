"""
新闻抓取模块
"""

from .base import BaseFetcher, NewsItem
from .html_fetcher import HTMLFetcher
from .rss_fetcher import RSSFetcher
from .github_fetcher import GitHubTrendingFetcher
from .huggingface_fetcher import HuggingFaceFetcher
from .manager import FetcherManager

__all__ = [
    'BaseFetcher',
    'NewsItem',
    'HTMLFetcher',
    'RSSFetcher',
    'GitHubTrendingFetcher',
    'HuggingFaceFetcher',
    'FetcherManager'
]
