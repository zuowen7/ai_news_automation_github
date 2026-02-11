"""
新闻抓取器基类
定义新闻抓取器的通用接口和基础功能
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from ..utils.logger import get_logger
from ..utils.helpers import is_ai_related, make_absolute_url, truncate_text
from ..utils.request_utils import RequestOptimizer, RetryConfig
from ..config.constants import (
    REQUEST_TIMEOUT, REQUEST_HEADERS, NEWS_SELECTORS,
    AI_KEYWORDS, EXCLUDE_KEYWORDS, MIN_TITLE_LENGTH, MAX_TITLE_LENGTH,
    ALL_AI_KEYWORDS, RETRY_CONFIG
)


class NewsItem:
    """新闻项数据类"""

    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        region: str = "global",
        summary: str = "",
        date: str = "",
        news_type: str = "news",
        score: float = 0.0
    ):
        self.title = title
        self.url = url
        self.source = source
        self.region = region
        self.summary = summary
        self.date = date
        self.news_type = news_type
        self.score = score  # AI评分，用于排序

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'title': self.title,
            'link': self.url,
            'source': self.source,
            'region': self.region,
            'summary': self.summary,
            'date': self.date,
            'type': self.news_type,
            'score': self.score
        }

    def __repr__(self) -> str:
        return f"NewsItem(title='{self.title[:50]}...', source='{self.source}')"


class BaseFetcher(ABC):
    """新闻抓取器基类"""

    def __init__(self, name: str, url: str, max_news: int = 5, use_retry: bool = True):
        self.name = name
        self.url = url
        self.max_news = max_news
        self.logger = get_logger(f"fetcher.{name}")

        # 使用请求优化器
        self.use_retry = use_retry
        if use_retry:
            retry_config = RetryConfig(
                max_retries=RETRY_CONFIG["max_retries"],
                initial_backoff=RETRY_CONFIG["initial_backoff"],
                max_backoff=RETRY_CONFIG["max_backoff"],
                exponential_base=RETRY_CONFIG["exponential_base"],
                jitter=RETRY_CONFIG["jitter"],
                jitter_factor=RETRY_CONFIG["jitter_factor"]
            )
            self.request_optimizer = RequestOptimizer(retry_config)
            self.session = self.request_optimizer.create_session()
        else:
            self.request_optimizer = None
            self.session = requests.Session()
            self.session.headers.update(REQUEST_HEADERS)
            self.session.verify = False
            self.session.max_redirects = 5

    @abstractmethod
    def fetch(self) -> List[NewsItem]:
        """
        抓取新闻

        Returns:
            新闻项列表
        """
        pass

    def _make_request(self, url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
        """
        发送HTTP请求

        Args:
            url: 请求URL
            timeout: 超时时间

        Returns:
            响应对象，失败返回None
        """
        if self.use_retry and self.request_optimizer:
            # 使用带重试的请求
            return self.request_optimizer.make_request_with_retry(
                url=url,
                method='GET',
                session=self.session,
                timeout=timeout
            )
        else:
            # 普通请求
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.error(f"请求失败 {url}: {e}")
                return None

    def _parse_html(self, html: str, base_url: str) -> List[NewsItem]:
        """
        解析HTML内容提取新闻

        Args:
            html: HTML内容
            base_url: 基础URL用于处理相对链接

        Returns:
            新闻项列表
        """
        soup = BeautifulSoup(html, 'html.parser')
        news_list = []

        for selector, attr in NEWS_SELECTORS:
            elements = soup.select(selector)
            for elem in elements[:self.max_news * 2]:  # 多取一些，后续过滤
                if elem.name == 'a':
                    title = elem.get_text(strip=True)
                    url = elem.get('href', '')
                else:
                    # 可能是标题元素，找内部链接
                    link = elem.find('a')
                    if link:
                        title = elem.get_text(strip=True)
                        url = link.get('href', '')
                    else:
                        continue

                # 处理相对URL
                if url and not url.startswith('http'):
                    url = make_absolute_url(base_url, url)

                # 验证并创建新闻项
                if self._is_valid_news(title):
                    news_list.append(NewsItem(
                        title=title,
                        url=url,
                        source=self.name,
                        region=self._get_region()
                    ))

                    if len(news_list) >= self.max_news:
                        break

            if len(news_list) >= self.max_news:
                break

        return news_list

    def _is_valid_news(self, title: str) -> bool:
        """
        验证是否为有效的AI新闻

        Args:
            title: 新闻标题

        Returns:
            是否有效
        """
        # 检查标题长度
        if len(title) < MIN_TITLE_LENGTH or len(title) > MAX_TITLE_LENGTH:
            return False

        # 检查排除关键词
        title_lower = title.lower()
        for keyword in EXCLUDE_KEYWORDS:
            if keyword.lower() in title_lower:
                return False

        # 检查AI相关性
        return is_ai_related(title, ALL_AI_KEYWORDS)

    def _get_region(self) -> str:
        """获取新闻区域（国内或国际）"""
        return "domestic" if self._is_domestic() else "global"

    def _is_domestic(self) -> bool:
        """判断是否为国内新闻源"""
        # 检查URL或名称是否包含中文网站特征
        domestic_indicators = [
            '36kr', 'huxiu', 'qbitai', 'jiqizhixin', 'tmtpost',  # 原有
            'leiphone', 'yanxishe', 'ai-era', '36dsj',         # 新增
            'cbdio', 'sina', '163', 'sohu', 'ifeng',           # 门户
            'infoq.cn', 'geekpark', 'chinadaily', 'people'     # 其他
        ]
        url_lower = self.url.lower()
        name_lower = self.name.lower()
        return any(indicator in url_lower or indicator in name_lower for indicator in domestic_indicators)

    def close(self):
        """关闭会话"""
        if self.request_optimizer:
            self.request_optimizer.close()
        else:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
