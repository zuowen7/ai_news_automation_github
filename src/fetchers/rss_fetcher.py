"""
RSS新闻抓取器
用于抓取RSS/Atom feed新闻
"""

from datetime import datetime
from typing import List

try:
    import feedparser
except ImportError:
    feedparser = None

from .base import BaseFetcher, NewsItem
from ..utils.logger import get_logger
from ..utils.helpers import is_ai_related, truncate_text
from ..config.constants import ALL_AI_KEYWORDS


class RSSFetcher(BaseFetcher):
    """RSS Feed新闻抓取器"""

    def fetch(self) -> List[NewsItem]:
        """
        抓取RSS feed新闻

        Returns:
            新闻项列表
        """
        if feedparser is None:
            self.logger.error("feedparser未安装，无法抓取RSS")
            return []

        self.logger.info(f"开始抓取RSS: {self.name}")

        response = self._make_request(self.url)
        if not response:
            self.logger.warning(f"RSS抓取失败: {self.name}")
            return []

        try:
            feed = feedparser.parse(response.content)
            news_list = []

            for entry in feed.entries[:self.max_news * 2]:
                title = entry.get('title', '')
                url = entry.get('link', '')
                summary = entry.get('summary', entry.get('description', ''))

                # 清理HTML标签
                if summary:
                    from bs4 import BeautifulSoup
                    summary = BeautifulSoup(summary, 'html.parser').get_text(strip=True)
                    summary = truncate_text(summary, 200)

                # 获取日期
                published = entry.get('published', '')
                if not published:
                    published = datetime.now().strftime('%Y-%m-%d')

                # 验证AI相关性
                if is_ai_related(title, ALL_AI_KEYWORDS):
                    news_list.append(NewsItem(
                        title=title,
                        url=url,
                        source=self.name,
                        region=self._get_region(),
                        summary=summary,
                        date=published,
                        news_type="rss"
                    ))

                    if len(news_list) >= self.max_news:
                        break

            self.logger.info(f"从 {self.name} RSS获取到 {len(news_list)} 篇新闻")
            return news_list

        except Exception as e:
            self.logger.error(f"RSS解析失败 {self.name}: {e}")
            return []
