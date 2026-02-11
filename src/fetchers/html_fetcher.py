"""
HTML新闻抓取器
用于抓取常规HTML页面的新闻
"""

from typing import List

from .base import BaseFetcher, NewsItem


class HTMLFetcher(BaseFetcher):
    """HTML页面新闻抓取器"""

    def fetch(self) -> List[NewsItem]:
        """
        抓取HTML页面新闻

        Returns:
            新闻项列表
        """
        self.logger.info(f"开始抓取: {self.name}")

        response = self._make_request(self.url)
        if not response:
            self.logger.warning(f"抓取失败: {self.name}")
            return []

        try:
            news_list = self._parse_html(response.text, self.url)
            self.logger.info(f"从 {self.name} 获取到 {len(news_list)} 篇新闻")
            return news_list
        except Exception as e:
            self.logger.error(f"解析失败 {self.name}: {e}")
            return []
