"""
新闻抓取管理器
管理多个新闻源，统一抓取和去重
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .base import NewsItem, BaseFetcher
from .html_fetcher import HTMLFetcher
from .rss_fetcher import RSSFetcher
from .github_fetcher import GitHubTrendingFetcher
from .huggingface_fetcher import HuggingFaceFetcher
from ..utils.logger import get_logger
from ..utils.helpers import generate_hash
from ..utils.cache_manager import CacheManager
from ..utils.request_utils import DynamicConcurrencyManager
from ..config.constants import DEFAULT_SOURCES, CACHE_CONFIG, DYNAMIC_CONCURRENCY_CONFIG


class FetcherManager:
    """新闻抓取管理器"""

    def __init__(
        self,
        sources: Dict[str, List[Dict[str, Any]]] = None,
        max_workers: int = 3,
        enable_github: bool = True,
        enable_huggingface: bool = True,
        enable_cache: bool = True,
        enable_dynamic_concurrency: bool = False,
        incremental_fetch: bool = False
    ):
        """
        初始化抓取管理器

        Args:
            sources: 新闻源配置，格式为 {"domestic": [...], "global": [...]}
            max_workers: 最大并发数
            enable_github: 是否启用GitHub Trending
            enable_huggingface: 是否启用Hugging Face
            enable_cache: 是否启用缓存
            enable_dynamic_concurrency: 是否启用动态并发
            incremental_fetch: 是否启用增量抓取
        """
        self.logger = get_logger("fetcher")
        self.sources = sources or DEFAULT_SOURCES
        self.max_workers = max_workers
        self.enable_github = enable_github
        self.enable_huggingface = enable_huggingface
        self._fetchers: List[BaseFetcher] = []

        # 初始化缓存管理器
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache_manager = CacheManager(
                cache_dir=CACHE_CONFIG["cache_dir"],
                news_expire_hours=CACHE_CONFIG["news_expire_hours"],
                ai_cache_hours=CACHE_CONFIG["ai_cache_hours"],
                max_news_cache=CACHE_CONFIG.get("max_news_cache", 1000),
                max_ai_cache=CACHE_CONFIG.get("max_ai_cache", 100)
            )
            self.logger.info(f"缓存管理器已启用（过期时间: {CACHE_CONFIG['news_expire_hours']}h新闻, {CACHE_CONFIG['ai_cache_hours']}h AI）")
        else:
            self.cache_manager = None

        # 初始化动态并发管理器
        self.enable_dynamic_concurrency = enable_dynamic_concurrency
        if enable_dynamic_concurrency:
            self.concurrency_manager = DynamicConcurrencyManager(
                min_workers=DYNAMIC_CONCURRENCY_CONFIG["min_workers"],
                max_workers=DYNAMIC_CONCURRENCY_CONFIG["max_workers"],
                initial_workers=max_workers
            )
            self.logger.info(f"动态并发已启用（范围: {DYNAMIC_CONCURRENCY_CONFIG['min_workers']}-{DYNAMIC_CONCURRENCY_CONFIG['max_workers']}）")
        else:
            self.concurrency_manager = None

        self.incremental_fetch = incremental_fetch

    def create_fetchers(self) -> List[BaseFetcher]:
        """
        创建新闻抓取器实例

        Returns:
            抓取器列表
        """
        self._fetchers = []

        # 国内新闻源
        for source in self.sources.get("domestic", []):
            if source.get("enabled", True):
                self._fetchers.append(HTMLFetcher(
                    name=source["name"],
                    url=source["url"],
                    max_news=5,
                    use_retry=True
                ))

        # 国际新闻源
        for source in self.sources.get("global", []):
            if source.get("enabled", True):
                # RSS源使用RSS抓取器
                if "feed" in source["url"]:
                    self._fetchers.append(RSSFetcher(
                        name=source["name"],
                        url=source["url"],
                        max_news=5,
                        use_retry=True
                    ))
                else:
                    self._fetchers.append(HTMLFetcher(
                        name=source["name"],
                        url=source["url"],
                        max_news=5,
                        use_retry=True
                    ))

        # Hugging Face (热门模型)
        if self.enable_huggingface:
            self._fetchers.append(HuggingFaceFetcher(max_news=5, use_retry=True))

        self.logger.info(f"创建了 {len(self._fetchers)} 个新闻抓取器（启用重试）")
        return self._fetchers

    def fetch_all(self, concurrent: bool = True) -> List[NewsItem]:
        """
        抓取所有新闻源

        Args:
            concurrent: 是否并发抓取

        Returns:
            去重后的新闻列表
        """
        if not self._fetchers:
            self.create_fetchers()

        all_news = []

        # 获取最近新闻哈希（增量抓取）
        recent_hashes = set()
        if self.incremental_fetch and self.cache_manager:
            check_hours = 24  # 默认检查最近24小时
            recent_hashes = self.cache_manager.get_recent_news_hashes(check_hours)
            self.logger.info(f"增量抓取模式：已加载最近{check_hours}小时的{len(recent_hashes)}条新闻哈希")

        # 先单独抓取GitHub（避免并发问题）
        if self.enable_github:
            from .github_fetcher import GitHubTrendingFetcher
            github_fetcher = GitHubTrendingFetcher(token="", max_news=5, use_retry=True)
            try:
                news_list = github_fetcher.fetch()
                all_news.extend(news_list)
            except Exception as e:
                self.logger.error(f"抓取器 GitHub Trending 出错: {e}")

        # 过滤常规抓取器
        regular_fetchers = [f for f in self._fetchers if f.name != "GitHub Trending"]

        # 并发执行常规抓取器
        if concurrent and regular_fetchers:
            # 动态获取工作线程数
            if self.concurrency_manager:
                workers = self.concurrency_manager.get_current_workers()
            else:
                workers = self.max_workers

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_fetcher = {
                    executor.submit(fetcher.fetch): fetcher
                    for fetcher in regular_fetchers
                }

                for future in as_completed(future_to_fetcher):
                    fetcher = future_to_fetcher[future]

                    # 更新动态并发统计
                    success = True
                    try:
                        news_list = future.result()

                        # 增量过滤：只添加新新闻
                        if self.incremental_fetch and self.cache_manager:
                            new_news = []
                            for news in news_list:
                                if not self.cache_manager.is_news_cached(news.title):
                                    new_news.append(news)
                                    # 添加到缓存
                                    self.cache_manager.add_news(
                                        title=news.title,
                                        url=news.url,
                                        source=news.source,
                                        region=news.region
                                    )
                            news_list = new_news
                            self.logger.info(f"{fetcher.name}: 过滤后新增 {len(news_list)} 条新闻")

                        all_news.extend(news_list)
                    except Exception as e:
                        success = False
                        self.logger.error(f"抓取器 {fetcher.name} 出错: {e}")

                    # 更新动态并发
                    if self.concurrency_manager:
                        self.concurrency_manager.adjust_workers(success)

        elif regular_fetchers:
            for fetcher in regular_fetchers:
                try:
                    news_list = fetcher.fetch()

                    # 增量过滤
                    if self.incremental_fetch and self.cache_manager:
                        new_news = []
                        for news in news_list:
                            if not self.cache_manager.is_news_cached(news.title):
                                new_news.append(news)
                                self.cache_manager.add_news(
                                    title=news.title,
                                    url=news.url,
                                    source=news.source,
                                    region=news.region
                                )
                        news_list = new_news
                        self.logger.info(f"{fetcher.name}: 过滤后新增 {len(news_list)} 条新闻")

                    all_news.extend(news_list)
                except Exception as e:
                    self.logger.error(f"抓取器 {fetcher.name} 出错: {e}")

        # 保存缓存
        if self.cache_manager:
            self.cache_manager.save_news_cache()

        # 去重
        unique_news = self._deduplicate(all_news)

        # 分配基础分数
        unique_news = self.assign_scores(unique_news)

        self.logger.info(f"总共获取到 {len(unique_news)} 篇不重复新闻")

        return unique_news

    def _fetch_concurrent(self) -> List[NewsItem]:
        """并发抓取"""
        all_news = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_fetcher = {
                executor.submit(fetcher.fetch): fetcher
                for fetcher in self._fetchers
            }

            for future in as_completed(future_to_fetcher):
                fetcher = future_to_fetcher[future]
                try:
                    news_list = future.result()
                    all_news.extend(news_list)
                except Exception as e:
                    self.logger.error(f"抓取器 {fetcher.name} 出错: {e}")

        return all_news

    def _fetch_sequential(self) -> List[NewsItem]:
        """顺序抓取"""
        all_news = []

        for fetcher in self._fetchers:
            try:
                news_list = fetcher.fetch()
                all_news.extend(news_list)
            except Exception as e:
                self.logger.error(f"抓取器 {fetcher.name} 出错: {e}")

        return all_news

    def _deduplicate(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """
        去重新闻

        Args:
            news_list: 新闻列表

        Returns:
            去重后的新闻列表
        """
        seen_hashes = set()
        unique_news = []

        for news in news_list:
            # 使用标题哈希作为去重依据
            title_hash = generate_hash(news.title.lower().strip())

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique_news.append(news)

        return unique_news

    def get_statistics(self, news_list: List[NewsItem]) -> Dict[str, Any]:
        """
        获取新闻统计信息

        Args:
            news_list: 新闻列表

        Returns:
            统计信息字典
        """
        domestic = [n for n in news_list if n.region == "domestic"]
        global_news = [n for n in news_list if n.region == "global"]

        sources = {}
        for news in news_list:
            sources[news.source] = sources.get(news.source, 0) + 1

        return {
            "total": len(news_list),
            "domestic": len(domestic),
            "global": len(global_news),
            "sources": sources,
            "date": datetime.now().strftime('%Y-%m-%d')
        }

    def close(self):
        """关闭所有抓取器"""
        for fetcher in self._fetchers:
            fetcher.close()

        # 保存缓存
        if self.cache_manager:
            self.cache_manager.save_all()

    def get_cache_manager(self) -> Optional[CacheManager]:
        """获取缓存管理器"""
        return self.cache_manager

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache_manager:
            stats = self.cache_manager.get_cache_stats()
            stats["cache_enabled"] = True
        else:
            stats = {"cache_enabled": False}

        # 添加并发信息
        if self.concurrency_manager:
            stats["dynamic_concurrency_enabled"] = True
            stats["current_workers"] = self.concurrency_manager.get_current_workers()
        else:
            stats["dynamic_concurrency_enabled"] = False
            stats["current_workers"] = self.max_workers

        return stats

    def sort_by_score(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """
        按AI评分排序新闻，高分在前

        Args:
            news_list: 新闻列表

        Returns:
            排序后的新闻列表
        """
        # GitHub和Hugging Face项目默认高分
        for news in news_list:
            if news.news_type in ["github", "huggingface"]:
                news.score = min(news.score + 3.0, 10.0)  # 加3分

        # 按分数降序排序
        sorted_news = sorted(news_list, key=lambda x: x.score, reverse=True)

        self.logger.info(f"新闻已按评分排序，最高分: {sorted_news[0].score if sorted_news else 0:.1f}")
        return sorted_news

    def assign_scores(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """
        根据新闻类型和来源自动分配基础分数

        Args:
            news_list: 新闻列表

        Returns:
            已分配分数的新闻列表
        """
        for news in news_list:
            # 基础分数
            base_score = 5.0

            # 根据新闻类型调整
            if news.news_type == "github":
                base_score = 7.0  # GitHub项目默认高分
            elif news.news_type == "huggingface":
                base_score = 7.0  # Hugging Face模型默认高分
            elif news.news_type == "rss":
                base_score = 5.5  # RSS源略高

            # 根据来源调整
            high_value_sources = ["量子位AI", "MIT Technology Review", "TechCrunch AI"]
            if news.source in high_value_sources:
                base_score += 1.0

            news.score = min(base_score, 10.0)

        return news_list
