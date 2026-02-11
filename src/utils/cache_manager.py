"""
缓存管理器
负责管理新闻历史缓存和AI处理结果缓存
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from .logger import get_logger


@dataclass
class NewsCacheItem:
    """新闻缓存项"""
    title_hash: str
    url: str
    title: str
    source: str
    timestamp: str
    region: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'NewsCacheItem':
        return cls(**data)


@dataclass
class AICacheItem:
    """AI处理结果缓存项"""
    content_hash: str
    summary: str
    trends: str
    timestamp: str
    news_hashes: List[str]  # 关联的新闻哈希列表

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AICacheItem':
        return cls(**data)


class CacheManager:
    """缓存管理器"""

    def __init__(
        self,
        cache_dir: str = "cache",
        news_expire_hours: int = 24,
        ai_cache_hours: int = 6,
        max_news_cache: int = 1000,
        max_ai_cache: int = 100
    ):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
            news_expire_hours: 新闻缓存过期时间（小时）
            ai_cache_hours: AI结果缓存过期时间（小时）
            max_news_cache: 最大新闻缓存数量
            max_ai_cache: 最大AI缓存数量
        """
        self.logger = get_logger("cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.news_cache_file = self.cache_dir / "news_cache.json"
        self.ai_cache_file = self.cache_dir / "ai_cache.json"

        self.news_expire_hours = news_expire_hours
        self.ai_cache_hours = ai_cache_hours
        self.max_news_cache = max_news_cache
        self.max_ai_cache = max_ai_cache

        # 内存缓存
        self._news_cache: Dict[str, NewsCacheItem] = {}
        self._ai_cache: Dict[str, AICacheItem] = {}

        # 加载缓存（自动清理）
        self._load_news_cache()
        self._load_ai_cache()
        self._auto_cleanup()

    def _generate_hash(self, text: str) -> str:
        """生成文本哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_news_cache(self):
        """加载新闻缓存"""
        if not self.news_cache_file.exists():
            return

        try:
            with open(self.news_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item_data in data.values():
                item = NewsCacheItem.from_dict(item_data)

                # 检查是否过期
                if self._is_news_expired(item):
                    continue

                self._news_cache[item.title_hash] = item

            self.logger.info(f"加载了 {len(self._news_cache)} 条新闻缓存")

        except Exception as e:
            self.logger.error(f"加载新闻缓存失败: {e}")

    def _load_ai_cache(self):
        """加载AI缓存"""
        if not self.ai_cache_file.exists():
            return

        try:
            with open(self.ai_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item_data in data.values():
                item = AICacheItem.from_dict(item_data)

                # 检查是否过期
                if self._is_ai_cache_expired(item):
                    continue

                self._ai_cache[item.content_hash] = item

            self.logger.info(f"加载了 {len(self._ai_cache)} 条AI缓存")

        except Exception as e:
            self.logger.error(f"加载AI缓存失败: {e}")

    def _is_news_expired(self, item: NewsCacheItem) -> bool:
        """检查新闻是否过期"""
        try:
            item_time = datetime.fromisoformat(item.timestamp)
            expire_time = datetime.now() - timedelta(hours=self.news_expire_hours)
            return item_time < expire_time
        except:
            return True

    def _is_ai_cache_expired(self, item: AICacheItem) -> bool:
        """检查AI缓存是否过期"""
        try:
            item_time = datetime.fromisoformat(item.timestamp)
            expire_time = datetime.now() - timedelta(hours=self.ai_cache_hours)
            return item_time < expire_time
        except:
            return True

    def is_news_cached(self, title: str) -> bool:
        """
        检查新闻是否已缓存

        Args:
            title: 新闻标题

        Returns:
            是否已缓存且未过期
        """
        title_hash = self._generate_hash(title.lower().strip())
        return title_hash in self._news_cache

    def add_news(self, title: str, url: str, source: str, region: str = "global"):
        """
        添加新闻到缓存

        Args:
            title: 新闻标题
            url: 新闻URL
            source: 新闻来源
            region: 新闻区域
        """
        title_hash = self._generate_hash(title.lower().strip())

        item = NewsCacheItem(
            title_hash=title_hash,
            url=url,
            title=title,
            source=source,
            timestamp=datetime.now().isoformat(),
            region=region
        )

        self._news_cache[title_hash] = item

        # 超出限制时自动清理
        if len(self._news_cache) > self.max_news_cache:
            self._cleanup_news_cache_by_size()

    def get_recent_news_hashes(self, hours: int = 24) -> Set[str]:
        """
        获取最近N小时内的新闻哈希集合

        Args:
            hours: 小时数

        Returns:
            新闻哈希集合
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_hashes = set()

        for item in self._news_cache.values():
            try:
                item_time = datetime.fromisoformat(item.timestamp)
                if item_time >= cutoff_time:
                    recent_hashes.add(item.title_hash)
            except:
                continue

        return recent_hashes

    def get_ai_cached_result(self, news_hashes: List[str]) -> Optional[tuple[str, str]]:
        """
        获取AI缓存的摘要和趋势分析

        Args:
            news_hashes: 新闻哈希列表

        Returns:
            (摘要, 趋势分析) 元组，如果缓存未命中返回None
        """
        # 生成内容哈希（基于新闻哈希的排序组合）
        sorted_hashes = sorted(news_hashes)
        content_hash = self._generate_hash(''.join(sorted_hashes))

        if content_hash in self._ai_cache:
            item = self._ai_cache[content_hash]
            self.logger.info(f"AI缓存命中（{len(news_hashes)}条新闻）")
            return item.summary, item.trends

        return None

    def save_ai_result(self, news_hashes: List[str], summary: str, trends: str):
        """
        保存AI处理结果到缓存

        Args:
            news_hashes: 新闻哈希列表
            summary: 摘要
            trends: 趋势分析
        """
        # 生成内容哈希
        sorted_hashes = sorted(news_hashes)
        content_hash = self._generate_hash(''.join(sorted_hashes))

        item = AICacheItem(
            content_hash=content_hash,
            summary=summary,
            trends=trends,
            timestamp=datetime.now().isoformat(),
            news_hashes=news_hashes
        )

        self._ai_cache[content_hash] = item
        self.logger.info(f"AI结果已缓存（{len(news_hashes)}条新闻）")

        # 超出限制时自动清理
        if len(self._ai_cache) > self.max_ai_cache:
            self._cleanup_ai_cache_by_size()

    def save_news_cache(self):
        """保存新闻缓存到文件"""
        try:
            # 清理过期缓存
            self._clean_expired_news_cache()

            data = {
                item.title_hash: item.to_dict()
                for item in self._news_cache.values()
            }

            with open(self.news_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"新闻缓存已保存 ({len(self._news_cache)} 条)")

        except Exception as e:
            self.logger.error(f"保存新闻缓存失败: {e}")

    def save_ai_cache(self):
        """保存AI缓存到文件"""
        try:
            # 清理过期缓存
            self._clean_expired_ai_cache()

            data = {
                item.content_hash: item.to_dict()
                for item in self._ai_cache.values()
            }

            with open(self.ai_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"AI缓存已保存 ({len(self._ai_cache)} 条)")

        except Exception as e:
            self.logger.error(f"保存AI缓存失败: {e}")

    def save_all(self):
        """保存所有缓存"""
        self.save_news_cache()
        self.save_ai_cache()

    def _clean_expired_news_cache(self):
        """清理过期的新闻缓存"""
        expired_keys = []

        for key, item in self._news_cache.items():
            if self._is_news_expired(item):
                expired_keys.append(key)

        for key in expired_keys:
            del self._news_cache[key]

        if expired_keys:
            self.logger.debug(f"清理了 {len(expired_keys)} 条过期新闻缓存")

    def _clean_expired_ai_cache(self):
        """清理过期的AI缓存"""
        expired_keys = []

        for key, item in self._ai_cache.items():
            if self._is_ai_cache_expired(item):
                expired_keys.append(key)

        for key in expired_keys:
            del self._ai_cache[key]

        if expired_keys:
            self.logger.debug(f"清理了 {len(expired_keys)} 条过期AI缓存")

    def _cleanup_news_cache_by_size(self):
        """按LRU策略清理新闻缓存（保留最新的）"""
        # 按时间戳排序，删除最老的
        items_by_time = sorted(
            self._news_cache.items(),
            key=lambda x: x[1].timestamp
        )

        # 删除超出部分
        num_to_delete = len(self._news_cache) - self.max_news_cache
        if num_to_delete > 0:
            for i in range(num_to_delete):
                key_to_delete = items_by_time[i][0]
                del self._news_cache[key_to_delete]

            self.logger.info(f"LRU清理了 {num_to_delete} 条新闻缓存（保留最新{self.max_news_cache}条）")

    def _cleanup_ai_cache_by_size(self):
        """按LRU策略清理AI缓存（保留最新的）"""
        # 按时间戳排序，删除最老的
        items_by_time = sorted(
            self._ai_cache.items(),
            key=lambda x: x[1].timestamp
        )

        # 删除超出部分
        num_to_delete = len(self._ai_cache) - self.max_ai_cache
        if num_to_delete > 0:
            for i in range(num_to_delete):
                key_to_delete = items_by_time[i][0]
                del self._ai_cache[key_to_delete]

            self.logger.info(f"LRU清理了 {num_to_delete} 条AI缓存（保留最新{self.max_ai_cache}条）")

    def _auto_cleanup(self):
        """启动时自动清理"""
        # 清理过期缓存
        self._clean_expired_news_cache()
        self._clean_expired_ai_cache()

        # 清理超出大小限制的缓存
        if len(self._news_cache) > self.max_news_cache:
            self._cleanup_news_cache_by_size()
        if len(self._ai_cache) > self.max_ai_cache:
            self._cleanup_ai_cache_by_size()

        # 显示清理后的统计
        total_news = len(self._news_cache)
        total_ai = len(self._ai_cache)
        self.logger.info(f"缓存自动清理完成 - 新闻: {total_news}/{self.max_news_cache}, AI: {total_ai}/{self.max_ai_cache}")

    def clear_all_cache(self):
        """清空所有缓存"""
        self._news_cache.clear()
        self._ai_cache.clear()
        self.logger.info("所有缓存已清空")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "news_cache_count": len(self._news_cache),
            "ai_cache_count": len(self._ai_cache),
            "max_news_cache": self.max_news_cache,
            "max_ai_cache": self.max_ai_cache,
            "news_usage_percent": round(len(self._news_cache) / self.max_news_cache * 100, 1),
            "ai_usage_percent": round(len(self._ai_cache) / self.max_ai_cache * 100, 1),
            "news_expire_hours": self.news_expire_hours,
            "ai_cache_hours": self.ai_cache_hours
        }
