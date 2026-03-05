"""
数据库自动保存服务
在新闻抓取完成后自动保存到数据库
"""

from datetime import datetime, date
from typing import List, Tuple
from src.web.models import News, DailySummary, RunHistory, db
from src.fetchers.base import NewsItem
from src.utils.logger import get_logger


class DatabaseSaver:
    """数据库自动保存服务"""

    def __init__(self):
        self.logger = get_logger('database_saver')

    def save_news_list(self, news_list: List[NewsItem], ai_summary: str = "", ai_trends: str = "") -> dict:
        """
        保存新闻列表到数据库

        Args:
            news_list: 新闻项列表
            ai_summary: AI摘要
            ai_trends: AI趋势分析

        Returns:
            dict: 保存统计信息
        """
        if not news_list:
            self.logger.warning("新闻列表为空，跳过保存")
            return {
                'total': 0,
                'added': 0,
                'skipped': 0,
                'errors': 0
            }

        stats = {
            'total': len(news_list),
            'added': 0,
            'skipped': 0,
            'errors': 0
        }

        try:
            today = date.today()

            # 保存每条新闻
            for news in news_list:
                try:
                    # 检查是否已存在（通过链接去重）
                    existing = News.query.filter_by(link=news.url).first()

                    if existing:
                        stats['skipped'] += 1
                        self.logger.debug(f"新闻已存在，跳过: {news.title[:50]}...")
                    else:
                        # 创建新的新闻记录
                        news_record = News(
                            title=news.title,
                            link=news.url,
                            source=news.source,
                            region=news.region,
                            category='AI',
                            summary=news.summary or '',
                            date=today,
                            type=news.news_type,
                            score=news.score
                        )

                        db.session.add(news_record)
                        stats['added'] += 1
                        self.logger.debug(f"保存新闻: {news.title[:50]}...")

                except Exception as e:
                    stats['errors'] += 1
                    self.logger.error(f"保存新闻失败: {news.title[:50]}... - {e}")

            # 保存每日汇总
            self._save_daily_summary(today, news_list, ai_summary, ai_trends)

            # 提交所有更改
            db.session.commit()

            self.logger.info(
                f"数据库保存完成: 新增 {stats['added']} 条, "
                f"跳过 {stats['skipped']} 条, "
                f"错误 {stats['errors']} 条"
            )

        except Exception as e:
            db.session.rollback()
            self.logger.error(f"数据库保存失败: {e}", exc_info=True)
            stats['errors'] = stats['total']

        return stats

    def _save_daily_summary(self, today: date, news_list: List[NewsItem],
                           ai_summary: str, ai_trends: str):
        """
        保存每日汇总

        Args:
            today: 今天日期
            news_list: 新闻列表
            ai_summary: AI摘要
            ai_trends: AI趋势
        """
        try:
            # 统计数据
            domestic_count = len([n for n in news_list if n.region == 'domestic'])
            global_count = len([n for n in news_list if n.region == 'global'])
            github_count = len([n for n in news_list if n.news_type == 'github'])
            huggingface_count = len([n for n in news_list if n.news_type == 'huggingface'])

            # 查找或创建每日汇总
            summary = DailySummary.query.filter_by(date=today).first()

            if summary:
                # 更新现有汇总
                summary.total_news = len(news_list)
                summary.domestic_count = domestic_count
                summary.global_count = global_count
                summary.github_count = github_count
                summary.huggingface_count = huggingface_count

                # 只有当有新内容时才更新AI摘要
                if ai_summary and not summary.ai_summary:
                    summary.ai_summary = ai_summary
                if ai_trends and not summary.ai_trends:
                    summary.ai_trends = ai_trends

                self.logger.info(f"更新每日汇总: {today}")
            else:
                # 创建新的每日汇总
                summary = DailySummary(
                    date=today,
                    total_news=len(news_list),
                    domestic_count=domestic_count,
                    global_count=global_count,
                    github_count=github_count,
                    huggingface_count=huggingface_count,
                    ai_summary=ai_summary,
                    ai_trends=ai_trends
                )

                db.session.add(summary)
                self.logger.info(f"创建每日汇总: {today}")

        except Exception as e:
            self.logger.error(f"保存每日汇总失败: {e}")

    def save_run_history(self, start_time: datetime, end_time: datetime,
                         status: str, total_fetched: int, unique_news: int,
                         final_selected: int, sources_success: int,
                         sources_total: int, error_message: str = None) -> int:
        """
        保存运行历史

        Args:
            start_time: 开始时间
            end_time: 结束时间
            status: 状态 (success/failed/error)
            total_fetched: 抓取总数
            unique_news: 去重后数量
            final_selected: 最终筛选数量
            sources_success: 成功源数量
            sources_total: 总源数量
            error_message: 错误信息

        Returns:
            int: 运行历史ID
        """
        try:
            duration = int((end_time - start_time).total_seconds()) if end_time else 0

            history = RunHistory(
                start_time=start_time,
                end_time=end_time,
                status=status,
                total_fetched=total_fetched,
                unique_news=unique_news,
                final_selected=final_selected,
                sources_success=sources_success,
                sources_total=sources_total,
                error_message=error_message,
                duration_seconds=duration
            )

            db.session.add(history)
            db.session.commit()

            self.logger.info(f"保存运行历史: ID={history.id}, 状态={status}")
            return history.id

        except Exception as e:
            db.session.rollback()
            self.logger.error(f"保存运行历史失败: {e}")
            return None

    def get_today_news_count(self) -> int:
        """
        获取今天的新闻数量

        Returns:
            int: 新闻数量
        """
        try:
            today = date.today()
            count = News.query.filter_by(date=today).count()
            return count
        except Exception as e:
            self.logger.error(f"获取今天新闻数量失败: {e}")
            return 0

    def is_news_exists(self, url: str) -> bool:
        """
        检查新闻是否已存在

        Args:
            url: 新闻链接

        Returns:
            bool: 是否存在
        """
        try:
            existing = News.query.filter_by(link=url).first()
            return existing is not None
        except Exception as e:
            self.logger.error(f"检查新闻存在性失败: {e}")
            return False
