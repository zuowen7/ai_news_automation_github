"""
AI新闻自动化系统 - 主程序入口
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from src.utils.logger import get_logger, setup_logging
from src.utils.helpers import format_date
from src.config.settings import get_config_manager
from src.fetchers import FetcherManager
from src.ai import NewsAIProcessor, NewsAIFilter
from src.email import EmailSender


class NewsAutomationApp:
    """新闻自动化应用"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化应用

        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger("app")
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.load()

        # 初始化各模块
        self.fetcher_manager = None
        self.ai_processor = None
        self.ai_filter = None
        self.email_sender = None

    def initialize(self):
        """初始化各模块"""
        self.logger.info("=" * 50)
        self.logger.info("AI新闻自动化系统启动")
        self.logger.info(f"版本: 2.1.0 (优化版)")
        self.logger.info("=" * 50)

        # 初始化抓取管理器（启用缓存和可选的动态并发）
        self.fetcher_manager = FetcherManager(
            max_workers=self.config.fetcher.concurrent_requests,
            enable_github=self.config.fetcher.enable_github,
            enable_huggingface=self.config.fetcher.enable_huggingface,
            enable_cache=True,  # 默认启用缓存
            enable_dynamic_concurrency=getattr(self.config.fetcher, 'dynamic_concurrency', False),
            incremental_fetch=getattr(self.config.fetcher, 'incremental_fetch', False)
        )

        # 显示缓存统计
        cache_stats = self.fetcher_manager.get_cache_stats()
        if cache_stats.get('cache_enabled'):
            news_count = cache_stats.get('news_cache_count', 0)
            ai_count = cache_stats.get('ai_cache_count', 0)
            max_news = cache_stats.get('max_news_cache', 1000)
            max_ai = cache_stats.get('max_ai_cache', 100)
            news_usage = cache_stats.get('news_usage_percent', 0)
            ai_usage = cache_stats.get('ai_usage_percent', 0)
            self.logger.info(f"缓存已启用 - 新闻: {news_count}/{max_news} ({news_usage}%), AI: {ai_count}/{max_ai} ({ai_usage}%)")
        if cache_stats.get('dynamic_concurrency_enabled'):
            self.logger.info(f"动态并发已启用 - 当前工作线程: {cache_stats.get('current_workers', 3)}")

        # 初始化AI处理器（传入缓存管理器）
        if self.config.ai.enabled:
            cache_manager = self.fetcher_manager.get_cache_manager()
            self.ai_processor = NewsAIProcessor(self.config.ai, cache_manager=cache_manager)
            self.ai_filter = NewsAIFilter(self.config.ai)
            if self.ai_processor.is_available():
                self.logger.info("AI服务已就绪")
            else:
                self.logger.warning("AI服务不可用，将跳过AI处理")

        # 初始化邮件发送器
        self.email_sender = EmailSender(self.config)

        if not self.email_sender._is_email_configured():
            self.logger.warning("邮箱未配置，只保存到文件")
        else:
            self.logger.info(f"邮件将发送至: {self.config.email.recipient_email}")

    def run(self, send_email: bool = True) -> bool:
        """
        运行主流程

        Args:
            send_email: 是否发送邮件

        Returns:
            是否执行成功
        """
        try:
            self.initialize()

            # 1. 抓取新闻
            self.logger.info("\n开始抓取新闻...")
            news_list = self.fetcher_manager.fetch_all(concurrent=True)

            if not news_list:
                self.logger.warning("未获取到任何新闻")
                return False

            stats = self.fetcher_manager.get_statistics(news_list)
            self.logger.info(f"抓取完成 - 总计: {stats['total']}, 国内: {stats['domestic']}, 国际: {stats['global']}")

            # 1.5. AI智能筛选（可选）
            if self.ai_filter and self.ai_filter.is_available() and self.config.ai.enable_filter:
                self.logger.info("\n开始AI智能筛选...")
                original_count = len(news_list)
                news_list = self.ai_filter.filter_news(news_list)
                self.logger.info(f"AI筛选完成：{original_count} -> {len(news_list)} 条")

            # 1.6. 按评分排序（高分在前）
            self.logger.info("\n按评分排序新闻...")
            news_list = self.fetcher_manager.sort_by_score(news_list)

            # 2. AI处理（国内外分别使用不同模型）
            ai_summary = ""
            ai_trends = ""
            if self.ai_processor and self.ai_processor.is_available():
                self.logger.info("\n开始AI处理（国内外分别处理）...")
                ai_summary, ai_trends = self.ai_processor.generate_all_separated(news_list)

            # 3. 保存文件
            self.logger.info("\n保存文件...")
            output_dir = self.config.output.output_dir
            self.email_sender.save_to_file(news_list, ai_summary, ai_trends, output_dir)

            # 4. 发送邮件
            if send_email and self.email_sender._is_email_configured():
                self.logger.info("\n发送邮件...")
                success = self.email_sender.send_news(news_list, ai_summary, ai_trends)
                if success:
                    self.logger.info("邮件发送成功！")
                else:
                    self.logger.error("邮件发送失败")
                    return False
            else:
                self.logger.info("\n跳过邮件发送")

            # 清理资源
            self.cleanup()

            self.logger.info("\n任务完成！")
            return True

        except Exception as e:
            self.logger.error(f"运行出错: {e}", exc_info=True)
            return False

    def cleanup(self):
        """清理资源"""
        if self.fetcher_manager:
            self.fetcher_manager.close()

    def test_ai(self):
        """测试AI功能"""
        self.initialize()

        if not self.ai_processor or not self.ai_processor.is_available():
            self.logger.error("AI服务不可用")
            return

        self.logger.info("测试AI功能...")

        # 测试对话
        test_news = [
            type('NewsItem', (), {'title': 'OpenAI发布GPT-5模型', 'source': 'TechCrunch'})(),
            type('NewsItem', (), {'title': '百度发布文心一言4.0', 'source': '36氪'})()
        ]

        summary = self.ai_processor.generate_summary(test_news)
        self.logger.info(f"摘要测试结果: {summary}")

        trends = self.ai_processor.generate_trends(test_news)
        self.logger.info(f"趋势测试结果: {trends}")

        self.cleanup()


def main():
    """主函数"""
    # 设置控制台编码（Windows）
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    parser = argparse.ArgumentParser(description="AI新闻自动化系统")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--no-email", action="store_true", help="不发送邮件")
    parser.add_argument("--test-ai", action="store_true", help="测试AI功能")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")

    args = parser.parse_args()

    # 设置日志级别
    setup_logging(args.log_level)

    # 创建应用实例
    app = NewsAutomationApp(args.config)

    if args.test_ai:
        app.test_ai()
    else:
        app.run(send_email=not args.no_email)


if __name__ == "__main__":
    main()
