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
from src.web.app import create_app


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
        self.database_saver = None
        self.flask_app = None

    def initialize(self):
        """初始化各模块"""
        self.logger.info("=" * 50)
        self.logger.info("AI新闻自动化系统启动")
        self.logger.info(f"版本: 2.3.0 (智能去重+质量评分)")
        self.logger.info("=" * 50)

        # 初始化Flask应用和数据库
        self.flask_app = create_app()
        self.logger.info("数据库已初始化")

        # 初始化数据库保存器
        from src.web.services.database_saver import DatabaseSaver
        self.database_saver = DatabaseSaver()
        self.logger.info("数据库保存器已初始化")

        # 初始化抓取管理器（启用缓存和可选的动态并发）
        self.fetcher_manager = FetcherManager(
            max_workers=self.config.fetcher.concurrent_requests,
            enable_github=self.config.fetcher.enable_github,
            enable_huggingface=self.config.fetcher.enable_huggingface,
            enable_cache=True,  # 默认启用缓存
            enable_dynamic_concurrency=getattr(self.config.fetcher, 'dynamic_concurrency', False),
            incremental_fetch=getattr(self.config.fetcher, 'incremental_fetch', False),
            enable_smart_dedup=getattr(self.config.fetcher, 'enable_smart_dedup', True),
            enable_quality_scoring=getattr(self.config.fetcher, 'enable_quality_scoring', True)
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
        start_time = datetime.now()
        run_status = "success"
        error_message = None

        try:
            self.initialize()

            # 1. 抓取新闻
            self.logger.info("\n开始抓取新闻...")
            news_list = self.fetcher_manager.fetch_all(concurrent=True)

            if not news_list:
                self.logger.warning("未获取到任何新闻")
                run_status = "failed"
                error_message = "未获取到任何新闻"
                self._save_run_history(start_time, datetime.now(), run_status, 0, 0, 0, 0, 0, error_message)
                return False

            stats = self.fetcher_manager.get_statistics(news_list)
            self.logger.info(f"抓取完成 - 总计: {stats['total']}, 国内: {stats['domestic']}, 国际: {stats['global']}")

            # 1.5. AI智能筛选（可选）
            original_count = len(news_list)
            if self.ai_filter and self.ai_filter.is_available() and self.config.ai.enable_filter:
                self.logger.info("\n开始AI智能筛选...")
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

            # 3.5. 保存到数据库（新增）
            self.logger.info("\n保存到数据库...")
            with self.flask_app.app_context():
                db_stats = self.database_saver.save_news_list(news_list, ai_summary, ai_trends)
                self.logger.info(f"数据库保存完成 - 新增: {db_stats['added']}, 跳过: {db_stats['skipped']}")
                today_count = self.database_saver.get_today_news_count()
                self.logger.info(f"今天数据库中共有 {today_count} 条新闻")

            # 4. 发送邮件
            if send_email and self.email_sender._is_email_configured():
                self.logger.info("\n发送邮件...")
                success = self.email_sender.send_news(news_list, ai_summary, ai_trends)
                if success:
                    self.logger.info("邮件发送成功！")
                else:
                    self.logger.error("邮件发送失败")
                    run_status = "error"
                    error_message = "邮件发送失败"
                    self._save_run_history(start_time, datetime.now(), run_status, stats['total'],
                                         original_count, len(news_list), 0, 0, error_message)
                    return False
            else:
                self.logger.info("\n跳过邮件发送")

            # 清理资源
            self.cleanup()

            # 保存运行历史
            self._save_run_history(start_time, datetime.now(), run_status, stats['total'],
                                 original_count, len(news_list), 0, 0, None)

            self.logger.info("\n任务完成！")
            return True

        except Exception as e:
            self.logger.error(f"运行出错: {e}", exc_info=True)
            run_status = "error"
            error_message = str(e)
            self._save_run_history(start_time, datetime.now(), run_status, 0, 0, 0, 0, 0, error_message)
            return False

    def _save_run_history(self, start_time: datetime, end_time: datetime, status: str,
                         total_fetched: int, unique_news: int, final_selected: int,
                         sources_success: int, sources_total: int, error_message: str = None):
        """
        保存运行历史

        Args:
            start_time: 开始时间
            end_time: 结束时间
            status: 状态
            total_fetched: 抓取总数
            unique_news: 去重后数量
            final_selected: 最终筛选数量
            sources_success: 成功源数量
            sources_total: 总源数量
            error_message: 错误信息
        """
        if self.database_saver and self.flask_app:
            try:
                with self.flask_app.app_context():
                    self.database_saver.save_run_history(
                        start_time=start_time,
                        end_time=end_time,
                        status=status,
                        total_fetched=total_fetched,
                        unique_news=unique_news,
                        final_selected=final_selected,
                        sources_success=sources_success,
                        sources_total=sources_total,
                        error_message=error_message
                    )
            except Exception as e:
                self.logger.error(f"保存运行历史失败: {e}")

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


def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """
    运行Web服务器

    Args:
        host: 主机地址
        port: 端口
        debug: 调试模式
    """
    try:
        from src.web.app import run_web_server as start_web
        start_web(host=host, port=port, debug=debug)
    except ImportError as e:
        print(f"错误: 无法导入Web模块 - {e}")
        print("请确保已安装Flask及相关依赖: pip install Flask Flask-SQLAlchemy")
        sys.exit(1)


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
    parser.add_argument("--web", action="store_true", help="启动Web服务器")
    parser.add_argument("--web-host", default="127.0.0.1", help="Web服务器主机地址")
    parser.add_argument("--web-port", type=int, default=5000, help="Web服务器端口")
    parser.add_argument("--web-debug", action="store_true", help="Web调试模式")
    parser.add_argument("--import-data", action="store_true", help="导入历史数据到数据库")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")

    args = parser.parse_args()

    # 设置日志级别
    setup_logging(args.log_level)

    # Web模式
    if args.web:
        run_web_server(host=args.web_host, port=args.web_port, debug=args.web_debug)
        return

    # 数据导入模式
    if args.import_data:
        try:
            from src.web.app import create_app
            from src.web.services.data_importer import DataImporter
            from src.utils.logger import get_logger

            logger = get_logger('import')
            logger.info("开始导入历史数据...")

            # 创建Flask应用并运行在应用上下文中
            app = create_app()
            with app.app_context():
                importer = DataImporter('output')
                stats = importer.sync_from_output()

                logger.info(f"导入完成: {stats}")
                print(f"\n导入统计:")
                print(f"  文件: {stats['success_files']}/{stats['total_files']}")
                print(f"  新闻: {stats['imported_news']}/{stats['total_news']}")
                print(f"  跳过: {stats['skipped_news']}")
        except ImportError as e:
            print(f"错误: 无法导入数据模块 - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"导入失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # 创建应用实例
    app = NewsAutomationApp(args.config)

    if args.test_ai:
        app.test_ai()
    else:
        app.run(send_email=not args.no_email)


if __name__ == "__main__":
    main()
