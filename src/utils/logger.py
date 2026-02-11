"""
日志配置模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logger:
    """日志管理器"""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_logger(cls, name: str = "app", log_dir: str = "logs") -> logging.Logger:
        """
        获取日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志目录

        Returns:
            配置好的日志记录器
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # 避免重复添加handler
        if logger.handlers:
            return logger

        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 日志格式
        detailed_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # 文件处理器 - 详细日志
        file_handler = RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_format)

        # 文件处理器 - 错误日志
        error_handler = RotatingFileHandler(
            log_path / f"{name}_error.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_format)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_format)

        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)

        cls._loggers[name] = logger
        return logger


def setup_logging(level: str = "INFO"):
    """
    设置全局日志级别

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)


# 便捷函数
def get_logger(name: str = "app") -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return Logger.get_logger(name)
