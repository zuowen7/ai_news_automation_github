"""
配置管理模块
负责加载、验证和管理应用配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..utils.helpers import is_valid_email, merge_dicts
from .constants import DEFAULT_CONFIG


@dataclass
class EmailConfig:
    """邮件配置"""
    sender_email: str
    sender_password: str
    smtp_server: str
    smtp_port: int
    recipient_email: str


@dataclass
class SettingsConfig:
    """设置配置"""
    email_subject: str = "AI新闻日报 - {date}"
    html_email: bool = True
    qq_mail_format: bool = True


@dataclass
class AIConfig:
    """AI配置"""
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    enable_filter: bool = True


@dataclass
class FetcherConfig:
    """抓取器配置"""
    max_news_per_source: int = 5
    concurrent_requests: int = 3
    retry_times: int = 2
    retry_delay: int = 1
    enable_github: bool = True
    enable_huggingface: bool = True
    # 新增：优化配置
    use_exponential_backoff: bool = True  # 是否使用指数退避
    max_backoff_time: float = 60.0  # 最大退避时间（秒）
    dynamic_concurrency: bool = False  # 是否启用动态并发
    incremental_fetch: bool = False  # 是否启用增量抓取
    check_hours: int = 24  # 增量抓取检查时间范围（小时）


@dataclass
class OutputConfig:
    """输出配置"""
    save_json: bool = True
    save_html: bool = True
    output_dir: str = "output"


@dataclass
class AppConfig:
    """应用配置"""
    email: EmailConfig
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "config.json"):
        self.logger = get_logger("config")
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        """
        加载配置文件

        Returns:
            应用配置对象
        """
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            self.logger.warning(f"配置文件不存在: {self.config_path}")
            self._config = self._create_default_config()
            return self._config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 从环境变量获取密码
            env_password = os.getenv('AI_NEWS_EMAIL_PASSWORD')
            if env_password and env_password.strip():
                config_data['email']['sender_password'] = env_password.strip()
                self.logger.info("邮箱密码已从环境变量加载")

            # 验证并创建配置对象
            self._config = self._parse_config(config_data)
            self.logger.info("配置加载成功")
            return self._config

        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件JSON格式错误: {e}")
            self._config = self._create_default_config()
            return self._config
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            self._config = self._create_default_config()
            return self._config

    def _parse_config(self, config_data: Dict[str, Any]) -> AppConfig:
        """
        解析配置数据

        Args:
            config_data: 配置字典

        Returns:
            应用配置对象
        """
        # 合并默认配置
        merged = merge_dicts(DEFAULT_CONFIG, config_data)

        # 创建各配置对象
        email_config = EmailConfig(**merged['email'])
        settings_config = SettingsConfig(**merged['settings'])
        ai_config = AIConfig(**merged['ai'])
        fetcher_config = FetcherConfig(**merged.get('fetcher', {}))
        output_config = OutputConfig(**merged.get('output', {}))

        # 验证邮箱
        if email_config.sender_email and not is_valid_email(email_config.sender_email):
            self.logger.warning(f"发件人邮箱格式可能不正确: {email_config.sender_email}")

        if email_config.recipient_email and not is_valid_email(email_config.recipient_email):
            self.logger.warning(f"收件人邮箱格式可能不正确: {email_config.recipient_email}")

        return AppConfig(
            email=email_config,
            settings=settings_config,
            ai=ai_config,
            fetcher=fetcher_config,
            output=output_config
        )

    def _create_default_config(self) -> AppConfig:
        """创建默认配置"""
        self.logger.info("使用默认配置")

        merged = merge_dicts(DEFAULT_CONFIG, {})

        return AppConfig(
            email=EmailConfig(**merged['email']),
            settings=SettingsConfig(**merged['settings']),
            ai=AIConfig(**merged['ai']),
            fetcher=FetcherConfig(**merged.get('fetcher', {})),
            output=OutputConfig(**merged.get('output', {}))
        )

    def save(self, config: AppConfig) -> None:
        """
        保存配置到文件

        Args:
            config: 应用配置对象
        """
        config_data = {
            "email": {
                "sender_email": config.email.sender_email,
                "sender_password": "",  # 不保存密码到文件
                "smtp_server": config.email.smtp_server,
                "smtp_port": config.email.smtp_port,
                "recipient_email": config.email.recipient_email
            },
            "settings": {
                "email_subject": config.settings.email_subject,
                "html_email": config.settings.html_email,
                "qq_mail_format": config.settings.qq_mail_format
            },
            "ai": {
                "enabled": config.ai.enabled,
                "ollama_url": config.ai.ollama_url,
                "model_name": config.ai.model_name,
                "temperature": config.ai.temperature,
                "max_tokens": config.ai.max_tokens,
                "timeout": config.ai.timeout
            },
            "fetcher": {
                "max_news_per_source": config.fetcher.max_news_per_source,
                "concurrent_requests": config.fetcher.concurrent_requests,
                "retry_times": config.fetcher.retry_times,
                "retry_delay": config.fetcher.retry_delay
            },
            "output": {
                "save_json": config.output.save_json,
                "save_html": config.output.save_html,
                "output_dir": config.output.output_dir
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"配置已保存到: {self.config_path}")

    def reload(self) -> AppConfig:
        """重新加载配置"""
        self._config = None
        return self.load()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = "config.json") -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager
