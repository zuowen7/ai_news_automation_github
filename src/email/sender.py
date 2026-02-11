"""
邮件发送模块
负责生成和发送新闻邮件
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from ..utils.logger import get_logger
from ..config.settings import AppConfig, EmailConfig, SettingsConfig
from ..fetchers.base import NewsItem
from .templates import EditorialNewsTemplate, MinimalNewsTemplate, ModernNewsTemplate, TextEmailTemplate


class EmailSender:
    """邮件发送器"""

    def __init__(self, config: AppConfig):
        """
        初始化邮件发送器

        Args:
            config: 应用配置
        """
        self.config = config
        self.email_config = config.email
        self.settings = config.settings
        self.logger = get_logger("email.sender")

        # 初始化模板（新的编辑风格作为默认）
        self.editorial_template = EditorialNewsTemplate()
        self.minimal_template = MinimalNewsTemplate()
        self.modern_template = ModernNewsTemplate()
        self.text_template = TextEmailTemplate()

    def generate_content(
        self,
        news_list: list[NewsItem],
        ai_summary: str = "",
        ai_trends: str = ""
    ) -> Tuple[str, str]:
        """
        生成邮件内容

        Args:
            news_list: 新闻列表
            ai_summary: AI摘要
            ai_trends: AI趋势分析

        Returns:
            (主题, 内容) 元组
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        subject = self.settings.email_subject.format(date=date_str)

        # 根据配置选择模板
        if self.settings.qq_mail_format:
            # QQ邮箱使用编辑风格（更好的兼容性）
            content = self.editorial_template.render(news_list, ai_summary, ai_trends, date_str)
        elif self.settings.html_email:
            # HTML邮件使用编辑风格
            content = self.editorial_template.render(news_list, ai_summary, ai_trends, date_str)
        else:
            content = self.text_template.render(news_list, ai_summary, ai_trends, date_str)

        return subject, content

    def send(self, subject: str, content: str, is_html: bool = True) -> bool:
        """
        发送邮件

        Args:
            subject: 邮件主题
            content: 邮件内容
            is_html: 是否为HTML格式

        Returns:
            是否发送成功
        """
        # 检查邮箱配置
        if not self._is_email_configured():
            self.logger.warning("邮箱未配置，跳过发送")
            return False

        try:
            self.logger.info(f"正在发送邮件到: {self.email_config.recipient_email}")

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.email_config.sender_email
            msg['To'] = self.email_config.recipient_email
            msg['Subject'] = subject

            # 添加内容
            msg.attach(MIMEText(content, 'html' if is_html else 'plain', 'utf-8'))

            # 发送邮件
            with smtplib.SMTP_SSL(
                self.email_config.smtp_server,
                self.email_config.smtp_port
            ) as server:
                server.login(self.email_config.sender_email, self.email_config.sender_password)
                server.send_message(msg)

            self.logger.info("邮件发送成功")
            return True

        except smtplib.SMTPAuthenticationError:
            self.logger.error("SMTP认证失败，请检查邮箱和授权码")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP错误: {e}")
            return False
        except Exception as e:
            self.logger.error(f"邮件发送失败: {e}")
            return False

    def send_news(
        self,
        news_list: list[NewsItem],
        ai_summary: str = "",
        ai_trends: str = ""
    ) -> bool:
        """
        发送新闻邮件

        Args:
            news_list: 新闻列表
            ai_summary: AI摘要
            ai_trends: AI趋势分析

        Returns:
            是否发送成功
        """
        subject, content = self.generate_content(news_list, ai_summary, ai_trends)
        is_html = self.settings.html_email or self.settings.qq_mail_format
        return self.send(subject, content, is_html)

    def save_to_file(
        self,
        news_list: list[NewsItem],
        ai_summary: str = "",
        ai_trends: str = "",
        output_dir: str = "output"
    ) -> None:
        """
        保存邮件到文件

        Args:
            news_list: 新闻列表
            ai_summary: AI摘要
            ai_trends: AI趋势分析
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime('%Y%m%d')

        # 保存JSON
        json_file = output_path / f"ai_news_{date_str}.json"
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([n.to_dict() for n in news_list], f, indent=2, ensure_ascii=False)

        # 保存HTML
        _, html_content = self.generate_content(news_list, ai_summary, ai_trends)
        html_file = output_path / f"ai_news_{date_str}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"已保存文件: {json_file.name}, {html_file.name}")

    def _is_email_configured(self) -> bool:
        """检查邮箱是否配置"""
        return bool(
            self.email_config.sender_email and
            self.email_config.sender_password and
            self.email_config.recipient_email and
            self.email_config.sender_email != "your_email@qq.com"
        )


class EmailPreviewGenerator:
    """邮件预览生成器"""

    @staticmethod
    def generate_preview(
        news_list: list[NewsItem],
        ai_summary: str = "",
        ai_trends: str = ""
    ) -> str:
        """
        生成邮件预览

        Args:
            news_list: 新闻列表
            ai_summary: AI摘要
            ai_trends: AI趋势分析

        Returns:
            预览HTML
        """
        template = ModernNewsTemplate()
        return template.render(news_list, ai_summary, ai_trends)
