"""
AI新闻处理器
使用AI模型进行新闻摘要和趋势分析
"""

import re
from typing import List, Optional

from ..utils.logger import get_logger
from ..config.constants import AI_PROMPTS
from .ollama_client import OllamaClient
from ..fetchers.base import NewsItem
from ..config.settings import AIConfig


def clean_markdown(text: str) -> str:
    """
    清理Markdown格式标记

    Args:
        text: 包含markdown格式的文本

    Returns:
        清理后的纯文本
    """
    if not text:
        return text

    # 清理标题标记 (# ## ### 等)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 清理加粗标记 (**text** 或 __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    # 清理斜体标记 (*text* 或 _text_)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)

    # 清理删除线标记
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    # 清理代码标记
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 清理链接格式 [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

    # 清理多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


class NewsAIProcessor:
    """新闻AI处理器"""

    def __init__(self, config: AIConfig, cache_manager=None):
        """
        初始化AI处理器

        Args:
            config: AI配置
            cache_manager: 缓存管理器（可选）
        """
        self.config = config
        self.logger = get_logger("ai.processor")
        self.client = OllamaClient(config)
        self.cache_manager = cache_manager
        self._summary_cache: Optional[str] = None
        self._trends_cache: Optional[str] = None

    def is_available(self) -> bool:
        """检查AI服务是否可用"""
        return self.client.is_available()

    def generate_summary(self, news_list: List[NewsItem]) -> str:
        """
        生成新闻摘要（只基于真实新闻，不包括GitHub/HF项目）

        Args:
            news_list: 新闻列表

        Returns:
            摘要文本
        """
        if not self.is_available() or not news_list:
            return "AI摘要功能暂不可用"

        # 只使用真实新闻，排除GitHub和Hugging Face项目，包括type: "news" 和 "rss"
        real_news = [n for n in news_list if n.news_type in ["news", "rss"]]

        if not real_news:
            return "今日暂无AI新闻"

        # 提取新闻标题（最多5条）
        top_news = real_news[:5]
        numbered_titles = [f"{i}. {news.title}" for i, news in enumerate(top_news, 1)]
        titles_text = "\n".join(numbered_titles)

        # 构建提示词
        prompt = AI_PROMPTS["summary"].format(news_titles=titles_text)

        self.logger.info(f"正在生成AI新闻摘要（基于{len(real_news)}条真实新闻）...")
        result = self.client.generate(prompt, temperature=0.3)

        # 清理markdown格式
        result = clean_markdown(result) if result else ""

        # 缓存结果
        self._summary_cache = result if result else ""

        if result:
            self.logger.info(f"摘要生成完成: {result[:50]}...")
        else:
            self.logger.warning("摘要生成为空")

        return result if result else f"今日AI新闻摘要：共{len(real_news)}条新闻，涵盖大模型、AI应用等前沿动态。"

    def generate_trends(self, news_list: List[NewsItem]) -> str:
        """
        生成趋势分析（只基于真实新闻，不包括GitHub/HF项目）

        Args:
            news_list: 新闻列表

        Returns:
            趋势分析文本
        """
        if not self.is_available() or not news_list:
            return "趋势分析功能暂不可用"

        # 只使用真实新闻，排除GitHub和Hugging Face项目，包括type: "news" 和 "rss"
        real_news = [n for n in news_list if n.news_type in ["news", "rss"]]

        if not real_news:
            return "暂无趋势分析"

        # 提取所有新闻标题
        all_titles = "\n".join([f"- {news.title}" for news in real_news])

        # 构建提示词
        prompt = AI_PROMPTS["trends"].format(news_titles=all_titles)

        self.logger.info(f"正在生成AI趋势分析（基于{len(real_news)}条真实新闻）...")
        result = self.client.generate(prompt, temperature=0.5)

        # 清理markdown格式
        result = clean_markdown(result) if result else ""

        # 缓存结果
        self._trends_cache = result if result else ""

        if result:
            self.logger.info(f"趋势分析完成: {result[:50]}...")
        else:
            self.logger.warning("趋势分析为空")

        return result if result else "AI趋势分析：大语言模型持续发展，多模态AI应用成为新热点，AI在各行业加速落地。"

    def generate_all(self, news_list: List[NewsItem]) -> tuple[str, str]:
        """
        生成所有AI内容

        Args:
            news_list: 新闻列表

        Returns:
            (摘要, 趋势分析) 元组
        """
        if not self.is_available():
            return "", ""

        summary = self.generate_summary(news_list)
        trends = self.generate_trends(news_list)

        return summary, trends

    def get_cached_summary(self) -> str:
        """获取缓存的摘要"""
        return self._summary_cache or ""

    def get_cached_trends(self) -> str:
        """获取缓存的趋势分析"""
        return self._trends_cache or ""

    def clear_cache(self):
        """清除缓存"""
        self._summary_cache = None
        self._trends_cache = None

    def generate_summary_for_region(
        self,
        news_list: List[NewsItem],
        region: str,
        model: str
    ) -> str:
        """
        为指定区域生成新闻摘要

        Args:
            news_list: 新闻列表
            region: 区域 ("domestic" 或 "global")
            model: 使用的模型名称

        Returns:
            摘要文本
        """
        if not self.is_available() or not news_list:
            return ""

        # 只使用指定区域的真实新闻
        real_news = [n for n in news_list if n.news_type in ["news", "rss"] and n.region == region]

        if not real_news:
            return f"{'国内' if region == 'domestic' else '国际'}暂无AI新闻"

        # 提取新闻标题
        top_news = real_news[:10]  # 最多10条
        numbered_titles = [f"{i}. {news.title}" for i, news in enumerate(top_news, 1)]
        titles_text = "\n".join(numbered_titles)

        # 构建提示词
        region_name = "国内" if region == "domestic" else "国际"
        prompt = f"""请为以下{region_name}AI新闻生成简洁摘要（100字以内）：

{titles_text}

要求：
1. 提炼核心信息，突出重点
2. 语言简洁专业
3. 不要分点，用连贯的段落

摘要："""

        self.logger.info(f"正在生成{region_name}AI新闻摘要（模型: {model}，基于{len(real_news)}条新闻）...")
        result = self.client.generate(prompt, model=model, temperature=0.3)

        # 清理markdown格式
        result = clean_markdown(result) if result else ""

        if result:
            self.logger.info(f"{region_name}摘要生成完成: {result[:50]}...")
        else:
            self.logger.warning(f"{region_name}摘要生成为空")

        return result if result else f"{region_name}AI新闻摘要：共{len(real_news)}条新闻"

    def generate_trends_for_region(
        self,
        news_list: List[NewsItem],
        region: str,
        model: str
    ) -> str:
        """
        为指定区域生成趋势分析

        Args:
            news_list: 新闻列表
            region: 区域 ("domestic" 或 "global")
            model: 使用的模型名称

        Returns:
            趋势分析文本
        """
        if not self.is_available() or not news_list:
            return ""

        # 只使用指定区域的真实新闻
        real_news = [n for n in news_list if n.news_type in ["news", "rss"] and n.region == region]

        if not real_news:
            return f"{'国内' if region == 'domestic' else '国际'}暂无趋势分析"

        # 提取所有新闻标题
        all_titles = "\n".join([f"- {news.title}" for news in real_news[:15]])

        # 构建提示词
        region_name = "国内" if region == "domestic" else "国际"
        prompt = f"""基于以下{region_name}AI新闻，分析当前AI发展趋势（150字以内）：

{all_titles}

要求：
1. 识别主要技术趋势
2. 分析行业发展方向
3. 语言简洁专业

趋势分析："""

        self.logger.info(f"正在生成{region_name}AI趋势分析（模型: {model}，基于{len(real_news)}条新闻）...")
        result = self.client.generate(prompt, model=model, temperature=0.5)

        # 清理markdown格式
        result = clean_markdown(result) if result else ""

        if result:
            self.logger.info(f"{region_name}趋势分析完成: {result[:50]}...")
        else:
            self.logger.warning(f"{region_name}趋势分析为空")

        return result if result else f"{region_name}AI趋势：涵盖大模型、AI应用等前沿动态"

    def generate_combined_summary(
        self,
        domestic_summary: str,
        domestic_trends: str,
        global_summary: str,
        global_trends: str
    ) -> tuple[str, str]:
        """
        合并国内外摘要和趋势，生成最终版本

        Args:
            domestic_summary: 国内新闻摘要
            domestic_trends: 国内趋势分析
            global_summary: 国际新闻摘要
            global_trends: 国际趋势分析

        Returns:
            (合并后的摘要, 合并后的趋势)
        """
        if not self.is_available():
            return domestic_summary or global_summary or "", domestic_trends or global_trends or ""

        prompt = f"""请将以下国内外AI新闻摘要和趋势分析合并成一份综合报告：

【国内新闻摘要】
{domestic_summary or "暂无"}

【国际新闻摘要】
{global_summary or "暂无"}

【国内趋势分析】
{domestic_trends or "暂无"}

【国际趋势分析】
{global_trends or "暂无"}

请分别输出：
1. 综合摘要（150字以内，涵盖国内外重点）
2. 综合趋势（200字以内，分析整体发展方向）

请按以下格式输出：
摘要：[你的摘要]
趋势：[你的趋势分析]"""

        self.logger.info("正在生成综合摘要和趋势分析...")
        result = self.client.generate(prompt, model="qwen2.5:7b-instruct", temperature=0.4)

        if result:
            # 解析结果
            summary_match = re.search(r'摘要[：:]\s*(.+?)(?=趋势[：:]|$)', result, re.DOTALL)
            trends_match = re.search(r'趋势[：:]\s*(.+?)$', result, re.DOTALL)

            summary = clean_markdown(summary_match.group(1)) if summary_match else (domestic_summary + "\n" + global_summary if domestic_summary and global_summary else domestic_summary or global_summary or "")
            trends = clean_markdown(trends_match.group(1)) if trends_match else (domestic_trends + "\n" + global_trends if domestic_trends and global_trends else domestic_trends or global_trends or "")

            self.logger.info(f"综合分析完成")
            return summary, trends

        # 失败时返回合并版本
        return (
            domestic_summary + "\n" + global_summary if domestic_summary and global_summary else domestic_summary or global_summary or "",
            domestic_trends + "\n" + global_trends if domestic_trends and global_trends else domestic_trends or global_trends or ""
        )

    def generate_all_separated(self, news_list: List[NewsItem]) -> tuple[str, str]:
        """
        生成所有AI内容（国内外分别处理）

        Args:
            news_list: 新闻列表

        Returns:
            (摘要, 趋势分析) 元组
        """
        if not self.is_available():
            return "", ""

        # 生成新闻哈希列表用于缓存检查
        from ..utils.helpers import generate_hash
        news_hashes = [generate_hash(news.title.lower().strip()) for news in news_list]

        # 检查缓存
        if self.cache_manager:
            cached_result = self.cache_manager.get_ai_cached_result(news_hashes)
            if cached_result:
                self.logger.info("使用AI缓存结果")
                self._summary_cache = cached_result[0]
                self._trends_cache = cached_result[1]
                return cached_result

        # 分别生成国内外的摘要和趋势
        domestic_summary = self.generate_summary_for_region(news_list, "domestic", "qwen2.5:7b-instruct")
        domestic_trends = self.generate_trends_for_region(news_list, "domestic", "qwen2.5:7b-instruct")
        global_summary = self.generate_summary_for_region(news_list, "global", "llama3.1:8b")
        global_trends = self.generate_trends_for_region(news_list, "global", "llama3.1:8b")

        # 合并生成最终版本
        final_summary, final_trends = self.generate_combined_summary(
            domestic_summary, domestic_trends,
            global_summary, global_trends
        )

        # 缓存结果
        self._summary_cache = final_summary
        self._trends_cache = final_trends

        # 保存到缓存管理器
        if self.cache_manager:
            self.cache_manager.save_ai_result(news_hashes, final_summary, final_trends)

        return final_summary, final_trends
