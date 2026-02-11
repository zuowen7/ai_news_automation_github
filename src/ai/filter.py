"""
AI新闻筛选器 - 优化版
使用本地AI模型判断新闻是否值得发送
增强关键词匹配和打分机制
"""

from typing import List, Tuple
import re
from ..utils.logger import get_logger
from ..fetchers.base import NewsItem
from ..config.settings import AIConfig
from ..config.constants import AI_KEYWORDS, EXCLUDE_KEYWORDS, ALL_AI_KEYWORDS
from .ollama_client import OllamaClient


class NewsAIFilter:
    """AI新闻筛选器 - 优化版"""

    # 关键词类别权重（用于打分）
    KEYWORD_WEIGHTS = {
        "core": 3.0,        # 核心AI术语 - 最高权重
        "llm": 3.0,         # 大语言模型 - 最高权重
        "multimodal": 2.5,  # 多模态AI - 高权重
        "trends": 2.5,      # 新兴趋势 - 高权重
        "frameworks": 2.0,  # AI框架 - 中高权重
        "rag": 2.0,         # RAG技术 - 中高权重
        "applications": 1.5, # AI应用 - 中等权重
        "infrastructure": 1.5, # AI基础设施 - 中等权重
        "companies": 1.0,   # AI公司 - 基础权重
        "ethics": 1.5,      # AI伦理 - 中等权重
        "chinese": 2.0,     # 中文术语 - 中高权重
        "academic": 1.0,    # 学术术语 - 基础权重
    }

    # 高价值关键词（直接影响是否保留）
    HIGH_VALUE_KEYWORDS = {
        # 2025年热门模型和产品
        'claude 4', 'gemini 2.0', 'gpt-5', 'o3', 'llama 4', 'mistral large',
        'deepseek', 'qwen 2', 'sora', 'runway', 'midjourney', 'stable diffusion 3',

        # 重要技术突破
        'agi', 'reasoning model', 'world model', 'embodied ai',
        'reinforcement learning', 'diffusion model', 'transformer',

        # 重要应用领域
        'autonomous driving', 'drug discovery', 'protein folding',
        'ai research', 'ai breakthrough', 'ai safety',

        # 中文重要术语
        '人工智能突破', '大模型发布', 'ai芯片', '算力',
    }

    def __init__(self, config: AIConfig):
        """
        初始化AI筛选器

        Args:
            config: AI配置
        """
        self.config = config
        self.logger = get_logger("ai.filter")
        self.client = OllamaClient(config)

        # 构建关键词查找索引
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建关键词索引，快速查找"""
        self.keyword_index = {}
        self.category_keywords = {}

        for category, keywords in AI_KEYWORDS.items():
            self.category_keywords[category] = []
            for kw in keywords:
                kw_lower = kw.lower()
                self.keyword_index[kw_lower] = category
                self.category_keywords[category].append(kw_lower)

        # 编译排除关键词正则
        self.exclude_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in EXCLUDE_KEYWORDS),
            re.IGNORECASE
        )

    def is_available(self) -> bool:
        """检查AI是否可用"""
        return self.client.is_available()

    def calculate_keyword_score(self, title: str) -> Tuple[float, List[str]]:
        """
        计算标题的关键词得分

        Args:
            title: 新闻标题

        Returns:
            (得分, 匹配的关键词列表)
        """
        if not title:
            return 0.0, []

        title_lower = title.lower()
        total_score = 0.0
        matched_keywords = []

        # 按类别计算分数
        for category, keywords in self.category_keywords.items():
            category_score = 0
            weight = self.KEYWORD_WEIGHTS.get(category, 1.0)

            for kw in keywords:
                if kw in title_lower:
                    # 检查是否是高价值关键词
                    if kw in self.HIGH_VALUE_KEYWORDS:
                        category_score += weight * 2
                    else:
                        category_score += weight
                    matched_keywords.append(kw)

            total_score += category_score

        return total_score, matched_keywords

    def should_exclude(self, title: str, summary: str = "") -> bool:
        """
        检查是否应该排除该新闻

        Args:
            title: 新闻标题
            summary: 新闻摘要

        Returns:
            是否应该排除
        """
        text = f"{title} {summary}".lower()

        # 检查排除关键词
        for exclude_kw in EXCLUDE_KEYWORDS:
            if exclude_kw.lower() in text:
                return True

        # 检查标题长度
        if len(title) < 20 or len(title) > 300:
            return True

        return False

    def pre_filter_news(self, news_list: List[NewsItem], min_score: float = 2.0) -> List[NewsItem]:
        """
        使用关键词预筛选新闻，减少AI调用

        Args:
            news_list: 新闻列表
            min_score: 最低关键词得分

        Returns:
            预筛选后的新闻列表
        """
        if not news_list:
            return []

        filtered = []
        excluded_count = 0
        low_score_count = 0

        for news in news_list:
            # 跳过高价值项目
            if news.news_type in ["github", "huggingface"]:
                filtered.append(news)
                continue

            # 检查是否应该排除
            if self.should_exclude(news.title, news.summary or ""):
                excluded_count += 1
                continue

            # 计算关键词得分
            score, matched = self.calculate_keyword_score(news.title)

            # 得分太低也排除
            if score < min_score and not matched:
                low_score_count += 1
                continue

            # 保存得分到新闻对象（用于后续排序）
            news.ai_score = score
            news.matched_keywords = matched
            filtered.append(news)

        self.logger.info(f"关键词预筛选: 排除 {excluded_count} 条不相关，{low_score_count} 条低分，保留 {len(filtered)} 条")

        return filtered

    def filter_news(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """
        使用AI筛选新闻，只保留值得发送的
        国内外分别处理：国内前5名，国际前10名

        Args:
            news_list: 新闻列表

        Returns:
            筛选后的新闻列表
        """
        if not news_list:
            return []

        self.logger.info(f"开始AI筛选，共 {len(news_list)} 条新闻")

        # GitHub和Hugging Face内容直接保留，不参与筛选
        high_value_items = [n for n in news_list if n.news_type in ["github", "huggingface"]]
        regular_news = [n for n in news_list if n.news_type not in ["github", "huggingface"]]

        self.logger.info(f"高价值项目(GitHub/HF): {len(high_value_items)} 条，直接保留")

        if not regular_news:
            self.logger.info("没有常规新闻需要筛选")
            return high_value_items

        # 先进行关键词预筛选
        regular_news = self.pre_filter_news(regular_news, min_score=1.5)

        if not regular_news:
            self.logger.info("预筛选后没有剩余新闻")
            return high_value_items

        # 分离国内外新闻
        domestic_news = [n for n in regular_news if n.region == "domestic"]
        global_news = [n for n in regular_news if n.region == "global"]

        self.logger.info(f"国内新闻: {len(domestic_news)} 条，国际新闻: {len(global_news)} 条")

        # 按关键词得分排序
        domestic_news.sort(key=lambda x: getattr(x, 'ai_score', 0), reverse=True)
        global_news.sort(key=lambda x: getattr(x, 'ai_score', 0), reverse=True)

        # 如果AI不可用，直接返回预筛选结果（国内前5，国际前10）
        if not self.is_available():
            self.logger.warning("AI不可用，返回预筛选结果")
            final_domestic = domestic_news[:5] if domestic_news else []
            final_global = global_news[:10] if global_news else []
            final_regular = final_domestic + final_global
            self.logger.info(f"筛选完成（无AI）：国内 {len(final_domestic)} 条，国际 {len(final_global)} 条")
            return high_value_items + final_regular

        # 对国内新闻进行AI筛选（最多15条给AI评估）
        filtered_domestic = self._filter_region_news(domestic_news, "domestic", max_input=15, max_output=5)
        filtered_global = self._filter_region_news(global_news, "global", max_input=30, max_output=10)

        filtered = filtered_domestic + filtered_global

        self.logger.info(f"AI筛选完成：国内保留 {len(filtered_domestic)} 条，国际保留 {len(filtered_global)} 条")

        # 合并高价值项目和筛选后的新闻
        final_result = high_value_items + filtered
        self.logger.info(f"最终保留：{len(final_result)} 条 (高价值项目: {len(high_value_items)}, 常规新闻: {len(filtered)})")

        return final_result

    def _filter_region_news(
        self,
        news_list: List[NewsItem],
        region: str,
        max_input: int = 30,
        max_output: int = 10
    ) -> List[NewsItem]:
        """
        对指定区域的新闻进行AI筛选

        Args:
            news_list: 新闻列表（已按得分排序）
            region: 区域名称
            max_input: 最多输入多少条给AI
            max_output: 最多保留多少条

        Returns:
            筛选后的新闻列表
        """
        if not news_list:
            return []

        # 只取前N条给AI评估
        top_news = news_list[:max_input]

        # 构建新闻文本
        news_text = "\n".join([
            f"{i+1}. {news.title}" +
            (f" (关键词得分: {getattr(news, 'ai_score', 0):.1f})" if hasattr(news, 'ai_score') else "")
            for i, news in enumerate(top_news)
        ])

        region_name = "国内" if region == "domestic" else "国际"
        prompt = f"""请从以下{region_name}AI相关新闻中筛选出最值得发送的{max_output}条。

筛选标准（严格执行）：
【必须保留】
- 技术突破、研究进展、新模型/算法发布
- 重要产品发布、重大更新
- 行业领军公司的重大动态
- 融资、收购、合作等商业动态
- 深度分析、行业报告、数据洞察

【必须排除】
- 招聘信息、求职指南
- 广告推广、营销软文
- 内容空洞、缺乏实质信息的文章
- 标题党、夸大其词
- 重复内容、旧闻重提
- 纯产品评测、使用教程（除非有重大更新）

【宁缺毋滥】
- 如果新闻质量一般，宁可少发也不要发低质量内容

新闻列表：
{news_text}

请返回值得发送的新闻编号，用逗号分隔，例如：1,3,5,7,9
根据质量决定数量，最多保留{max_output}条
只返回编号："""

        result = self.client.generate(prompt, temperature=0.3)

        if result:
            # 解析结果
            try:
                numbers = re.findall(r'\d+', result)
                valid_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(top_news)]

                filtered = []
                for idx in valid_indices:
                    if 0 <= idx < len(top_news):
                        filtered.append(top_news[idx])

                self.logger.info(f"{region_name}AI筛选完成：保留 {len(filtered)} 条")
                return filtered

            except Exception as e:
                self.logger.error(f"解析{region_name}AI筛选结果失败: {e}")
                # 失败时按排名返回
                return news_list[:max_output]

        self.logger.warning(f"{region_name}AI筛选失败，按关键词得分保留前{max_output}条")
        return news_list[:max_output]

    def score_news(self, news: NewsItem) -> float:
        """
        为单条新闻打分（0-10分）
结合关键词得分和AI评分

        Args:
            news: 新闻项

        Returns:
            评分
        """
        # 先计算关键词得分
        keyword_score, _ = self.calculate_keyword_score(news.title)

        # 归一化到0-5分
        normalized_score = min(keyword_score / 3, 5.0)

        # 如果AI不可用，直接返回关键词得分
        if not self.is_available():
            return normalized_score

        # 获取AI评分
        prompt = f"""请为以下AI新闻打分（0-5分），从以下维度考虑：
1. 新闻的重要性（对行业的影响）- 权重40%
2. 内容的深度和独特性 - 权重35%
3. 来源的权威性 - 权重25%

新闻标题：{news.title}
新闻来源：{news.source}

只返回一个数字分数（0-5），保留一位小数："""

        result = self.client.generate(prompt, temperature=0.2)

        if result:
            try:
                score_match = re.search(r'(\d+(?:\.\d+)?)', result)
                if score_match:
                    ai_score = float(score_match.group(1))
                    ai_score = min(max(ai_score, 0), 5)

                    # 组合得分：关键词得分占40%，AI得分占60%
                    final_score = normalized_score * 0.4 + ai_score * 0.6
                    return min(final_score, 10.0)
            except:
                pass

        return normalized_score
