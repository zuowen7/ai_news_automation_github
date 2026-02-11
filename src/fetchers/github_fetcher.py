"""
GitHub Trending 抓取器 - 优化版
使用GitHub官方REST API获取热门AI项目
使用统一的关键词配置
"""

from typing import List
from datetime import datetime, timedelta
import requests

from .base import NewsItem
from ..utils.logger import get_logger
from ..config.constants import AI_KEYWORDS


class GitHubTrendingFetcher:
    """GitHub Trending 抓取器 - 使用官方API"""

    # 关键词类别权重
    KEYWORD_WEIGHTS = {
        "core": 3.0,
        "llm": 3.0,
        "multimodal": 2.5,
        "trends": 2.5,
        "frameworks": 2.0,
        "rag": 2.0,
        "applications": 1.5,
        "infrastructure": 1.5,
        "companies": 1.0,
        "ethics": 1.0,
        "chinese": 2.0,
        "academic": 0.5,
    }

    # 高价值关键词（双倍权重）
    HIGH_VALUE_KEYWORDS = {
        'gpt', 'claude', 'gemini', 'llama', 'mistral', 'deepseek',
        'qwen', 'transformer', 'diffusion model', 'reinforcement learning',
        'machine learning', 'deep learning', 'neural network',
        'pytorch', 'tensorflow', 'hugging face', 'langchain',
        'computer vision', 'nlp', 'rag', 'agent', 'multimodal'
    }

    def __init__(self, token: str = "", max_news: int = 5, use_retry: bool = True):
        self.name = "GitHub Trending"
        self.max_news = max_news
        self.token = token
        self.use_retry = use_retry
        self.logger = get_logger("fetcher.github")
        self.api_url = "https://api.github.com/search/repositories"

        # 展平所有AI关键词
        self.all_keywords = []
        for category_keywords in AI_KEYWORDS.values():
            self.all_keywords.extend(category_keywords)

        # 构建关键词索引
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建关键词索引"""
        self.category_keywords = {}
        for category, keywords in AI_KEYWORDS.items():
            self.category_keywords[category] = [kw.lower() for kw in keywords]

    def _calculate_ai_score(self, text: str) -> float:
        """
        计算项目的AI相关度分数

        Args:
            text: 项目名称和描述

        Returns:
            AI相关度分数
        """
        text_lower = text.lower()
        total_score = 0.0

        # 按类别计算分数
        for category, keywords in self.category_keywords.items():
            weight = self.KEYWORD_WEIGHTS.get(category, 1.0)

            for kw in keywords:
                if kw in text_lower:
                    # 高价值关键词双倍权重
                    if kw in self.HIGH_VALUE_KEYWORDS:
                        total_score += weight * 2
                    else:
                        total_score += weight

        return total_score

    def fetch(self) -> List[NewsItem]:
        """抓取GitHub热门AI项目"""
        self.logger.info("使用GitHub API抓取热门项目")

        try:
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            # 首先尝试搜索包含AI关键词的项目
            ai_queries = [
                f"machine learning stars:>100 pushed:>{week_ago}",
                f"deep learning stars:>100 pushed:>{week_ago}",
                f"llm stars:>100 pushed:>{week_ago}",
                f"pytorch stars:>100 pushed:>{week_ago}",
                f"transformer stars:>100 pushed:>{week_ago}",
            ]

            all_ai_projects = []

            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AI-News-Automation'
            }

            if self.token:
                headers['Authorization'] = f"token {self.token}"

            # 尝试多个AI相关查询
            for query in ai_queries:
                params = {
                    'q': query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 20
                }

                try:
                    response = requests.get(self.api_url, params=params, headers=headers, timeout=30)

                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])

                        for item in items:
                            try:
                                name = item.get('name', '')
                                description = item.get('description', '') or ''
                                language = item.get('language', '')
                                stars = item.get('stargazers_count', 0)
                                url = item.get('html_url', '')

                                # 计算AI相关度分数
                                combined_text = f"{name} {description}"
                                ai_score = self._calculate_ai_score(combined_text)

                                # 最低分数要求
                                if ai_score >= 2.0:
                                    # 检查是否已存在
                                    if not any(p['url'] == url for p in all_ai_projects):
                                        all_ai_projects.append({
                                            'title': name,
                                            'url': url,
                                            'description': description[:150] if description else '',
                                            'language': language or 'Unknown',
                                            'stars': stars,
                                            'ai_score': ai_score
                                        })
                            except Exception:
                                continue

                except Exception:
                    continue

                # 如果已经找到足够项目，停止搜索
                if len(all_ai_projects) >= self.max_news * 2:
                    break

            # 如果AI项目不足，补充热门项目（只要和AI相关）
            if len(all_ai_projects) < self.max_news:
                fallback_query = f"stars:>5000 pushed:>{week_ago}"
                params = {
                    'q': fallback_query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 50
                }

                try:
                    response = requests.get(self.api_url, params=params, headers=headers, timeout=30)

                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])

                        for item in items:
                            try:
                                name = item.get('name', '')
                                description = item.get('description', '') or ''
                                language = item.get('language', '')
                                stars = item.get('stargazers_count', 0)
                                url = item.get('html_url', '')

                                combined_text = f"{name} {description}"
                                ai_score = self._calculate_ai_score(combined_text)

                                # 更严格的分数要求
                                if ai_score >= 4.0:
                                    if not any(p['url'] == url for p in all_ai_projects):
                                        all_ai_projects.append({
                                            'title': name,
                                            'url': url,
                                            'description': description[:150] if description else '',
                                            'language': language or 'Unknown',
                                            'stars': stars,
                                            'ai_score': ai_score
                                        })
                            except Exception:
                                continue
                except Exception:
                    pass

            if all_ai_projects:
                # 先按AI分数排序，再按star数排序
                all_ai_projects.sort(key=lambda x: (x['ai_score'], x['stars']), reverse=True)
                selected = all_ai_projects[:self.max_news]

                self.logger.info(f"从GitHub API获取到 {len(selected)} 个AI项目")

                # 构建NewsItem
                news_list = []
                for project in selected:
                    # 标题只包含项目名和描述，不包含star数
                    display_title = project['title']
                    if project['description']:
                        display_title += f" - {project['description']}"

                    # star数放在summary里，方便模板解析
                    summary_parts = [f"stars: {project['stars']}", project['language']]

                    news_list.append(NewsItem(
                        title=display_title,
                        url=project['url'],
                        source="GitHub Trending",
                        region="global",
                        summary=" | ".join(summary_parts),
                        date=datetime.now().strftime('%Y-%m-%d'),
                        news_type="github"
                    ))

                return news_list
            else:
                self.logger.info("GitHub API未找到AI项目")
                return []

        except Exception as e:
            self.logger.error(f"GitHub API抓取失败: {e}")
            return []

    def close(self):
        """关闭方法"""
        pass
