"""
Hugging Face Trending 抓取器
抓取Hugging Face上热门的AI模型
"""

from typing import List
from datetime import datetime
import requests

from ..utils.logger import get_logger
from .base import BaseFetcher, NewsItem


class HuggingFaceFetcher(BaseFetcher):
    """Hugging Face Trending 抓取器"""

    def __init__(self, max_news: int = 5, use_retry: bool = True):
        super().__init__("Hugging Face", "https://huggingface.co", max_news, use_retry=use_retry)

    def fetch(self) -> List[NewsItem]:
        """抓取Hugging Face热门模型 - 按最近更新排序"""
        self.logger.info("开始抓取Hugging Face热门模型")

        try:
            # 使用API获取热门模型 - 按下载量排序
            api_url = "https://huggingface.co/api/models"
            params = {
                "limit": 100,  # 增加候选数量
                "sort": "downloads",  # 按下载量排序，获取热门模型
                "filter": "pytorch"  # 只获取PyTorch模型
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(api_url, params=params, headers=headers, timeout=15)

            if response.status_code != 200:
                self.logger.warning(f"Hugging Face API请求失败: {response.status_code}")
                return []

            models = response.json()
            news_list = []

            for model in models[:self.max_news * 10]:  # 增加候选数量
                try:
                    model_id = model.get('id', '')
                    if not model_id:
                        continue

                    # 获取模型详情
                    model_data = self._get_model_details(model_id)
                    if not model_data:
                        continue

                    # 从model_data获取下载量和点赞数
                    downloads = model_data.get('downloads', 0)
                    likes = model_data.get('likes', 0)

                    # 跳过下载量太低的模型（可能是测试/个人模型）
                    if downloads < 100:
                        continue

                    # 跳过太老的基础模型（除非有大量下载）
                    model_name = model_id.lower()
                    skip_models = ['bert', 'gpt2', 'resnet', 'mobilenet', 'efficientnet', 'vit']
                    if any(skip in model_name for skip in skip_models):
                        # 如果是最近更新的保留
                        if downloads < 100000:
                            continue

                    # 格式化下载量
                    if downloads >= 1000000:
                        dl_str = f"{downloads/1000000:.1f}M"
                    elif downloads >= 1000:
                        dl_str = f"{downloads/1000:.1f}K"
                    else:
                        dl_str = str(downloads)

                    # 标题只用模型名称，简洁清晰
                    title = model_id

                    # 只显示有趣的描述，避免太技术化
                    description = model_data.get('description', '')
                    if description and len(description) < 100:
                        # 清理描述中的特殊字符
                        description = description.replace('\n', ' ').replace('\r', '')
                        title += f" - {description}"

                    # 构建摘要 - 下载量放在最前面，用于在模板中显示
                    summary_parts = [f"下载: {dl_str}"]
                    if model_data.get('pipeline_tag'):
                        summary_parts.append(f"任务: {model_data['pipeline_tag']}")
                    if likes:
                        summary_parts.append(f"👍 {likes}")

                    news_list.append(NewsItem(
                        title=title,
                        url=f"https://huggingface.co/{model_id}",
                        source="Hugging Face",
                        region="global",
                        summary=" | ".join(summary_parts) if summary_parts else "",
                        date=datetime.now().strftime('%Y-%m-%d'),
                        news_type="huggingface"
                    ))

                    if len(news_list) >= self.max_news:
                        break

                except Exception as e:
                    self.logger.debug(f"解析模型失败: {e}")
                    continue

            if news_list:
                self.logger.info(f"从 Hugging Face 获取到 {len(news_list)} 个热门模型")
            else:
                self.logger.info("Hugging Face没有找到合适的模型")

            return news_list

        except Exception as e:
            self.logger.error(f"Hugging Face抓取失败: {e}")
            return []

    def _get_model_details(self, model_id: str) -> dict:
        """获取模型详情"""
        try:
            url = f"https://huggingface.co/api/models/{model_id}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
