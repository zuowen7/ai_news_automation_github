"""
Ollama客户端模块
与本地Ollama服务交互
"""

import time
import requests
from typing import Optional, Dict, Any, List

from ..utils.logger import get_logger
from ..config.settings import AIConfig


class OllamaClient:
    """Ollama API客户端"""

    def __init__(self, config: AIConfig):
        """
        初始化Ollama客户端

        Args:
            config: AI配置
        """
        self.config = config
        self.logger = get_logger("ai.ollama")
        self.base_url = config.ollama_url.rstrip('/')
        self.model = config.model_name
        self.available = False

        # 检查连接
        self._check_connection()

    def _check_connection(self) -> bool:
        """
        检查Ollama服务是否可用

        Returns:
            是否可用
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            self.available = response.status_code == 200
            if self.available:
                self.logger.info(f"Ollama连接成功，模型: {self.model}")
            else:
                self.logger.warning(f"Ollama响应异常: {response.status_code}")
            return self.available
        except requests.RequestException as e:
            self.logger.warning(f"无法连接到Ollama: {e}")
            self.available = False
            return False

    def is_available(self) -> bool:
        """检查Ollama是否可用"""
        return self.available and self.config.enabled

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示词
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            model: 指定模型（不指定则使用默认模型）

        Returns:
            生成的文本
        """
        if not self.is_available():
            self.logger.warning("Ollama不可用，返回空结果")
            return ""

        try:
            # 构建消息
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # 请求参数
            request_data = {
                "model": model if model else self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "stream": False
            }

            if max_tokens:
                request_data["num_predict"] = max_tokens

            self.logger.debug(f"发送请求到Ollama，模型: {request_data['model']}，提示词长度: {len(prompt)}")

            # 发送请求
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.config.timeout
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                self.logger.info(f"Ollama生成成功，模型: {request_data['model']}，耗时: {elapsed:.2f}s，返回长度: {len(content)}")
                return content
            else:
                self.logger.error(f"Ollama API错误: {response.status_code} - {response.text}")
                return ""

        except requests.Timeout:
            self.logger.error(f"Ollama请求超时 (>{self.config.timeout}s)")
            return ""
        except requests.RequestException as e:
            self.logger.error(f"Ollama请求失败: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Ollama生成出错: {e}")
            return ""

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话接口

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        if not self.is_available():
            return ""

        try:
            request_data = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            return ""

        except Exception as e:
            self.logger.error(f"Ollama对话失败: {e}")
            return ""

    def check_model(self) -> bool:
        """
        检查指定模型是否可用

        Returns:
            模型是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # 检查模型名匹配（可能包含版本标签）
                for name in model_names:
                    if name.startswith(self.model.split(":")[0]):
                        self.logger.info(f"找到模型: {name}")
                        return True
                self.logger.warning(f"未找到模型 {self.model}，可用模型: {model_names}")
                return False
        except Exception as e:
            self.logger.error(f"检查模型失败: {e}")
        return False
