"""
通用大模型客户端模块
支持 OpenAI 兼容接口：OpenAI、DeepSeek、通义千问、智谱AI、月之暗面等
同时保留对本地 Ollama 原生接口的支持
"""

import time
import requests
from typing import Optional, List, Dict, Any

from ..utils.logger import get_logger
from ..config.settings import AIConfig

# 预定义的远程服务商 base_url（OpenAI 兼容）
PROVIDER_URLS: Dict[str, str] = {
    "openai":       "https://api.openai.com/v1",
    "deepseek":     "https://api.deepseek.com/v1",
    "qwen":         "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu":        "https://open.bigmodel.cn/api/paas/v4",
    "moonshot":     "https://api.moonshot.cn/v1",
    "siliconflow":  "https://api.siliconflow.cn/v1",
    "ollama_compat": "http://localhost:11434/v1",   # Ollama OpenAI 兼容层
}


class LLMClient:
    """
    通用 LLM 客户端（OpenAI 兼容接口）

    支持所有实现了 /v1/chat/completions 接口的服务商。
    通过 config.json 中的 ai.provider 和 ai.api_key 字段来配置。
    """

    def __init__(self, config: AIConfig):
        self.config = config
        self.logger = get_logger("ai.llm_client")
        self.available = False

        # 解析 base_url
        self.base_url = self._resolve_base_url()
        self.api_key = config.api_key or "ollama"   # Ollama 不需要真实 key
        self.model = config.model_name

        self._check_connection()

    def _resolve_base_url(self) -> str:
        """根据 provider 名称或 custom_base_url 确定 API 地址"""
        # 优先使用用户自定义的 base_url
        if self.config.custom_base_url:
            return self.config.custom_base_url.rstrip('/')

        provider = (self.config.provider or "").lower()
        if provider in PROVIDER_URLS:
            return PROVIDER_URLS[provider]

        # 未知 provider，尝试把 ollama_url 当做 OpenAI 兼容层
        base = self.config.ollama_url.rstrip('/')
        if not base.endswith('/v1'):
            base = f"{base}/v1"
        return base

    def _check_connection(self) -> bool:
        """检查服务是否可达（HEAD /models 或 GET /models）"""
        try:
            headers = self._build_headers()
            # 绝大多数 OpenAI 兼容接口都有 /models 端点
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=8
            )
            self.available = resp.status_code in (200, 401, 403)
            # 401/403 说明服务可达，只是 key 问题；200 则完全 OK
            if resp.status_code == 200:
                self.logger.info(
                    f"[LLMClient] 连接成功 → {self.base_url}  model={self.model}"
                )
            elif self.available:
                self.logger.warning(
                    f"[LLMClient] 服务可达但鉴权失败({resp.status_code}) → {self.base_url}"
                )
            else:
                self.logger.warning(
                    f"[LLMClient] 连接异常({resp.status_code}) → {self.base_url}"
                )
            return self.available
        except Exception as e:
            self.logger.warning(f"[LLMClient] 无法连接 {self.base_url}: {e}")
            self.available = False
            return False

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self) -> bool:
        return self.available and self.config.enabled

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        生成文本（与 OllamaClient.generate 接口完全兼容）

        Args:
            prompt:      用户提示词
            system:      系统提示词（可选）
            temperature: 温度参数
            max_tokens:  最大输出 token 数
            model:       指定模型名，不填则用配置中的 model_name

        Returns:
            生成的文本，失败返回空字符串
        """
        if not self.is_available():
            self.logger.warning("[LLMClient] 服务不可用，返回空结果")
            return ""

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._chat_completion(
            messages=messages,
            model=model or self.model,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话接口（与 OllamaClient.chat 接口完全兼容）
        """
        if not self.is_available():
            return ""
        return self._chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用 /v1/chat/completions 并返回 content 字符串"""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        self.logger.debug(
            f"[LLMClient] POST {self.base_url}/chat/completions  "
            f"model={model}  prompt_len={len(str(messages))}"
        )

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
                timeout=self.config.timeout,
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                self.logger.info(
                    f"[LLMClient] 生成成功 model={model} "
                    f"耗时={elapsed:.2f}s 返回长度={len(content)}"
                )
                return content
            else:
                self.logger.error(
                    f"[LLMClient] API 错误 {resp.status_code}: {resp.text[:300]}"
                )
                return ""

        except requests.Timeout:
            self.logger.error(f"[LLMClient] 请求超时 (>{self.config.timeout}s)")
            return ""
        except requests.RequestException as e:
            self.logger.error(f"[LLMClient] 请求失败: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"[LLMClient] 未知错误: {e}")
            return ""

    def check_model(self) -> bool:
        """检查模型是否在服务商模型列表中"""
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._build_headers(),
                timeout=8,
            )
            if resp.status_code == 200:
                models_data = resp.json()
                # OpenAI 格式：{"data": [{"id": "gpt-4o", ...}]}
                ids = [m.get("id", "") for m in models_data.get("data", [])]
                if any(self.model in mid or mid.startswith(self.model.split(":")[0]) for mid in ids):
                    self.logger.info(f"[LLMClient] 找到模型: {self.model}")
                    return True
                self.logger.warning(
                    f"[LLMClient] 未在模型列表中找到 {self.model}，可用: {ids[:10]}"
                )
                # 部分服务商不列出全部模型，返回 True 让调用者继续尝试
                return True
        except Exception as e:
            self.logger.error(f"[LLMClient] 检查模型失败: {e}")
        return False
