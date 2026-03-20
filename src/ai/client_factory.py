"""
AI 客户端工厂
根据配置自动选择 OllamaClient（本地原生）或 LLMClient（OpenAI 兼容远程）
"""

from ..config.settings import AIConfig
from ..utils.logger import get_logger

_logger = get_logger("ai.client_factory")


def create_ai_client(config: AIConfig):
    """
    根据配置创建合适的 AI 客户端。

    判断逻辑：
      1. 若 config.provider 非空，或 config.custom_base_url 非空，或 config.api_key 非空
         → 使用 LLMClient（OpenAI 兼容接口）
      2. 否则 → 使用原 OllamaClient（本地 Ollama 原生接口，向后兼容）

    Returns:
        OllamaClient 或 LLMClient 实例（两者接口一致）
    """
    use_remote = bool(
        (config.provider and config.provider.strip()) or
        (getattr(config, 'custom_base_url', '') and config.custom_base_url.strip()) or
        (getattr(config, 'api_key', '') and config.api_key.strip())
    )

    if use_remote:
        from .llm_client import LLMClient
        _logger.info(
            f"[ClientFactory] 使用远程 LLMClient  "
            f"provider={config.provider or '(custom)'}  model={config.model_name}"
        )
        return LLMClient(config)
    else:
        from .ollama_client import OllamaClient
        _logger.info(
            f"[ClientFactory] 使用本地 OllamaClient  "
            f"url={config.ollama_url}  model={config.model_name}"
        )
        return OllamaClient(config)
