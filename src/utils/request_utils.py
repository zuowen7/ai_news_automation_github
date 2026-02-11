"""
请求工具模块
提供指数退避重试、User-Agent轮换等请求优化功能
"""

import time
import random
from typing import Optional, Callable, Any
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .logger import get_logger


# User-Agent池
USER_AGENTS = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    # Firefox
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0',
    # Safari
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    # Edge
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
    # Opera
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 OPR/107.0.0.0',
    # 移动端
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36',
]


class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1
    ):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            initial_backoff: 初始退避时间（秒）
            max_backoff: 最大退避时间（秒）
            exponential_base: 指数退避基数
            jitter: 是否添加随机抖动
            jitter_factor: 抖动因子（0-1之间）
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = jitter_factor


class RequestOptimizer:
    """请求优化器"""

    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        初始化请求优化器

        Args:
            retry_config: 重试配置
        """
        self.logger = get_logger("request_optimizer")
        self.retry_config = retry_config or RetryConfig()
        self.session = None

    def calculate_backoff(self, attempt: int) -> float:
        """
        计算指数退避时间

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            退避时间（秒）
        """
        # 指数退避: initial_backoff * (base ^ attempt)
        backoff = self.retry_config.initial_backoff * (
            self.retry_config.exponential_base ** attempt
        )

        # 限制最大退避时间
        backoff = min(backoff, self.retry_config.max_backoff)

        # 添加随机抖动
        if self.retry_config.jitter:
            jitter_range = backoff * self.retry_config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            backoff += jitter

        return max(0, backoff)

    def get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(USER_AGENTS)

    def get_random_headers(self, base_headers: Optional[dict] = None) -> dict:
        """
        生成随机请求头

        Args:
            base_headers: 基础请求头

        Returns:
            包含随机User-Agent的请求头
        """
        headers = base_headers.copy() if base_headers else {}
        headers['User-Agent'] = self.get_random_user_agent()

        # 添加其他常见请求头
        if 'Accept' not in headers:
            headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        if 'Accept-Language' not in headers:
            headers['Accept-Language'] = 'zh-CN,zh;q=0.9,en;q=0.8'
        if 'Accept-Encoding' not in headers:
            headers['Accept-Encoding'] = 'gzip, deflate'
        if 'DNT' not in headers:
            headers['DNT'] = '1'
        if 'Connection' not in headers:
            headers['Connection'] = 'keep-alive'
        if 'Upgrade-Insecure-Requests' not in headers:
            headers['Upgrade-Insecure-Requests'] = '1'

        return headers

    def create_session(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 100,
        max_retries: int = 3
    ) -> requests.Session:
        """
        创建优化的Session

        Args:
            pool_connections: 连接池大小
            pool_maxsize: 连接池最大数量
            max_retries: 自动重试次数

        Returns:
            配置好的Session对象
        """
        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 禁用SSL验证（仅用于开发环境）
        session.verify = False

        self.session = session
        return session

    def make_request_with_retry(
        self,
        url: str,
        method: str = 'GET',
        session: Optional[requests.Session] = None,
        headers: Optional[dict] = None,
        timeout: int = 10,
        **kwargs
    ) -> Optional[requests.Response]:
        """
        带指数退避重试的请求

        Args:
            url: 请求URL
            method: 请求方法
            session: Session对象
            headers: 请求头
            timeout: 超时时间
            **kwargs: 其他请求参数

        Returns:
            响应对象，失败返回None
        """
        if session is None:
            if self.session is None:
                self.session = self.create_session()
            session = self.session

        # 使用随机User-Agent
        if headers:
            headers = self.get_random_headers(headers)
        else:
            headers = self.get_random_headers()

        last_exception = None

        for attempt in range(self.retry_config.max_retries):
            try:
                response = session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                last_exception = e

                # 最后一次尝试不再重试
                if attempt == self.retry_config.max_retries - 1:
                    break

                # 计算退避时间
                backoff_time = self.calculate_backoff(attempt)

                self.logger.warning(
                    f"请求失败 ({url}), 第{attempt + 1}次重试, "
                    f"等待 {backoff_time:.1f} 秒后重试... 错误: {e}"
                )

                time.sleep(backoff_time)

        # 所有重试都失败
        self.logger.error(f"请求失败，已重试{self.retry_config.max_retries}次: {url}")
        return None

    def close(self):
        """关闭Session"""
        if self.session:
            self.session.close()
            self.session = None


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    指数退避重试装饰器

    Args:
        max_retries: 最大重试次数
        initial_backoff: 初始退避时间
        max_backoff: 最大退避时间
        exponential_base: 指数基数
        exceptions: 需要重试的异常类型

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(func.__name__)

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} 失败，已重试{max_retries}次: {e}")
                        raise

                    # 计算退避时间
                    backoff = initial_backoff * (exponential_base ** attempt)
                    backoff = min(backoff, max_backoff)

                    # 添加抖动
                    jitter = backoff * 0.1 * (2 * random.random() - 1)
                    backoff += jitter

                    logger.warning(
                        f"{func.__name__} 失败 ({e}), "
                        f"等待 {backoff:.1f} 秒后重试..."
                    )

                    time.sleep(backoff)

        return wrapper
    return decorator


# 动态并发调整器
class DynamicConcurrencyManager:
    """动态并发管理器"""

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        initial_workers: int = 3
    ):
        """
        初始化动态并发管理器

        Args:
            min_workers: 最小工作线程数
            max_workers: 最大工作线程数
            initial_workers: 初始工作线程数
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = initial_workers
        self.success_count = 0
        self.failure_count = 0
        self.logger = get_logger("concurrency_manager")

    def adjust_workers(self, success: bool):
        """
        根据请求成功率动态调整工作线程数

        Args:
            success: 最后一次请求是否成功

        Returns:
            调整后的工作线程数
        """
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # 每10次请求评估一次
        total = self.success_count + self.failure_count
        if total >= 10:
            success_rate = self.success_count / total

            # 成功率高，增加并发
            if success_rate > 0.9 and self.current_workers < self.max_workers:
                old_workers = self.current_workers
                self.current_workers = min(self.current_workers + 1, self.max_workers)
                if self.current_workers != old_workers:
                    self.logger.info(
                        f"成功率 {success_rate:.1%}, 增加并发到 {self.current_workers}"
                    )

            # 成功率低，减少并发
            elif success_rate < 0.7 and self.current_workers > self.min_workers:
                old_workers = self.current_workers
                self.current_workers = max(self.current_workers - 1, self.min_workers)
                if self.current_workers != old_workers:
                    self.logger.info(
                        f"成功率 {success_rate:.1%}, 减少并发到 {self.current_workers}"
                    )

            # 重置计数
            self.success_count = 0
            self.failure_count = 0

        return self.current_workers

    def get_current_workers(self) -> int:
        """获取当前工作线程数"""
        return self.current_workers

    def reset(self):
        """重置统计"""
        self.success_count = 0
        self.failure_count = 0
