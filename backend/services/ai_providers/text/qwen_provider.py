"""
Qwen SDK implementation for text generation
"""
import logging
import requests
from .base import TextProvider
from config import get_config

logger = logging.getLogger(__name__)


class QwenTextProvider(TextProvider):
    """Text generation using Qwen SDK (DashScope API)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "qwen-plus"):
        """
        Initialize Qwen text provider
        
        Args:
            api_key: API key
            api_base: API base URL
            model: Model name to use
        """
        self.api_key = api_key
        self.api_base = api_base or "https://dashscope.aliyuncs.com/api/v1"
        self.model = model
        self.timeout = get_config().QWEN_TIMEOUT
        self.max_retries = get_config().QWEN_MAX_RETRIES
    
    def generate_text(self, prompt: str, thinking_budget: int = 0) -> str:
        """
        Generate text using Qwen SDK
        
        Args:
            prompt: The input prompt
            thinking_budget: Not used in Qwen format, kept for interface compatibility (0 = default)
            
        Returns:
            Generated text
        """
        try:
            # 构建 API 请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建请求体
            data = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
            
            # 发送请求 - 使用正确的阿里云 DashScope API 路径
            response = requests.post(
                f"{self.api_base}/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # 处理响应
            response_data = response.json()
            return response_data["output"]["text"]
            
        except Exception as e:
            logger.error(f"Error generating text with Qwen: {str(e)}")
            raise