"""
Qwen SDK implementation for image generation
"""
import logging
import requests
from io import BytesIO
from typing import Optional, List
from PIL import Image
from .base import ImageProvider
from config import get_config

logger = logging.getLogger(__name__)


class QwenImageProvider(ImageProvider):
    """Image generation using Qwen SDK (DashScope API)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "qwen-image-max"):
        self.api_key = api_key
        self.api_base = api_base or "https://dashscope.aliyuncs.com/api/v1"
        self.model = model
        self.timeout = get_config().QWEN_TIMEOUT
    
    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        enable_thinking: bool = False,
        thinking_budget: int = 0
    ) -> Optional[Image.Image]:

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 转换分辨率 - 使用阿里云DashScope API支持的格式
            resolution_map = {
                "1K": "1024*1024",
                "2K": "1664*928",
                "4K": "2048*2048"
            }
            size = resolution_map.get(resolution, "1024*1024")

            # 构建请求体 - 参考项目实现，不处理参考图片以避免API限制
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
                    "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，AI感强，构图混乱，文字模糊",
                    "prompt_extend": True,
                    "watermark": False,
                    "size": size
                }
            }

            response = requests.post(
                f"{self.api_base}/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=data,
                timeout=self.timeout
            )

            # 详细的错误处理
            if response.status_code != 200:
                logger.error(f"Qwen API error: {response.status_code} - {response.text}")
                response.raise_for_status()

            response_data = response.json()

            # 处理响应格式
            if "output" not in response_data or "choices" not in response_data["output"]:
                logger.error(f"Invalid Qwen API response: {response_data}")
                return None

            image_url = response_data["output"]["choices"][0]["message"]["content"][0]["image"]

            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()

            return Image.open(BytesIO(image_response.content))

        except Exception as e:
            logger.error(f"Error generating image with Qwen: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None