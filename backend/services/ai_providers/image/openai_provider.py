"""
OpenAI SDK implementation for image generation

Supports multiple resolution parameter formats for different OpenAI-compatible providers:
- Flat style: extra_body.aspect_ratio + extra_body.resolution
- Nested style: extra_body.generationConfig.imageConfig.aspectRatio + imageSize

Note: Not all providers support 2K/4K resolution in OpenAI format.
Some may only return 1K regardless of settings.
Resolution validation is handled at the task_manager level for all providers.
"""
import logging
import base64
import re
import requests
from io import BytesIO
from typing import Optional, List
from openai import OpenAI
from PIL import Image
from .base import ImageProvider
from config import get_config

logger = logging.getLogger(__name__)


class OpenAIImageProvider(ImageProvider):
    """
    Image generation using OpenAI SDK (compatible with Gemini via proxy)
    
    Supports multiple resolution parameter formats for different providers.
    Resolution support varies by provider:
    - Some providers support 2K/4K via extra_body parameters
    - Some providers only support 1K regardless of settings
    
    The provider will try multiple parameter formats to maximize compatibility.
    """
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-3-pro-image-preview"):
        """
        Initialize OpenAI image provider
        
        Args:
            api_key: API key
            api_base: API base URL (e.g., https://aihubmix.com/v1)
            model: Model name to use
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=get_config().OPENAI_TIMEOUT,  # set timeout from config
            max_retries=get_config().OPENAI_MAX_RETRIES  # set max retries from config
        )
        self.api_base = api_base or ""
        self.model = model
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        # Convert to RGB if necessary (e.g., RGBA images)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _map_resolution_to_gemini(self, resolution: str, aspect_ratio: str) -> str:
        """
        Map standard resolution to Gemini-specific size format
        
        Args:
            resolution: Standard resolution ("1K", "2K", "4K")
            aspect_ratio: Image aspect ratio
            
        Returns:
            Gemini-specific size string
        """
        # Map resolution based on aspect ratio
        if aspect_ratio == "16:9":
            resolution_map = {
                "1K": "1024x576",
                "2K": "1920x1080",  # Full HD
                "4K": "3840x2160"   # 4K
            }
        elif aspect_ratio == "9:16":
            resolution_map = {
                "1K": "576x1024",
                "2K": "1080x1920",
                "4K": "2160x3840"
            }
        elif aspect_ratio == "1:1":
            resolution_map = {
                "1K": "1024x1024",
                "2K": "1536x1536",
                "4K": "2048x2048"
            }
        else:
            # Default to 16:9 for other aspect ratios
            resolution_map = {
                "1K": "1024x576",
                "2K": "1920x1080",
                "4K": "3840x2160"
            }
        
        return resolution_map.get(resolution, "1024x576")  # Default to 1K

    def _build_extra_body(self, aspect_ratio: str, resolution: str) -> dict:
        """
        Build extra_body parameters for resolution control.
        
        Uses multiple format strategies to support different providers:
        1. Flat style: aspect_ratio + resolution at top level
        2. Nested style: generationConfig.imageConfig structure
        3. Gemini-specific style: image object with size
        
        Args:
            aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16")
            resolution: Image resolution ("1K", "2K", "4K")
            
        Returns:
            Dict with extra_body parameters
        """
        # Ensure resolution is uppercase (some providers require "4K" not "4k")
        resolution_upper = resolution.upper()
        
        # Build comprehensive extra_body that works with multiple providers
        extra_body = {
            # Flat style parameters
            "aspect_ratio": aspect_ratio,
            "resolution": resolution_upper,
            
            # Nested style structure (compatible with some providers)
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution_upper,
                }
            },
            
            # Gemini-specific format (for resolution control)
            "image": {
                "aspect_ratio": aspect_ratio,
                "quality": "hd",  # High quality
                "size": self._map_resolution_to_gemini(resolution_upper, aspect_ratio)
            }
        }
        
        return extra_body

    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        enable_thinking: bool = False,
        thinking_budget: int = 0
    ) -> Optional[Image.Image]:
        """
        Generate image using OpenAI SDK
        
        Supports resolution control via extra_body parameters for compatible providers.
        Note: Not all providers support 2K/4K resolution - some may return 1K regardless.
        Note: enable_thinking and thinking_budget are ignored (OpenAI format doesn't support thinking mode)
        
        The provider will:
        1. Try to use extra_body parameters (API易/AvalAI style) for resolution control
        2. Use system message for aspect_ratio as fallback
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio
            resolution: Image resolution ("1K", "2K", "4K") - support depends on provider
            enable_thinking: Ignored, kept for interface compatibility
            thinking_budget: Ignored, kept for interface compatibility
            
        Returns:
            Generated PIL Image object, or None if failed
        """
        try:
            # Build message content
            content = []
            
            # Add reference images first (if any)
            if ref_images:
                for ref_img in ref_images:
                    base64_image = self._encode_image_to_base64(ref_img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            # Add text prompt
            content.append({"type": "text", "text": prompt})
            
            logger.debug(f"Calling OpenAI API for image generation with {len(ref_images) if ref_images else 0} reference images...")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio}, resolution: {resolution}")
            
            # Build extra_body with resolution parameters for compatible providers
            extra_body = self._build_extra_body(aspect_ratio, resolution)
            # Add OpenRouter required modalities parameter
            extra_body["modalities"] = ["image", "text"]
            logger.debug(f"Using extra_body for resolution control: {extra_body}")
            
            # Use chat completions API for image generation (compatible with most providers)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"aspect_ratio={aspect_ratio}, resolution={resolution}"},
                    {"role": "user", "content": content},
                ],
                extra_body=extra_body
            )
            
            logger.info("OpenAI API call completed")
            
            # Log full response for debugging
            logger.info(f"Full response: {response}")
            logger.info(f"Response model: {self.model}")
            logger.info(f"Request prompt: {prompt[:100]}...")
            logger.info(f"Request aspect_ratio: {aspect_ratio}")
            logger.info(f"Request resolution: {resolution}")
            
            # Extract image from response - handle chat completions format
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                
                # Debug: log available attributes
                logger.info(f"Response message attributes: {dir(message)}")
                logger.info(f"Message dict: {message.model_dump() if hasattr(message, 'model_dump') else message}")
                logger.info(f"Message content: '{message.content}'")
                logger.info(f"Message content length: {len(message.content) if message.content else 0}")
                
                # Try multi_mod_content first (custom format from some proxies)
                if hasattr(message, 'multi_mod_content') and message.multi_mod_content:
                    parts = message.multi_mod_content
                    for part in parts:
                        if "text" in part:
                            logger.debug(f"Response text: {part['text'][:100] if len(part['text']) > 100 else part['text']}")
                        if "inline_data" in part:
                            image_data = base64.b64decode(part["inline_data"]["data"])
                            image = Image.open(BytesIO(image_data))
                            logger.debug(f"Successfully extracted image: {image.size}, {image.mode}")
                            return image
                
                # Try standard OpenAI content format (list of content parts)
                if hasattr(message, 'content') and message.content:
                    # If content is a list (multimodal response)
                    if isinstance(message.content, list):
                        for part in message.content:
                            if isinstance(part, dict):
                                # Handle image_url type
                                if part.get('type') == 'image_url':
                                    image_url = part.get('image_url', {}).get('url', '')
                                    if image_url.startswith('data:image'):
                                        # Extract base64 data from data URL
                                        base64_data = image_url.split(',', 1)[1]
                                        image_data = base64.b64decode(base64_data)
                                        image = Image.open(BytesIO(image_data))
                                        logger.debug(f"Successfully extracted image from content: {image.size}, {image.mode}")
                                        return image
                            # Handle text type
                            elif part.get('type') == 'text':
                                text = part.get('text', '')
                                if text:
                                    logger.debug(f"Response text: {text[:100] if len(text) > 100 else text}")
                            elif hasattr(part, 'type'):
                                # Handle as object with attributes
                                if part.type == 'image_url':
                                    image_url = getattr(part, 'image_url', {})
                                    if isinstance(image_url, dict):
                                        url = image_url.get('url', '')
                                    else:
                                        url = getattr(image_url, 'url', '')
                                    if url.startswith('data:image'):
                                        base64_data = url.split(',', 1)[1]
                                        image_data = base64.b64decode(base64_data)
                                        image = Image.open(BytesIO(image_data))
                                        logger.debug(f"Successfully extracted image from content object: {image.size}, {image.mode}")
                                        return image
                
                # Try OpenRouter format - check message.images
                if hasattr(message, 'images') and message.images:
                    logger.debug(f"Found images in message: {message.images}")
                    for image in message.images:
                        if isinstance(image, dict) and 'image_url' in image:
                            image_url = image['image_url'].get('url', '')
                            if image_url.startswith('data:image'):
                                # Extract base64 data from data URL
                                base64_data = image_url.split(',', 1)[1]
                                image_data = base64.b64decode(base64_data)
                                image = Image.open(BytesIO(image_data))
                                logger.debug(f"Successfully extracted image from message.images: {image.size}, {image.mode}")
                                return image
                
                # Try checking for images attribute as list directly
                if hasattr(message, 'images'):
                    images = getattr(message, 'images')
                    logger.debug(f"Checking images attribute: {images}")
                    if isinstance(images, list) and images:
                        for img in images:
                            # Check different image formats
                            if isinstance(img, dict):
                                if 'image_url' in img:
                                    image_url = img['image_url'].get('url', '')
                                    if image_url.startswith('data:image'):
                                        base64_data = image_url.split(',', 1)[1]
                                        image_data = base64.b64decode(base64_data)
                                        image = Image.open(BytesIO(image_data))
                                        logger.debug(f"Successfully extracted image from images list: {image.size}, {image.mode}")
                                        return image
                            elif hasattr(img, 'image_url'):
                                image_url_obj = getattr(img, 'image_url')
                                if hasattr(image_url_obj, 'url'):
                                    image_url = getattr(image_url_obj, 'url')
                                    if image_url.startswith('data:image'):
                                        base64_data = image_url.split(',', 1)[1]
                                        image_data = base64.b64decode(base64_data)
                                        image = Image.open(BytesIO(image_data))
                                        logger.debug(f"Successfully extracted image from images object: {image.size}, {image.mode}")
                                        return image
                
                # If content is a string, try to extract image from it
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content_str = message.content
                    logger.debug(f"Response content (string): {content_str[:200] if len(content_str) > 200 else content_str}")
                    
                    # Try to extract Markdown image URL: ![...](url)
                    markdown_pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)'
                    markdown_matches = re.findall(markdown_pattern, content_str)
                    if markdown_matches:
                        image_url = markdown_matches[0]  # Use the first image URL found
                        logger.debug(f"Found Markdown image URL: {image_url}")
                        try:
                            response = requests.get(image_url, timeout=30, stream=True)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            image.load()  # Ensure image is fully loaded
                            logger.debug(f"Successfully downloaded image from Markdown URL: {image.size}, {image.mode}")
                            return image
                        except Exception as download_error:
                            logger.warning(f"Failed to download image from Markdown URL: {download_error}")
                    
                    # Try to extract plain URL (not in Markdown format)
                    url_pattern = r'(https?://[^\s\)\]]+\.(?:png|jpg|jpeg|gif|webp|bmp)(?:\?[^\s\)\]]*)?)'
                    url_matches = re.findall(url_pattern, content_str, re.IGNORECASE)
                    if url_matches:
                        image_url = url_matches[0]
                        logger.debug(f"Found plain image URL: {image_url}")
                        try:
                            response = requests.get(image_url, timeout=30, stream=True)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            image.load()
                            logger.debug(f"Successfully downloaded image from plain URL: {image.size}, {image.mode}")
                            return image
                        except Exception as download_error:
                            logger.warning(f"Failed to download image from plain URL: {download_error}")
                    
                    # Try to extract base64 data URL from string
                    base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                    base64_matches = re.findall(base64_pattern, content_str)
                    if base64_matches:
                        base64_data = base64_matches[0]
                        logger.debug(f"Found base64 image data in string")
                        try:
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(BytesIO(image_data))
                            logger.debug(f"Successfully extracted base64 image from string: {image.size}, {image.mode}")
                            return image
                        except Exception as decode_error:
                            logger.warning(f"Failed to decode base64 image from string: {decode_error}")
                
                # Try to check annotations for image data
                if hasattr(message, 'annotations') and message.annotations:
                    logger.debug(f"Checking annotations for image data: {message.annotations}")
                    for annotation in message.annotations:
                        if hasattr(annotation, 'type') and annotation.type == 'image':
                            logger.debug(f"Found image annotation: {annotation}")
                            # Check if annotation has image data
                            if hasattr(annotation, 'image_data'):
                                image_data = base64.b64decode(annotation.image_data)
                                image = Image.open(BytesIO(image_data))
                                logger.debug(f"Successfully extracted image from annotations: {image.size}, {image.mode}")
                                return image
                            elif hasattr(annotation, 'url') and annotation.url:
                                image_url = annotation.url
                                if image_url.startswith('data:image'):
                                    base64_data = image_url.split(',', 1)[1]
                                    image_data = base64.b64decode(base64_data)
                                    image = Image.open(BytesIO(image_data))
                                    logger.debug(f"Successfully extracted image from annotation URL: {image.size}, {image.mode}")
                                    return image
                
                # Try to check for custom fields that might contain image data
                if hasattr(message, 'model_dump'):
                    message_dict = message.model_dump()
                    logger.debug(f"Message dict keys: {list(message_dict.keys())}")
                    # Check for any fields that might contain image data
                    for key, value in message_dict.items():
                        if key not in ['content', 'role', 'reasoning', 'refusal']:
                            logger.debug(f"Checking field {key}: {value}")
                            # Check if this field contains image data
                            if isinstance(value, str) and value.startswith('data:image'):
                                base64_data = value.split(',', 1)[1]
                                image_data = base64.b64decode(base64_data)
                                image = Image.open(BytesIO(image_data))
                                logger.debug(f"Successfully extracted image from field {key}: {image.size}, {image.mode}")
                                return image
                
                # Try to check the raw response structure for image data
                logger.debug(f"Checking raw response structure...")
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    logger.debug(f"Response dict keys: {list(response_dict.keys())}")
                    # Check if there's any image data in the response
                    if 'choices' in response_dict:
                        for choice in response_dict['choices']:
                            if 'message' in choice:
                                message_data = choice['message']
                                logger.debug(f"Message data keys: {list(message_data.keys())}")
                                # Check for any image-related fields
                                for msg_key, msg_value in message_data.items():
                                    if msg_key not in ['content', 'role', 'reasoning', 'refusal']:
                                        logger.debug(f"Checking message field {msg_key}: {msg_value}")
                                        if isinstance(msg_value, str) and msg_value.startswith('data:image'):
                                            base64_data = msg_value.split(',', 1)[1]
                                            image_data = base64.b64decode(base64_data)
                                            image = Image.open(BytesIO(image_data))
                                            logger.debug(f"Successfully extracted image from raw response: {image.size}, {image.mode}")
                                            return image
                
                # Try a different approach - use chat completions with proper image generation parameters
                logger.info("Trying chat completions with image generation parameters...")
                try:
                    # Use chat completions with specific parameters for image generation
                    # Some providers require different parameter structure
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are an image generation assistant. Generate a high-quality image according to the user's request. Aspect ratio: {aspect_ratio}, Resolution: {resolution}."
                            },
                            {
                                "role": "user",
                                "content": prompt  # Use just the prompt text for simplicity
                            }
                        ],
                        # Different parameter structure for image generation
                        extra_body={
                            "task_type": "image_generation",
                            "image_generation": {
                                "aspect_ratio": aspect_ratio,
                                "resolution": resolution
                            }
                        }
                    )
                    
                    logger.info("Alternative chat completions API call completed")
                    logger.info(f"Full alternative response: {response}")
                    
                    # Check response structure
                    if hasattr(response, 'choices') and response.choices:
                        choice = response.choices[0]
                        if hasattr(choice, 'message'):
                            msg = choice.message
                            logger.info(f"Alternative message: {msg}")
                            # Check for image data in the message
                            if hasattr(msg, 'content') and msg.content:
                                # Process content as before
                                if isinstance(msg.content, list):
                                    for part in msg.content:
                                        if isinstance(part, dict) and part.get('type') == 'image_url':
                                            image_url = part.get('image_url', {}).get('url', '')
                                            if image_url.startswith('data:image'):
                                                base64_data = image_url.split(',', 1)[1]
                                                image_data = base64.b64decode(base64_data)
                                                image = Image.open(BytesIO(image_data))
                                                logger.debug(f"Successfully extracted image from alternative API: {image.size}, {image.mode}")
                                                return image
                except Exception as alt_error:
                    logger.warning(f"Alternative API approach failed: {alt_error}")
                
                # Try one more approach - direct image generation endpoint
                logger.info("Trying direct image generation approach...")
                try:
                    # Some providers expose a direct image generation endpoint
                    # This is a fallback for providers that don't support image generation through chat completions
                    import json
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.client.api_key}"
                    }
                    
                    # Build request body for direct image generation
                    payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "n": 1
                    }
                    
                    # Determine the correct endpoint URL
                    api_base = self.api_base or "https://api.openai.com/v1"
                    endpoint = f"{api_base}/images/generations"
                    
                    logger.info(f"Calling direct image generation endpoint: {endpoint}")
                    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
                    
                    # Make direct HTTP request
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    logger.info(f"Direct API response status: {response.status_code}")
                    logger.info(f"Direct API response: {response.text}")
                    
                    # Parse response
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and data['data']:
                            image_data = data['data'][0]
                            if 'b64_json' in image_data:
                                # Direct base64 image data
                                image_data = base64.b64decode(image_data['b64_json'])
                                image = Image.open(BytesIO(image_data))
                                logger.debug(f"Successfully extracted image from direct API: {image.size}, {image.mode}")
                                return image
                            elif 'url' in image_data:
                                # Image URL
                                image_url = image_data['url']
                                try:
                                    img_response = requests.get(image_url, timeout=30, stream=True)
                                    img_response.raise_for_status()
                                    image = Image.open(BytesIO(img_response.content))
                                    image.load()
                                    logger.debug(f"Successfully downloaded image from direct API URL: {image.size}, {image.mode}")
                                    return image
                                except Exception as download_error:
                                    logger.warning(f"Failed to download image from direct API URL: {download_error}")
                except Exception as direct_error:
                    logger.warning(f"Direct image generation approach failed: {direct_error}")
            
            # Log raw response for debugging
            logger.warning(f"Unable to extract image. Raw response: {str(response)[:500]}{'...(truncated)' if len(str(response)) > 500 else ''}")
            
            raise ValueError("No valid image response received from OpenAI API")
            
        except Exception as e:
            error_detail = f"Error generating image with OpenAI (model={self.model}): {type(e).__name__}: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise Exception(error_detail) from e
