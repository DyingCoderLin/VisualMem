# core/understand/api_vlm.py
import base64
import os
from io import BytesIO
import requests
import json
import time
from datetime import datetime, timezone
from typing import Optional
from PIL.Image import Image
from .base_vlm import AbstractVLM
from config import config
from utils.logger import setup_logger, setup_generate_logger

# 普通日志（输出到终端）
logger = setup_logger(__name__)

# VLM生成信息日志（只写入文件）
generate_logger = setup_generate_logger("logs/generate_info.log")


class ApiVLM(AbstractVLM):
    """
    通过API调用VLM
    支持两种后端类型:
    1. vllm: 使用 OpenAI 格式接口 (/v1/chat/completions)
    2. transformer: 使用 generate 接口 (/generate)
    
    通过环境变量 VLM_BACKEND_TYPE 选择:
    - "vllm": 使用 OpenAI 格式 (默认)
    - "transformer": 使用 generate 格式
    """
    
    # 两套 API 端点路径（硬编码）
    VLLM_ENDPOINT = "/v1/chat/completions"
    TRANSFORMER_ENDPOINT = "/generate"
    
    def __init__(self):
        self.api_key = config.VLM_API_KEY
        self.model = config.VLM_API_MODEL
        # 基础 URL（只包含 host:port）
        self.base_url = config.VLM_API_URI.rstrip('/')
        # 根据 VLM_BACKEND_TYPE 选择接口格式
        self.backend_type = config.VLM_BACKEND_TYPE.lower()
        
        # 根据 backend_type 选择正确的端点
        if self.backend_type == "vllm":
            self.api_uri = f"{self.base_url}{self.VLLM_ENDPOINT}"
        else:  # transformer
            self.api_uri = f"{self.base_url}{self.TRANSFORMER_ENDPOINT}"
        
        logger.debug(f"ApiVLM initialized")
        logger.debug(f"  • Base URL: {self.base_url}")
        logger.debug(f"  • Endpoint: {self.api_uri}")
        logger.debug(f"  • Model: {self.model}")
        logger.debug(f"  • Backend: {self.backend_type}")
    
    def _image_to_base64(self, image: Image) -> str:
        """
        将PIL Image转换为base64编码的字符串（JPEG格式，质量80%）
        """
        # 确保图片是RGB模式（JPEG不支持透明通道）
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=config.IMAGE_QUALITY, optimize=True)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    
    def _build_openai_payload(
        self, 
        prompt: str, 
        images_base64: list, 
        timestamps: list = None,
        system_prompt: str = None
    ) -> dict:
        """
        构建支持时间戳穿插的 OpenAI payload
        
        结构: Text(Time1) -> Image1 -> Text(Time2) -> Image2 -> Final Prompt
        
        Args:
            prompt: 用户查询文本
            images_base64: base64 编码的图片列表
            timestamps: 时间戳列表（datetime 对象），长度应与 images_base64 相同
            system_prompt: 系统提示词（可选）
        """
        content = []
        
        # 遍历图片和时间戳
        for idx, img_base64 in enumerate(images_base64):
            # 1. 如果有时间戳，先插入时间戳文本
            if timestamps and idx < len(timestamps):
                ts = timestamps[idx]
                
                # 确保 ts 是 datetime 对象
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except:
                        pass
                
                if isinstance(ts, datetime):
                    # 将存储的 UTC 时间转换为本地时间
                    if ts.tzinfo is None:
                        # 假设无时区信息的 datetime 是 UTC
                        ts = ts.replace(tzinfo=timezone.utc)
                    local_ts = ts.astimezone()  # 自动转换为本地时区
                    
                    # 格式化时间戳为可读格式
                    ts_str = local_ts.strftime("%Y-%m-%d %H:%M:%S")
                    ts_text = f"Image {idx+1} Timestamp (Local Time): {ts_str}"
                    content.append({
                        "type": "text",
                        "text": ts_text
                    })
            
            # 2. 插入图片
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_base64
                }
            })
        
        # 3. 最后插入用户的总指令
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建 messages 列表
        messages = []
        
        # 如果有 system prompt，添加到 messages 开头
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 添加 user message
        messages.append({
            "role": "user",
            "content": content
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        return payload
    
    def _build_transformer_payload(self, prompt: str, images_base64: list) -> dict:
        """
        构建 transformer/generate 格式的 payload
        
        格式:
        {
            "images": ["data:image/jpeg;base64,...", ...],
            "text": "prompt"
        }
        """
        return {
            "images": images_base64,
            "text": prompt
        }
    
    def _parse_openai_response(self, response_json: dict) -> str:
        """解析OpenAI格式的响应"""
        try:
            # OpenAI格式: {"choices": [{"message": {"content": "..."}}]}
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            # 如果格式不对，尝试其他字段
            return str(response_json)
    
    def _parse_transformer_response(self, response_json) -> str:
        """解析 transformer/generate 格式的响应"""
        if isinstance(response_json, dict):
            # 尝试常见的响应字段
            return (
                response_json.get("response") or 
                response_json.get("text") or 
                response_json.get("output") or
                response_json.get("content") or
                str(response_json)
            )
        return str(response_json)
    
    def _call_vlm(
        self, 
        prompt: str, 
        images: list, 
        num_images: int = None, 
        image_timestamps: list[datetime] = None,
        system_prompt: str = None
    ) -> str:
        """
        调用VLM API（支持多图片和时间戳）
        
        Args:
            prompt: 用户查询文本
            images: PIL Image列表
            num_images: 参与分析的图像总数（用于日志，如果None则使用len(images)）
            image_timestamps: 图片时间戳列表（datetime对象），长度应与images相同
            system_prompt: 系统提示词（可选）
        """
        try:
            # 如果传入的是单个Image而不是列表，兼容旧接口
            if isinstance(images, Image):
                images = [images]
            
            if num_images is None:
                num_images = len(images)
            
            # 为 Prompt 补充当前本地时间，方便 VLM 理解相对时间（如“刚才”、“昨天”）
            local_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"Current Local Time: {local_now}\n\n{prompt}"
            
            # 记录开始时间
            start_time = time.time()
            
            # 将所有图像转换为base64（JPEG格式）
            images_base64 = []
            for idx, image in enumerate(images):
                img_base64 = self._image_to_base64(image)
                images_base64.append(f"data:image/jpeg;base64,{img_base64}")
                logger.debug(f"Image {idx+1}: base64 length = {len(img_base64)}")
            
            logger.info(f"Sending {len(images_base64)} images to VLM (backend: {self.backend_type})")
            
            # 构造API请求头
            headers = {
                "Content-Type": "application/json"
            }
            
            # 如果有API key，添加到header
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 根据后端类型构造payload
            if self.backend_type == "vllm":
                payload = self._build_openai_payload(
                    prompt, 
                    images_base64, 
                    timestamps=image_timestamps,
                    system_prompt=system_prompt
                )
            else:  # transformer
                payload = self._build_transformer_payload(prompt, images_base64)
            
            # 计算payload大小
            payload_size = len(json.dumps(payload))
            logger.debug(f"Payload size: {payload_size / 1024 / 1024:.2f} MB")
            
            # 发送请求
            logger.debug(f"Sending request to {self.api_uri}")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            logger.debug(f"Images count: {len(images_base64)}")
            
            # print(f"prompt preview: {prompt[:100]}...")
            
            response = requests.post(
                self.api_uri,
                headers=headers,
                json=payload,
                timeout=360,  # 多图片可能需要更长时间
                verify=False  # 如果是自签名证书
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 记录VLM调用日志（响应码200时）
            # print(f"Response: {response.status_code}")
            if response.status_code == 200:
                # 写入generate_info.log文件
                generate_logger.info(
                    f"VLM_CALL | "
                    f"Images: {num_images} | "
                    f"Response_Time: {response_time:.2f}s | "
                    f"Status: 200"
                )
                # 同时在终端显示（简化版）
                logger.info(f"VLM call successful | time: {response_time:.2f}s")
            
            response.raise_for_status()
            
            # 解析响应
            try:
                response_json = response.json()
                if self.backend_type == "vllm":
                    content = self._parse_openai_response(response_json)
                else:  # transformer
                    content = self._parse_transformer_response(response_json)
            except:
                # 如果不是JSON，直接返回文本
                content = response.text
            
            logger.debug(f"Received response: {content[:200]}...")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"API调用失败: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"错误: {str(e)}"
    
    def _call_vlm_text_only(self, prompt: str, system_prompt: str = None) -> str:
        """
        纯文本查询（不带图片）
        
        Args:
            prompt: 提示文本
            system_prompt: 系统提示词（可选）
        """
        try:
            start_time = time.time()
            
            logger.info(f"纯文本查询 (后端: {self.backend_type})")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 纯文本模式只支持 OpenAI 格式
            if self.backend_type == "vllm":
                messages = []
                
                # 如果有 system prompt，添加到 messages 开头
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                # 添加 user message
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            else:
                # transformer 格式的纯文本
                payload = {
                    "text": prompt,
                    "images": []
                }
            
            logger.debug(f"Sending text-only request to {self.api_uri}")
            
            response = requests.post(
                self.api_uri,
                headers=headers,
                json=payload,
                timeout=120,
                verify=False
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                generate_logger.info(
                    f"VLM_TEXT_CALL | "
                    f"Response_Time: {response_time:.2f}s | "
                    f"Status: 200"
                )
                logger.info(f"纯文本VLM调用成功 | 耗时{response_time:.2f}s")
            
            response.raise_for_status()
            
            try:
                response_json = response.json()
                if self.backend_type == "vllm":
                    content = self._parse_openai_response(response_json)
                else:
                    content = self._parse_transformer_response(response_json)
            except:
                content = response.text
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Text-only API request failed: {e}")
            return f"API调用失败: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in text-only call: {e}")
            return f"错误: {str(e)}"

    def chat_text(self, messages: list, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        """
        纯文本对话（OpenAI chat/completions 兼容），messages: [{"role": "...", "content": "..."}]
        
        Args:
            messages: 消息列表
            max_tokens: 最大 token 数，None 时不设置（让模型自己决定）
            temperature: 温度参数，默认 0.7
        """
        if self.backend_type != "vllm":
            raise ValueError("chat_text 仅在 VLLM(OpenAI) 后端可用")
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        # 只有当明确指定 max_tokens 时才添加
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        resp = requests.post(self.api_uri, headers=headers, json=payload, timeout=120, verify=False)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

