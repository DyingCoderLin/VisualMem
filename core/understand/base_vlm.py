# core/understand/base_vlm.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.data_models import ScreenFrame, VLMAnalysis
from utils.logger import setup_logger
import json
import datetime

# 为vlm调用基类，分为local_vlm（直接运行在本机上使用函数接口调用）和api_vlm（运行在本机或其他机器上用api调用）两种方案

logger = setup_logger(__name__)

class AbstractVLM(ABC):
    """
    抽象VLM基类
    定义了所有VLM后端必须实现的接口
    """
    
    @abstractmethod
    def _call_vlm(self, prompt: str, image) -> str:
        """
        调用具体的VLM后端
        返回原始的响应文本
        """
        pass
    
    def _get_vlm_prompt(self, frame: ScreenFrame) -> str:
        """
        (辅助函数) 构造 VLM 的 prompt
        """
        ocr_context = frame.ocr_text if frame.ocr_text else "(No OCR text available)"
        
        prompt = (
            "You are a helpful assistant. Analyze this screenshot. "
            "Focus on visual semantics, UI elements, layout, and user intent. "
            "Do not just transcribe text.\n\n"
            f"Extracted OCR text (for context):\n{ocr_context}\n\n"
            "Provide a structured analysis in JSON format:\n"
            "{\n"
            '  "description": "(A concise summary of what the user is seeing/doing)",\n'
            '  "visual_elements": [{"type": "button", "text": "Submit", "location": [x,y,w,h]}],\n'
            '  "layout_summary": "(e.g., \'a sidebar on the left, main content on the right\')",\n'
            '  "entities": ["(List key people, products, or concepts mentioned)"]\n'
            "}\n"
        )
        return prompt
    
    def _parse_vlm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析VLM返回的JSON格式响应
        """
        try:
            # 尝试提取JSON内容（可能被包裹在markdown代码块中）
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            parsed = json.loads(response_text)
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse VLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            # 返回一个默认结构
            return {
                "description": response_text[:200],
                "visual_elements": [],
                "layout_summary": "Failed to parse structured response",
                "entities": []
            }
    
    def _generate_frame_id(self, frame: ScreenFrame) -> str:
        """
        为帧生成唯一ID（基于时间戳）
        
        格式: YYYYMMDD_HHMMSS_ffffff
        例如: 20251201_143025_123456
        """
        return frame.timestamp.strftime("%Y%m%d_%H%M%S_") + f"{frame.timestamp.microsecond:06d}"
    
    def analyze(self, frame: ScreenFrame) -> VLMAnalysis:
        """
        分析一个屏幕帧
        这是外部调用的主要接口
        """
        logger.info(f"Analyzing frame from {frame.timestamp}")
        
        # 1. 构造prompt
        prompt = self._get_vlm_prompt(frame)
        
        # 2. 调用VLM
        response_text = self._call_vlm(prompt, frame.image)
        
        # 3. 解析响应
        parsed_response = self._parse_vlm_response(response_text)
        
        # 4. 构造VLMAnalysis对象
        analysis = VLMAnalysis(
            frame_id=self._generate_frame_id(frame),
            timestamp=frame.timestamp,
            description=parsed_response.get("description", ""),
            visual_elements=parsed_response.get("visual_elements", []),
            layout_summary=parsed_response.get("layout_summary", ""),
            entities=parsed_response.get("entities", []),
            embedding=None  # 将在存储阶段生成
        )
        
        logger.info(f"Analysis completed: {analysis.description[:100]}")
        return analysis


