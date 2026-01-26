# utils/image_utils.py
from PIL import Image
from config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def resize_image_if_needed(image: Image.Image, max_width: int = None) -> Image.Image:
    """
    按比例压缩图片宽度到指定的最大宽度
    
    Args:
        image: PIL 图片对象
        max_width: 最大宽度。如果为 None 则使用 config.MAX_IMAGE_WIDTH。
                  如果为 0 或 None，则不压缩。
                  
    Returns:
        压缩后（或原始）的图片对象
    """
    target_width = max_width if max_width is not None else config.MAX_IMAGE_WIDTH
    
    if target_width <= 0 or image.width <= target_width:
        return image
    
    # 计算新的尺寸（保持宽高比）
    ratio = target_width / image.width
    new_height = int(image.height * ratio)
    
    # 使用高质量的缩放算法
    resized = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image for storage: {image.size} -> {resized.size}")
    return resized
