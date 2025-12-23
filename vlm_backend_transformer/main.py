from flask import Flask, request, jsonify
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LogitsProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
import io
from PIL import Image
import logging
import argparse
import pickle
from datetime import datetime
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 自定义 LogitsProcessor 用于记录 prefill 结束时间
class TimingLogitsProcessor(LogitsProcessor):
    """记录 prefill 和 decode 阶段的时间"""
    def __init__(self):
        self.prefill_end_time = None
        self.first_token = True
        
    def __call__(self, input_ids, scores):
        # 第一次调用时记录 prefill 结束时间（此时第一个 token 已经生成）
        if self.first_token:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保 GPU 操作完成
            self.prefill_end_time = time.perf_counter()
            self.first_token = False
        return scores

# 增加最大请求大小限制（支持多图片，默认16MB太小）
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Global model variables
model = None
processor = None
memory_evaluate = False  # 全局标志：是否启用内存评估

def load_model():
    """Load the Qwen2.5-VL model and processor"""
    global model, processor
    try:
        logger.info("Loading Qwen2.5-VL-3B-Instruct model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("Model loaded on CUDA")
        else:
            logger.info("Model loaded on CPU")

        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def warmup_model():
    """Warmup the model with a simple text-image pair"""
    global model, processor
    
    if model is None or processor is None:
        logger.warning("Model not loaded, skipping warmup")
        return False
    
    try:
        logger.info("Warming up model...")
        import time
        start_time = time.time()
        
        # 创建一个简单的测试图片 (224x224 红色方块)
        dummy_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
        
        # 简单的文本提示
        dummy_text = "What color is this?"
        
        # 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": dummy_text},
                ],
            }
        ]
        
        # 处理输入
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # 创建时间记录器
        timing_processor = TimingLogitsProcessor()
        
        # 执行一次推理 (使用较少的tokens)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=10,
                logits_processor=[timing_processor]
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        warmup_time = time.time() - start_time
        logger.info(f"Warmup completed in {warmup_time:.2f}s")
        logger.info(f"Warmup output: {output_text[0][:50]}...")  # 显示前50个字符
        
        # 如果使用CUDA，清理warmup产生的缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return False

def process_image_from_base64(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # 确保图片是RGB模式（VLM处理需要）
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Failed to process base64 image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate response from image(s) and text"""
    try:
        if model is None or processor is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({"error": "No JSON data provided"}), 400

        # 支持单张图片（兼容旧格式）或多张图片
        image_base64 = data.get('image')
        images_base64 = data.get('images', [])
        text = data.get('text', '')
        
        logger.info(f"Received request - has 'image': {image_base64 is not None}, has 'images': {len(images_base64)} items")

        # 统一处理为图片列表
        if image_base64 and not images_base64:
            # 旧格式：单张图片
            images_base64 = [image_base64]
        
        if not images_base64:
            logger.error("No images provided in request")
            return jsonify({"error": "No images provided"}), 400

        logger.info(f"Will process {len(images_base64)} images")

        # Process all base64 images
        images = []
        for idx, img_base64 in enumerate(images_base64):
            image = process_image_from_base64(img_base64)
            if image is None:
                logger.error(f"Failed to decode image at index {idx}")
                return jsonify({"error": f"Invalid image format at index {idx}"}), 400
            images.append(image)
            logger.info(f"Successfully decoded image {idx+1}/{len(images_base64)}")
        
        logger.info(f"Processing {len(images)} images")

        # ========== 内存评估开始 ==========
        if memory_evaluate and torch.cuda.is_available():
            # 清理缓存并重置统计
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # 记录初始内存
            mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"Memory before generation: {mem_before:.2f} GB")
            
            # 开始记录内存快照
            torch.cuda.memory._record_memory_history(max_entries=100000)

        # Create messages for the model - 多图片格式
        content = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Prepare inputs for the model
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the same device as model
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # 创建时间记录器
        timing_processor = TimingLogitsProcessor()
        
        # 记录开始时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_start_time = time.perf_counter()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                logits_processor=[timing_processor]  # 添加自定义 processor
            )
        
        # 记录结束时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_end_time = time.perf_counter()
        
        # 计算时间
        total_time = generation_end_time - generation_start_time
        prefill_time = timing_processor.prefill_end_time - generation_start_time if timing_processor.prefill_end_time else 0
        decode_time = generation_end_time - timing_processor.prefill_end_time if timing_processor.prefill_end_time else 0
        
        # 记录时间统计
        logger.info("="*60)
        logger.info("Generation Timing Breakdown:")
        logger.info(f"  • Total time: {total_time*1000:.2f} ms")
        logger.info(f"  • Prefill time: {prefill_time*1000:.2f} ms ({prefill_time/total_time*100:.1f}%)")
        logger.info(f"  • Decode time: {decode_time*1000:.2f} ms ({decode_time/total_time*100:.1f}%)")
        logger.info("="*60)

        # Trim the generated ids to remove input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0] if output_text else ""

        # ========== 内存评估结束 ==========
        if memory_evaluate and torch.cuda.is_available():
            torch.cuda.synchronize()
            
            # 获取内存统计
            mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            logger.info("="*60)
            logger.info("CUDA Memory Statistics:")
            logger.info(f"  • Images processed: {len(images)}")
            logger.info(f"  • Memory before: {mem_before:.2f} GB")
            logger.info(f"  • Memory after: {mem_after:.2f} GB")
            logger.info(f"  • Memory used: {mem_after - mem_before:.2f} GB")
            logger.info(f"  • Peak memory: {mem_peak:.2f} GB")
            logger.info(f"  • Reserved memory: {mem_reserved:.2f} GB")
            logger.info("="*60)
            
            # 保存内存快照
            try:
                snapshot = torch.cuda.memory._snapshot()
                
                # 创建快照目录
                snapshot_dir = Path("memory_snapshots")
                snapshot_dir.mkdir(exist_ok=True)
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_file = snapshot_dir / f"memory_snapshot_{len(images)}imgs_{timestamp}.pickle"
                
                # 保存快照
                with open(snapshot_file, 'wb') as f:
                    pickle.dump(snapshot, f)
                
                logger.info(f"Memory snapshot saved to: {snapshot_file}")
                
                # 停止记录
                torch.cuda.memory._record_memory_history(enabled=None)
                
            except Exception as e:
                logger.error(f"Failed to save memory snapshot: {e}")

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    logger.error("Request entity too large (413)")
    return jsonify({"error": "Request too large"}), 413

@app.errorhandler(400)
def bad_request(error):
    logger.error(f"Bad request (400): {error}")
    return jsonify({"error": "Bad request"}), 400

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VLM Backend Server')
    parser.add_argument('--memory_evaluate', action='store_true', 
                        help='Enable CUDA memory evaluation and snapshot')
    parser.add_argument('--port', type=int, default=8081,
                        help='Port to run the server on (default: 8081)')
    args = parser.parse_args()
    
    # 设置全局内存评估标志
    memory_evaluate = args.memory_evaluate
    
    if memory_evaluate:
        logger.info("="*60)
        logger.info("CUDA Memory Evaluation Mode ENABLED")
        logger.info("  • Memory snapshots will be saved to ./memory_snapshots/")
        logger.info("  • Peak memory usage will be logged for each request")
        logger.info("="*60)
    
    # Load model on startup
    if load_model():
        # Warmup model
        if not memory_evaluate:
            logger.info("="*60)
            warmup_model()
            logger.info("="*60)
        
        logger.info(f"Starting server on port {args.port}...")
        logger.info(f"Max content length: {app.config.get('MAX_CONTENT_LENGTH', 'default')} bytes")
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
