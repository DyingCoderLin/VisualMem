# core/retrieval/local_reranker.py
"""
Local Reranker: Directly loads model and uses function calls (non-API mode)

Based on paper method: Uses VLM to judge whether images contain query content
Score = exp(logit_Yes) / (exp(logit_Yes) + exp(logit_No))
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL.Image import Image
import threading
from tqdm import tqdm

from config import config
from utils.logger import setup_logger

logger = setup_logger("local_reranker")

# Global singleton: model is loaded only once
_rerank_model = None
_rerank_processor = None
_rerank_lock = threading.Lock()


def _load_rerank_model():
    """
    Load rerank model (singleton pattern, loaded only once)
    """
    global _rerank_model, _rerank_processor
    
    if _rerank_model is not None:
        return _rerank_model, _rerank_processor
    
    with _rerank_lock:
        # Double-check
        if _rerank_model is not None:
            return _rerank_model, _rerank_processor
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            
            model_name = config.RERANK_MODEL or "Qwen/Qwen3-VL-2B-Instruct"
            logger.info(f"Loading rerank model: {model_name} (this may take a while if downloading for the first time)")
            
            try:
                import flash_attn  # noqa: F401
                logger.debug("✓ flash_attn found, using flash_attention_2 for acceleration")
                _rerank_model = Qwen3VLForConditionalGeneration.from_pretrained(
                            "Qwen/Qwen3-VL-2B-Instruct",
                            dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            device_map="auto",
                    )
            except ImportError:
                logger.warning("flash_attn not found, using default attention implementation")
                _rerank_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            try:
                _rerank_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            except ImportError as e:
                logger.warning(f"Fast processor not available ({e}), falling back to slow processor")
                _rerank_processor = AutoProcessor.from_pretrained(model_name)
            
            return _rerank_model, _rerank_processor
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Please install: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load rerank model: {e}")
            raise


class LocalReranker:
    """
    Local Reranker: Directly uses model for function calls
    """
    
    def __init__(self):
        """Initialize LocalReranker (lazy load model)"""
        self.model = None
        self.processor = None
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded"""
        if self.model is None or self.processor is None:
            self.model, self.processor = _load_rerank_model()
    
    def _get_yes_no_token_ids(self) -> Tuple[int, int]:
        """
        Get "yes" and "no" token IDs (strictly following Qwen3-Reranker method)
        
        Returns:
            (yes_token_id, no_token_id)
        """
        try:
            tok = self.processor.tokenizer
            # Directly use convert_tokens_to_ids to avoid prefix space issues from encode
            yes_token_id = tok.convert_tokens_to_ids("yes")
            no_token_id = tok.convert_tokens_to_ids("no")
            
            # Simple robustness check
            if yes_token_id is None or no_token_id is None:
                raise ValueError(f"convert_tokens_to_ids returned None: yes={yes_token_id}, no={no_token_id}")
            
            return yes_token_id, no_token_id
            
        except Exception as e:
            logger.error(f"Failed to get yes/no token IDs: {e}")
            raise
    
    def _call_rerank_model(
        self,
        query: str,
        image: Image,
        return_logits: bool = True
    ) -> Tuple[float, Optional[str]]:
        """
        Call local rerank model to get yes/no logits
        
        Args:
            query: Query text
            image: Image
            return_logits: Whether to return logits (for score calculation)
        
        Returns:
            (score, response_text)
            score: Score between 0.0-1.0, >0.5 means Yes
            response_text: Model-generated text response (for debugging)
        """
        try:
            self._ensure_model_loaded()
            
            # Build system prompt (state task type, reference Qwen3-Reranker)
            system_prompt = (
                "Judge whether the image (Document) meets the requirements based on the Query. "
                "Note that the answer can only be \"yes\" or \"no\"."
            )
            
            # User part: pass Query and image, briefly explain Query's role
            user_text = f"<Query>: {query}\nPlease answer whether this image satisfies the query."
            
            # Build messages: system states task, user provides query + image
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            
            # Process input (using new apply_chat_template API)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to model's device
            inputs = inputs.to(self.model.device)
            
            # Get Yes/No token IDs
            yes_token_id, no_token_id = self._get_yes_no_token_ids()
            
            # Generate and get logits
            with torch.no_grad():
                if return_logits:
                    # Use generate and get logits
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,  # Only need Yes/No, don't need too many tokens
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=False,  # Greedy decoding (deterministic output)
                    )
                    
                    # Get logits of first generated token
                    if outputs.scores and len(outputs.scores) > 0:
                        first_token_logits = outputs.scores[0][0]  # [vocab_size]
                        
                        # Extract Yes and No logits
                        logit_yes = first_token_logits[yes_token_id].item()
                        logit_no = first_token_logits[no_token_id].item()
                        
                        # Get actual first generated token ID (for verification)
                        generated_ids = outputs.sequences
                        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
                        first_generated_token_id = generated_ids[0][len(input_ids[0])].item()
                        
                        # Get logit of first generated token (for debugging)
                        first_generated_logit = first_token_logits[first_generated_token_id].item()
                        
                        # Calculate score: s = sy/(sy + sn)
                        sy = np.exp(logit_yes)
                        sn = np.exp(logit_no)
                        score = sy / (sy + sn)
                        
                        # Get response text (for debugging)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                        ]
                        response_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        
                        # Verification: if generated token is not yes or no, log warning
                        if first_generated_token_id not in [yes_token_id, no_token_id]:
                            logger.warning(
                                f"First generated token ({first_generated_token_id}) is not yes ({yes_token_id}) or no ({no_token_id}). "
                                f"Generated token logit: {first_generated_logit:.4f}, "
                                f"yes logit: {logit_yes:.4f}, no logit: {logit_no:.4f}"
                            )
                        
                        logger.debug(
                            f"Rerank score from logits: {score:.4f} "
                            f"(yes logit: {logit_yes:.4f}, no logit: {logit_no:.4f}, "
                            f"generated token: {first_generated_token_id}, generated logit: {first_generated_logit:.4f})"
                        )
                    else:
                        # If no scores, judge from generated text
                        generated_ids = outputs.sequences
                        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                        ]
                        response_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        score = self._parse_text_response(response_text)
                        logger.debug(f"Rerank score from text: {score:.4f} (unable to get logits)")
                else:
                    # Don't use logits, judge directly from text
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,  # Greedy decoding (deterministic output)
                    )
                    
                    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                    ]
                    response_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    score = self._parse_text_response(response_text)
                    logger.debug(f"Rerank score from text: {score:.4f}")
            
            return score, response_text
            
        except Exception as e:
            logger.error(f"Rerank model call failed: {e}", exc_info=True)
            # Return 0 score on failure (not passed)
            return 0.0, None
    
    def _parse_text_response(self, text: str) -> float:
        """
        Parse yes/no from text response (lowercase, reference Qwen3-Reranker)
        
        Args:
            text: Model-generated text
        
        Returns:
            Score (0.0 or 1.0)
        """
        text_lower = text.strip().lower()
        
        # More precise judgment: check if starts with yes/no
        if text_lower.startswith("yes"):
            return 1.0
        elif text_lower.startswith("no"):
            return 0.0
        elif "yes" in text_lower and "no" not in text_lower:
            return 1.0
        elif "no" in text_lower and "yes" not in text_lower:
            return 0.0
        else:
            logger.warning(f"Unable to extract yes/no from response: {text}")
            return 0.0  # Default not passed
    
    def rerank(
        self,
        query: str,
        frames: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Re-rank search results
        
        Args:
            query: Query text
            frames: Search result list (each frame must contain 'image' field)
            top_k: Return top-k results (default 10)
        
        Returns:
            Sorted top-k frames list (sorted by score in descending order)
        """
        if not frames:
            return []
        
        total = len(frames)
        logger.info(f"Starting rerank of {total} results")
        
        scored_frames = []
        
        # Process each frame sequentially, with progress bar
        for i, frame in enumerate(
            tqdm(frames, total=total, desc="Reranking", unit="img")
        ):
            image = frame.get("image")
            if image is None:
                logger.warning(f"Frame {i+1} has no image, skipping rerank")
                continue
            
            try:
                score, response_text = self._call_rerank_model(query, image, return_logits=True)

                logger.debug(f"rerank score: {score}, response_text: {response_text}")
                
                # Record score for all frames
                frame_with_score = frame.copy()
                frame_with_score["rerank_score"] = score
                if response_text:
                    frame_with_score["rerank_response"] = response_text
                scored_frames.append(frame_with_score)
                logger.debug(f"Frame {i+1}: score={score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Frame {i+1} rerank failed: {e}")
                continue
        
        # Sort by score in descending order
        scored_frames.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        # Take top-k
        top_k_frames = scored_frames[:top_k]
        
        logger.debug(f"Rerank complete: returning top-{len(top_k_frames)}/{len(frames)} results")
        
        return top_k_frames
