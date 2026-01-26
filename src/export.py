"""
Export Module
Handles exporting fine-tuned models in multiple formats (LoRA, merged, GGUF)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from unsloth import FastLanguageModel
import torch


logger = logging.getLogger(__name__)


def save_lora(
    model,
    tokenizer,
    output_dir: Path,
    config: Dict[str, Any]
) -> None:
    """
    Save LoRA adapter (if enabled in config)
    
    Args:
        model: Trained model with LoRA
        tokenizer: Tokenizer
        output_dir: Output directory
        config: Configuration dictionary
    """
    if not config.get('export', {}).get('export_lora', False):
        logger.info("LoRA export disabled in config, skipping")
        return
    
    lora_dir = Path(output_dir) / "lora_adapter"
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving LoRA adapter to: {lora_dir}")
    
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    
    logger.info("LoRA adapter saved")


def merge_and_save(
    model,
    tokenizer,
    output_dir: Path,
    config: Dict[str, Any]
) -> None:
    """
    Merge LoRA weights to base model and save as 16-bit (if enabled in config)
    
    Args:
        model: Trained model with LoRA
        tokenizer: Tokenizer
        output_dir: Output directory
        config: Configuration dictionary
    """
    if not config.get('export', {}).get('export_merged', False):
        logger.info("Merged model export disabled in config, skipping")
        return
    
    merged_dir = Path(output_dir) / "merged_16bit"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Merging LoRA weights and saving to: {merged_dir}")
    
    # Merge LoRA weights
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(
        str(merged_dir),
        tokenizer=tokenizer,
        safe_serialization=True
    )
    
    logger.info("Merged 16-bit model saved")


def export_to_gguf(
    model,
    tokenizer,
    output_dir: Path,
    config: Dict[str, Any],
    model_name: str = "model"
) -> None:
    """
    Export model to GGUF format (if enabled in config)
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
        config: Configuration dictionary
        model_name: Name for the exported model
    """
    if not config.get('export', {}).get('export_gguf', False):
        logger.info("GGUF export disabled in config, skipping")
        return
    
    try:
        from unsloth import is_bfloat16_supported
    except ImportError:
        logger.warning("GGUF export requires unsloth with GGUF support")
        return
    
    gguf_dir = Path(output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting to GGUF format: {gguf_dir}")
    
    # Merge if needed
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    
    # Export to GGUF
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m"  # 4-bit quantization for GGUF
        )
        logger.info("GGUF model exported successfully")
    except Exception as e:
        logger.error(f"Failed to export GGUF: {e}")
        logger.info("You may need to use llama.cpp directly for GGUF export")


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    num_samples: int = 10
) -> Dict[str, Any]:
    """
    Basic model evaluation
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        num_samples: Number of samples to evaluate
    
    Returns:
        Evaluation metrics dictionary
    """
    logger.info(f"Evaluating model on {num_samples} samples")
    
    model.eval()
    results = {
        "num_samples": num_samples,
        "predictions": []
    }
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset[:num_samples]):
            if isinstance(example, dict) and 'text' in example:
                # Simple generation test
                inputs = tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results["predictions"].append({
                    "input": example['text'][:100],
                    "output": generated[:200]
                })
    
    logger.info("Evaluation completed")
    return results


def export_all(
    model,
    tokenizer,
    output_dir: Path,
    config: Dict[str, Any],
    model_name: str = "model"
) -> None:
    """
    Export model in all enabled formats
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
        config: Configuration dictionary
        model_name: Name for exported models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting model export...")
    
    # Save LoRA adapter
    save_lora(model, tokenizer, output_dir, config)
    
    # Merge and save 16-bit
    merge_and_save(model, tokenizer, output_dir, config)
    
    # Export to GGUF
    export_to_gguf(model, tokenizer, output_dir, config, model_name)
    
    logger.info("All exports completed")
