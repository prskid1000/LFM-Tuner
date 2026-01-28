"""
Training Module
Handles model loading, LoRA setup, and training with Unsloth
Supports Flash Attention 2 and SAGE Attention
"""

import logging
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.utils import validate_attention_backend, validate_bitsandbytes, get_gpu_memory_info

_original_map = Dataset.map
def _patched_map(self, *args, **kwargs):
    kwargs['num_proc'] = None
    return _original_map(self, *args, **kwargs)
Dataset.map = _patched_map

logger = logging.getLogger(__name__)


def load_tokenizer_only(
    model_name: str,
    config: Dict[str, Any]
):
    """
    Load only the tokenizer (lightweight, for dataset formatting)
    
    This is useful when you only need the tokenizer for formatting datasets
    and don't need to load the full model yet.
    
    Args:
        model_name: Model name or path
        config: Configuration dictionary
    
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer
    
    logger.info(f"Loading tokenizer from: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.get('trust_remote_code', False),
            token=config.get('hf_token', None),
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def load_model(
    model_name: str,
    config: Dict[str, Any],
    max_seq_length: int = 2048
) -> Tuple[Any, Any]:
    """
    Load model with Unsloth using specified attention backend and quantization
    
    Args:
        model_name: Model name or path
        config: Configuration dictionary
        max_seq_length: Maximum sequence length
    
    Returns:
        (model, tokenizer)
    """
    attention_backend = config.get('attention_backend', 'sageattention')
    load_in_4bit = config.get('quantization', {}).get('load_in_4bit', False)
    load_in_8bit = config.get('quantization', {}).get('load_in_8bit', False)
    load_in_16bit = config.get('quantization', {}).get('load_in_16bit', True)
    full_finetuning = config.get('full_finetuning', False)
    
    # Validate attention backend
    validate_attention_backend(attention_backend)
    
    # Validate bitsandbytes if using quantization
    if load_in_4bit or load_in_8bit:
        if not validate_bitsandbytes():
            raise ImportError(
                "bitsandbytes is required for 4-bit/8-bit quantization. "
                "Install it or use 16-bit quantization instead."
            )
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Attention backend: {attention_backend}")
    logger.info(f"Quantization: 4bit={load_in_4bit}, 8bit={load_in_8bit}, 16bit={load_in_16bit}")
    logger.info(f"Max sequence length: {max_seq_length}")
    
    # Setup attention backend
    attention_kwargs = setup_attention(attention_backend)
    
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            load_in_16bit=load_in_16bit,
            full_finetuning=full_finetuning,
            trust_remote_code=config.get('trust_remote_code', False),
            token=config.get('hf_token', None),
            **attention_kwargs
        )
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def setup_attention(attention_backend: str) -> Dict[str, Any]:
    """
    Setup attention backend configuration
    
    Args:
        attention_backend: 'flash_attention' or 'sageattention'
    
    Returns:
        Dictionary of attention kwargs for model loading
    """
    # Unsloth handles attention internally, we just validate here
    # The actual attention backend is set via environment or Unsloth's internal mechanisms
    if attention_backend == "flash_attention":
        try:
            import flash_attn
            logger.info("Using Flash Attention 2")
            # Unsloth will use Flash Attention if available
            return {}
        except ImportError:
            raise ImportError("Flash Attention 2 is not installed")
    
    elif attention_backend == "sageattention":
        try:
            import sageattention
            logger.info("Using SAGE Attention")
            # Unsloth will use SAGE Attention if available
            return {}
        except ImportError:
            raise ImportError("SAGE Attention is not installed")
    
    else:
        raise ValueError(
            f"Unknown attention backend: {attention_backend}. "
            "Must be 'flash_attention' or 'sageattention'"
        )


def setup_lora(
    model,
    config: Dict[str, Any],
    max_seq_length: int = 2048
) -> Any:
    """
    Setup LoRA adapter with 12GB VRAM optimizations
    
    Args:
        model: Unsloth model
        config: Configuration dictionary
        max_seq_length: Maximum sequence length
    
    Returns:
        Model with LoRA adapter
    """
    lora_config = config.get('lora', {})
    
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 16)
    lora_dropout = lora_config.get('lora_dropout', 0)
    bias = lora_config.get('bias', 'none')
    target_modules = lora_config.get('target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing = lora_config.get('use_gradient_checkpointing', 'unsloth')
    use_rslora = lora_config.get('use_rslora', False)
    
    logger.info(f"Setting up LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=use_rslora,
        loftq_config=None,
    )
    
    logger.info("LoRA adapter configured")
    return model


def optimize_for_12gb(
    config: Dict[str, Any],
    model_size_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize training configuration for 12GB VRAM
    
    Args:
        config: Training configuration
        model_size_gb: Estimated model size in GB
    
    Returns:
        Optimized configuration
    """
    optimized = config.copy()
    training_config = optimized.get('training', {})
    
    # Default 12GB VRAM settings
    if 'per_device_train_batch_size' not in training_config:
        training_config['per_device_train_batch_size'] = 1
    
    if 'gradient_accumulation_steps' not in training_config:
        training_config['gradient_accumulation_steps'] = 4
    
    if 'max_seq_length' not in training_config:
        training_config['max_seq_length'] = 2048
    
    # Adjust based on model size if provided
    if model_size_gb:
        if model_size_gb > 7:
            training_config['per_device_train_batch_size'] = 1
            training_config['gradient_accumulation_steps'] = 8
            training_config['max_seq_length'] = 1024
        elif model_size_gb > 3:
            training_config['per_device_train_batch_size'] = 1
            training_config['gradient_accumulation_steps'] = 4
            training_config['max_seq_length'] = 2048
    
    optimized['training'] = training_config
    
    logger.info("Optimized for 12GB VRAM:")
    logger.info(f"  Batch size: {training_config['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    logger.info(f"  Max sequence length: {training_config['max_seq_length']}")
    
    return optimized


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: Dict[str, Any],
    output_dir: Path,
    resume_from_checkpoint: Optional[str] = None
) -> SFTTrainer:
    """
    Train model using SFTTrainer
    
    Args:
        model: Unsloth model with LoRA
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Training configuration
        output_dir: Output directory for checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
    
    Returns:
        Trainer object
    """
    training_config = config.get('training', {})
    
    # Optimize for 12GB VRAM
    config = optimize_for_12gb(config)
    training_config = config.get('training', {})
    
    # Check GPU memory
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        logger.info(f"GPU Memory: {gpu_info['used']:.2f}GB / {gpu_info['total']:.2f}GB used")
    
    # Setup training arguments
    training_args = SFTConfig(
        max_seq_length=training_config.get('max_seq_length', 2048),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        warmup_steps=training_config.get('warmup_steps', 10),
        max_steps=training_config.get('max_steps', 100),
        logging_steps=training_config.get('logging_steps', 1),
        output_dir=str(output_dir),
        optim=training_config.get('optim', 'adamw_8bit'),
        seed=training_config.get('seed', 3407),
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        report_to=training_config.get('report_to', None),
        dataloader_num_workers=0,
        dataset_num_proc=1,
    )
    
    # Add validation dataset if provided
    eval_dataset = val_dataset if val_dataset else None
    
    logger.info("Starting training...")
    logger.info(f"Training steps: {training_args.max_steps}")
    logger.info(f"Output directory: {output_dir}")
    
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Train (with optional checkpoint resume)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info("Training completed")
    return trainer


def save_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    checkpoint_name: str = "checkpoint"
) -> None:
    """Save training checkpoint"""
    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to: {checkpoint_dir}")
    
    # Save LoRA adapter
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    
    logger.info("Checkpoint saved")
