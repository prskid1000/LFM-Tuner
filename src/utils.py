"""
Utility functions for the fine-tuning pipeline
Includes Windows path handling, configuration loading, and validation
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = get_project_root() / "configs" / "default_config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model_config(model_name: str) -> Dict[str, Any]:
    """Load model-specific configuration"""
    config_path = get_project_root() / "configs" / "model_configs.yaml"
    
    if not config_path.exists():
        logging.warning(f"Model config file not found: {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        model_configs = yaml.safe_load(f)
    
    return model_configs.get(model_name, {})


def validate_attention_backend(backend: str) -> bool:
    """
    Validate that the requested attention backend is installed
    
    Args:
        backend: 'flash_attention' or 'sageattention'
    
    Returns:
        True if backend is available, False otherwise
    
    Raises:
        ImportError: If backend is not available
    """
    if backend == "flash_attention":
        try:
            import flash_attn
            logging.info("Flash Attention 2 is available")
            return True
        except ImportError:
            raise ImportError(
                "Flash Attention 2 is not installed. "
                "Install it using: pip install flash-attn"
            )
    
    elif backend == "sageattention":
        try:
            import sageattention
            logging.info("SAGE Attention is available")
            return True
        except ImportError:
            raise ImportError(
                "SAGE Attention is not installed. "
                "Install it using the Windows installation script"
            )
    
    else:
        raise ValueError(
            f"Unknown attention backend: {backend}. "
            "Must be 'flash_attention' or 'sageattention'"
        )


def validate_triton() -> bool:
    """Check if Triton Windows is installed (optional)"""
    try:
        import triton
        logging.info("Triton Windows is available")
        return True
    except ImportError:
        logging.warning("Triton Windows is not installed (optional)")
        return False


def validate_bitsandbytes() -> bool:
    """Check if bitsandbytes is installed (required for 4-bit/8-bit quantization)"""
    try:
        import bitsandbytes
        logging.info("bitsandbytes is available")
        return True
    except ImportError:
        logging.warning("bitsandbytes is not installed (required for 4-bit/8-bit quantization)")
        return False


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Get GPU memory information using pynvml if available"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total": info.total / 1024**3,  # GB
            "used": info.used / 1024**3,     # GB
            "free": info.free / 1024**3,     # GB
        }
    except (ImportError, Exception) as e:
        logging.debug(f"Could not get GPU memory info: {e}")
        return None


def format_path(path: str) -> Path:
    """Format path for cross-platform compatibility (Windows-friendly)"""
    return Path(path).resolve()


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Chain-of-Thought / reasoning token masking
# ---------------------------------------------------------------------------

REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"


def mask_reasoning_labels_for_sequence(
    input_ids: List[int],
    labels: List[int],
    tokenizer,
) -> List[int]:
    """
    Set labels to -100 for all tokens between <reasoning> and </reasoning> (inclusive).
    Keeps <final> and the rest trainable. Works on lists; use for single sequences or
    inside a batch collator.

    Assumes the same tokenizer was used to produce input_ids.
    """
    start_ids = tokenizer(REASONING_START, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(REASONING_END, add_special_tokens=False)["input_ids"]
    labels = list(labels)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(start_ids)] == start_ids:
            for j in range(i, i + len(start_ids)):
                labels[j] = -100
            i += len(start_ids)
            while i < len(input_ids):
                if input_ids[i : i + len(end_ids)] == end_ids:
                    for j in range(i, i + len(end_ids)):
                        labels[j] = -100
                    i += len(end_ids)
                    break
                labels[i] = -100
                i += 1
        else:
            i += 1
    return labels


def mask_reasoning_tokens(dataset, tokenizer):
    """
    Mask <reasoning>...</reasoning> spans by setting labels to -100.
    Keeps <final> tokens trainable.

    Use when the dataset already has 'input_ids' and 'labels' columns
    (e.g. after pre-tokenization). Returns a new dataset with labels modified.

    For SFTTrainer with 'text' column, reasoning masking is applied in the
    data collator instead; this function is for pre-tokenized datasets.
    """
    from datasets import Dataset as HFDataset

    def _mask(example):
        input_ids = example["input_ids"]
        labels = example["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        example["labels"] = mask_reasoning_labels_for_sequence(input_ids, labels, tokenizer)
        return example

    return dataset.map(_mask, desc="Masking <reasoning> tokens")
