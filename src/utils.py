"""
Utility functions for the fine-tuning pipeline
Includes Windows path handling, configuration loading, and validation
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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
