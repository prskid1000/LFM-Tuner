"""
Export Module
Handles exporting fine-tuned models in multiple formats (LoRA, merged, GGUF)

Note: GGUF export uses llama.cpp directly (not Unsloth) for Windows compatibility
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import sys
import shutil

logger = logging.getLogger(__name__)

# llama.cpp location (relative to project root)
LLAMA_CPP_DIR = Path(__file__).parent.parent / "notebooks" / "llama.cpp"

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
        safe_serialization=True
    )
    
    # Save tokenizer separately (IMPORTANT!)
    tokenizer.save_pretrained(str(merged_dir))
    
    logger.info("Merged 16-bit model saved")


def check_llama_cpp() -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Check if llama.cpp is installed and built
    
    Returns:
        Tuple of (is_ready, quantize_exe, convert_script)
    """
    if not LLAMA_CPP_DIR.exists():
        return False, None, None
    
    # Check for quantizer (Windows or Unix)
    quantize_exe = None
    for possible_path in [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",  # Windows
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",  # Unix
        LLAMA_CPP_DIR / "llama-quantize",  # Unix (direct build)
    ]:
        if possible_path.exists():
            quantize_exe = possible_path
            break
    
    # Check for converter script
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    
    is_ready = quantize_exe is not None and convert_script.exists()
    return is_ready, quantize_exe, convert_script


def build_llama_cpp() -> bool:
    """
    Clone and build llama.cpp
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info("Setting up llama.cpp for GGUF conversion")
    logger.info("=" * 70)
    
    # Clone if needed
    if not LLAMA_CPP_DIR.exists():
        logger.info(f"Cloning llama.cpp to {LLAMA_CPP_DIR}...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)],
                check=True,
                capture_output=True
            )
            logger.info("OK: llama.cpp cloned")
        except Exception as e:
            logger.error(f"Failed to clone llama.cpp: {e}")
            return False
    
    # Build
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building llama.cpp (this may take a few minutes)...")
    
    try:
        # Configure with CMake
        subprocess.run(
            ["cmake", "..", "-G", "Visual Studio 17 2022", "-A", "x64"],
            cwd=build_dir,
            check=True,
            capture_output=True
        )
        
        # Build
        result = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j", "8"],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        
        # Check if build succeeded (ignore MSVC warnings)
        output = result.stdout + result.stderr
        has_real_error = any(e in output for e in ['error C', 'error LNK', 'fatal error'])
        build_succeeded = '.lib' in output or 'llama.lib' in output
        
        if result.returncode != 0 and has_real_error and not build_succeeded:
            logger.error(f"Build failed: {result.stderr[:500]}")
            return False
        
        logger.info("OK: llama.cpp built successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build llama.cpp: {e}")
        return False


def convert_to_gguf_direct(
    model_dir: Path,
    output_dir: Path,
    quantize_exe: Path,
    convert_script: Path,
    quantization_methods: list = None
) -> bool:
    """
    Convert model to GGUF using llama.cpp directly
    
    Args:
        model_dir: Directory containing the merged 16-bit model
        output_dir: Output directory for GGUF files
        quantize_exe: Path to llama-quantize executable
        convert_script: Path to convert_hf_to_gguf.py
        quantization_methods: List of quantization methods (default: ["q4_k_m"])
    
    Returns:
        True if successful, False otherwise
    """
    if quantization_methods is None:
        quantization_methods = ["q4_k_m"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert to FP16 GGUF
    logger.info("[1/2] Converting to FP16 GGUF...")
    fp16_gguf = output_dir / "model-fp16.gguf"
    
    try:
        subprocess.run(
            [sys.executable, str(convert_script), str(model_dir),
             "--outfile", str(fp16_gguf), "--outtype", "f16"],
            check=True,
            capture_output=True
        )
        
        if fp16_gguf.exists():
            size_gb = fp16_gguf.stat().st_size / (1024**3)
            logger.info(f"OK: Created {fp16_gguf.name} ({size_gb:.2f} GB)")
        else:
            logger.error("FP16 GGUF file not created")
            return False
            
    except Exception as e:
        logger.error(f"Failed to convert to FP16 GGUF: {e}")
        return False
    
    # Step 2: Quantize
    logger.info(f"[2/2] Quantizing to {quantization_methods}...")
    
    for method in quantization_methods:
        quant_gguf = output_dir / f"model-{method}.gguf"
        logger.info(f"  Creating {method.upper()} quantization...")
        
        try:
            subprocess.run(
                [str(quantize_exe), str(fp16_gguf), str(quant_gguf), method],
                check=True,
                capture_output=True
            )
            
            if quant_gguf.exists():
                size_gb = quant_gguf.stat().st_size / (1024**3)
                logger.info(f"  OK: Created {quant_gguf.name} ({size_gb:.2f} GB)")
            else:
                logger.warning(f"  Quantized file not created: {method}")
                
        except Exception as e:
            logger.error(f"  Failed to quantize {method}: {e}")
    
    logger.info("GGUF conversion completed")
    return True


def export_to_gguf(
    model,
    tokenizer,
    output_dir: Path,
    config: Dict[str, Any],
    model_name: str = "model"
) -> None:
    """
    Export model to GGUF format using llama.cpp directly
    
    This bypasses Unsloth's GGUF export (which has issues on Windows) and
    uses llama.cpp tools directly for reliable conversion.
    
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
    
    logger.info("=" * 70)
    logger.info("GGUF Export (using llama.cpp directly)")
    logger.info("=" * 70)
    
    # Check if llama.cpp is ready
    is_ready, quantize_exe, convert_script = check_llama_cpp()
    
    if not is_ready:
        logger.info("llama.cpp not found or not built, setting up...")
        if not build_llama_cpp():
            logger.error("Failed to setup llama.cpp, skipping GGUF export")
            logger.info("You can manually build it by running: notebooks/build_llama_cpp.py")
            return
        
        # Re-check
        is_ready, quantize_exe, convert_script = check_llama_cpp()
        if not is_ready:
            logger.error("llama.cpp build completed but tools not found")
            return
    
    logger.info("OK: llama.cpp is ready")
    logger.info(f"  Quantizer: {quantize_exe}")
    logger.info(f"  Converter: {convert_script}")
    
    # Ensure merged 16-bit model exists
    merged_dir = Path(output_dir) / "merged_16bit"
    if not merged_dir.exists():
        logger.error(f"Merged model not found at {merged_dir}")
        logger.error("Run merge_and_save() first before GGUF export")
        return
    
    # Convert to GGUF
    gguf_dir = Path(output_dir) / "gguf"
    quantization_methods = config.get('export', {}).get('gguf_quantization_methods', ["q4_k_m"])
    
    convert_to_gguf_direct(
        model_dir=merged_dir,
        output_dir=gguf_dir,
        quantize_exe=quantize_exe,
        convert_script=convert_script,
        quantization_methods=quantization_methods
    )
    
    logger.info("=" * 70)


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
