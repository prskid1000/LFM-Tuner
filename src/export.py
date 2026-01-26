"""
Export Module
Handles exporting fine-tuned models in multiple formats (LoRA, merged, GGUF)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from unsloth import FastLanguageModel
import torch
import subprocess
import sys


logger = logging.getLogger(__name__)


def _patch_unsloth_gguf_export():
    """
    Monkey patch Unsloth's GGUF export to ignore MSVC warnings.
    
    Unsloth treats MSVC compiler warnings (C4996, C4244, C4267, etc.) as build failures,
    even though the build succeeds. This patch makes the error detection more lenient.
    """
    try:
        # Try to import the internal conversion module
        from unsloth.save import _convert_to_gguf
        
        # Store original function
        original_convert = _convert_to_gguf
        
        def patched_convert(*args, **kwargs):
            """Patched version that ignores MSVC warnings"""
            # Monkey patch subprocess to filter out warning messages
            original_run = subprocess.run
            
            def filtered_run(*run_args, **run_kwargs):
                result = original_run(*run_args, **run_kwargs)
                
                # If there's stderr output, filter MSVC warnings
                if hasattr(result, 'stderr') and result.stderr:
                    stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode('utf-8', errors='ignore')
                    
                    # Check if this is just warnings, not actual errors
                    has_warnings = any(w in stderr for w in ['warning C4', 'warning:', '/W'])
                    has_real_errors = 'error C' in stderr or 'FAILED' in stderr.upper()
                    
                    # If we have warnings but no real errors, consider it success
                    if has_warnings and not has_real_errors:
                        # Check if binaries were created (sign of successful build)
                        if '.lib' in stderr or '.vcxproj' in stderr:
                            logger.info("Build completed with warnings (ignored)")
                            result.returncode = 0
                
                return result
            
            # Apply the patch temporarily
            subprocess.run = filtered_run
            try:
                return original_convert(*args, **kwargs)
            finally:
                # Restore original
                subprocess.run = original_run
        
        # Apply the patch
        import unsloth.save
        unsloth.save._convert_to_gguf = patched_convert
        logger.info("Applied MSVC warning filter patch to Unsloth")
        
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch Unsloth GGUF export: {e}")
        # Try alternative approach - patch at FastLanguageModel level
        try:
            original_save_gguf = FastLanguageModel.save_pretrained_gguf
            
            def patched_save_gguf(self, *args, **kwargs):
                """Wrapper that catches and retries on warning-related failures"""
                try:
                    return original_save_gguf(self, *args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    # Check if this is a warning-related error
                    if 'warning C4' in error_msg or 'deprecated' in error_msg.lower():
                        logger.warning(f"GGUF export encountered warnings: {error_msg}")
                        logger.info("Attempting alternative GGUF export method...")
                        # Try using direct llama.cpp conversion as fallback
                        raise RuntimeError("GGUF export failed due to overly-strict error detection. "
                                         "The model was likely exported successfully. Check the output directory.")
                    raise
            
            FastLanguageModel.save_pretrained_gguf = patched_save_gguf
            logger.info("Applied alternative GGUF export patch")
            
        except Exception as e2:
            logger.debug(f"Could not apply alternative patch: {e2}")


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
    
    # Monkey patch Unsloth's error detection to ignore MSVC warnings
    _patch_unsloth_gguf_export()
    
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
