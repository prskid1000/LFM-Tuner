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
    cleanups = []
    
    try:
        # Patch 1: Patch subprocess.run globally during GGUF export
        original_run = subprocess.run
        
        def patched_subprocess_run(*run_args, **run_kwargs):
            """Patched subprocess.run that filters MSVC warnings"""
            result = original_run(*run_args, **run_kwargs)
            
            # Check for MSVC warning-only failures
            if result.returncode != 0:
                stderr_text = ""
                stdout_text = ""
                
                if hasattr(result, 'stderr') and result.stderr:
                    if isinstance(result.stderr, bytes):
                        stderr_text = result.stderr.decode('utf-8', errors='ignore')
                    elif isinstance(result.stderr, str):
                        stderr_text = result.stderr
                
                if hasattr(result, 'stdout') and result.stdout:
                    if isinstance(result.stdout, bytes):
                        stdout_text = result.stdout.decode('utf-8', errors='ignore')
                    elif isinstance(result.stdout, str):
                        stdout_text = result.stdout
                
                combined_output = stderr_text + "\n" + stdout_text
                
                # Check if this is a warning-only failure
                has_msvc_warnings = any(w in combined_output for w in ['warning C4', 'warning C5', 'warning:', 'Warning:'])
                has_deprecation = 'deprecated' in combined_output.lower() and 'isatty' in combined_output
                has_actual_errors = ('error C' in combined_output or 
                                    'error LNK' in combined_output or 
                                    'fatal error' in combined_output.lower() or
                                    'Error:' in combined_output and 'warning' not in combined_output.lower())
                
                # Check if build actually succeeded (libraries were created)
                build_artifacts = ('.lib' in combined_output or 
                                 '.vcxproj ->' in combined_output or 
                                 'llama.lib' in combined_output or
                                 'llama.vcxproj' in combined_output)
                
                # If we have warnings/deprecation but no actual errors AND build artifacts exist, override
                if (has_msvc_warnings or has_deprecation) and not has_actual_errors and build_artifacts:
                    logger.info("✓ Build completed with MSVC warnings (ignored)")
                    result.returncode = 0
            
            return result
        
        subprocess.run = patched_subprocess_run
        cleanups.append(lambda: setattr(subprocess, 'run', original_run))
        logger.info("✓ Applied subprocess.run patch for MSVC warnings")
        
    except Exception as e:
        logger.warning(f"Could not patch subprocess.run: {e}")
    
    try:
        # Patch 2: Patch unsloth_zoo's command execution
        import unsloth_zoo.saving_utils as saving_utils
        
        if hasattr(saving_utils, 'check_output'):
            original_check = saving_utils.check_output
            
            def patched_check(*args, **kwargs):
                """Patched check_output"""
                try:
                    return original_check(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    # If error contains warnings but build succeeded, suppress
                    if ('warning C4' in error_str or 'deprecated' in error_str) and '.lib' in error_str:
                        logger.info("Suppressed warning-based error")
                        return ""
                    raise
            
            saving_utils.check_output = patched_check
            cleanups.append(lambda: setattr(saving_utils, 'check_output', original_check))
            logger.info("✓ Applied check_output patch")
            
    except Exception as e:
        logger.debug(f"Could not patch check_output: {e}")
    
    # Return cleanup function
    def cleanup_all():
        for cleanup_fn in cleanups:
            try:
                cleanup_fn()
            except:
                pass
    
    return cleanup_all


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
    cleanup = _patch_unsloth_gguf_export()
    
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
    finally:
        # Restore original subprocess.run
        if cleanup:
            cleanup()


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
