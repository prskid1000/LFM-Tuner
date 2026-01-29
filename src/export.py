"""
Export Module
Handles exporting fine-tuned models in multiple formats (LoRA, merged, GGUF)

Cross-platform: Windows, Linux, macOS. Paths and subprocess calls are OS-agnostic.

- LoRA / merged: Use Unsloth's save methods when available (no in-memory merge).
- GGUF: Uses llama.cpp directly (not Unsloth). Unsloth's GGUF export often fails
  on Windows; we run convert_hf_to_gguf + llama-quantize ourselves on all OSes.
"""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "llama.cpp"


def _subprocess_run_utf8(*args, **kwargs) -> subprocess.CompletedProcess:
    """
    Windows-friendly subprocess runner.

    When capture_output/text=True, Python decodes using the active code page
    (often cp1252) which can crash on non-ASCII bytes from tools/progress bars.
    Force UTF-8 decoding and replace undecodable bytes.
    """
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    env = kwargs.get("env")
    if env is None:
        env = os.environ.copy()
    else:
        env = dict(env)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    kwargs["env"] = env
    return subprocess.run(*args, **kwargs)

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
    Merge LoRA weights to base model and save as 16-bit (if enabled in config).
    Uses Unsloth's save_pretrained_merged when available (no in-memory merge);
    otherwise merge_and_unload + save_pretrained.
    """
    if not config.get('export', {}).get('export_merged', False):
        logger.info("Merged model export disabled in config, skipping")
        return
    
    merged_dir = Path(output_dir) / "merged_16bit"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Merging LoRA weights and saving to: {merged_dir}")
    
    use_unsloth = getattr(model, "save_pretrained_merged", None)
    if callable(use_unsloth):
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit"
        )
    else:
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
    
    logger.info("Merged 16-bit model saved")


def _quantize_candidates() -> list:
    """Paths to check for llama-quantize (build/bin, build/bin/Release, legacy)."""
    base = LLAMA_CPP_DIR / "build" / "bin"
    ext = ".exe" if IS_WINDOWS else ""
    return [
        base / "Release" / f"llama-quantize{ext}",  # VS multi-config
        base / f"llama-quantize{ext}",              # Ninja / default
        LLAMA_CPP_DIR / f"llama-quantize{ext}",     # legacy
    ]


def check_llama_cpp() -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Check if llama.cpp is installed and built.
    Looks for llama-quantize in build/bin (and build/bin/Release on Windows).
    """
    if not LLAMA_CPP_DIR.exists():
        return False, None, None
    
    quantize_exe = None
    for p in _quantize_candidates():
        if p.exists():
            quantize_exe = p
            break
    
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    is_ready = quantize_exe is not None and convert_script.exists()
    return is_ready, quantize_exe, convert_script


def build_llama_cpp() -> bool:
    """
    Clone and build llama.cpp. Uses default CMake on all OSes; on Windows,
    falls back to Visual Studio generator if the default build fails.
    """
    logger.info("=" * 70)
    logger.info("Setting up llama.cpp for GGUF conversion")
    logger.info("=" * 70)
    
    if not LLAMA_CPP_DIR.exists():
        logger.info(f"Cloning llama.cpp to {LLAMA_CPP_DIR}...")
        try:
            _subprocess_run_utf8(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("OK: llama.cpp cloned")
        except Exception as e:
            logger.error(f"Failed to clone llama.cpp: {e}")
            return False
    
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    def run_cmake(config_cmd: list, build_cmd: list) -> bool:
        try:
            _subprocess_run_utf8(config_cmd, cwd=str(build_dir), check=True, capture_output=True, text=True)
            r = _subprocess_run_utf8(build_cmd, cwd=str(build_dir), capture_output=True, text=True)
            out = (r.stdout or "") + (r.stderr or "")
            errs = ["error C", "error LNK", "fatal error"]
            if r.returncode != 0 and any(e in out for e in errs):
                if "llama.lib" not in out and ".lib" not in out:
                    logger.warning(out[-800:] if len(out) > 800 else out)
                    return False
            return True
        except Exception as e:
            logger.warning(f"CMake attempt failed: {e}")
            return False
    
    logger.info("Building llama.cpp (this may take a few minutes)...")
    # 1) Default: cmake -B build, then --build (Ninja or platform default)
    config_default = ["cmake", ".."]
    build_default = ["cmake", "--build", ".", "--config", "Release", "-j", "8"]
    if run_cmake(config_default, build_default):
        if any(p.exists() for p in _quantize_candidates()):
            logger.info("OK: llama.cpp built successfully")
            return True
    # 2) Windows: Visual Studio generator
    if IS_WINDOWS:
        config_vs = ["cmake", "..", "-G", "Visual Studio 17 2022", "-A", "x64"]
        if run_cmake(config_vs, build_default) and any(p.exists() for p in _quantize_candidates()):
            logger.info("OK: llama.cpp built successfully (VS)")
            return True
    logger.error("llama.cpp build failed or llama-quantize not found")
    return False


def convert_to_gguf_direct(
    model_dir: Path,
    output_dir: Path,
    quantize_exe: Path,
    convert_script: Path,
    quantization_methods: list = None,
) -> bool:
    """
    Convert model to GGUF using llama.cpp directly. Runs converter from
    llama.cpp root (cwd) so convert_hf_to_gguf imports work; uses absolute paths.
    """
    if quantization_methods is None:
        quantization_methods = ["q4_k_m"]
    
    output_dir = Path(output_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fp16_gguf = output_dir / "model-fp16.gguf"
    cwd = str(LLAMA_CPP_DIR)
    script = str(convert_script)
    model_str = str(model_dir)
    out_str = str(fp16_gguf)
    
    logger.info("[1/2] Converting to FP16 GGUF...")
    try:
        _subprocess_run_utf8(
            [sys.executable, script, model_str, "--outfile", out_str, "--outtype", "f16"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert to FP16 GGUF: {e.stderr or e.stdout or str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert to FP16 GGUF: {e}")
        return False
    
    if not fp16_gguf.exists():
        logger.error("FP16 GGUF file not created")
        return False
    size_gb = fp16_gguf.stat().st_size / (1024**3)
    logger.info(f"OK: Created {fp16_gguf.name} ({size_gb:.2f} GB)")
    
    logger.info(f"[2/2] Quantizing to {quantization_methods}...")
    q_exe = str(quantize_exe)
    fp16_str = str(fp16_gguf)
    for method in quantization_methods:
        quant_gguf = output_dir / f"model-{method}.gguf"
        logger.info(f"  Creating {method.upper()} quantization...")
        try:
            _subprocess_run_utf8(
                [q_exe, fp16_str, str(quant_gguf), method],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"  Failed to quantize {method}: {e.stderr or e.stdout or str(e)}")
            continue
        except Exception as e:
            logger.error(f"  Failed to quantize {method}: {e}")
            continue
        if quant_gguf.exists():
            size_gb = quant_gguf.stat().st_size / (1024**3)
            logger.info(f"  OK: Created {quant_gguf.name} ({size_gb:.2f} GB)")
        else:
            logger.warning(f"  Quantized file not created: {method}")
    
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
    
    Uses llama.cpp tools directly (convert_hf_to_gguf + llama-quantize) for
    reliable conversion on Windows, Linux, and macOS.
    
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


def _hf_token(config: Dict[str, Any]) -> Optional[str]:
    """Resolve HuggingFace token from config or environment."""
    t = config.get("hf_token") or config.get("model", {}).get("hf_token")
    if t:
        return str(t).strip() or None
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def push_to_hub(
    model,
    tokenizer,
    output_dir: Path,
    repo_id: str,
    config: Dict[str, Any],
    token: Optional[str] = None,
) -> None:
    """
    Push exported artifacts to Hugging Face Hub. Uses upload_folder for
    merged/GGUF (disk); model/tokenizer push_to_hub for LoRA.
    """
    export_cfg = config.get("export", {})
    if not export_cfg.get("push_to_hub") or not repo_id:
        return
    token = token or _hf_token(config)
    if not token:
        logger.warning("push_to_hub enabled but no HF token (hf_token or HF_TOKEN). Skipping.")
        return

    output_dir = Path(output_dir)
    push_lora = export_cfg.get("push_lora", True)
    push_merged = export_cfg.get("push_merged", False)
    push_gguf = export_cfg.get("push_gguf", False)

    try:
        from huggingface_hub import upload_folder
    except ImportError:
        logger.warning("huggingface_hub not installed; cannot push to Hub.")
        return

    if push_lora:
        lora_dir = output_dir / "lora_adapter"
        if lora_dir.exists():
            logger.info("Pushing LoRA adapter to Hub...")
            try:
                upload_folder(
                    folder_path=str(lora_dir),
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
                logger.info("LoRA adapter pushed")
            except Exception as e:
                logger.error(f"Failed to push LoRA: {e}")
        else:
            logger.info("LoRA dir missing, skip pushing LoRA")

    if push_merged:
        merged_dir = output_dir / "merged_16bit"
        if merged_dir.exists():
            logger.info("Pushing merged 16-bit model to Hub...")
            try:
                upload_folder(
                    folder_path=str(merged_dir),
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
                logger.info("Merged 16-bit model pushed")
            except Exception as e:
                logger.error(f"Failed to push merged model: {e}")
        else:
            logger.info("Merged dir missing, skip pushing merged")

    if push_gguf:
        gguf_dir = output_dir / "gguf"
        if gguf_dir.exists():
            logger.info("Pushing GGUF models to Hub...")
            try:
                upload_folder(
                    folder_path=str(gguf_dir),
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
                logger.info("GGUF models pushed")
            except Exception as e:
                logger.error(f"Failed to push GGUF: {e}")
        else:
            logger.info("GGUF dir missing, skip pushing GGUF")


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
    
    # Export to GGUF (llama.cpp direct; cross-platform)
    export_to_gguf(model, tokenizer, output_dir, config, model_name)

    export_cfg = config.get("export", {})
    if export_cfg.get("push_to_hub"):
        repo_id = export_cfg.get("hub_repo_id")
        if repo_id:
            push_to_hub(model, tokenizer, output_dir, repo_id, config)
        else:
            logger.warning("push_to_hub enabled but hub_repo_id not set; skipping")

    logger.info("All exports completed")
