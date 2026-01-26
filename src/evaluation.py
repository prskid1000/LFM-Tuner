"""
Model Evaluation Module
Handles loading and testing saved models in different formats
Supports both HuggingFace models and GGUF models (via llama-cpp-python)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python (optional)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.debug("llama-cpp-python not available")


def is_gguf_model(model) -> bool:
    """
    Check if model is a GGUF model (llama-cpp-python)
    
    Args:
        model: Model to check
    
    Returns:
        True if GGUF model, False if HuggingFace model
    """
    return LLAMA_CPP_AVAILABLE and isinstance(model, Llama)


def load_saved_model(
    model_path: Path,
    model_format: str = "lora",
    base_model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Load a saved model in different formats
    
    Supports HuggingFace models (lora, checkpoint, merged) and GGUF models
    
    Args:
        model_path: Path to saved model
        model_format: Format type - "lora", "checkpoint", "merged", or "gguf"
        base_model_name: Base model name (required for LoRA)
        config: Configuration dictionary
    
    Returns:
        Tuple of (model, tokenizer)
        Note: For GGUF models, tokenizer will be None
    """
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Format: {model_format}")
    
    if model_format == "lora":
        # Load LoRA adapter on top of base model
        if not base_model_name:
            raise ValueError("base_model_name is required for LoRA format")
        
        logger.info(f"Loading base model: {base_model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=config.get('training', {}).get('max_seq_length', 2048) if config else 2048,
            dtype=None,
            load_in_4bit=False,
        )
        
        logger.info(f"Loading LoRA adapter from: {model_path}")
        model = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=config.get('training', {}).get('max_seq_length', 2048) if config else 2048,
            dtype=None,
            load_in_4bit=False,
        )[0]
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
    elif model_format == "checkpoint":
        # Load full training checkpoint
        logger.info("Loading training checkpoint")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=config.get('training', {}).get('max_seq_length', 2048) if config else 2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        
    elif model_format == "merged":
        # Load merged 16-bit model
        logger.info("Loading merged 16-bit model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=config.get('training', {}).get('max_seq_length', 2048) if config else 2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        
    elif model_format == "gguf":
        # Load GGUF model using llama-cpp-python
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "GGUF evaluation requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            )
        
        logger.info("Loading GGUF model with llama.cpp")
        
        # Find GGUF file
        gguf_path = Path(model_path)
        if gguf_path.is_dir():
            # If directory provided, look for a GGUF file
            gguf_files = list(gguf_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {gguf_path}")
            gguf_path = gguf_files[0]  # Use first GGUF file
            logger.info(f"Using GGUF file: {gguf_path.name}")
        
        # Load GGUF model with settings from config
        # Get evaluation settings from config, with fallback defaults
        eval_config = config.get('evaluation', {}) if config else {}
        n_ctx = eval_config.get('gguf_n_ctx', 2048)
        n_batch = eval_config.get('gguf_n_batch', 512)
        
        logger.info(f"GGUF settings: n_ctx={n_ctx}, n_batch={n_batch}")
        
        model = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_threads=8,
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False,
            n_batch=n_batch,
        )
        
        # GGUF models don't have a separate tokenizer
        tokenizer = None
        
        logger.info("✓ GGUF model loaded successfully")
        return model, tokenizer
    
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
    
    logger.info("✓ Model loaded successfully")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate text from a prompt
    
    Supports both HuggingFace models and GGUF models (llama-cpp-python)
    
    Args:
        model: Loaded model (HuggingFace or Llama)
        tokenizer: Tokenizer (None for GGUF models)
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        config: Configuration dictionary (optional)
    
    Returns:
        Generated text
    """
    # Check if it's a GGUF model (llama-cpp-python)
    if is_gguf_model(model):
        # Get max prompt length from config
        eval_config = config.get('evaluation', {}) if config else {}
        max_prompt_length = eval_config.get('gguf_max_prompt_length', 1000)
        
        # Truncate prompt if too long for GGUF
        if len(prompt) > max_prompt_length:
            logger.debug(f"Truncating prompt from {len(prompt)} to {max_prompt_length} chars")
            prompt = prompt[:max_prompt_length]
        
        # Reset context before generation to avoid overflow
        try:
            model.reset()
        except:
            pass  # Some versions don't have reset()
        
        try:
            # Use llama-cpp-python generation
            output = model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False  # Don't echo the prompt
            )
            return output['choices'][0]['text']
        except RuntimeError as e:
            if 'llama_decode' in str(e):
                # Context overflow - try with even shorter prompt
                logger.warning(f"Context overflow, retrying with shorter prompt")
                short_prompt = prompt[:500]  # Very short
                model.reset()
                output = model(
                    short_prompt,
                    max_tokens=min(max_new_tokens, 100),  # Shorter generation too
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    echo=False
                )
                return output['choices'][0]['text']
            raise
    
    # HuggingFace model generation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset,
    num_samples: int = 10,
    max_new_tokens: int = 200,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Evaluate model on a dataset
    
    Supports both HuggingFace models and GGUF models (llama-cpp-python)
    
    Args:
        model: Loaded model (HuggingFace or Llama)
        tokenizer: Tokenizer (None for GGUF models)
        dataset: Dataset to evaluate on
        num_samples: Number of samples to test
        max_new_tokens: Maximum tokens to generate
        config: Configuration dictionary (optional)
    
    Returns:
        List of evaluation results
    """
    # Detect model type
    model_type = "GGUF (llama.cpp)" if is_gguf_model(model) else "HuggingFace"
    logger.info(f"Evaluating on {num_samples} samples using {model_type} model...")
    
    results = []
    samples_processed = 0
    
    for i in range(min(num_samples, len(dataset))):
        try:
            example = dataset[i]
            
            # Extract input text
            input_text = None
            if isinstance(example, dict):
                if 'text' in example:
                    # For text format, extract the input part
                    text = example['text']
                    # Try to find the first user message or prompt
                    if '<|im_start|>user' in text:
                        input_text = text.split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()
                    elif 'Question:' in text:
                        input_text = text.split('Question:')[1].split('Answer:')[0].strip()
                    else:
                        input_text = text[:500]  # First 500 chars as fallback
                        
                elif 'messages' in example:
                    # Extract user message from chat format
                    messages = example['messages']
                    if isinstance(messages, list):
                        user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                        if user_msg:
                            input_text = user_msg.get('content', '')
            
            if not input_text:
                logger.debug(f"Skipping sample {i}: no input found")
                continue
            
            # Generate response
            generated = generate_text(
                model, 
                tokenizer, 
                input_text,
                max_new_tokens=max_new_tokens,
                config=config
            )
            
            results.append({
                "sample_id": i,
                "input": input_text,
                "output": generated
            })
            samples_processed += 1
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue
    
    logger.info(f"✓ Evaluated {samples_processed} samples")
    return results


def interactive_test(model, tokenizer, config: Optional[Dict[str, Any]] = None):
    """
    Interactive testing loop
    
    Supports both HuggingFace models and GGUF models (llama-cpp-python)
    
    Args:
        model: Loaded model (HuggingFace or Llama)
        tokenizer: Tokenizer (None for GGUF models)
        config: Configuration dictionary (optional)
    """
    # Detect model type
    model_type = "GGUF (llama.cpp)" if is_gguf_model(model) else "HuggingFace"
    
    print("\n" + "="*70)
    print("Interactive Testing Mode")
    print("="*70)
    print(f"Model type: {model_type}")
    print("Enter prompts to test the model. Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating response...\n")
            response = generate_text(model, tokenizer, prompt, config=config)
            
            print("="*70)
            print("Response:")
            print("-"*70)
            print(response)
            print("="*70)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
