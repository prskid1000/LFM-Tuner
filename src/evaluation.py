"""
Model Evaluation Module
Handles loading and testing saved models in different formats
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from unsloth import FastLanguageModel


logger = logging.getLogger(__name__)


def load_saved_model(
    model_path: Path,
    model_format: str = "lora",
    base_model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Load a saved model in different formats
    
    Args:
        model_path: Path to saved model
        model_format: Format type - "lora", "checkpoint", "merged", or "gguf"
        base_model_name: Base model name (required for LoRA)
        config: Configuration dictionary
    
    Returns:
        Tuple of (model, tokenizer)
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
        raise NotImplementedError(
            "GGUF evaluation requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python\n"
            "Then use llama_cpp.Llama to load the model."
        )
    
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
) -> str:
    """
    Generate text from a prompt
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Generated text
    """
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
    max_new_tokens: int = 200
) -> List[Dict[str, str]]:
    """
    Evaluate model on a dataset
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        dataset: Dataset to evaluate on
        num_samples: Number of samples to test
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of evaluation results
    """
    logger.info(f"Evaluating on {num_samples} samples...")
    
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
                max_new_tokens=max_new_tokens
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


def interactive_test(model, tokenizer):
    """
    Interactive testing loop
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
    """
    print("\n" + "="*70)
    print("Interactive Testing Mode")
    print("="*70)
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
            response = generate_text(model, tokenizer, prompt)
            
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
