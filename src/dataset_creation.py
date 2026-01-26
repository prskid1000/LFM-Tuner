"""
Dataset Creation Module
Handles loading initial datasets and generating synthetic data using LM Studio
"""

import json
import logging
from random import random
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import time


logger = logging.getLogger(__name__)


def load_initial_dataset(
    dataset_path: Path,
    schema: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load initial JSON dataset with flexible schema detection
    
    Args:
        dataset_path: Path to JSON file
        schema: Optional schema type ('instruction', 'chat', 'completion', 'auto')
    
    Returns:
        List of data examples
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.suffix == '.jsonl':
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                # Try to find the data array
                if 'data' in data:
                    data = data['data']
                elif 'examples' in data:
                    data = data['examples']
                else:
                    raise ValueError("Could not find data array in JSON file")
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def detect_schema(data: List[Dict[str, Any]]) -> str:
    """Auto-detect dataset schema"""
    if not data:
        return "unknown"
    
    sample = data[0]
    keys = set(sample.keys())
    
    # Instruction-response format
    if 'instruction' in keys and 'response' in keys:
        return 'instruction'
    
    # Chat format
    if 'messages' in keys or 'conversations' in keys:
        return 'chat'
    
    # Completion format
    if 'prompt' in keys and 'completion' in keys:
        return 'completion'
    
    # Text format
    if 'text' in keys:
        return 'text'
    
    return 'unknown'


def generate_with_lm_studio(
    system_prompt: str,
    user_prompt: str,
    api_url: str = "http://localhost:1234",
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 60,
    response_format: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate text using LM Studio API with chat completions endpoint
    
    Args:
        system_prompt: System instruction for the model
        user_prompt: User message/prompt
        api_url: LM Studio API URL (default: http://localhost:1234)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        response_format: Optional structured output format (e.g., {"type": "json_object"})
    
    Returns:
        Generated text
    """
    endpoint = f"{api_url}/v1/chat/completions"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": int(random() * (2**32 - 1)),
        "stream": False
    }
    
    # Add structured output format if specified
    if response_format:
        payload["response_format"] = response_format
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        return generated_text
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to LM Studio API at {api_url}. "
            "Make sure LM Studio is running and API server is enabled."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"LM Studio API request timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Error calling LM Studio API: {e}")


def create_augmentation_prompts(
    example: Dict[str, Any], 
    schema: str, 
    strategy: str
) -> tuple[str, str]:
    """
    Create system and user prompts for augmentation
    
    Augmentation Strategies:
    - 'paraphrase': Rewrites instructions/prompts with different wording while keeping the same meaning
                   Works with: instruction, completion schemas
    
    - 'expand': Generates new detailed responses/completions for existing instructions/prompts
               Works with: instruction, completion schemas
    
    - 'variation': Creates new instructions that ask for similar information in different ways
                  Works with: instruction schema only
    
    - 'response_variation': Generates alternative responses using different approaches or styles
                           Works with: instruction schema only
    
    Args:
        example: Dataset example to augment
        schema: Detected schema ('instruction', 'chat', 'completion', 'text')
        strategy: Augmentation strategy to apply
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    
    if strategy == "paraphrase":
        if schema == 'instruction':
            system_prompt = """You are a dataset augmentation assistant. Your task is to paraphrase instructions while preserving their meaning and intent. 
Rules:
- Maintain the same semantic meaning
- Use different wording and sentence structure
- Keep the same level of detail
- Output ONLY the paraphrased instruction, nothing else
- Do not add explanations or extra text"""
            
            user_prompt = f"""Paraphrase this instruction:

{example.get('instruction', '')}

Paraphrased instruction:"""
            
        elif schema == 'completion':
            system_prompt = """You are a dataset augmentation assistant. Paraphrase the given prompt while maintaining its core meaning and purpose.
Output ONLY the paraphrased prompt without any additional text."""
            
            user_prompt = f"""Paraphrase this prompt:

{example.get('prompt', '')}

Paraphrased prompt:"""
        
        else:
            # Fallback for unsupported schemas
            system_prompt = "You are a helpful dataset augmentation assistant."
            user_prompt = f"Paraphrase this text: {str(example)}"
    
    elif strategy == "expand":
        if schema == 'instruction':
            system_prompt = """You are a dataset augmentation assistant. Your task is to generate detailed, high-quality responses to instructions.
Rules:
- Provide comprehensive and accurate responses
- Maintain professional tone
- Be specific and actionable
- Output ONLY the response, no preamble or explanations
- Do not restate the instruction"""
            
            user_prompt = f"""Instruction: {example.get('instruction', '')}

Generate a detailed response:"""
            
        elif schema == 'completion':
            system_prompt = """You are a dataset augmentation assistant. Generate expanded, detailed completions for the given prompt.
Output ONLY the completion text."""
            
            user_prompt = f"""Prompt: {example.get('prompt', '')}

Completion:"""
        
        else:
            system_prompt = "You are a helpful dataset augmentation assistant."
            user_prompt = f"Generate a detailed completion for: {str(example)}"
    
    elif strategy == "variation":
        if schema == 'instruction':
            system_prompt = """You are a dataset augmentation assistant. Create variations of instructions that ask for similar information in different ways.
Rules:
- Change the phrasing and approach
- Maintain the core intent
- Add or modify context slightly
- Output ONLY the varied instruction
- Keep it natural and realistic"""
            
            user_prompt = f"""Create a variation of this instruction:

{example.get('instruction', '')}

Varied instruction:"""
        
        else:
            # 'variation' strategy only works well with instruction schema
            logger.warning(f"'variation' strategy works best with 'instruction' schema, current schema is '{schema}'")
            system_prompt = "You are a helpful dataset augmentation assistant."
            user_prompt = f"Create a variation of: {str(example)}"
    
    elif strategy == "response_variation":
        if schema == 'instruction':
            system_prompt = """You are a dataset augmentation assistant. Generate alternative responses to the given instruction that are correct but approach the answer differently.
Rules:
- Provide accurate information
- Use a different structure or emphasis
- Maintain quality and completeness
- Output ONLY the response"""
            
            user_prompt = f"""Instruction: {example.get('instruction', '')}

Original response: {example.get('response', '')}

Generate an alternative response:"""
        
        else:
            # 'response_variation' strategy only works with instruction schema
            logger.warning(f"'response_variation' strategy requires 'instruction' schema, current schema is '{schema}'")
            system_prompt = "You are a helpful dataset augmentation assistant."
            user_prompt = f"Generate an alternative response for: {str(example)}"
    
    else:
        system_prompt = "You are a helpful dataset augmentation assistant."
        user_prompt = str(example)
    
    return system_prompt, user_prompt


def augment_dataset(
    initial_data: List[Dict[str, Any]],
    api_url: str = "http://localhost:1234",
    augmentation_strategy: str = "paraphrase",
    num_augmentations_per_example: int = 2,
    delay: float = 0.5,
    use_structured_output: bool = False,
    save_path: Optional[Path] = None,
    save_format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Augment dataset using LM Studio
    
    Args:
        initial_data: Initial dataset examples
        api_url: LM Studio API URL
        augmentation_strategy: Strategy to use:
            - 'paraphrase': Rewrite the instruction/prompt with different wording (keeps same meaning)
            - 'expand': Generate new detailed responses for existing instructions
            - 'variation': Create new instructions that ask similar things in different ways
            - 'response_variation': Generate alternative responses with different approaches/styles
        num_augmentations_per_example: Number of augmentations per example
        delay: Delay between API calls (seconds)
        use_structured_output: Use JSON structured output format
        save_path: Optional path to save augmented data (default: data/generated/augmented_dataset.json)
        save_format: Format to save ('json' or 'jsonl')
    
    Returns:
        Augmented dataset
        
    Note:
        Strategy compatibility by schema:
        - instruction schema: All strategies work
        - completion schema: 'paraphrase' (prompts) and 'expand' (completions) work
        - chat schema: Limited support (appends new assistant messages)
    """
    augmented_data = []
    schema = detect_schema(initial_data)
    
    logger.info(f"Augmenting dataset with '{augmentation_strategy}' strategy")
    logger.info(f"Detected schema: {schema}")
    logger.info(f"Generating {num_augmentations_per_example} augmentations per example")
    
    response_format = {"type": "json_object"} if use_structured_output else None
    
    for i, example in enumerate(initial_data):
        augmented_data.append(example)  # Keep original
        
        for aug_idx in range(num_augmentations_per_example):
            try:
                system_prompt, user_prompt = create_augmentation_prompts(
                    example, schema, augmentation_strategy
                )
                
                # Modify prompts for structured output
                if use_structured_output and schema == 'instruction':
                    if augmentation_strategy == "paraphrase":
                        user_prompt += '\n\nRespond in JSON format: {"instruction": "your paraphrased instruction here"}'
                    elif augmentation_strategy == "expand":
                        user_prompt += '\n\nRespond in JSON format: {"response": "your response here"}'
                
                generated = generate_with_lm_studio(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_url=api_url,
                    response_format=response_format
                )
                
                # Parse structured output
                if use_structured_output:
                    try:
                        generated_json = json.loads(generated)
                        if augmentation_strategy == "paraphrase":
                            generated = generated_json.get("instruction", generated)
                        elif augmentation_strategy == "expand":
                            generated = generated_json.get("response", generated)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON output, using raw text")
                
                # Create augmented example based on schema and strategy
                aug_example = example.copy()
                
                if schema == 'instruction':
                    if augmentation_strategy == "paraphrase":
                        aug_example['instruction'] = generated
                    elif augmentation_strategy == "expand":
                        aug_example['response'] = generated
                    elif augmentation_strategy == "variation":
                        aug_example['instruction'] = generated
                    elif augmentation_strategy == "response_variation":
                        aug_example['response'] = generated
                
                elif schema == 'chat':
                    messages = example.get('messages', example.get('conversations', []))
                    if messages:
                        aug_example = example.copy()
                        if 'messages' in aug_example:
                            aug_example['messages'] = messages.copy()
                            aug_example['messages'].append({
                                "role": "assistant",
                                "content": generated
                            })
                
                elif schema == 'completion':
                    if augmentation_strategy == "paraphrase":
                        aug_example['prompt'] = generated
                    else:
                        aug_example['completion'] = generated
                
                augmented_data.append(aug_example)
                time.sleep(delay)
                
            except Exception as e:
                logger.warning(f"Failed to augment example {i}, augmentation {aug_idx}: {e}")
                continue
        
        if (i + 1) % 10 == 0:
            logger.info(f"Augmented {i + 1}/{len(initial_data)} examples")
    
    logger.info(f"Augmentation complete: {len(initial_data)} -> {len(augmented_data)} examples")
    
    # Auto-save if save_path is provided or use default
    if save_path is None:
        # Default save path
        save_path = Path("data/generated/augmented_dataset.json")
    
    save_path = Path(save_path)
    save_dataset(augmented_data, save_path, format=save_format)
    
    return augmented_data


def filter_quality(
    data: List[Dict[str, Any]],
    min_length: int = 10,
    max_length: int = 2000,
    remove_duplicates: bool = True,
    save_path: Optional[Path] = None,
    save_format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Filter dataset for quality
    
    Args:
        data: Dataset examples
        min_length: Minimum text length
        max_length: Maximum text length
        remove_duplicates: Remove duplicate examples
        save_path: Optional path to save filtered data (default: data/generated/filtered_dataset.json)
        save_format: Format to save ('json' or 'jsonl')
    
    Returns:
        Filtered dataset
    """
    filtered = []
    seen = set()
    
    for example in data:
        # Extract text for length check
        text = ""
        if 'instruction' in example:
            text += str(example.get('instruction', ''))
        if 'response' in example:
            text += str(example.get('response', ''))
        if 'text' in example:
            text = str(example.get('text', ''))
        
        # Length filter
        if len(text) < min_length or len(text) > max_length:
            continue
        
        # Duplicate filter
        if remove_duplicates:
            text_hash = hash(text)
            if text_hash in seen:
                continue
            seen.add(text_hash)
        
        filtered.append(example)
    
    logger.info(f"Quality filtering: {len(data)} -> {len(filtered)} examples")
    
    # Auto-save if save_path is provided or use default
    if save_path is None:
        save_path = Path("data/generated/filtered_dataset.json")
    
    save_path = Path(save_path)
    save_dataset(filtered, save_path, format=save_format)
    
    return filtered


def save_dataset(data: List[Dict[str, Any]], output_path: Path, format: str = "json") -> None:
    """Save dataset to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved dataset to: {output_path}")