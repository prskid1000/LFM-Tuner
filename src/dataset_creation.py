"""
Dataset Creation Module
Handles loading initial datasets and generating synthetic data using LM Studio

All input/output data must be in messages format:
[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""

import hashlib
import json
import logging
from random import random
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import time


logger = logging.getLogger(__name__)


def _get_example_text(example: Dict[str, Any]) -> str:
    """Extract concatenated text from an example for length/filter checks."""
    messages = example.get("messages", example.get("conversations", []))
    if messages and isinstance(messages, list):
        return " ".join(
            msg.get("content", "") for msg in messages if isinstance(msg, dict)
        )
    if "text" in example:
        return str(example.get("text", ""))
    return ""


def _content_hash(text: str) -> str:
    """Deterministic hash for deduplication (stable across runs)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_initial_dataset(
    dataset_path: Path
) -> List[Dict[str, Any]]:
    """
    Load initial JSON dataset in messages format
    
    Expected format: Each example must have a 'messages' key containing a list of
    message dictionaries with 'role' and 'content' keys.
    
    Example:
    [
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language..."}
            ]
        }
    ]
    
    Args:
        dataset_path: Path to JSON or JSONL file
    
    Returns:
        List of data examples in messages format
    
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If data format is invalid
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
    
    # Validate format
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of examples")
    
    if len(data) == 0:
        raise ValueError("Dataset is empty")
    
    # Check that all examples have messages
    for i, example in enumerate(data):
        if not isinstance(example, dict):
            raise ValueError(f"Example {i} must be a dictionary")
        if 'messages' not in example and 'conversations' not in example:
            raise ValueError(
                f"Example {i} must have 'messages' key. "
                "Expected format: {{'messages': [{{'role': 'user', 'content': '...'}}, ...]}}"
            )
    
    logger.info(f"Loaded {len(data)} examples in messages format")
    return data


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
    strategy: str
) -> tuple[str, str]:
    """
    Create system and user prompts for augmentation
    
    Augmentation Strategies:
    - 'paraphrase': Rewrites user messages with different wording while keeping the same meaning
    - 'expand': Generates new detailed assistant responses for existing user messages
    - 'variation': Creates new user messages that ask for similar information in different ways
    - 'response_variation': Generates alternative assistant responses with different approaches/styles
    
    Args:
        example: Dataset example in messages format (must have 'messages' key)
        strategy: Augmentation strategy to apply
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Extract messages from example
    messages = example.get('messages', example.get('conversations', []))
    
    if not messages or not isinstance(messages, list):
        raise ValueError("Example must have 'messages' key with a list of messages")
    
    # Extract user and assistant content from messages
    user_messages = [msg['content'] for msg in messages if msg.get('role') == 'user']
    assistant_messages = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
    
    user_content = user_messages[-1] if user_messages else ""
    assistant_content = assistant_messages[-1] if assistant_messages else ""
    
    if strategy == "paraphrase":
        system_prompt = """You are a dataset augmentation assistant. Your task is to paraphrase user messages while preserving their meaning and intent. 
Rules:
- Maintain the same semantic meaning
- Use different wording and sentence structure
- Keep the same level of detail
- Output ONLY the paraphrased message, nothing else
- Do not add explanations or extra text"""
        
        user_prompt = f"""Paraphrase this user message:

{user_content}

Paraphrased message:"""
    
    elif strategy == "expand":
        system_prompt = """You are a dataset augmentation assistant. Your task is to generate detailed, high-quality assistant responses.
Rules:
- Provide comprehensive and accurate responses
- Maintain professional tone
- Be specific and actionable
- Output ONLY the response, no preamble or explanations
- Do not restate the user message"""
        
        user_prompt = f"""User message: {user_content}

Generate a detailed assistant response:"""
    
    elif strategy == "variation":
        system_prompt = """You are a dataset augmentation assistant. Create variations of user messages that ask for similar information in different ways.
Rules:
- Change the phrasing and approach
- Maintain the core intent
- Add or modify context slightly
- Output ONLY the varied message
- Keep it natural and realistic"""
        
        user_prompt = f"""Create a variation of this user message:

{user_content}

Varied message:"""
    
    elif strategy == "response_variation":
        if not assistant_content:
            logger.warning(f"'response_variation' strategy requires an assistant response, but none found")
            system_prompt = "You are a helpful dataset augmentation assistant."
            user_prompt = f"Generate an alternative response for: {user_content}"
        else:
            system_prompt = """You are a dataset augmentation assistant. Generate alternative assistant responses that are correct but approach the answer differently.
Rules:
- Provide accurate information
- Use a different structure or emphasis
- Maintain quality and completeness
- Output ONLY the response"""
            
            user_prompt = f"""User message: {user_content}

Original assistant response: {assistant_content}

Generate an alternative assistant response:"""
    
    else:
        system_prompt = "You are a helpful dataset augmentation assistant."
        user_prompt = f"Augment this conversation: {str(messages)}"
    
    return system_prompt, user_prompt


def _passes_filter(
    example: Dict[str, Any],
    min_length: int,
    max_length: int,
    remove_duplicates: bool,
    seen: set,
) -> bool:
    """Return True if example passes length and (optionally) duplicate check."""
    text = _get_example_text(example)
    if len(text) < min_length or len(text) > max_length:
        return False
    if remove_duplicates:
        h = _content_hash(text)
        if h in seen:
            return False
        seen.add(h)
    return True


def augment_dataset(
    initial_data: List[Dict[str, Any]],
    api_url: str = "http://localhost:1234",
    augmentation_strategy: str = "paraphrase",
    num_augmentations_per_example: int = 2,
    delay: float = 0.5,
    use_structured_output: bool = False,
    min_length: int = 10,
    max_length: int = 2000,
    remove_duplicates: bool = True,
    max_attempts_per_example: Optional[int] = None,
    save_path: Optional[Path] = None,
    save_format: str = "json",
) -> List[Dict[str, Any]]:
    """
    Augment dataset using LM Studio with built-in quality filtering.

    Only examples that pass length and duplicate checks are kept, so
    num_augmentations_per_example is the number of unique valid samples per seed.
    Generation continues per example until that many valid samples are collected
    or max_attempts_per_example is reached.

    Input and output data must be in messages format.

    Args:
        initial_data: Initial dataset examples in messages format (each must have 'messages' key)
        api_url: LM Studio API URL
        augmentation_strategy: Strategy to use:
            - 'paraphrase': Rewrite user messages with different wording (keeps same meaning)
            - 'expand': Generate new detailed assistant responses
            - 'variation': Create new user messages that ask similar things in different ways
            - 'response_variation': Generate alternative assistant responses with different approaches/styles
        num_augmentations_per_example: Number of unique valid augmentations to keep per example
        delay: Delay between API calls (seconds)
        use_structured_output: Use JSON structured output format
        min_length: Minimum total text length; samples below are dropped (default 10)
        max_length: Maximum total text length; samples above are dropped (default 2000)
        remove_duplicates: If True, only keep one copy of each content (default True)
        max_attempts_per_example: Max API calls per example when chasing num_augmentations_per_example
            (default 3 * num_augmentations_per_example)
        save_path: Optional path to save augmented data (default: data/generated/augmented_dataset.json)
        save_format: Format to save ('json' or 'jsonl')

    Returns:
        Augmented and filtered dataset in messages format (no separate filter step needed).
    """
    if max_attempts_per_example is None:
        max_attempts_per_example = max(num_augmentations_per_example * 3, 10)

    augmented_data = []
    seen: set = set()

    logger.info(f"Augmenting dataset with '{augmentation_strategy}' strategy")
    logger.info(
        f"Target: {num_augmentations_per_example} unique valid samples per example "
        f"(filter: length [{min_length}, {max_length}], remove_duplicates={remove_duplicates})"
    )

    response_format = {"type": "json_object"} if use_structured_output else None

    for i, example in enumerate(initial_data):
        # Include original only if it passes the same filter
        if _passes_filter(example, min_length, max_length, remove_duplicates, seen):
            augmented_data.append(example)

        collected = 0
        attempts = 0

        while collected < num_augmentations_per_example and attempts < max_attempts_per_example:
            attempts += 1
            try:
                system_prompt, user_prompt = create_augmentation_prompts(
                    example, augmentation_strategy
                )

                if use_structured_output:
                    if augmentation_strategy == "paraphrase":
                        user_prompt += '\n\nRespond in JSON format: {"message": "your paraphrased message here"}'
                    elif augmentation_strategy == "expand":
                        user_prompt += '\n\nRespond in JSON format: {"response": "your response here"}'

                generated = generate_with_lm_studio(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_url=api_url,
                    response_format=response_format,
                )

                if use_structured_output:
                    try:
                        generated_json = json.loads(generated)
                        if augmentation_strategy == "paraphrase":
                            generated = generated_json.get("message", generated)
                        elif augmentation_strategy == "expand":
                            generated = generated_json.get("response", generated)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON output, using raw text")

                messages = list(
                    example.get("messages", example.get("conversations", []))
                )
                aug_example = example.copy()

                if augmentation_strategy == "paraphrase":
                    for j in range(len(messages) - 1, -1, -1):
                        if messages[j].get("role") == "user":
                            messages[j] = {"role": "user", "content": generated}
                            break
                elif augmentation_strategy == "expand":
                    messages.append({"role": "assistant", "content": generated})
                elif augmentation_strategy == "variation":
                    for j in range(len(messages) - 1, -1, -1):
                        if messages[j].get("role") == "user":
                            messages[j] = {"role": "user", "content": generated}
                            break
                elif augmentation_strategy == "response_variation":
                    for j in range(len(messages) - 1, -1, -1):
                        if messages[j].get("role") == "assistant":
                            messages[j] = {"role": "assistant", "content": generated}
                            break

                aug_example["messages"] = messages

                if _passes_filter(
                    aug_example, min_length, max_length, remove_duplicates, seen
                ):
                    augmented_data.append(aug_example)
                    collected += 1

                time.sleep(delay)

            except Exception as e:
                logger.warning(
                    f"Failed to augment example {i}, attempt {attempts}: {e}"
                )

        if (i + 1) % 10 == 0:
            logger.info(
                f"Augmented {i + 1}/{len(initial_data)} examples "
                f"({len(augmented_data)} total unique valid so far)"
            )

    logger.info(
        f"Augmentation complete: {len(initial_data)} seeds -> {len(augmented_data)} unique valid examples"
    )
    
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
        text = _get_example_text(example)
        if len(text) < min_length or len(text) > max_length:
            continue
        if remove_duplicates:
            h = _content_hash(text)
            if h in seen:
                continue
            seen.add(h)
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