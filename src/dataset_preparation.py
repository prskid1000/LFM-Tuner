"""
Dataset Preparation Module
Converts datasets to Unsloth-compatible format and validates data
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, DatasetDict
import json


logger = logging.getLogger(__name__)


def convert_to_unsloth_format(
    data: List[Dict[str, Any]],
    model_type: str = "llama",
    format_type: str = "chat",
    save_path: Optional[Path] = None,
    save_format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Convert dataset to Unsloth-compatible format
    
    Args:
        data: Raw dataset examples
        model_type: Model type ('llama', 'qwen', 'gemma', 'lfm')
        format_type: Format type ('chat', 'instruction')
        save_path: Optional path to save converted data (default: data/processed/converted_dataset.json)
        save_format: Format to save ('json' or 'jsonl')
    
    Returns:
        Converted dataset
    """
    converted = []
    schema = detect_schema(data)
    
    logger.info(f"Converting dataset from {schema} to {format_type} format for {model_type}")
    
    for example in data:
        if format_type == "chat":
            converted_example = convert_to_chat_format(example, model_type, schema)
        elif format_type == "instruction":
            converted_example = convert_to_instruction_format(example, model_type, schema)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        if converted_example:
            converted.append(converted_example)
    
    logger.info(f"Converted {len(data)} -> {len(converted)} examples")
    
    # Auto-save if save_path is provided or use default
    if save_path is None:
        save_path = Path("data/processed/converted_dataset.json")
    
    save_path = Path(save_path)
    _save_dataset_single(converted, save_path, format=save_format)
    
    return converted


def detect_schema(data: List[Dict[str, Any]]) -> str:
    """Detect dataset schema"""
    if not data:
        return "unknown"
    
    sample = data[0]
    keys = set(sample.keys())
    
    if 'instruction' in keys and 'response' in keys:
        return 'instruction'
    if 'messages' in keys or 'conversations' in keys:
        return 'chat'
    if 'prompt' in keys and 'completion' in keys:
        return 'completion'
    if 'text' in keys:
        return 'text'
    
    return 'unknown'


def convert_to_chat_format(
    example: Dict[str, Any],
    model_type: str,
    source_schema: str
) -> Optional[Dict[str, str]]:
    """Convert example to chat format"""
    messages = []
    
    if source_schema == 'chat':
        messages = example.get('messages', example.get('conversations', []))
        if not isinstance(messages, list):
            messages = [messages]
    
    elif source_schema == 'instruction':
        instruction = example.get('instruction', '')
        response = example.get('response', example.get('output', ''))
        
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    
    elif source_schema == 'completion':
        prompt = example.get('prompt', '')
        completion = example.get('completion', example.get('response', ''))
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    
    elif source_schema == 'text':
        text = example.get('text', '')
        # Simple split (can be improved)
        messages = [
            {"role": "user", "content": text[:len(text)//2]},
            {"role": "assistant", "content": text[len(text)//2:]}
        ]
    
    if not messages:
        return None
    
    # Format according to model type
    if model_type in ['llama', 'qwen', 'gemma', 'lfm']:
        # Standard chat format
        text = format_chat_messages(messages, model_type)
        return {"text": text}
    
    return None


def convert_to_instruction_format(
    example: Dict[str, Any],
    model_type: str,
    source_schema: str
) -> Optional[Dict[str, str]]:
    """Convert example to instruction format"""
    instruction = ""
    response = ""
    
    if source_schema == 'instruction':
        instruction = example.get('instruction', '')
        response = example.get('response', example.get('output', ''))
    
    elif source_schema == 'chat':
        messages = example.get('messages', example.get('conversations', []))
        if isinstance(messages, list) and len(messages) >= 2:
            instruction = messages[-2].get('content', '')
            response = messages[-1].get('content', '')
    
    elif source_schema == 'completion':
        instruction = example.get('prompt', '')
        response = example.get('completion', example.get('response', ''))
    
    if not instruction or not response:
        return None
    
    # Format according to model type
    if model_type in ['llama', 'qwen', 'gemma', 'lfm']:
        text = format_instruction(instruction, response, model_type)
        return {"text": text}
    
    return None


def format_chat_messages(messages: List[Dict[str, str]], model_type: str) -> str:
    """Format messages for chat template"""
    if model_type == "llama":
        # Llama 3 format
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                formatted.append(f"<|user|>\n{content}<|end|>\n")
            elif role == 'assistant':
                formatted.append(f"<|assistant|>\n{content}<|end|>\n")
        return "".join(formatted)
    
    elif model_type == "qwen":
        # Qwen format
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(formatted)
    
    else:
        # Generic format
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted.append(f"{role.capitalize()}: {content}\n")
        return "".join(formatted)


def format_instruction(instruction: str, response: str, model_type: str) -> str:
    """Format instruction-response pair"""
    if model_type == "llama":
        return f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>\n"
    elif model_type == "qwen":
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"


def validate_dataset(
    data: List[Dict[str, Any]],
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate dataset format and quality
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not data:
        errors.append("Dataset is empty")
        return False, errors
    
    if required_keys:
        sample = data[0]
        for key in required_keys:
            if key not in sample:
                errors.append(f"Missing required key: {key}")
    
    # Check for empty examples
    empty_count = 0
    for i, example in enumerate(data):
        if 'text' in example:
            if not example['text'] or len(example['text'].strip()) == 0:
                empty_count += 1
                errors.append(f"Empty text in example {i}")
    
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty examples")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    save_dir: Optional[Path] = None,
    save_format: str = "json"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train and validation sets
    
    Args:
        data: Dataset examples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        save_dir: Optional directory to save split data (default: data/processed/)
        save_format: Format to save ('json' or 'jsonl')
    
    Returns:
        (train_data, val_data)
    """
    import random
    
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val")
    
    # Auto-save if save_dir is provided or use default
    if save_dir is None:
        save_dir = Path("data/processed")
    
    save_dir = Path(save_dir)
    save_processed_dataset(train_data, val_data, save_dir, format=save_format)
    
    return train_data, val_data


def preview_tokenization(
    examples: List[Dict[str, Any]],
    tokenizer,
    num_examples: int = 3
) -> None:
    """Preview tokenization of examples"""
    logger.info(f"Previewing tokenization for {num_examples} examples:")
    
    for i, example in enumerate(examples[:num_examples]):
        text = example.get('text', '')
        tokens = tokenizer(text, return_tensors="pt")
        token_count = tokens['input_ids'].shape[1]
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Token count: {token_count}")
        logger.info(f"Preview: {text[:100]}...")


def save_processed_dataset(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    output_dir: Path,
    format: str = "json"
) -> None:
    """Save processed dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"train.{format}"
    val_path = output_dir / f"val.{format}"
    
    if format == "json":
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
    elif format == "jsonl":
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved processed dataset to: {output_dir}")


def _save_dataset_single(data: List[Dict[str, Any]], output_path: Path, format: str = "json") -> None:
    """Internal helper to save a single dataset file"""
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