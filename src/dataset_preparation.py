"""
Dataset Preparation Module
Converts datasets to Unsloth-compatible format using tokenizer's chat template

All input data must be in messages format:
[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, DatasetDict
import json


logger = logging.getLogger(__name__)

def _infer_chat_markers_from_tokenizer(tokenizer) -> Optional[Tuple[str, str]]:
    """
    Infer raw string markers for user/instruction and assistant/response
    by rendering a minimal conversation with tokenizer.apply_chat_template.

    Returns:
        (instruction_part, response_part) or None if unavailable.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None

    user_sentinel = "__USER_SENTINEL__"
    assistant_sentinel = "__ASSISTANT_SENTINEL__"
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_sentinel},
                {"role": "assistant", "content": assistant_sentinel},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None

    if not isinstance(rendered, str):
        return None

    user_idx = rendered.find(user_sentinel)
    asst_idx = rendered.find(assistant_sentinel)
    if user_idx == -1 or asst_idx == -1 or asst_idx <= user_idx:
        return None

    instruction_part = rendered[:user_idx]
    between = rendered[user_idx + len(user_sentinel) : asst_idx]
    if not between:
        return None

    # Prefer the exact assistant turn header if it exists inside the boundary.
    response_part = between
    for candidate in ("<|im_start|>assistant\n", "<|assistant|>\n", "assistant\n"):
        if candidate in between:
            response_part = candidate
            break

    return instruction_part, response_part


def _token_ids_contain_subsequence(haystack, needle) -> bool:
    """Return True if `needle` appears as a contiguous subsequence in `haystack`."""
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def convert_to_unsloth_format(
    data: List[Dict[str, Any]],
    tokenizer,
    save_path: Optional[Path] = None,
    save_format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Convert dataset to Unsloth-compatible format using tokenizer's chat template
    
    Input data must be in messages format. Each example should have a 'messages' key
    containing a list of message dictionaries with 'role' and 'content' keys.
    
    Example input format:
    {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]
    }
    
    Args:
        data: List of examples, each with a 'messages' key containing list of message dicts
        tokenizer: Tokenizer with apply_chat_template method (from the model)
        save_path: Optional path to save converted data (default: data/processed/converted_dataset.json)
        save_format: Format to save ('json' or 'jsonl')
    
    Returns:
        Converted dataset with 'text' field formatted using tokenizer's chat template
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required. Load the model first to get the tokenizer.")
    
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError("tokenizer must have apply_chat_template method. Use a proper model tokenizer.")

    inferred = _infer_chat_markers_from_tokenizer(tokenizer)
    if inferred is None:
        raise ValueError(
            "Could not infer chat markers from tokenizer.apply_chat_template. "
            "This pipeline requires a tokenizer with a valid chat template so the dataset text "
            "and training masking logic stay aligned."
        )
    _, response_part = inferred
    
    converted = []
    
    logger.info(f"Converting {len(data)} examples using tokenizer's chat template")
    
    for i, example in enumerate(data):
        # Extract messages from example
        messages = example.get('messages', example.get('conversations', []))
        
        if not messages:
            logger.warning(f"Example {i} has no 'messages' key, skipping")
            continue
        
        if not isinstance(messages, list):
            logger.warning(f"Example {i} has invalid messages format (not a list), skipping")
            continue
        
        # Validate and normalize messages format
        normalized_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Example {i} has invalid message (not a dict), skipping")
                break
            if 'role' not in msg or 'content' not in msg:
                logger.warning(f"Example {i} has message missing 'role' or 'content', skipping")
                break
            if not msg.get('content', '').strip():
                continue  # Skip empty messages
            role = str(msg["role"]).strip()
            content = str(msg["content"]).strip()
            if role not in {"system", "user", "assistant"}:
                logger.warning(f"Example {i} has unsupported role={role!r}, skipping")
                break
            normalized_messages.append({"role": role, "content": content})
        else:
            # Only process if we didn't break (all messages valid)
            if not normalized_messages:
                logger.warning(f"Example {i} has no valid messages after normalization, skipping")
                continue

            # Strict: require at least 1 user turn and an assistant answer.
            roles = [m["role"] for m in normalized_messages]
            if "user" not in roles or "assistant" not in roles:
                logger.warning(f"Example {i} missing required user/assistant turns, skipping")
                continue
            if normalized_messages[-1]["role"] != "assistant":
                logger.warning(f"Example {i} does not end with assistant message, skipping")
                continue
            
            # Use tokenizer's apply_chat_template to format according to model's native format
            try:
                text = tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                # Token-level sanity check: response marker must appear in tokenized text.
                marker_ids = tokenizer(response_part, add_special_tokens=False)["input_ids"]
                text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if not _token_ids_contain_subsequence(text_ids, marker_ids):
                    raise ValueError(
                        "Assistant response marker not found in tokenized rendered text. "
                        "This indicates a tokenizer/template mismatch."
                    )

                # Store both for traceability/debugging.
                converted.append({"messages": normalized_messages, "text": text})
            except Exception as e:
                logger.warning(f"Failed to apply chat template to example {i}: {e}")
                continue
    
    logger.info(f"Converted {len(data)} -> {len(converted)} examples")
    
    # Auto-save if save_path is provided or use default
    if save_path is None:
        save_path = Path("data/processed/converted_dataset.json")
    
    save_path = Path(save_path)
    _save_dataset_single(converted, save_path, format=save_format)
    
    return converted


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
    """
    Preview tokenization of examples
    
    Args:
        examples: List of examples with 'text' field (already formatted)
        tokenizer: Tokenizer to use for tokenization
        num_examples: Number of examples to preview
    """
    logger.info(f"Previewing tokenization for {num_examples} examples:")
    
    for i, example in enumerate(examples[:num_examples]):
        text = example.get('text', '')
        if not text:
            logger.warning(f"Example {i+1} has no text field")
            continue
            
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