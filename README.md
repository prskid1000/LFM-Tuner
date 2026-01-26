# LFM-Tuner: LLM Fine-tuning Pipeline

A comprehensive end-to-end fine-tuning framework using Unsloth with Flash Attention 2 and SAGE Attention support, optimized for Windows and 12GB VRAM GPUs. Pre-configured for **LFM 2.5 1.2B Thinking** model.

## Features

- **Complete Pipeline**: Dataset creation → Preparation → Training → Export
- **LFM 2.5 Thinking Optimized**: Pre-configured for LFM 2.5 1.2B Thinking model
- **LM Studio Integration**: Generate synthetic data using local models
- **Dual Attention Support**: Flash Attention 2 or SAGE Attention (user choice, no fallback)
- **12GB VRAM Optimized**: Automatic memory management and batch size tuning
- **Configurable Export**: LoRA adapter, merged 16-bit, and GGUF formats
- **Windows Optimized**: Pre-built wheels, automated installation scripts
- **Flexible Dataset Input**: Supports JSON with multiple schemas

## Requirements

- Windows 10/11
- Python 3.12 (recommended for pre-built wheels)
- CUDA 13.0 toolkit
- NVIDIA GPU with 12GB+ VRAM
- LM Studio (optional, for dataset generation)

## Quick Start

### 1. Setup Virtual Environment

```batch
python -m venv .venv
```

### 2. Install Dependencies

```batch
install.bat
```

This will install:
- PyTorch (nightly with CUDA 13.0)
- Flash Attention 2 (pre-built wheel)
- SAGE Attention (pre-built wheel)
- Unsloth and core dependencies
- Optional: bitsandbytes, Triton Windows

### 3. Configure Settings

Edit `configs/default_config.yaml`:

```yaml
# Choose attention backend (one only)
attention_backend: "sageattention"  # or "flash_attention"

# Model selection - LFM 2.5 Thinking
model:
  name: "LiquidAI/LFM2.5-1.2B-Thinking"

# Quantization (16-bit is sufficient for LFM 2.5 with Flash/SAGE Attention)
quantization:
  load_in_16bit: true
  load_in_4bit: false  # Not needed for LFM 2.5 (small model)

# Export formats
export:
  export_lora: true
  export_merged: true
  export_gguf: false
```

### 4. Activate Virtual Environment and Run Pipeline

```batch
.venv\Scripts\activate
jupyter notebook notebooks\00_complete_pipeline.ipynb
```

Or use PowerShell:
```powershell
.venv\Scripts\Activate.ps1
jupyter notebook notebooks\00_complete_pipeline.ipynb
```

## Project Structure

```
LFM-Tuner/
├── notebooks/          # Jupyter notebooks for workflow
├── src/               # Core modules
│   ├── dataset_creation.py
│   ├── dataset_preparation.py
│   ├── training.py
│   ├── export.py
│   └── utils.py
├── configs/           # Configuration files
│   ├── default_config.yaml  # Main config (pre-set for LFM 2.5 Thinking)
│   └── model_configs.yaml  # Model-specific settings
├── install.bat        # Installation script (root folder)
├── data/              # Dataset storage
│   ├── raw/          # Initial datasets (place your dataset here)
│   ├── processed/    # Processed datasets
│   └── generated/    # LM Studio generated data
└── outputs/           # Training outputs and exports
```

## Configuration

### Attention Backend

Choose **one** attention backend in `configs/default_config.yaml`:

- **Flash Attention 2**: Better performance for Llama, Qwen, Mistral models
- **SAGE Attention**: Universal support, good memory efficiency

**No fallback** - system validates installation and errors if not available.

### Quantization

With Flash/SAGE Attention, **16-bit quantization is often sufficient** for smaller models (1-7B) on 12GB VRAM:

- **16-bit**: Default, no bitsandbytes needed
- **4-bit/8-bit**: Only if model doesn't fit or for larger models (requires bitsandbytes)

### Export Formats

Enable/disable each format in config:

```yaml
export:
  export_lora: true      # LoRA adapter only
  export_merged: true    # Merged 16-bit model
  export_gguf: true     # GGUF format (for local inference)
```

## Dataset Format for LFM 2.5 Thinking

Place your initial dataset in `data/raw/initial_dataset.json`. LFM 2.5 Thinking is optimized for reasoning tasks, so include step-by-step thinking:

**Instruction-Response Format (Recommended for Thinking Tasks):**
```json
[
  {
    "instruction": "Solve this step by step: What is 25 * 17?",
    "response": "Let me think through this:\n1. 25 * 10 = 250\n2. 25 * 7 = 175\n3. 250 + 175 = 425\n\nAnswer: 425"
  },
  {
    "instruction": "Explain why we need sleep, using reasoning.",
    "response": "Let me think through why sleep is important:\n1. During sleep, the brain consolidates memories\n2. The body repairs tissues\n3. The immune system strengthens\n4. Energy is restored\n\nTherefore, sleep is essential for health and well-being."
  }
]
```

**Chat Format:**
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi! How can I help?"}
    ]
  }
]
```

## LM Studio Integration

1. Start LM Studio
2. Enable API server (default: `http://localhost:1234`)
3. Load a model
4. Use `01_dataset_creation.ipynb` to generate synthetic data

## Memory Optimization for LFM 2.5

The pipeline is optimized for LFM 2.5 (1.2B model) on 12GB VRAM:

- **Batch size**: 2 (LFM 2.5 is small, allows larger batches)
- **Gradient accumulation**: 4 steps (effective batch size = 8)
- **Max sequence length**: 4096 (LFM supports long context)
- **Gradient checkpointing**: "unsloth" mode
- **LoRA rank**: 32 (higher rank for better performance on small models)
- **Memory usage**: ~6-8GB (plenty of headroom on 12GB GPU)

## Troubleshooting

### Flash Attention / SAGE Attention Not Found

Make sure you ran `install.bat` and the installation succeeded. Check your config matches the installed backend.

### CUDA Out of Memory

- Reduce `max_seq_length` in config
- Use 4-bit quantization (requires bitsandbytes)
- Reduce batch size or increase gradient accumulation

### LM Studio Connection Failed

- Ensure LM Studio is running
- Check API server is enabled (Settings → Server)
- Verify URL in config matches LM Studio port

## LFM 2.5 Thinking Specific Notes

- **Model Size**: 1.2B parameters (very efficient, fits easily in 12GB VRAM)
- **Context Length**: Supports up to 32,768 tokens (4k used for training)
- **Best For**: Reasoning tasks, step-by-step thinking, problem solving
- **Training Speed**: Very fast (small model trains quickly)
- **Memory Usage**: ~6-8GB (plenty of headroom)

## License

Apache 2.0

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning framework
- [Liquid AI](https://www.liquid.ai/) - LFM 2.5 Thinking model
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
- [SAGE Attention](https://github.com/woct0rdho/SageAttention) - Alternative attention mechanism
