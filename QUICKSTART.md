# Quick Start Guide - LFM 2.5 Thinking

## Step-by-Step Setup for LFM 2.5 1.2B Thinking Model

### 1. Prerequisites
- Windows 10/11
- Python 3.12 (recommended)
- CUDA 13.0 toolkit installed
- NVIDIA GPU with 12GB+ VRAM

### 2. Setup Virtual Environment

```batch
python -m venv .venv
```

### 3. Install Dependencies

```batch
install.bat
```

This installs:
- PyTorch (CUDA 13.0)
- Flash Attention 2 (pre-built wheel)
- SAGE Attention (pre-built wheel)
- Unsloth and dependencies
- Optional: bitsandbytes, Triton

### 4. Configure

Edit `configs/default_config.yaml`:

```yaml
# Choose ONE attention backend
attention_backend: "sageattention"  # or "flash_attention"

# LFM 2.5 Thinking model (pre-configured)
model:
  name: "LiquidAI/LFM2.5-1.2B-Thinking"

# 16-bit is sufficient for LFM 2.5 (small model)
quantization:
  load_in_16bit: true
```

### 5. Prepare Dataset

Place your dataset in `data/raw/initial_dataset.json`:

```json
[
  {
    "instruction": "Your question here",
    "response": "Your answer here"
  }
]
```

See `data/raw/example_dataset.json` for format.

### 6. Run Pipeline

**Complete Pipeline**
```batch
REM Activate virtual environment
.venv\Scripts\activate

REM Start Jupyter
jupyter notebook notebooks\00_complete_pipeline.ipynb
```

Or in PowerShell:
```powershell
.venv\Scripts\Activate.ps1
jupyter notebook notebooks\00_complete_pipeline.ipynb
```

### 7. LM Studio (Optional)

For dataset augmentation:
1. Start LM Studio
2. Enable API server (Settings → Server)
3. Load a model
4. Use `01_dataset_creation.ipynb` to generate data

## Troubleshooting

**Attention backend not found?**
- Run `install.bat` again
- Check config matches installed backend

**CUDA Out of Memory?**
- Reduce `max_seq_length` in config
- Use 4-bit quantization (requires bitsandbytes)
- Reduce batch size

**LM Studio connection failed?**
- Ensure LM Studio is running
- Check API server is enabled
- Verify URL in config

## Next Steps

- Read `README.md` for detailed documentation
- Check `configs/model_configs.yaml` for model-specific settings
- Customize training parameters in `configs/default_config.yaml`
