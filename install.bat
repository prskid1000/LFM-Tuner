@echo off
setlocal
echo ========================================
echo Installing All Requirements
echo ========================================
echo.

REM Check if .venv exists
if not exist ".venv" (
    echo ERROR: .venv folder not found!
    echo Please create a virtual environment first by running: python -m venv .venv
    pause
    exit /b 1
)

REM Install core requirements
echo [0] Installing core requirements...
if exist "requirements.txt" (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if not %errorlevel% == 0 (
        echo ERROR: Failed to install core requirements
        pause
        exit /b 1
    )
    echo SUCCESS: Core requirements installed
) else (
    echo WARNING: requirements.txt not found, installing minimal requirements...
    .venv\Scripts\python.exe -m pip install unsloth pyyaml requests jupyter
    if not %errorlevel% == 0 (
        echo ERROR: Failed to install minimal requirements
        pause
        exit /b 1
    )
)
echo.

REM Install wheel first (required for flash-attn)
echo Installing wheel (required for flash-attn)...
.venv\Scripts\python.exe -m pip install --upgrade pip wheel
if not %errorlevel% == 0 (
    echo ERROR: Failed to install wheel
    pause
    exit /b 1
)
echo SUCCESS: wheel installed
echo.

REM Install PyTorch first
echo [1/6] Installing PyTorch (nightly with CUDA 13.0)...
.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
if not %errorlevel% == 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo SUCCESS: PyTorch installed
echo.

REM Install flash-attn (pre-built wheel for Python 3.12 + CUDA 13.0)
echo [2/6] Installing flash-attn (pre-built wheel for CUDA 13.0)...
echo NOTE: This is optional - only install if you want to use Flash Attention 2
echo You can skip this and use SAGE Attention instead
.venv\Scripts\python.exe -m pip install https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3+cu130torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
if not %errorlevel% == 0 (
    echo WARNING: Failed to install flash-attn
    echo You can use SAGE Attention instead (will be installed next)
) else (
    echo SUCCESS: flash-attn installed
)
echo.

REM Install sage-attn (pre-built wheel for Python 3.12 + CUDA 13.0)
echo [3/6] Installing sage-attn (pre-built wheel for CUDA 13.0)...
.venv\Scripts\python.exe -m pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
if not %errorlevel% == 0 (
    echo ERROR: Failed to install sage-attn
    pause
    exit /b 1
)
echo SUCCESS: sage-attn installed
echo.

REM Install Triton Windows (optional)
echo [4/6] Installing Triton Windows (optional)...
.venv\Scripts\python.exe -m pip install triton-windows
if not %errorlevel% == 0 (
    echo WARNING: Failed to install Triton (optional, continuing...)
) else (
    echo SUCCESS: Triton installed
)
echo.

REM Install bitsandbytes (optional, only if using 4-bit/8-bit quantization)
echo [6/6] Installing bitsandbytes (optional, for 4-bit/8-bit quantization)...
echo NOTE: With Flash/SAGE Attention and 16-bit quantization, bitsandbytes is not needed
echo You can skip this if you plan to use 16-bit quantization only
.venv\Scripts\python.exe -m pip install bitsandbytes
if not %errorlevel% == 0 (
    echo WARNING: Failed to install bitsandbytes
    echo You can still use 16-bit quantization without it
) else (
    echo SUCCESS: bitsandbytes installed
)
echo.

echo ========================================
echo Installation completed!
echo ========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment: .venv\Scripts\activate
echo 2. Configure your settings in: configs\default_config.yaml
echo 3. Choose attention backend: flash_attention or sageattention
echo 4. Start with notebook: notebooks\00_complete_pipeline.ipynb
echo.
