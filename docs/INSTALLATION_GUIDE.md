# Installation & Setup Guide — BoneSense AI™

This guide walks through setting up **BoneSense AI** on Windows, Linux, or macOS.

---

## 1. System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel Core i5 / AMD Ryzen 5 or better).
- **RAM**: Minimum 8 GB (16 GB recommended for multi-image benchmarking).
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.8 / 12.x support (PyTorch GPU acceleration). CPU mode is fully supported out-of-the-box.
- **Disk Space**: At least 2 GB free disk space for models, datasets, and cache.

### Software Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python**: Version 3.10, 3.11, 3.12, or 3.13
- **Git & Git LFS**: Installed and added to system PATH

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/karandange70/Ai-Fracture-Detection.git
cd Ai-Fracture-Detection
```

### Step 2: Set Up a Python Virtual Environment
We strongly recommend using a dedicated virtual environment:

#### Windows (PowerShell / Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies
Install the pinned requirements using `pip`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` is pre-configured with CPU-optimized PyTorch wheels from `https://download.pytorch.org/whl/cpu`.
>
> If you have an NVIDIA GPU and want GPU acceleration, install the CUDA-enabled PyTorch build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Step 4: Environment Variables (Optional)
Copy `.env.example` to `.env` if you wish to customize port, host, or cache locations:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

---

### Step 5: Verify Model Checkpoints
Verify that the model weights are located in the `checkpoints/` directory:
- `checkpoints/best_fracatlas_model.pt` (Primary FracAtlas trained network)
- `checkpoints/anatomy_classifier.pt` (4-Class Anatomy classifier)
- `checkpoints/baseline_resnet18.pt` (Baseline ResNet-18)
- `checkpoints/baseline_densenet121.pt` (Baseline DenseNet-121)

---

### Step 6: Verify the Installation
Run a quick import and forward-pass test:
```bash
python -c "import torch, cv2, flask; from src.model import DualBoneFractureNetwork; net = DualBoneFractureNetwork(pretrained=False); print('BoneSense AI Installation Verified!')"
```

---

## 3. Running the Application

### Option A: Using the Windows Batch Script
Double-click `run_bonesense.bat` or run:
```cmd
run_bonesense.bat
```

### Option B: Direct Python Execution
```bash
python src/app.py
```

Once started, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

