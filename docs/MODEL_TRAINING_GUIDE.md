# Model Training & Fine-Tuning Guide — BoneSense AI™

This guide details how to train, fine-tune, and evaluate the **BoneSense AI** neural networks on custom or public radiograph datasets.

---

## 1. Available Training Scripts

| Script | Purpose |
| :--- | :--- |
| [`src/train.py`](file:///c:/Users/karan/OneDrive/Music/Desktop/ai%20frcture%20anti/src/train.py) | Standard training script for binary fracture detection with BCE and Cosine Annealing. |
| [`src/train_fracatlas.py`](file:///c:/Users/karan/OneDrive/Music/Desktop/ai%20frcture%20anti/src/train_fracatlas.py) | Trains `DualBoneFractureNetwork` on the multi-class FracAtlas benchmark. |
| [`src/train_unified_loc.py`](file:///c:/Users/karan/OneDrive/Music/Desktop/ai%20frcture%20anti/src/train_unified_loc.py) | Joint multi-task training: Binary Detection + Subtype Classification + Crack Localization Head. |
| [`retrain.py`](file:///c:/Users/karan/OneDrive/Music/Desktop/ai%20frcture%20anti/retrain.py) | Standalone automated retraining script with dataset validation and checkpoint backup. |
| [`src/anatomy_engine.py`](file:///c:/Users/karan/OneDrive/Music/Desktop/ai%20frcture%20anti/src/anatomy_engine.py) | Trains the specialized 4-class Anatomy Classifier (Shoulder, Leg, Hand, Hip). |

---

## 2. Preparing Data Before Training

Ensure your dataset is organized under `data/` or provide a custom directory:
```
data/dataset_splits/
├── train/
│   ├── fracture/
│   └── normal/
└── test/
    ├── fracture/
    └── normal/
```
If starting from unorganized raw radiographs, use the split preparation utility:
```bash
python src/prepare_splits.py --data_dir data/cleaned/unified_512 --split_ratio 0.8
```

---

## 3. Training the Fracture Detection Network

### Option A: Standard Training (`src/train.py`)
```bash
python src/train.py \
  --epochs 15 \
  --batch_size 8 \
  --lr 0.0003 \
  --weight_decay 0.0001
```

### Option B: Unified Multi-Task & Localization Training (`src/train_unified_loc.py`)
```bash
python src/train_unified_loc.py \
  --epochs 20 \
  --batch_size 4 \
  --lr 0.0001 \
  --loc_weight 0.8 \
  --subtype_weight 0.5
```

### Option C: Using the Windows Batch Retrain Script
```cmd
run_retrain.bat
```

---

## 4. Hyperparameter Recommendations

| Parameter | Recommended Value | Description |
| :--- | :---: | :--- |
| **Learning Rate** | `1e-4` to `3e-4` | Best paired with AdamW optimizer and Cosine Annealing scheduler. |
| **Batch Size** | `4` to `16` | Larger batch sizes require sufficient GPU VRAM or system RAM. |
| **Weight Decay** | `1e-4` | Prevents overfitting on subtle trabecular bone patterns. |
| **Input Resolution** | $512 \times 512$ | Native resolution preserving sub-millimeter fissures. |
| **Dropout Rate** | `0.30` | Applied in shared projection layer. |

---

## 5. Training the Anatomy Classifier

To retrain the 4-class anatomy recognition engine (`checkpoints/anatomy_classifier.pt`):
```bash
python -c "
from src.anatomy_engine import train_and_save_anatomy_classifier
from src.model import DualBoneFractureNetwork
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = DualBoneFractureNetwork(pretrained=False)
train_and_save_anatomy_classifier('data/fracatlas_prepared', backbone, device=device)
"
```

---

## 6. Evaluating Trained Models

### Running the Clinical Verification Suite
```bash
python src/evaluate.py
```

### Running the Multi-Model Comparison Benchmark
To generate updated benchmark tables and comparison charts (`data/model_comparison_results.png`):
```bash
python src/model_comparison.py
```

### Running the Autonomous 6-Pillar ML Audit
To stress-test model calibration, noise robustness, and latency:
```bash
python src/autocheck.py
```

