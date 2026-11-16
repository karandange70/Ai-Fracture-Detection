import os
import sys
import argparse
import time
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_ROOT)

from experiments.mura_transfer_learning.mura_dataset import MURADataset
from src.spatial_modules import CBAM

class MURABackboneNetwork(nn.Module):
    """
    Representation learning network for MURA radiographs.
    Uses the exact ResNet18 + CBAM architecture of DualBoneFractureNetwork
    so that learned convolutional filters directly transfer into BoneSense AI.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        
        # 1-channel grayscale conv1
        orig_conv1 = base.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
                
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1 # 64 ch
        self.layer2 = base.layer2 # 128 ch
        self.layer3 = base.layer3 # 256 ch
        self.cbam = CBAM(in_planes=256, reduction_ratio=16, kernel_size=7)
        self.layer4 = base.layer4 # 512 ch
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        f3 = self.layer3(x)
        f3 = self.cbam(f3)
        
        f4 = self.layer4(f3)
        
        pooled = self.avgpool(f4)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

def train_mura(
    mura_root=r"D:\MURA dataset\MURA-v1.1",
    epochs=10,
    batch_size=16,
    lr=1e-4,
    body_parts=None,
    dry_run=False,
    resume_path=None
):
    output_dir = os.path.join(WORKSPACE_ROOT, "experiments", "mura_transfer_learning")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        
    # Dataset loaders
    train_ds = MURADataset(mura_root=mura_root, split="train", body_parts=body_parts, img_size=256, is_train=True)
    valid_ds = MURADataset(mura_root=mura_root, split="valid", body_parts=body_parts, img_size=256, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    
    model = MURABackboneNetwork(pretrained=True).to(device)
    
    start_epoch = 1
    best_val_auc = 0.0
    
    if resume_path and os.path.exists(resume_path):
        print(f"[*] Resuming from checkpoint: {resume_path}")
        ckpt_data = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_val_auc = ckpt_data.get("val_auc", 0.0)

    # Calculate class weighting for BCEWithLogitsLoss
    pos_weight = torch.tensor([1.2], device=device) # Based on MURA ~40% positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    log_csv = os.path.join(log_dir, "mura_pretrain_history.csv")
    csv_header = ["epoch", "train_loss", "val_loss", "val_auc", "val_acc", "val_f1", "val_sens", "val_spec", "duration_sec"]
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    print(f"\n[*] Starting MURA Representation Learning ({epochs} Epochs, Batch Size: {batch_size})...")
    print("-" * 85)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        batch_idx = 0
        
        for batch in train_loader:
            batch_idx += 1
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            
            if batch_idx % 250 == 0:
                print(f"  -> Batch [{batch_idx:04d}/{len(train_loader):04d}] | Step Loss: {loss.item():.4f}", flush=True)
                
            if dry_run and batch_idx >= 3:
                print(f"[*] Dry run mode: stopped after {batch_idx} batches.", flush=True)
                break
                
        scheduler.step()
        train_loss /= (batch_idx * batch_size if dry_run else len(train_ds))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        val_batch_idx = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                val_batch_idx += 1
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.50).astype(float)
                targets = labels.cpu().numpy().flatten()
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(targets)
                
                if dry_run and val_batch_idx >= 3:
                    break
                    
        val_loss /= (val_batch_idx * batch_size if dry_run else len(valid_ds))
        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        val_preds = np.array(val_preds)
        
        try:
            val_auc = roc_auc_score(val_targets, val_probs) if len(np.unique(val_targets)) > 1 else 0.5
        except Exception:
            val_auc = 0.5
            
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_sens = recall_score(val_targets, val_preds, zero_division=0)
        # Specificity
        neg_mask = (val_targets == 0)
        val_spec = (val_preds[neg_mask] == 0).mean() if np.sum(neg_mask) > 0 else 0.0
        
        dur = round(time.time() - t0, 2)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.1f}% | Sens: {val_sens*100:.1f}% | Spec: {val_spec*100:.1f}% | Time: {dur}s", flush=True)

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_auc, val_acc, val_f1, val_sens, val_spec, dur])

        # Save checkpoint
        is_best = val_auc > best_val_auc
        if is_best or dry_run:
            best_val_auc = val_auc
            save_path = os.path.join(ckpt_dir, "best_mura_backbone.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_accuracy": val_acc,
                "val_loss": val_loss
            }, save_path)
            print(f"  -> [Saved Best MURA Backbone: {save_path}]", flush=True)
            
        if dry_run:
            print("[*] Dry run completed successfully! Architecture, DataLoader, AMP, and Memory verified.", flush=True)
            break

    print("-" * 85, flush=True)
    print(f"[*] MURA Training Completed. Artifacts saved in {ckpt_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--body_parts", type=str, default=None, help="Comma separated body parts e.g. XR_FOREARM,XR_WRIST")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    bp_list = [p.strip() for p in args.body_parts.split(",")] if args.body_parts else None
    train_mura(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        body_parts=bp_list,
        dry_run=args.dry_run,
        resume_path=args.resume
    )
