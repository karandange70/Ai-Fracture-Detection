import os
import sys
import argparse
import time
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork
from src.dataset import BoneFractureDataset

def finetune_with_mura(
    mura_backbone_path=None,
    data_root=None,
    epochs=12,
    batch_size=4,
    lr=1e-4,
    dry_run=False
):
    if data_root is None:
        data_root = os.path.join(WORKSPACE_ROOT, "data")
        
    output_dir = os.path.join(WORKSPACE_ROOT, "experiments", "mura_transfer_learning")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute device for Tibia-Fibula Transfer Fine-Tuning: {device}")
    
    # 1. Initialize DualBoneFractureNetwork
    model = DualBoneFractureNetwork(pretrained=True, num_classes=1, num_subtypes=5)
    
    # 2. Transfer MURA pre-trained backbone if provided
    if mura_backbone_path and os.path.exists(mura_backbone_path):
        print(f"[*] Transferring MURA representation weights from: {mura_backbone_path}")
        mura_ckpt = torch.load(mura_backbone_path, map_location=device)
        sd = mura_ckpt.get("model_state_dict", mura_ckpt)
        
        # Transfer matching backbone layers (conv1, bn1, layer1, layer2, layer3, cbam, layer4)
        model_sd = model.state_dict()
        transferred = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k].copy_(v)
                transferred += 1
        print(f"[OK] Transferred {transferred} tensor weights from MURA backbone to DualBoneFractureNetwork.")
    else:
        print("[*] No custom MURA backbone path provided; starting from ImageNet adapted 1-ch ResNet-18.")
        
    model.to(device)
    
    # 3. Load Tibia-Fibula & Paired Dual-Bone Datasets
    train_ds = BoneFractureDataset(data_root=data_root, mode="train", is_train=True)
    test_ds = BoneFractureDataset(data_root=data_root, mode="test", is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"[*] Loaded {len(train_ds)} training samples and {len(test_ds)} evaluation samples.")
    
    # Layer-wise discriminative learning rate:
    # lower LR for pretrained backbone, higher for clinical attention heads
    optimizer = torch.optim.AdamW([
        {"params": model.conv1.parameters(), "lr": lr * 0.1},
        {"params": model.layer1.parameters(), "lr": lr * 0.1},
        {"params": model.layer2.parameters(), "lr": lr * 0.2},
        {"params": model.layer3.parameters(), "lr": lr * 0.5},
        {"params": model.cbam.parameters(), "lr": lr},
        {"params": model.layer4.parameters(), "lr": lr * 0.5},
        {"params": model.inter_bone_att.parameters(), "lr": lr},
        {"params": model.cross_view_att.parameters(), "lr": lr},
        {"params": model.shared_fc.parameters(), "lr": lr},
        {"params": model.detection_head.parameters(), "lr": lr},
        {"params": model.subtype_head.parameters(), "lr": lr}
    ], weight_decay=1e-4)
    
    criterion_det = nn.BCEWithLogitsLoss()
    criterion_sub = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    save_path = os.path.join(ckpt_dir, "best_mura_tibfib_model.pt")
    
    log_csv = os.path.join(log_dir, "tibfib_transfer_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_subtype_acc"])
        
    print(f"\n[*] Starting Fine-Tuning ({epochs} Epochs)...")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            img_ap = batch["image_ap"].to(device)
            img_lat = batch["image_lat"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            subtypes = batch["subtype"].to(device)
            
            optimizer.zero_grad()
            det_logits, sub_logits = model(img_ap, img_lat, return_subtypes=True)
            
            l_det = criterion_det(det_logits, labels)
            frac_mask = (labels.squeeze(1) == 1.0)
            if frac_mask.sum() > 0:
                l_sub = criterion_sub(sub_logits[frac_mask], subtypes[frac_mask])
            else:
                l_sub = torch.tensor(0.0, device=device)
                
            loss = l_det + 0.5 * l_sub
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img_ap.size(0)
            
            if dry_run:
                break
                
        train_loss /= len(train_ds) if not dry_run else batch_size
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        corr_det = 0
        corr_sub = 0
        total_frac = 0
        
        with torch.no_grad():
            for batch in test_loader:
                img_ap = batch["image_ap"].to(device)
                img_lat = batch["image_lat"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                subtypes = batch["subtype"].to(device)
                
                det_logits, sub_logits = model(img_ap, img_lat, return_subtypes=True)
                l_det = criterion_det(det_logits, labels)
                
                frac_mask = (labels.squeeze(1) == 1.0)
                if frac_mask.sum() > 0:
                    l_sub = criterion_sub(sub_logits[frac_mask], subtypes[frac_mask])
                    pred_subs = sub_logits[frac_mask].argmax(dim=-1)
                    corr_sub += (pred_subs == subtypes[frac_mask]).sum().item()
                    total_frac += frac_mask.sum().item()
                else:
                    l_sub = torch.tensor(0.0, device=device)
                    
                val_loss += (l_det + 0.5 * l_sub).item() * img_ap.size(0)
                preds = (torch.sigmoid(det_logits) >= 0.50).float()
                corr_det += (preds == labels).sum().item()
                
                if dry_run:
                    break
                    
        val_loss /= len(test_ds) if not dry_run else batch_size
        val_acc = (corr_det / len(test_ds)) * 100.0 if not dry_run else 100.0
        val_sub_acc = (corr_sub / max(1, total_frac)) * 100.0 if not dry_run else 100.0
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | Subtype Acc: {val_sub_acc:.1f}%")
        
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc, val_sub_acc])
            
        if val_loss < best_val_loss or dry_run:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_subtype_accuracy": val_sub_acc,
                "mura_transfer_applied": bool(mura_backbone_path)
            }, save_path)
            print(f"  -> [Saved Checkpoint: {save_path}]")
            
        if dry_run:
            print("[*] Dry-run completed successfully.")
            break

    print("-" * 80)
    print(f"[OK] Fine-tuning finished. Best model saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mura_backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    finetune_with_mura(
        mura_backbone_path=args.mura_backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run
    )
