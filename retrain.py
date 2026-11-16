"""
========================================================================================
                  BONESENSE AI - UNIFIED DATASET RETRAINING PIPELINE
========================================================================================
Supports:
  1. FracAtlas Unified Model (Detection + Crack Localization Heatmap) -> checkpoints/best_fracatlas_model.pt
  2. Clinical Paired-View Model (Dual AP/Lateral + 5 Fracture Subtypes) -> checkpoints/best_fracture_model.pt
  3. Both Models sequentially with automated checkpoint backup & health verification
========================================================================================
"""

import os
import sys
import time
import shutil
import argparse
import torch

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_ROOT)

def backup_checkpoint(checkpoint_name):
    """Safely back up existing checkpoint before retraining overwrites it."""
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", checkpoint_name)
    if os.path.exists(ckpt_path):
        backup_dir = os.path.join(WORKSPACE_ROOT, "checkpoints", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.splitext(checkpoint_name)[0]}_{timestamp}.pt")
        shutil.copy2(ckpt_path, backup_path)
        print(f"[Backup] Successfully backed up existing {checkpoint_name} -> {backup_path}")
        return backup_path
    return None

def retrain_fracatlas_unified(epochs=6, batch_size=16, lr=1e-3):
    print("\n" + "=" * 80)
    print(" [1/2] RETRAINING FRACATLAS UNIFIED DETECTION & LOCALIZATION MODEL")
    print("=" * 80)
    
    backup_checkpoint("best_fracatlas_model.pt")
    
    from src.train_unified_loc import (
        FracAtlasUnifiedDataset,
        DualBoneFractureNetwork,
        compute_combined_loc_loss,
        DataLoader
    )
    import json
    import torch.nn.functional as F
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")
    
    anno_path = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared", "annotations.json")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"FracAtlas annotations not found at: {anno_path}")
        
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        
    data_root = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared")
    train_ds = FracAtlasUnifiedDataset(data_root, annotations, split="train")
    val_ds = FracAtlasUnifiedDataset(data_root, annotations, split="val")
    test_ds = FracAtlasUnifiedDataset(data_root, annotations, split="test")
    
    print(f"[*] Dataset: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_fracatlas_model.pt")
    
    if os.path.exists(ckpt_path):
        prev_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(prev_ckpt['model_state_dict'], strict=False)
        print(f"[*] Initialized from existing checkpoint: {ckpt_path}")
        
    model.to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.loc_head.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': model.cbam.parameters(), 'lr': lr * 0.2},
        {'params': model.inter_bone_att.parameters(), 'lr': lr * 0.2},
        {'params': model.layer4.parameters(), 'lr': lr * 0.05},
        {'params': model.shared_fc.parameters(), 'lr': lr * 0.1},
        {'params': model.detection_head.parameters(), 'lr': lr * 0.2},
        {'params': model.subtype_head.parameters(), 'lr': lr * 0.1}
    ])
    
    best_val_loss = float('inf')
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            targets = batch["target_map"].to(device)
            weights = batch["weight"].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            det_logits, sub_logits, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
            
            raw_cls = F.binary_cross_entropy_with_logits(det_logits, labels, reduction='none')
            l_cls = (raw_cls * weights).mean()
            l_loc = compute_combined_loc_loss(loc_heat, targets)
            
            loss = l_cls + 2.5 * l_loc
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                targets = batch["target_map"].to(device)
                weights = batch["weight"].to(device).unsqueeze(1)
                
                det_logits, _, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
                raw_cls = F.binary_cross_entropy_with_logits(det_logits, labels, reduction='none')
                l_cls = (raw_cls * weights).mean()
                l_loc = compute_combined_loc_loss(loc_heat, targets)
                val_loss += (l_cls + 2.5 * l_loc).item() * imgs.size(0)
                
        val_loss /= len(val_ds)
        elapsed = time.time() - t0
        
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "dataset": "FracAtlas_Unified_Calibrated",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(save_dict, ckpt_path)
            saved_tag = f" --> [Saved Checkpoint (Val Loss: {val_loss:.4f})]"
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{saved_tag}", flush=True)
        
    print(f"\n[DONE] Training complete. Evaluating best checkpoint on unseen Test Set ({len(test_ds)} samples)...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    errors = []
    frac_probs = []
    norm_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            filenames = batch["filename"]
            
            det_logits, _, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
            probs = torch.sigmoid(det_logits).squeeze().cpu().numpy()
            pred_heat = torch.sigmoid(loc_heat).cpu().numpy()
            
            for i in range(imgs.size(0)):
                p = float(probs[i]) if imgs.size(0) > 1 else float(probs)
                if labels[i] > 0.5:
                    frac_probs.append(p)
                    fn = filenames[i]
                    gt_boxes = annotations.get(fn, {}).get("bboxes_256", [])
                    if gt_boxes:
                        heat = pred_heat[i, 0]
                        py, px = np.unravel_index(np.argmax(heat), heat.shape)
                        pred_cx = px * 4.0
                        pred_cy = py * 4.0
                        
                        gt = gt_boxes[0]
                        gt_cx = (gt[0] + gt[2]) / 2.0
                        gt_cy = (gt[1] + gt[3]) / 2.0
                        dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
                        errors.append(dist)
                else:
                    norm_probs.append(p)
                    
    frac_recall = (np.array(frac_probs) >= 0.50).mean() * 100.0 if frac_probs else 0.0
    norm_acc = (np.array(norm_probs) < 0.50).mean() * 100.0 if norm_probs else 0.0
    overall_acc = ((np.array(frac_probs) >= 0.50).sum() + (np.array(norm_probs) < 0.50).sum()) / (len(frac_probs) + len(norm_probs)) * 100.0
    med_dist = float(np.median(errors)) if errors else 0.0
    
    print("-" * 65)
    print("           FRACATLAS TEST BENCHMARK RESULTS")
    print("-" * 65)
    print(f"Overall Accuracy       : {overall_acc:.1f}%")
    print(f"Fracture Sensitivity   : {frac_recall:.1f}%")
    print(f"Normal Specificity     : {norm_acc:.1f}%")
    if errors:
        print(f"Median Loc Distance    : {med_dist:.1f} px")
        print(f"Within 25px Accuracy   : {(np.array(errors) <= 25).mean()*100:.1f}%")
    print("-" * 65)
    
    # Save test metrics JSON
    metrics = {
        "dataset": "FracAtlas_Unified",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "overall_accuracy": round(overall_acc, 2),
        "fracture_sensitivity": round(frac_recall, 2),
        "normal_specificity": round(norm_acc, 2),
        "median_loc_distance_px": round(med_dist, 2),
        "training_time_min": round((time.time() - t0) / 60.0, 2),
        "best_epoch": checkpoint.get("epoch", epochs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = os.path.join(data_root, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
        
    print(f"[DONE] Saved test metrics to: {metrics_path}")
    print(f"[DONE] Saved model weights to: {ckpt_path}\n")
    return ckpt_path

def retrain_clinical_paired(epochs=10, batch_size=4, lr=3e-4):
    print("\n" + "=" * 80)
    print(" [2/2] RETRAINING CLINICAL PAIRED-VIEW (AP/LAT + SUBTYPES) MODEL")
    print("=" * 80)
    
    backup_checkpoint("best_fracture_model.pt")
    
    from src.train import train_model
    data_root = os.path.join(WORKSPACE_ROOT, "data")
    ckpt_dir = os.path.join(WORKSPACE_ROOT, "checkpoints")
    
    ckpt_path = train_model(
        data_root=data_root,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        checkpoint_dir=ckpt_dir
    )
    print(f"[DONE] Clinical Paired-View Retraining finished: {ckpt_path}\n")
    return ckpt_path

def main():
    parser = argparse.ArgumentParser(description="BoneSense AI Model Retraining Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "fracatlas", "paired", "fracatlas_cls"],
        help="Model pipeline to retrain: 'all', 'fracatlas' (unified), 'paired', or 'fracatlas_cls'"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("            BONESENSE AI - MODEL RETRAINING PIPELINE")
    print("#" * 80)
    print(f"[*] Workspace Root: {WORKSPACE_ROOT}")
    print(f"[*] Target Model(s): {args.model.upper()}")
    
    start_total = time.time()
    
    if args.model in ("all", "fracatlas"):
        ep = args.epochs or 6
        bs = args.batch_size or 16
        lr = args.lr or 1e-3
        retrain_fracatlas_unified(epochs=ep, batch_size=bs, lr=lr)
        
    if args.model in ("all", "paired"):
        ep = args.epochs or 10
        bs = args.batch_size or 4
        lr = args.lr or 3e-4
        retrain_clinical_paired(epochs=ep, batch_size=bs, lr=lr)
        
    if args.model == "fracatlas_cls":
        from src.train_fracatlas import train_fracatlas
        ep = args.epochs or 6
        bs = args.batch_size or 16
        lr = args.lr or 3e-4
        backup_checkpoint("best_fracatlas_model.pt")
        train_fracatlas(epochs=ep, batch_size=bs, learning_rate=lr)
        
    total_time = (time.time() - start_total) / 60.0
    print("#" * 80)
    print(f" [SUCCESS] Retraining Complete! Total Duration: {total_time:.2f} minutes")
    print(f" [SUCCESS] Checkpoints updated in: {os.path.join(WORKSPACE_ROOT, 'checkpoints')}")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
