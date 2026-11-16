import os
import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork
from src.dataset import BoneFractureDataset

def run_evaluation(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    sub_true = []
    sub_pred = []
    filenames = []
    
    with torch.no_grad():
        for batch in data_loader:
            img_ap = batch["image_ap"].to(device)
            img_lat = batch["image_lat"].to(device)
            labels = batch["label"].to(device)
            subtypes = batch["subtype"].to(device)
            fnames = batch["filename"]
            
            det_logits, sub_logits = model(img_ap, img_lat, return_subtypes=True)
            probs = torch.sigmoid(det_logits).squeeze().cpu().numpy()
            preds = (probs >= 0.50).astype(int)
            targets = labels.cpu().numpy().astype(int)
            
            if np.ndim(probs) == 0:
                probs = [float(probs)]
                preds = [int(preds)]
                targets = [int(targets)]
            else:
                probs = probs.tolist()
                preds = preds.tolist()
                targets = targets.tolist()
                
            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(targets)
            filenames.extend(fnames)
            
            # Subtype accuracy on positive cases
            for i in range(len(targets)):
                if targets[i] == 1:
                    sub_true.append(int(subtypes[i].item()))
                    pred_sub = int(sub_logits[i].argmax().item())
                    sub_pred.append(pred_sub)
                    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    neg_mask = (y_true == 0)
    spec = float((y_pred[neg_mask] == 0).mean()) if np.sum(neg_mask) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        auc = 1.0
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    sub_acc = accuracy_score(sub_true, sub_pred) if len(sub_true) > 0 else 0.0
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auc), 4),
        "subtype_accuracy": round(float(sub_acc), 4),
        "confusion_matrix": cm,
        "sample_count": len(y_true),
        "fracture_count": int(np.sum(y_true == 1)),
        "normal_count": int(np.sum(y_true == 0))
    }

def compare_models(
    original_ckpt_path=r"checkpoints\best_fracture_model.pt",
    experimental_ckpt_path=r"experiments\mura_transfer_learning\checkpoints\best_mura_tibfib_model.pt",
    data_root=r"data"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Comparing models on device: {device}")
    
    test_ds = BoneFractureDataset(data_root=os.path.join(WORKSPACE_ROOT, data_root), mode="test", is_train=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    results = {}
    
    # 1. Evaluate Original Model
    orig_full_path = os.path.join(WORKSPACE_ROOT, original_ckpt_path)
    if os.path.exists(orig_full_path):
        print(f"[*] Evaluating Original Model: {orig_full_path}")
        model_orig = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
        ckpt_orig = torch.load(orig_full_path, map_location=device)
        model_orig.load_state_dict(ckpt_orig["model_state_dict"], strict=False)
        model_orig.to(device)
        results["original_model"] = run_evaluation(model_orig, test_loader, device)
        results["original_model"]["checkpoint"] = original_ckpt_path
    else:
        print(f"[-] Original checkpoint not found at: {orig_full_path}")
        
    # 2. Evaluate Experimental Model
    exp_full_path = os.path.join(WORKSPACE_ROOT, experimental_ckpt_path)
    if os.path.exists(exp_full_path):
        print(f"[*] Evaluating Experimental Model: {exp_full_path}")
        model_exp = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
        ckpt_exp = torch.load(exp_full_path, map_location=device)
        model_exp.load_state_dict(ckpt_exp["model_state_dict"], strict=False)
        model_exp.to(device)
        results["experimental_model"] = run_evaluation(model_exp, test_loader, device)
        results["experimental_model"]["checkpoint"] = experimental_ckpt_path
    else:
        print(f"[-] Experimental checkpoint not found at: {exp_full_path}")
        
    out_json = os.path.join(WORKSPACE_ROOT, "experiments", "mura_transfer_learning", "model_comparison_mura.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"[OK] Evaluation comparison saved to: {out_json}")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    compare_models()
