import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from src.dataset import BoneFractureDataset
from src.model import DualBoneFractureNetwork

def train_model(
    data_root=None,
    num_epochs=12,
    batch_size=4,
    learning_rate=3e-4,
    checkpoint_dir=None
):
    if data_root is None:
        data_root = os.path.join(WORKSPACE_ROOT, "data")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(WORKSPACE_ROOT, "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # Load dataset splits (Balanced Fracture + Normal)
    train_dataset = BoneFractureDataset(data_root=data_root, mode="train", is_train=True)
    test_dataset = BoneFractureDataset(data_root=data_root, mode="test", is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Loaded {len(train_dataset)} Training samples ({len(train_loader)} batches/epoch)")
    print(f"Loaded {len(test_dataset)} Independent Test samples for validation")
    
    # Initialize model with Dual-View Cross-Attention & Multi-Task Heads
    model = DualBoneFractureNetwork(pretrained=True, num_classes=1, num_subtypes=5).to(device)
    
    criterion_det = nn.BCEWithLogitsLoss()
    criterion_sub = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "best_fracture_model.pt")
    
    print("\nStarting Dual-View Cross-Attention & Multi-Task Subtype Training...")
    print("-" * 85)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            img_ap = batch["image_ap"].to(device)
            img_lat = batch["image_lat"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            subtypes = batch["subtype"].to(device)
            
            optimizer.zero_grad()
            det_logits, sub_logits = model(img_ap, img_lat, return_subtypes=True)
            
            loss_det = criterion_det(det_logits, labels)
            
            # Subtype loss computed on fracture instances
            frac_mask = (labels.squeeze(1) == 1.0)
            if frac_mask.sum() > 0:
                loss_sub = criterion_sub(sub_logits[frac_mask], subtypes[frac_mask])
            else:
                loss_sub = torch.tensor(0.0, device=device)
                
            loss = loss_det + 0.5 * loss_sub
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img_ap.size(0)
            
        epoch_train_loss = train_loss / len(train_dataset)
        scheduler.step()
        
        # Validation on Independent Test Set
        model.eval()
        val_loss = 0.0
        correct_det = 0
        correct_sub = 0
        total_frac = 0
        
        with torch.no_grad():
            for batch in test_loader:
                img_ap = batch["image_ap"].to(device)
                img_lat = batch["image_lat"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                subtypes = batch["subtype"].to(device)
                
                det_logits, sub_logits = model(img_ap, img_lat, return_subtypes=True)
                loss_det = criterion_det(det_logits, labels)
                
                frac_mask = (labels.squeeze(1) == 1.0)
                if frac_mask.sum() > 0:
                    loss_sub = criterion_sub(sub_logits[frac_mask], subtypes[frac_mask])
                    pred_subs = sub_logits[frac_mask].argmax(dim=-1)
                    correct_sub += (pred_subs == subtypes[frac_mask]).sum().item()
                    total_frac += frac_mask.sum().item()
                else:
                    loss_sub = torch.tensor(0.0, device=device)
                    
                batch_loss = loss_det + 0.5 * loss_sub
                val_loss += batch_loss.item() * img_ap.size(0)
                
                preds = (torch.sigmoid(det_logits) >= 0.50).float()
                correct_det += (preds == labels).sum().item()
                
        epoch_val_loss = val_loss / len(test_dataset)
        val_acc = (correct_det / len(test_dataset)) * 100.0
        val_sub_acc = (correct_sub / max(1, total_frac)) * 100.0
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_train_loss,
                "val_loss": best_val_loss,
                "val_accuracy": val_acc,
                "val_subtype_accuracy": val_sub_acc
            }, checkpoint_path)
            saved_str = f"[Checkpoint Saved -> Best Val Loss: {best_val_loss:.4f}]"
        else:
            saved_str = ""
            
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.1f}% | Subtype Acc: {val_sub_acc:.0f}% {saved_str}")
        
    print("-" * 85)
    print(f"Training Complete! Best model saved to: {checkpoint_path}\n")
    return checkpoint_path

if __name__ == "__main__":
    train_model(num_epochs=12, batch_size=4)

