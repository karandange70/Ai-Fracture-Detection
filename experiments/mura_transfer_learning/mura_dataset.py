import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class MURADataset(Dataset):
    """
    Stanford MURA-v1.1 Radiograph Dataset Loader.
    Provides study-level and image-level musculoskeletal radiograph access
    with patient-level segregation, grayscale conversion, and medical augmentations.
    """
    def __init__(self, mura_root=r"D:\MURA dataset\MURA-v1.1", split="train", 
                 body_parts=None, img_size=256, is_train=True):
        self.mura_root = mura_root
        self.split = split
        self.is_train = is_train
        self.img_size = img_size
        self.base_dir = os.path.dirname(mura_root) # "D:\MURA dataset"

        # Load image paths
        paths_csv = os.path.join(mura_root, f"{split}_image_paths.csv")
        studies_csv = os.path.join(mura_root, f"{split}_labeled_studies.csv")
        
        if not os.path.exists(paths_csv) or not os.path.exists(studies_csv):
            raise FileNotFoundError(f"MURA CSV files not found in {mura_root}")
            
        studies_df = pd.read_csv(studies_csv, header=None, names=["study_path", "label"])
        self.study_label_map = {row["study_path"].rstrip("/\\").replace("\\", "/"): int(row["label"]) 
                                for _, row in studies_df.iterrows()}
        
        raw_paths = pd.read_csv(paths_csv, header=None)[0].tolist()
        
        self.samples = []
        for p in raw_paths:
            p_norm = p.replace("\\", "/").strip()
            parts = p_norm.split("/")
            # Parts: ['MURA-v1.1', split, body_part, patient_id, study_dir, image_name]
            bp = parts[2]
            if body_parts and bp not in body_parts:
                continue
                
            study_key = "/".join(parts[:5])
            patient_id = parts[3]
            
            # Ground truth label
            if study_key in self.study_label_map:
                lbl = self.study_label_map[study_key]
            else:
                lbl = 1 if "positive" in parts[4].lower() else 0
                
            full_img_path = os.path.join(self.base_dir, p_norm.replace("/", os.sep))
            self.samples.append({
                "path": full_img_path,
                "label": float(lbl),
                "body_part": bp,
                "patient_id": patient_id,
                "study_key": study_key
            })
            
        print(f"[MURADataset] Loaded {len(self.samples):,} samples for split='{split}' (Body parts: {body_parts or 'ALL 7'})")
        
        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("L") # Enforce single-channel grayscale
        tensor = self.transform(img)
        
        return {
            "image": tensor,
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "body_part": item["body_part"],
            "patient_id": item["patient_id"],
            "path": item["path"]
        }

if __name__ == "__main__":
    ds = MURADataset(split="valid", img_size=256, is_train=False)
    sample = ds[0]
    print(f"Sample tensor shape: {sample['image'].shape}, label: {sample['label']}, body_part: {sample['body_part']}")
