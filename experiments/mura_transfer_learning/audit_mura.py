import os
import sys
import json
import csv
import time
from collections import defaultdict
from PIL import Image
import pandas as pd
import numpy as np

def audit_mura(mura_root=r"D:\MURA dataset\MURA-v1.1", output_dir=r"experiments\mura_transfer_learning"):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print(f"[*] Starting MURA Dataset Comprehensive Audit from: {mura_root}")
    
    # 1. Inspect CSV files
    train_paths_csv = os.path.join(mura_root, "train_image_paths.csv")
    train_studies_csv = os.path.join(mura_root, "train_labeled_studies.csv")
    valid_paths_csv = os.path.join(mura_root, "valid_image_paths.csv")
    valid_studies_csv = os.path.join(mura_root, "valid_labeled_studies.csv")

    train_paths = pd.read_csv(train_paths_csv, header=None)[0].tolist()
    valid_paths = pd.read_csv(valid_paths_csv, header=None)[0].tolist()
    
    train_studies_df = pd.read_csv(train_studies_csv, header=None, names=["study_path", "label"])
    valid_studies_df = pd.read_csv(valid_studies_csv, header=None, names=["study_path", "label"])
    
    study_label_map = {}
    for _, row in train_studies_df.iterrows():
        study_label_map[row["study_path"].rstrip("/\\")] = int(row["label"])
    for _, row in valid_studies_df.iterrows():
        study_label_map[row["study_path"].rstrip("/\\")] = int(row["label"])

    print(f"[*] Total recorded train images: {len(train_paths):,}")
    print(f"[*] Total recorded valid images: {len(valid_paths):,}")
    print(f"[*] Total recorded train studies: {len(train_studies_df):,}")
    print(f"[*] Total recorded valid studies: {len(valid_studies_df):,}")

    base_parent = os.path.dirname(mura_root) # "D:\MURA dataset"
    
    # Analyze splits
    splits_data = {
        "train": {"paths": train_paths, "studies_df": train_studies_df},
        "valid": {"paths": valid_paths, "studies_df": valid_studies_df}
    }
    
    audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mura_root": mura_root,
        "splits": {},
        "overall": {},
        "leakage_audit": {},
        "image_properties": {},
        "corrupted_images": [],
        "body_part_summary": []
    }
    
    all_patients = {"train": set(), "valid": set()}
    body_part_stats = defaultdict(lambda: {
        "train_images": 0, "train_normal_images": 0, "train_abnormal_images": 0,
        "train_studies": 0, "train_normal_studies": 0, "train_abnormal_studies": 0,
        "train_patients": set(),
        "valid_images": 0, "valid_normal_images": 0, "valid_abnormal_images": 0,
        "valid_studies": 0, "valid_normal_studies": 0, "valid_abnormal_studies": 0,
        "valid_patients": set(),
    })
    
    total_missing_files = 0
    checked_files = 0
    dims_sample = []
    modes_count = defaultdict(int)

    for split_name, split_info in splits_data.items():
        paths = split_info["paths"]
        print(f"[*] Auditing {split_name} split ({len(paths):,} images)...")
        
        split_img_count = 0
        split_normal_imgs = 0
        split_abnormal_imgs = 0
        
        for p in paths:
            # Normalized relative path
            # p looks like "MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png"
            parts = p.replace("\\", "/").split("/")
            # parts: ['MURA-v1.1', 'train', 'XR_SHOULDER', 'patient00001', 'study1_positive', 'image1.png']
            body_part = parts[2]
            patient_id = parts[3]
            study_folder = parts[4]
            study_key = "/".join(parts[:5])
            
            label = 1 if "positive" in study_folder else 0
            if study_key in study_label_map:
                label = study_label_map[study_key]
                
            all_patients[split_name].add(patient_id)
            body_part_stats[body_part][f"{split_name}_patients"].add(patient_id)
            
            split_img_count += 1
            if label == 1:
                split_abnormal_imgs += 1
                body_part_stats[body_part][f"{split_name}_abnormal_images"] += 1
            else:
                split_normal_imgs += 1
                body_part_stats[body_part][f"{split_name}_normal_images"] += 1
            body_part_stats[body_part][f"{split_name}_images"] += 1

            # Check existence on disk
            full_path = os.path.join(base_parent, p.replace("/", os.sep))
            checked_files += 1
            if not os.path.exists(full_path):
                total_missing_files += 1
                if total_missing_files <= 10:
                    print(f"[!] Missing file: {full_path}")
            else:
                # Sample 100 images per split for detailed resolution and mode check
                if len(dims_sample) < 300 and (checked_files % 130 == 0):
                    try:
                        with Image.open(full_path) as im:
                            w, h = im.size
                            mode = im.mode
                            dims_sample.append((w, h))
                            modes_count[mode] += 1
                    except Exception as e:
                        audit_results["corrupted_images"].append({"path": full_path, "error": str(e)})

        # Count studies
        studies_df = split_info["studies_df"]
        for _, row in studies_df.iterrows():
            st_path = row["study_path"].replace("\\", "/").rstrip("/")
            st_parts = st_path.split("/")
            bp = st_parts[2]
            st_lbl = int(row["label"])
            body_part_stats[bp][f"{split_name}_studies"] += 1
            if st_lbl == 1:
                body_part_stats[bp][f"{split_name}_abnormal_studies"] += 1
            else:
                body_part_stats[bp][f"{split_name}_normal_studies"] += 1

        audit_results["splits"][split_name] = {
            "total_images": split_img_count,
            "normal_images": split_normal_imgs,
            "abnormal_images": split_abnormal_imgs,
            "abnormal_ratio": round(split_abnormal_imgs / max(1, split_img_count), 4),
            "total_studies": len(studies_df),
            "total_patients": len(all_patients[split_name])
        }

    # Leakage check: check if any patient is in both train and valid
    patient_overlap = all_patients["train"].intersection(all_patients["valid"])
    audit_results["leakage_audit"] = {
        "train_patients_count": len(all_patients["train"]),
        "valid_patients_count": len(all_patients["valid"]),
        "patient_overlap_count": len(patient_overlap),
        "data_leakage_detected": len(patient_overlap) > 0,
        "overlapping_patients": list(patient_overlap)[:10]
    }
    print(f"[*] Patient-level data leakage audit: {len(patient_overlap)} overlapping patients (PASS: {len(patient_overlap) == 0})")

    # Dimensions statistics
    if dims_sample:
        ws = [d[0] for d in dims_sample]
        hs = [d[1] for d in dims_sample]
        audit_results["image_properties"] = {
            "sample_size": len(dims_sample),
            "width_min": int(np.min(ws)),
            "width_max": int(np.max(ws)),
            "width_mean": round(float(np.mean(ws)), 1),
            "height_min": int(np.min(hs)),
            "height_max": int(np.max(hs)),
            "height_mean": round(float(np.mean(hs)), 1),
            "modes_detected": dict(modes_count)
        }

    audit_results["overall"] = {
        "total_images_in_csv": len(train_paths) + len(valid_paths),
        "total_missing_files": total_missing_files,
        "corrupted_images_detected": len(audit_results["corrupted_images"]),
        "total_studies": len(train_studies_df) + len(valid_studies_df),
        "total_unique_patients": len(all_patients["train"].union(all_patients["valid"]))
    }

    # Prepare CSV summary
    csv_rows = []
    for bp in sorted(body_part_stats.keys()):
        stat = body_part_stats[bp]
        row = {
            "body_part": bp,
            "train_images": stat["train_images"],
            "train_normal_images": stat["train_normal_images"],
            "train_abnormal_images": stat["train_abnormal_images"],
            "train_abnormal_pct": round(stat["train_abnormal_images"] / max(1, stat["train_images"]) * 100, 2),
            "train_studies": stat["train_studies"],
            "train_patients": len(stat["train_patients"]),
            "valid_images": stat["valid_images"],
            "valid_normal_images": stat["valid_normal_images"],
            "valid_abnormal_images": stat["valid_abnormal_images"],
            "valid_abnormal_pct": round(stat["valid_abnormal_images"] / max(1, stat["valid_images"]) * 100, 2),
            "valid_studies": stat["valid_studies"],
            "valid_patients": len(stat["valid_patients"]),
            "total_images": stat["train_images"] + stat["valid_images"],
            "total_studies": stat["train_studies"] + stat["valid_studies"]
        }
        csv_rows.append(row)
        
        # Convert sets to count for JSON serialization
        stat_copy = {k: (len(v) if isinstance(v, set) else v) for k, v in stat.items()}
        stat_copy["body_part"] = bp
        audit_results["body_part_summary"].append(stat_copy)

    summary_csv_path = os.path.join(output_dir, "mura_dataset_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_json_path = os.path.join(output_dir, "mura_dataset_audit.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"[OK] MURA Audit Complete in {elapsed:.2f} seconds!")
    print(f"    - JSON report saved: {summary_json_path}")
    print(f"    - CSV summary saved: {summary_csv_path}")
    print(f"    - Total images verified: {audit_results['overall']['total_images_in_csv']}")
    print(f"    - Missing files: {total_missing_files}")
    print(f"    - Corrupted images: {len(audit_results['corrupted_images'])}")

if __name__ == "__main__":
    audit_mura()
