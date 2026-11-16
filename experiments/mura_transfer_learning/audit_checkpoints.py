import os
import sys
import json
import torch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork, CrackLocalizationHead

def audit_checkpoints():
    ckpt_dir = os.path.join(WORKSPACE_ROOT, "checkpoints")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    
    results = {}
    print(f"[*] Auditing {len(ckpts)} checkpoints in {ckpt_dir}...")
    
    for c in sorted(ckpts):
        cp_path = os.path.join(ckpt_dir, c)
        size_mb = round(os.path.getsize(cp_path) / (1024 * 1024), 2)
        try:
            data = torch.load(cp_path, map_location="cpu")
            if isinstance(data, dict) and "model_state_dict" in data:
                top_keys = list(data.keys())
                has_state_dict = True
                state_dict = data["model_state_dict"]
                param_count = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                num_tensors = len(state_dict)
                def clean_val(v):
                    if hasattr(v, "numel") and v.numel() == 1:
                        return v.item()
                    elif isinstance(v, (int, float, str, bool, list, dict)):
                        return v
                    return str(v)
                meta_info = {k: clean_val(v) for k, v in data.items() if k not in ["model_state_dict", "optimizer_state_dict"]}
            elif isinstance(data, dict):
                top_keys = ["raw_state_dict"]
                has_state_dict = False
                state_dict = data
                param_count = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                num_tensors = len(state_dict)
                meta_info = {"type": "raw_state_dict"}
            else:
                top_keys = ["model_object"]
                has_state_dict = False
                state_dict = data.state_dict() if hasattr(data, "state_dict") else {}
                param_count = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                num_tensors = len(state_dict)
                meta_info = {"type": "raw_object"}
                
            # Check DualBoneFractureNetwork compatibility
            dbfn = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
            dbfn_keys = set(dbfn.state_dict().keys())
            sd_keys = set(state_dict.keys())
            
            missing_in_sd = list(dbfn_keys - sd_keys)
            unexpected_in_sd = list(sd_keys - dbfn_keys)
            exact_match = (len(missing_in_sd) == 0 and len(unexpected_in_sd) == 0)
            backbone_match = all(k in sd_keys for k in dbfn_keys if k.startswith("layer") or k.startswith("conv1"))
            
            results[c] = {
                "file_size_mb": size_mb,
                "top_level_keys": top_keys,
                "metadata": meta_info,
                "num_tensor_weights": num_tensors,
                "total_parameters": param_count,
                "matches_dual_bone_network_exact": exact_match,
                "matches_backbone": backbone_match,
                "missing_keys_count": len(missing_in_sd),
                "unexpected_keys_count": len(unexpected_in_sd),
                "sample_missing_keys": missing_in_sd[:5],
                "sample_unexpected_keys": unexpected_in_sd[:5]
            }
            print(f"[+] {c} ({size_mb} MB) -> tensors: {num_tensors}, params: {param_count:,}, exact: {exact_match}, backbone: {backbone_match}")
        except Exception as e:
            results[c] = {"error": str(e), "file_size_mb": size_mb}
            print(f"[-] {c} ERROR: {e}")

    out_json = os.path.join(WORKSPACE_ROOT, "experiments", "mura_transfer_learning", "checkpoints_audit.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Checkpoints audit saved to: {out_json}")

if __name__ == "__main__":
    audit_checkpoints()
