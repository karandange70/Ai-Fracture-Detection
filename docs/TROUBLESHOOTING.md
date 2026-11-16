# Troubleshooting & FAQ — BoneSense AI™

This guide addresses common setup, runtime, and diagnostic issues encountered when operating **BoneSense AI**.

---

## 1. Installation & Dependency Issues

### Issue 1: `ModuleNotFoundError: No module named 'flask'` (or `cv2`, `torch`)
**Cause**: Dependencies are not installed in the active environment.  
**Solution**:
Activate your virtual environment and reinstall requirements:
```bash
# Activate virtual environment
source venv/bin/activate   # Linux / macOS
.\venv\Scripts\activate    # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue 2: Matplotlib Cache Directory Warning on Windows
**Symptom**: `C:\Users\username\.matplotlib is not a writable directory` warning printed at launch.  
**Cause**: The user home directory permissions prevent Matplotlib from writing font cache files.  
**Solution**: Set the `MPLCONFIGDIR` environment variable to a writable local path:
```powershell
# PowerShell
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
python src/app.py
```
Or define `MPLCONFIGDIR=.cache/matplotlib` in your `.env` file.

---

## 2. Server & Runtime Issues

### Issue 3: Port 5000 Already in Use
**Symptom**: `OSError: [Errno 48] Address already in use` or `[WinError 10048]`.  
**Cause**: Another instance of BoneSense or another background web server is already listening on port 5000.  
**Solution**:
1. Check which process is occupying port 5000:
   ```powershell
   # Windows PowerShell
   Get-NetTCPConnection -LocalPort 5000
   ```
2. Or change the port in `src/app.py` or run via environment variable:
   ```bash
   python -c "from src.app import app; app.run(port=5001)"
   ```

---

### Issue 4: `Checkpoint not found at checkpoints/best_fracatlas_model.pt`
**Cause**: Pre-trained model weights are missing or named incorrectly.  
**Solution**:
Ensure weights are present in `checkpoints/`:
- `best_fracatlas_model.pt` (53.8 MB)
- `anatomy_classifier.pt` (109 KB)
If you are setting up on a clean clone without Git LFS, check the GitHub Releases section to download pre-trained weights directly into `checkpoints/`.

---

### Issue 5: PyTorch CUDA Out-of-Memory (OOM) During Training
**Symptom**: `RuntimeError: CUDA out of memory`.  
**Solution**:
Reduce batch size and enable gradient accumulation:
```bash
python src/train.py --batch_size 4 --lr 1e-4
```
To force CPU execution:
```bash
python src/train.py --device cpu
```

---

## 3. GitHub & Version Control Issues

### Issue 6: Git Push Rejected — File Exceeds 100 MB Limit
**Symptom**:
`remote: error: File checkpoints/best_fracture_model.pt is 150.2 MB; this exceeds GitHub's file size limit of 100.00 MB`  
**Cause**: A file larger than 100 MB was added to a Git commit without Git LFS.  
**Solution**:
1. Track the large model using Git LFS:
   ```bash
   git lfs install
   git lfs track "checkpoints/*.pt"
   git add .gitattributes
   ```
2. Or exclude redundant backup checkpoints in `.gitignore`:
   ```gitignore
   checkpoints/backups/
   ```

