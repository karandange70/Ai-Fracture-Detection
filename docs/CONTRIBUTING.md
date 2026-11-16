# Contributing Guidelines — BoneSense AI™

Thank you for your interest in contributing to **BoneSense AI**! We welcome contributions in computer vision, orthopedic biomechanics, web PACS interfaces, and documentation.

---

## 1. Code of Conduct
We are committed to providing a welcoming, inclusive, and professional environment for all contributors. Please maintain respectful, constructive dialogue across issues, discussions, and pull requests.

---

## 2. Development Workflow

### Step 1: Fork and Clone
Fork the repository on GitHub and clone your fork locally:
```bash
git clone https://github.com/<your-username>/Ai-Fracture-Detection.git
cd Ai-Fracture-Detection
```

### Step 2: Create a Feature Branch
Create a descriptive branch for your changes:
```bash
git checkout -b feat/add-new-metric
# or: git checkout -b fix/caliper-angulation-bug
```

### Step 3: Install Development Environment
Set up your virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Coding Standards & Guidelines

1. **Python Standards**: Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
2. **Preserve Comments & Docstrings**: Ensure all functions, classes, and modules have informative docstrings.
3. **Deterministic Seeding**: When writing training or evaluation scripts, ensure random seeds (`torch.manual_seed`, `np.random.seed`) are set for reproducibility.
4. **No Raw Secrets / PHI**: Never commit API keys, personal credentials, or patient health information.

---

## 4. Testing & Verification Before Submitting

Before submitting a Pull Request, verify your changes by running the test suite:

1. **Syntax & Import Verification**:
   ```bash
   python -c "import src.model, src.localization, src.anatomy_engine, src.app; print('All core modules valid!')"
   ```
2. **Run Model Verification**:
   ```bash
   python src/evaluate.py
   ```
3. **Run 6-Pillar Health Audit**:
   ```bash
   python src/autocheck.py
   ```

---

## 5. Commit Message Conventions

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(model): add multi-scale feature pyramid to crack localization head`
- `fix(calipers): resolve angulation arc division by zero edge case`
- `docs(api): update sample request schema for dual-view radiographs`
- `refactor(preprocess): optimize banner cutoff detection speed`
- `test(autocheck): add quantum mottle noise validation scenario`

---

## 6. Submitting a Pull Request (PR)

1. Push your branch to your GitHub fork:
   ```bash
   git push origin feat/your-feature-name
   ```
2. Open a Pull Request against the `main` branch of `karandange70/Ai-Fracture-Detection`.
3. Provide a clear summary of your changes, motivation, and test results in the PR description.

