# Changelog

All notable changes to the **BoneSense AI** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-10-26

### Added
- **Specialized 4-Class Anatomy Classifier**: Automatically identifies Hand, Leg, Shoulder, and Hip radiographs with 96.5% accuracy (`src/anatomy_engine.py`).
- **Biomechanical Orthopedic Calipers**: Measures fissure gap (in mm), angulation arc (in degrees), and fragment length (`src/localization.py`).
- **Dual-View Cross-Attention**: Fuses orthogonal AP (Anteroposterior) and Lateral projections for comprehensive 3D fracture detection.
- **Dedicated Crack Localization Head**: High-resolution multi-scale decoder fusing Layer 3 and Layer 4 feature maps for continuous crack heatmaps.
- **FracAtlas Medical Benchmark Integration**: Prepared dataset pipeline for multi-class musculoskeletal radiographs (`src/prepare_fracatlas.py`).
- **Stanford MURA Transfer Learning**: Backbone transfer learning and domain adaptation pipelines (`experiments/mura_transfer_learning`).
- **Autonomous 6-Pillar Clinical ML Audit**: Stress testing across quantum mottle noise, contrast leveling, underexposure, and motion blur (`src/autocheck.py`).
- **Futuristic Robotic PACS Workstation UI**: Modern dark-mode interface with DICOM filters (Bone Window, Negative Invert, Edge Detect), typewriter telemetry, and interactive sample browser.

### Changed
- Replaced standard ResNet classification head with multi-task heads for binary detection, 5 clinical fracture subtypes, and localization heatmaps.
- Optimized radiograph preprocessing to strip white text banners and inpaint peripheral orientation markers ('L' / 'R').

---

## [1.5.0] - 2026-09-20

### Added
- **CBAM (Convolutional Block Attention Module)**: Channel and spatial attention integrated into Layer 3 of the backbone.
- **Inter-Bone Relational Attention**: Scaled dot-product attention to model parallel bones (Tibia/Fibula, Radius/Ulna) and suppress interosseous space artifacts.
- **Grad-CAM++ Explainability Engine**: Anatomically masked saliency visualization.

---

## [1.0.0] - 2026-09-09

### Added
- Initial project release with ResNet-18, MobileNet-V2, and DenseNet-121 baseline models.
- Core radiograph preprocessing and dataset splitting utilities.
- Flask prototype interface for X-ray inference.

