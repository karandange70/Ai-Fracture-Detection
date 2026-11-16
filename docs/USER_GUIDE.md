# User Guide — BoneSense AI™ PACS Station

Welcome to the **BoneSense AI Robotic Orthopedic PACS Station**. This guide details how to navigate the diagnostic interface, load clinical cases, interpret AI telemetry, and utilize interactive DICOM tools.

---

## 1. Navigating the PACS Interface

Upon navigating to `http://127.0.0.1:5000`, the station presents a futuristic dark-mode clinical dashboard structured into four key sections:
1. **Top Header**: Live system health, active compute device (`CPU` or `CUDA`), and model status.
2. **Left Sidebar (Sample Library & Upload)**: Curated clinical cases across 5 anatomical categories and a drag-and-drop file uploader.
3. **Center Viewport (3-Panel Diagnostic Canvas)**: Displays the Original Radiograph, Pinpoint Grad-CAM++ Saliency Overlay, and Orthopedic Calipers Panel.
4. **Right Panel (Clinical Telemetry & Metrics)**: Real-time confidence scores, 5-subtype probability breakdown, anatomical region analysis, and biomechanical surgical measurements.

---

## 2. Analyzing a Radiograph

### Method A: Selecting a Preloaded Clinical Sample
1. Expand any category in the left sidebar:
   - 🦵 *Lower Extremity / Leg & Dual Tib-Fib*
   - 🦴 *Shoulder Girdle & Clavicle*
   - ✋ *Upper Extremity / Hand & Wrist*
   - 🩻 *Pelvic Ring & Hip Joint*
2. Click on any study (e.g., *Study #08: Paired AP/LAT Tibia-Fibula Fracture* or *Study #0129: Normal Intact Lower Leg*).
3. The system automatically fetches, pre-processes, and analyzes the radiograph within ~200 ms.

### Method B: Uploading a Custom Patient Radiograph
1. In the **Upload Patient Radiograph** box, drag and drop any standard digital radiograph (`.png`, `.jpg`, `.jpeg`).
2. Alternatively, click **Browse Files** to select a radiograph from your local disk.
3. The application automatically normalizes dimensions to $512 \times 512$ with border-matched padding (preventing artificial contrast margins) and triggers inference immediately.

---

## 3. Interpreting AI Diagnostic Outputs

```
+---------------------------------------------------------------------------------------+
|  DIAGNOSIS: FRACTURE (98.4%) | Transverse / Oblique (Linear break)                   |
|  Surgeon: Tilt 45.0° | Gap 12.6 mm | Len 58.1 mm                                     |
|  Region: Lower Extremity [Diaphyseal Shaft - Cortical Alignment]                      |
+---------------------------------------------------------------------------------------+
```

### 1. Primary Diagnosis & Confidence
- **FRACTURE DETECTED (Red Banner)**: Indicates a confirmed cortical disruption. The confidence percentage reflects the sigmoid probability output from the detection head.
- **NORMAL / INTACT SKELETON (Green Banner)**: Indicates continuous, uninterrupted cortical margins.

### 2. Anatomy Recognition
- Displays the predicted anatomical compartment (**Shoulder**, **Leg**, **Hand**, or **Hip**) along with the classification confidence (up to 99.5%).

### 3. Fracture Subtype Classification
When a fracture is present, the system evaluates all 5 clinical subtypes and highlights the primary pattern:
- **Hairline / Fissure**: Subtle micro-cracks without cortical displacement.
- **Transverse / Oblique**: Direct horizontal or diagonal linear breaks across the shaft.
- **Comminuted**: Fragmentation into multiple bone pieces.
- **Greenstick**: Incomplete bending fracture common in pediatric cases.
- **Dislocation / Severe Displacement**: Complete loss of anatomical alignment.

### 4. Biomechanical Surgical Calipers
- **Angulation Arc ($\theta^\circ$)**: Angular tilt between the longitudinal axes of the proximal and distal fragments.
- **Displacement Gap ($\text{mm}$)**: Sub-millimeter physical separation distance between fractured margins.
- **Fragment Length ($\text{mm}$)**: Span of the detected fracture zone.

---

## 4. Interactive DICOM Viewing Tools

Located above the central diagnostic viewport:
- **Negative Invert (`INV`)**: Inverts pixel grayscale values, converting dense cortical bone into black and air into white (frequently preferred for spotting hairline fissures).
- **High-Contrast Bone Window (`CONT`)**: Applies localized CLAHE enhancement to accentuate fine trabecular bone patterns.
- **Edge Highlight (`EDGE`)**: Applies high-pass gradient filtering to trace cortical boundary continuities.
- **Full-Screen Expand (`EXP`)**: Expands the 3-panel composite image for high-resolution surgical theater review.

---

## 5. Exporting Diagnostic Findings

- Right-click on the center 3-panel composite canvas and select **Save Image As...** to download the comprehensive report including the original X-ray, Grad-CAM++ heatmap, and calibrated calipers.
- Telemetry values can be copied directly from the clinical summary card for inclusion in electronic health records (EHR / PACS).

