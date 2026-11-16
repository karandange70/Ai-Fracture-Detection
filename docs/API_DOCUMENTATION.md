# API Documentation — BoneSense AI™

The BoneSense AI web service provides a RESTful HTTP API for programmatic access to the diagnostic engine.

**Base URL**: `http://127.0.0.1:5000`

---

## 1. Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main PACS workstation single-page HTML interface. |
| `GET` | `/api/samples` | Retrieves the categorized directory of preloaded clinical sample studies. |
| `GET` | `/api/model_info` | Returns runtime architecture metadata, parameter counts, and active compute device. |
| `POST` | `/api/analyze` | Executes multi-task inference, anatomy classification, and caliper measurement on a given radiograph. |

---

## 2. Endpoint Details

### 2.1 Get Model Information
Returns runtime metadata regarding the active neural network backbone and spatial modules.

- **URL**: `/api/model_info`
- **Method**: `GET`
- **Response**: `200 OK` (application/json)

```json
{
  "architecture": "DualBoneFractureNetwork (Dual-View 3D Cross-Attention)",
  "backbone": "Pretrained ResNet-18 (Grayscale Adapted)",
  "spatial_modules": [
    "CBAM 2D Spatial Attention",
    "Inter-Bone Relational Attention",
    "Dual-View Cross-Attention",
    "CrackLocalizationHead"
  ],
  "parameters": "12,952,235",
  "size_mb": 49.41,
  "subtypes": [
    "Hairline / Fissure (Minor crack)",
    "Transverse / Oblique (Linear break)",
    "Comminuted (Multi-fragment)",
    "Greenstick (Incomplete bending)",
    "Dislocation / Severe Displacement"
  ],
  "device": "cpu"
}
```

---

### 2.2 Get Sample Studies
Returns an array of anatomical categories containing sample radiograph IDs and clinical labels.

- **URL**: `/api/samples`
- **Method**: `GET`
- **Response**: `200 OK` (application/json)

```json
[
  {
    "category": "🦵 Lower Extremity / Leg & Dual Tib-Fib",
    "samples": [
      {
        "id": "8.jpg.png",
        "filename": "Study #08: Paired AP/LAT Tibia-Fibula Fracture [Leg Fracture]"
      },
      {
        "id": "IMG0000129",
        "filename": "Study #0129: Normal Intact Lower Leg [Leg Normal]"
      }
    ]
  }
]
```

---

### 2.3 Analyze Radiograph
Executes the full diagnostic pipeline on either a pre-indexed sample ID or an uploaded image file.

- **URL**: `/api/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data` or `application/json`

#### Request Parameters:
| Field | Type | Description |
| :--- | :--- | :--- |
| `file` *(optional)* | Binary File | Uploaded radiograph (`.jpg`, `.jpeg`, `.png`). Required if `sample_id` is omitted. |
| `sample_id` *(optional)* | String | ID of a pre-indexed sample (e.g., `"IMG0002302"`, `"8.jpg.png"`). |

#### Response Schema (`200 OK`):
```json
{
  "sample_name": "IMG0002302 [Shoulder Fracture]",
  "is_fracture": true,
  "confidence_pct": 98.4,
  "body_part": "Shoulder",
  "anatomy_confidence": 96.5,
  "subtype": "Transverse / Oblique (Linear break)",
  "subtype_confidence": 78.2,
  "all_subtypes": [
    {"name": "Hairline / Fissure (Minor crack)", "prob": 4.1},
    {"name": "Transverse / Oblique (Linear break)", "prob": 78.2},
    {"name": "Comminuted (Multi-fragment)", "prob": 12.0},
    {"name": "Greenstick (Incomplete bending)", "prob": 2.3},
    {"name": "Dislocation / Severe Displacement", "prob": 3.4}
  ],
  "angulation_deg": 14.5,
  "displacement_gap_mm": 4.8,
  "length_mm": 32.1,
  "width_mm": 18.2,
  "height_mm": 24.6,
  "bbox": [120, 85, 280, 240],
  "anatomical_portion": "Shoulder Girdle [Proximal Humerus / Greater Tuberosity & Surgical Neck]",
  "images": {
    "raw": "data:image/jpeg;base64,...",
    "cam": "data:image/jpeg;base64,...",
    "calipers": "data:image/jpeg;base64,...",
    "composite": "data:image/jpeg;base64,..."
  }
}
```

#### Error Response (`400 Bad Request`):
```json
{
  "error": "No valid image provided"
}
```

---

## 3. Client Examples

### cURL (Analyze by File Upload)
```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "file=@sample_data/sample1.jpg"
```

### cURL (Analyze by Sample ID)
```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"sample_id": "8.jpg.png"}'
```

### Python (`requests`)
```python
import requests

url = "http://127.0.0.1:5000/api/analyze"

# Using file upload:
with open("sample_data/sample1.jpg", "rb") as fh:
    response = requests.post(url, files={"file": fh})
    result = response.json()
    print("Diagnosis:", "FRACTURE" if result["is_fracture"] else "NORMAL")
    print("Confidence:", result["confidence_pct"], "%")
    print("Anatomy:", result["body_part"])
    print("Displacement Gap:", result["displacement_gap_mm"], "mm")
```

