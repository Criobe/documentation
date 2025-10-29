# CVAT Label Configuration Templates

Copy-paste ready JSON templates for all CVAT label configurations used in the QUADRATSEG pipeline.

## Overview

This reference provides all label configurations needed for the complete coral annotation pipeline. Each template is ready to paste into CVAT's **Raw** label configuration editor.

!!! important "CVAT vs FiftyOne Label Type Naming"
    The system uses different terminology between CVAT (annotation interface) and FiftyOne (dataset management):

    **CVAT → FiftyOne mapping:**

    - **Skeleton** (CVAT) → **Keypoints** (FiftyOne) - Used for corner detection (4 points) and grid detection (117 points)
    - **Polygon** (CVAT) → **Polylines** with `closed=True, filled=True` (FiftyOne) - Used for coral segmentation

    **Important:**
    - Corner and grid detection use CVAT's **Skeleton** label type with numbered sublabels, not independent points
    - Coral segmentation uses CVAT's **Polygon** label type for each genus (closed shapes around coral colonies)

## Quick Reference

| Template | Use Case | CVAT Type | FiftyOne Type | Count | Guides |
|----------|----------|-----------|---------------|-------|--------|
| [4-Point Corners](#4-point-corner-detection) | Quadrat corner detection | Skeleton | Keypoints | 4 sublabels | B, C |
| [117-Point Grid](#117-point-grid-detection) | Grid intersection detection | Skeleton | Keypoints | 117 sublabels | C |
| [16 Genera Segmentation](#16-genera-coral-segmentation-finegrained) | Coral instance segmentation | Polygon | Polylines (closed) | 16 classes | A, B, C |
| [10 Class Extended](#10-class-extended-taxonomy) | Simplified coral segmentation | Polygon | Polylines (closed) | 10 classes | - |
| [7 Class Main Families](#7-class-main-families) | Family-level segmentation | Polygon | Polylines (closed) | 7 classes | - |
| [Binary Coral](#binary-coral-detection) | Coral vs background | Polygon | Polylines (closed) | 1 class | - |

---

## 4-Point Corner Detection

**Use case**: Detecting quadrat frame corners for perspective correction

**Pipeline stage**: Stage 1 (corner detection and warping)

**CVAT annotation type**: Skeleton with 4 numbered sublabels

**FiftyOne import type**: Keypoints

### JSON Template

```json
[
  {
    "name": "quadrat_corner",
    "color": "#19fdf3",
    "type": "skeleton",
    "sublabels": [
      {
        "name": "1",
        "type": "points",
        "color": "#d12345"
      },
      {
        "name": "2",
        "type": "points",
        "color": "#350dea"
      },
      {
        "name": "3",
        "type": "points",
        "color": "#479ffe"
      },
      {
        "name": "4",
        "type": "points",
        "color": "#4a649f"
      }
    ],
    "svg": "",
    "attributes": []
  }
]
```

### Configuration Notes

- **Label name**: `quadrat_corner` (not just "corner")
- **Label type**: `skeleton` (not independent points)
- **Sublabels**: 4 numbered points (1, 2, 3, 4)
- **Point order**: **CRITICAL** - Must follow clockwise order:
    1. Top-left corner
    2. Top-right corner
    3. Bottom-right corner
    4. Bottom-left corner
- **SVG field**: Can be empty `""` or contain SVG path definition for skeleton visualization

### Visual Guide

```
    1 (TL) ────────── 2 (TR)
      │                │
      │    Quadrat     │
      │                │
    4 (BL) ────────── 3 (BR)
```

!!! warning "Corner Order Matters"
    Incorrect corner order will result in distorted warped images. Always annotate clockwise starting from top-left!

### Common Mistakes

- ❌ **Counter-clockwise order** - Causes image flip
- ❌ **Starting from different corner** - Causes rotation
- ❌ **Placing points on wrong features** - Detect actual frame corners, not coral edges

---

## 117-Point Grid Detection

**Use case**: Detecting all grid intersection points for precise grid removal

**Pipeline stage**: Stage 2 (grid pose detection)

**CVAT annotation type**: Skeleton with 117 numbered sublabels

**FiftyOne import type**: Keypoints

### JSON Template

!!! info "Full Configuration File"
    The complete JSON with all 117 sublabels is available at:
    `docs/assets/cvat_project_label_config/grid_annotation_example.json`

**Simplified structure:**

```json
[
  {
    "name": "grid",
    "color": "#7571e6",
    "type": "skeleton",
    "sublabels": [
      {"name": "1", "type": "points", "color": "#d12345"},
      {"name": "2", "type": "points", "color": "#350dea"},
      {"name": "3", "type": "points", "color": "#479ffe"},
      ...
      {"name": "117", "type": "points", "color": "#______"}
    ],
    "svg": "...",
    "attributes": []
  }
]
```

### Configuration Notes

- **Label name**: `grid` (not "grid_point")
- **Label type**: `skeleton` (not independent points)
- **Sublabels**: 117 numbered points (1-117)
- **Grid structure**: 9 rows × 13 columns = 117 keypoints
- **SVG field**: Contains edge connections for grid visualization

### Grid Structure

```
Point numbering (9 rows × 13 columns):

  0   1   2   3   4   5   6   7   8   9  10  11  12
 13  14  15  16  17  18  19  20  21  22  23  24  25
 26  27  28  29  30  31  32  33  34  35  36  37  38
 39  40  41  42  43  44  45  46  47  48  49  50  51
 52  53  54  55  56  57  58  59  60  61  62  63  64
 65  66  67  68  69  70  71  72  73  74  75  76  77
 78  79  80  81  82  83  84  85  86  87  88  89  90
 91  92  93  94  95  96  97  98  99 100 101 102 103
104 105 106 107 108 109 110 111 112 113 114 115 116
```

### Complete Skeleton Edges

The skeleton connects adjacent points with edges. Here's the complete edge list for reference:

**Horizontal edges** (connect points in same row):
```python
# Row 0: 0-1, 1-2, 2-3, ..., 11-12
# Row 1: 13-14, 14-15, 15-16, ..., 24-25
# ... (9 rows total)
```

**Vertical edges** (connect points in same column):
```python
# Column 0: 0-13, 13-26, 26-39, ..., 91-104
# Column 1: 1-14, 14-27, 27-40, ..., 92-105
# ... (13 columns total)
```

!!! info "Automatic Detection Recommended"
    Manually annotating 117 points is extremely tedious. **Always use the pre-trained grid detection model** via webhook for automatic detection, then manually correct any misdetections.

---

## 16 Genera Coral Segmentation (Finegrained)

**Use case**: Genus-level coral instance segmentation (finegrained taxonomy)

**Pipeline stage**: Final stage (coral segmentation)

**CVAT annotation type**: Polygon (closed shapes around coral colonies)

**FiftyOne import type**: Polylines with `closed=True, filled=True`

### JSON Template

```json
[
  {
    "name": "Acanthastrea",
    "color": "#ff0000",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Acropora",
    "color": "#00ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Astreopora",
    "color": "#0000ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Atrea",
    "color": "#ffff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Fungia",
    "color": "#ff00ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Goniastrea",
    "color": "#00ffff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Leptastrea",
    "color": "#ff8000",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Merulinidae",
    "color": "#8000ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Millepora",
    "color": "#00ff80",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Montastrea",
    "color": "#ff0080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Montipora",
    "color": "#80ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Other",
    "color": "#808080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pavona/Leptoseris",
    "color": "#ff8080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pocillopora",
    "color": "#8080ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Porites",
    "color": "#80ff80",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Psammocora",
    "color": "#ff80ff",
    "attributes": [],
    "type": "polygon"
  }
]
```

### Coral Taxonomies

This configuration uses the **finegrained taxonomy** with 16 genera. See [Taxonomy Mapping](#taxonomy-hierarchies) below for how this relates to other classification schemes.

---

## 10 Class Extended Taxonomy

**Use case**: Simplified coral segmentation with merged genera

**CVAT annotation type**: Polygon (closed shapes around coral colonies)

**FiftyOne import type**: Polylines with `closed=True, filled=True`

### JSON Template

```json
[
  {
    "name": "Acropora",
    "color": "#00ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Astreopora",
    "color": "#0000ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Fungia",
    "color": "#ff00ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Group1",
    "color": "#ffff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Millepora",
    "color": "#00ff80",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Montipora",
    "color": "#80ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Other",
    "color": "#808080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pavona/Leptoseris",
    "color": "#ff8080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pocillopora",
    "color": "#8080ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Porites",
    "color": "#80ff80",
    "attributes": [],
    "type": "polygon"
  }
]
```

### Class Mapping

- **Group1**: Merges `Acanthastrea`, `Atrea`, `Goniastrea`, `Leptastrea`, `Merulinidae`
- **Other**: Includes `Montastrea`, `Psammocora`, and unidentified corals

---

## 7 Class Main Families

**Use case**: Family-level coral classification

**CVAT annotation type**: Polygon (closed shapes around coral colonies)

**FiftyOne import type**: Polylines with `closed=True, filled=True`

### JSON Template

```json
[
  {
    "name": "Acropora",
    "color": "#00ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Atrea",
    "color": "#ffff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Montipora",
    "color": "#80ff00",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Other",
    "color": "#808080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pavona/Leptoseris",
    "color": "#ff8080",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Pocillopora",
    "color": "#8080ff",
    "attributes": [],
    "type": "polygon"
  },
  {
    "name": "Porites",
    "color": "#80ff80",
    "attributes": [],
    "type": "polygon"
  }
]
```

---

## Binary Coral Detection

**Use case**: Coral vs background segmentation (coral cover analysis)

**CVAT annotation type**: Polygon (closed shapes around coral colonies)

**FiftyOne import type**: Polylines with `closed=True, filled=True`

### JSON Template

```json
[
  {
    "name": "Coral",
    "color": "#00ff00",
    "attributes": [],
    "type": "polygon"
  }
]
```

---

## Taxonomy Hierarchies

The CRIOBE coral segmentation system supports multiple taxonomic hierarchies. All classifications are at **genus level**.

### Finegrained (16 genera) → Extended (10 classes)

| Finegrained Genus | Extended Class |
|-------------------|----------------|
| Acanthastrea | Group1 |
| Acropora | Acropora |
| Astreopora | Astreopora |
| Atrea | Group1 |
| Fungia | Fungia |
| Goniastrea | Group1 |
| Leptastrea | Group1 |
| Merulinidae | Group1 |
| Millepora | Millepora |
| Montastrea | Other |
| Montipora | Montipora |
| Other | Other |
| Pavona/Leptoseris | Pavona/Leptoseris |
| Pocillopora | Pocillopora |
| Porites | Porites |
| Psammocora | Other |

### Extended (10 classes) → Main Families (7 classes)

| Extended Class | Main Family |
|----------------|-------------|
| Acropora | Acropora |
| Astreopora | Other |
| Fungia | Other |
| Group1 | Atrea |
| Millepora | Other |
| Montipora | Montipora |
| Other | Other |
| Pavona/Leptoseris | Pavona/Leptoseris |
| Pocillopora | Pocillopora |
| Porites | Porites |

### Main Families (7 classes) → Agnostic (1 class)

All 7 main families map to **Coral** (binary classification).

---

## Webhook Configurations

Common webhook configurations for CVAT projects.

### Model Detection Webhook

Automatically run ML models when jobs are opened.

**Template:**
```
URL: http://bridge:8000/detect-model-webhook?model_name={MODEL_NAME}&conv_mask_to_poly={BOOL}
Event: When a job state is changed to 'in progress'
Content-Type: application/json
```

**Available models:**

| Model Name | Task | conv_mask_to_poly |
|------------|------|-------------------|
| `pth-yolo-gridcorners` | 4-point corner detection | `false` |
| `pth-yolo-gridpose` | 117-point grid detection | `false` |
| `pth-yolo-coralsegv4` | Coral segmentation (finegrained) | `true` |
| `pth-yolo-coralsegbanggai` | Coral segmentation (Banggai) | `true` |
| `pth-mmseg-coralscopsegformer` | DINOv2 + CoralSCoP segmentation | `true` |

**Example:**
```
http://bridge:8000/detect-model-webhook?model_name=pth-yolo-coralsegv4&conv_mask_to_poly=true
```

### Task Completion Webhooks

Automate processing when tasks are completed.

#### Crop Quadrat and Create Task

**Use case**: After corner detection, automatically warp images and create segmentation tasks

**Template:**
```
URL: http://bridge:8000/crop-quadrat-and-create-new-task-webhook?target_proj_id={PROJECT_ID}
Event: When a task status is changed to 'completed'
Content-Type: application/json
```

**Example:**
```
http://bridge:8000/crop-quadrat-and-create-new-task-webhook?target_proj_id=7
```

#### Remove Grid and Create Task

**Use case**: After grid detection, automatically remove grid via inpainting and create segmentation tasks

**Template:**
```
URL: http://bridge:8000/remove-grid-and-create-new-task-webhook?target_proj_id={PROJECT_ID}
Event: When a task status is changed to 'completed'
Content-Type: application/json
```

**Example:**
```
http://bridge:8000/remove-grid-and-create-new-task-webhook?target_proj_id=8
```

---

## Annotation Best Practices

### Polygon Annotation Tips (Coral Segmentation)

1. **Colony boundaries**: Follow the actual living tissue boundary, not shadows or substrate
2. **Overlapping colonies**: Separate into individual instances, don't merge
3. **Minimum size**: Annotate colonies ≥2cm diameter (adjust based on image resolution)
4. **Partial colonies**: Annotate if ≥50% visible
5. **Uncertain identification**: Use "Other" label rather than guessing
6. **Closing polygons**: Press `N` key or double-click first point to close the polygon
7. **Tool selection**: Use CVAT's Polygon tool (not Polyline tool) for coral annotation

### Skeleton Annotation Tips

1. **Corner detection**: Place points at exact corner positions (frame edges, not tape/markers)
2. **Grid detection**: Points should be at line intersections, not on the lines themselves
3. **Verification**: Zoom in to ensure sub-pixel accuracy for critical points

### Quality Control Checklist

Before marking a task complete:

- [ ] All visible features are annotated
- [ ] No duplicate annotations
- [ ] Labels are correct
- [ ] Boundaries/points are accurate
- [ ] Annotations saved (`Ctrl+S`)

---

## Common Configuration Errors

??? danger "CVAT shows 'Invalid label configuration'"
    **Cause**: JSON syntax error

    **Solution**:
    - Copy template exactly (no modifications)
    - Ensure all brackets, quotes, commas are correct
    - Use CVAT's "Raw" mode, not the UI builder

??? warning "Skeleton not created for corner/grid detection"
    **Cause**: Need to manually add points in correct order

    **Solution**:
    - Skeleton structure is implicit in CVAT for point-based labels
    - Add points in the specified order during annotation
    - CVAT will automatically create edges between points

??? warning "Polygons don't close automatically"
    **Cause**: Expected behavior - must manually close

    **Solution**:
    - Press `N` key to close polygon
    - Or double-click on first point to complete the shape
    - Verify shape is closed (solid fill appears in the annotated region)

---

## See Also

- [Data Preparation Guides](../data-preparation/index.md) - Use these templates in practice
- [Training Guides](../training-and-deployment/index.md) - Train models using annotated data
- [CVAT Documentation](https://opencv.github.io/cvat/docs/) - Official CVAT user manual

---

**Related**: [Data Preparation](../data-preparation/index.md) · [Training & Deployment](../training-and-deployment/index.md)
