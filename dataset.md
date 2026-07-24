# Dataset Documentation for ArcFlowText

This project contains three scene-text datasets:

- `Total-Text`
- `CTW1500`
- `ICDAR2015`

The three datasets are not interchangeable at the annotation level. Use a dataset adapter per dataset, then convert all annotations into one internal format before training:

```python
{
    "image_path": str,
    "image_id": str | int,
    "width": int | None,
    "height": int | None,
    "instances": [
        {
            "polygon": [[x1, y1], [x2, y2], ...],
            "bbox_xywh": [x, y, w, h] | None,
            "bezier_pts": [float, ...] | None,
            "text": str | None,
            "ignore": bool,
            "source": "total_text" | "ctw1500" | "icdar2015",
        }
    ],
}
```

For detection or spotting, train from full images and polygons. For cropped recognition, generate crops from the polygons or quadrilaterals and keep the transcription as the label. Do not mix ignored regions into recognition labels.

## What You Need To Know Before Training

### Core Training Decisions

1. Task type:
   - Text detection: predict text regions only.
   - Text recognition: crop each word/text instance and predict text.
   - End-to-end spotting: predict region plus transcription.

2. Geometry type:
   - `ICDAR2015`: quadrilateral word boxes, 4 points per instance.
   - `Total-Text`: arbitrary polygons, usually curved text regions.
   - `CTW1500`: curved text, represented here as Bezier control points plus axis-aligned bounding boxes in COCO-style JSON.

3. Ignore labels:
   - `ICDAR2015` uses `###` for illegible/do-not-care text.
   - `Total-Text` uses transcription `#` or orientation `#` for ignored regions.
   - `CTW1500` JSON in this copy does not expose `###` text directly; `rec` is numeric encoded text.

4. Coordinate convention:
   - All coordinates are pixel coordinates in the original image coordinate system.
   - `x` grows left to right, `y` grows top to bottom.
   - Convert every polygon to float arrays during loading, then normalize only inside augmentation/model code if needed.

5. Evaluation:
   - Keep dataset-specific eval protocols separate.
   - Do not report one combined headline metric unless you also report each dataset separately.
   - For detection, ignored regions should be excluded from both loss and evaluation matching.
   - For recognition, labels should be normalized consistently, but raw labels should also be preserved.

## Local Dataset Inventory

Counts were inspected from the local project folder on 2026-05-17.

| Dataset | Split | Images | Annotation files | Text instances | Ignored instances | Usable instances |
|---|---:|---:|---:|---:|---:|---:|
| Total-Text | Train | 1,255 | 1,255 polygon `.txt` | 10,589 | 1,305 | 9,284 |
| Total-Text | Test | 300 | 300 polygon `.txt` | 2,552 | 343 | 2,209 |
| CTW1500 | Train | 1,000 | 1 JSON | 6,991 | not explicit | 6,991 |
| CTW1500 | Test | 500 | 1 JSON | 2,671 | not explicit | 2,671 |
| ICDAR2015 | Train | 1,000 | 1,000 `.txt` | 11,886 | 7,418 | 4,468 |
| ICDAR2015 | Test | 500 | 500 `.txt` | 5,230 | 3,153 | 2,077 |

Image-size ranges from local files:

| Dataset | Split | Width range | Height range |
|---|---:|---:|---:|
| Total-Text | Train | 180-4272 | 159-4640 |
| Total-Text | Test | 184-5184 | 162-4746 |
| CTW1500 | Train | 189-5591 | 144-4867 |
| CTW1500 | Test | 114-1024 | 154-500 |
| ICDAR2015 | Train | 1280 | 720 |
| ICDAR2015 | Test | 1280 | 720 |

## Total-Text

### Purpose

Total-Text is a curved scene-text dataset. It is useful for:

- curved text detection
- arbitrary-shape text segmentation
- end-to-end scene text spotting
- cropped word recognition after generating crops from polygons

### Local Structure

```text
Total-Text/
  Train/
    img*.jpg
  Test/
    img*.jpg
  Annotation/
    groundtruth_polygonal_annotation/
      Train/
        poly_gt_img*.txt
      Test/
        poly_gt_img*.txt
    groundtruth_pixel/
      Train/
      Test/
    groundtruth_textregion/
      Train/
      Test/
```

Use `Train/` and `Test/` as the full-image folders. Use `Annotation/groundtruth_polygonal_annotation/{Train,Test}` for training labels.

The `groundtruth_pixel` and `groundtruth_textregion` folders contain PNG masks/regions. They are useful for segmentation-style experiments or visualization, but polygon annotations are the main portable format for detection/spotting.

### File Naming

Image:

```text
Total-Text/Train/img1001.jpg
```

Polygon label:

```text
Total-Text/Annotation/groundtruth_polygonal_annotation/Train/poly_gt_img1001.txt
```

The image id is the number after `img`; the polygon annotation adds prefix `poly_gt_`.

### Annotation Format

Each line is one text instance. The format is Python/MATLAB-like text, not JSON:

```text
x: [[153 161 179 195 184 177]], y: [[347 323 305 315 331 357]], ornt: [u'c'], transcriptions: [u'the']
x: [[184 222 273 269 230 202]], y: [[293 269 270 296 297 317]], ornt: [u'c'], transcriptions: [u'alpaca']
x: [[298 334 335 294]], y: [[491 488 500 499]], ornt: [u'#'], transcriptions: [u'#']
```

Fields:

- `x`: list of x coordinates.
- `y`: list of y coordinates.
- `ornt`: orientation/category marker.
- `transcriptions`: text label.

Observed orientation markers include normal markers such as `c`, `h`, `m`, plus ignore marker `#`. Some local lines are irregular, so the parser should be tolerant and log malformed records.

### Parser Contract

For each line:

1. Extract all integers inside `x: [[...]]`.
2. Extract all integers inside `y: [[...]]`.
3. Require `len(x) == len(y)` and at least 3 points for a valid polygon.
4. Build polygon as `[[x[i], y[i]] for i in range(n)]`.
5. Extract transcription from `transcriptions: [u'...']`.
6. Mark `ignore=True` if transcription is `#` or orientation is `#`.

Recommended robust parser outline:

```python
import re

X_RE = re.compile(r"x:\s*\[\[([^\]]+)\]\]")
Y_RE = re.compile(r"y:\s*\[\[([^\]]+)\]\]")
T_RE = re.compile(r"transcriptions:\s*\[u?'([^']*)'\]")
O_RE = re.compile(r"ornt:\s*\[u?'([^']*)'\]")

def parse_total_text_line(line):
    mx, my = X_RE.search(line), Y_RE.search(line)
    mt, mo = T_RE.search(line), O_RE.search(line)
    if not mx or not my:
        return None
    xs = [int(v) for v in re.findall(r"-?\d+", mx.group(1))]
    ys = [int(v) for v in re.findall(r"-?\d+", my.group(1))]
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    text = mt.group(1) if mt else ""
    orient = mo.group(1) if mo else ""
    return {
        "polygon": [[float(x), float(y)] for x, y in zip(xs, ys)],
        "text": text,
        "ignore": text == "#" or orient == "#",
        "orientation": orient,
    }
```

### Training Notes

- Keep arbitrary polygons. Do not force them into quadrilaterals unless the model requires it.
- For recognizer crops, use polygon-aware crop/rectification if possible; plain axis-aligned crops include excess background for curved text.
- Filter ignored `#` instances from recognition training.
- Preserve case unless your experiment explicitly uses case-insensitive metrics.
- Some local annotation lines are malformed or irregular. Log them and skip invalid polygons rather than crashing a full run.

## CTW1500

### Purpose

CTW1500 is a curved text dataset. It is useful for:

- long curved text detection
- Bezier/curve-aware text spotting
- arbitrary-shape detection benchmarks

### Local Structure

```text
CTW1500/
  ctwtrain_text_image/
    0001.jpg
    ...
  ctwtest_text_image/
    1001.jpg
    ...
  annotations/
    train_ctw1500_maxlen100_v2.json
    test_ctw1500_maxlen100.json
  weak_voc_new.txt
  weak_voc_pair_list.txt
```

This local copy uses COCO-style JSON annotations. The original CTW1500 text-label-curve `.txt` files are not present at the dataset root.

### JSON Format

Top-level keys:

```json
{
  "licenses": [],
  "info": {},
  "images": [],
  "annotations": [],
  "categories": []
}
```

Image record sample:

```json
{
  "width": 369,
  "date_captured": "",
  "license": 0,
  "flickr_url": "",
  "file_name": "0001.jpg",
  "id": 1,
  "coco_url": "",
  "height": 549
}
```

Annotation record sample:

```json
{
  "image_id": 1,
  "bbox": [56.0, 506.0, 204.0, 26.0],
  "area": 5304.0,
  "rec": [45, 37, 56, 41, 35, 47, 12, 0, 36, 14, 0, 38, 14, 96, 96],
  "category_id": 1,
  "iscrowd": 0,
  "id": 1,
  "bezier_pts": [56.0, 506.0, 123.67, 506.0, 191.33, 506.0, 259.0, 506.0, 259.0, 531.0, 191.33, 531.0, 123.67, 531.0, 56.0, 531.0]
}
```

Important fields:

- `images[].id`: joins to `annotations[].image_id`.
- `images[].file_name`: image file name.
- `images[].width`, `images[].height`: image dimensions.
- `annotations[].bbox`: axis-aligned bounding box `[x, y, width, height]`.
- `annotations[].bezier_pts`: 16 numbers, interpreted as 8 `(x, y)` points. These represent curved text control geometry.
- `annotations[].rec`: numeric encoded transcription, length 100 in this local JSON. Padding value `96` appears repeatedly.
- `annotations[].area`: box area.
- `annotations[].category_id`: text category id.
- `annotations[].iscrowd`: COCO-style crowd flag.

Category record:

```json
{
  "supercategory": "beverage",
  "id": 1,
  "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
  "name": "text"
}
```

### Parser Contract

For each JSON file:

1. Load the JSON.
2. Build `image_id -> image record`.
3. Group annotations by `image_id`.
4. For each annotation:
   - use `bbox` for simple detector targets;
   - convert `bezier_pts` into 8 `(x, y)` points for curve-aware models;
   - keep `rec` as encoded text unless you have the exact vocabulary mapping.

Recommended conversion:

```python
def ctw_bezier_to_points(bezier_pts):
    assert len(bezier_pts) == 16
    return [
        [float(bezier_pts[i]), float(bezier_pts[i + 1])]
        for i in range(0, 16, 2)
    ]
```

For a polygon-only detector, a pragmatic fallback is to treat the 8 Bezier points as a polygon-like boundary. For best curved-text modeling, use Bezier-aware heads or sample points along the top and bottom curves.

### Recognition Labels

The `rec` field is numeric, not raw text. In this local copy:

- every `rec` sequence has length 100;
- `96` behaves like padding;
- the exact character vocabulary is not documented in the local files inspected.

The files `weak_voc_new.txt` and `weak_voc_pair_list.txt` contain weak vocabulary strings, for example:

```text
DOUGLASTON
E-313
L164
F.D.N.Y.
Pann's
```

Do not assume these weak vocabulary files are a one-to-one annotation list unless you verify the mapping for your training code. For detection-only training, you do not need decoded transcriptions.

### Training Notes

- Use CTW1500 for curved text detection/spotting rather than simple horizontal text only.
- If your model expects polygons, convert Bezier points carefully.
- If your model expects masks, generate masks from sampled curve boundaries.
- For recognition, either recover the exact `rec` vocabulary or use CTW1500 for detection-only until the mapping is confirmed.
- Train/test split is fixed by the JSON files and image folders.

## ICDAR2015

### Purpose

ICDAR2015 Challenge 4 is an incidental scene-text dataset. It is useful for:

- oriented quadrilateral text detection
- word-level scene text spotting
- cropped word recognition after crop generation

### Local Structure

```text
ICDAR2015/
  ch4_training_images/
    img_1.jpg
    ...
  ch4_training_localization_transcription_gt/
    gt_img_1.txt
    ...
  ch4_test_images/
    img_1.jpg
    ...
  ch4_test_localization_transcription_gt/
    gt_img_1.txt
    ...
  Detection/
    PSEnet/
      dataset/
        icdar2015_loader.py
        icdar2015_test_loader.py
        ctw1500_loader.py
        ctw1500_test_loader.py
  util/
```

The `Detection/PSEnet` folder is code, not raw dataset data. It includes useful loader examples for parsing ICDAR-style text files.

### Annotation Format

Each image has one `.txt` file. Each line is one word/text instance:

```text
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```

Example:

```text
377,117,463,117,465,130,378,130,Genaxis Theatre
493,115,519,115,519,131,493,131,[06]
374,155,409,155,409,170,374,170,###
```

Fields:

- first 8 comma-separated values are four polygon vertices;
- final field is transcription;
- `###` means ignored/do-not-care text.

The local PSEnet loader follows this logic:

```python
gt = util.str.split(line, ',')
if gt[-1][0] == '#':
    tags.append(False)
else:
    tags.append(True)
box = [int(gt[i]) for i in range(8)]
```

### Parser Contract

Use `split(",", 8)`, not a naive unrestricted split, because transcriptions can contain punctuation or commas in some scene-text datasets.

```python
def parse_icdar2015_line(line):
    parts = line.strip().lstrip("\ufeff").split(",", 8)
    if len(parts) != 9:
        return None
    coords = [int(v) for v in parts[:8]]
    text = parts[8]
    polygon = [
        [coords[0], coords[1]],
        [coords[2], coords[3]],
        [coords[4], coords[5]],
        [coords[6], coords[7]],
    ]
    return {
        "polygon": [[float(x), float(y)] for x, y in polygon],
        "text": text,
        "ignore": text.startswith("#"),
    }
```

### Training Notes

- The images are all 1280x720 in this local copy.
- The annotation is quadrilateral, so it is simpler than Total-Text and CTW1500.
- Do not train recognition on `###` entries.
- For detector loss, keep ignored polygons as ignore masks if your framework supports them.
- For end-to-end spotting, evaluate readable text separately from ignored regions.

## Unified Loader Strategy

### Recommended Internal Loader Steps

1. Scan dataset split roots:
   - Total-Text: `Total-Text/Train`, `Total-Text/Test`
   - CTW1500: `CTW1500/ctwtrain_text_image`, `CTW1500/ctwtest_text_image`
   - ICDAR2015: `ICDAR2015/ch4_training_images`, `ICDAR2015/ch4_test_images`

2. Parse annotations into the shared internal format.

3. Validate every record:
   - image exists;
   - each polygon has at least 3 points;
   - all coordinates are finite numbers;
   - ignored regions are flagged;
   - text is present for recognition/spotting unless the dataset lacks decoded text.

4. Write a manifest such as:

```json
{
  "dataset": "total_text",
  "split": "train",
  "image_path": "Total-Text/Train/img1001.jpg",
  "image_id": "img1001",
  "width": 640,
  "height": 480,
  "instances": [
    {
      "polygon": [[153, 347], [161, 323], [179, 305]],
      "text": "the",
      "ignore": false
    }
  ]
}
```

5. Train from manifests, not from raw folder scans. This makes counts reproducible.

### Ignore Policy

Use this policy consistently:

```python
def is_ignored(dataset, text=None, orient=None):
    if dataset == "icdar2015":
        return text is None or text.startswith("#")
    if dataset == "total_text":
        return text in {None, "", "#"} or orient == "#"
    if dataset == "ctw1500":
        return False
    raise ValueError(dataset)
```

### Geometry Normalization

Recommended geometry fields:

- `polygon`: always store polygon points when available.
- `bbox_xywh`: store axis-aligned box for fast filtering.
- `bezier_pts`: keep CTW1500 curve control points.
- `ignore`: always store this flag.
- `text_raw`: original transcription exactly as found.
- `text_norm`: normalized text for recognition metrics.

### Text Normalization

Keep two versions of every transcription:

```python
text_raw = original_text
text_norm = normalize(original_text)
```

Use `text_raw` for traceability and qualitative inspection. Use `text_norm` only when the official evaluation protocol or your experiment requires it.

Common normalization choices:

- case-sensitive: preserve case;
- case-insensitive: lowercase both prediction and label;
- alphanumeric-only: remove punctuation;
- ignore illegible: remove `###` and `#` entries from recognition loss.

Document the chosen normalization in every experiment report.

## Dataset-Specific Pitfalls

### Total-Text Pitfalls

- Annotation files are not JSON.
- Polygons may have variable point counts.
- Some local lines are malformed or irregular; parser should skip and log them.
- `#` marks ignored text.
- Axis-aligned crops are weak for curved text. Prefer polygon-aware crops.

### CTW1500 Pitfalls

- Local annotation is JSON, not the original CTW text-label-curve folder layout.
- `rec` is encoded numeric text; do not treat it as raw transcription.
- `bezier_pts` has 16 values, not a flat polygon with arbitrary length.
- Weak vocabulary files are not enough by themselves to guarantee decoded labels.

### ICDAR2015 Pitfalls

- `###` entries are abundant and should not become recognition labels.
- Use `split(",", 8)` so text containing commas does not break parsing.
- Coordinates are quadrilateral points, not `[x, y, w, h]`.
- Training and test images share names like `img_1.jpg`; keep split in the record id.

## Recommended Experiments

### Detection-Only Baseline

Use all three datasets:

- Total-Text: polygons.
- CTW1500: Bezier points converted to sampled polygon or curve representation.
- ICDAR2015: quadrilateral polygons.

Train with ignore masks for Total-Text and ICDAR2015.

### Recognition Baseline

Use:

- Total-Text usable transcriptions from polygons.
- ICDAR2015 non-ignored transcriptions.

Use CTW1500 only if the `rec` vocabulary mapping is confirmed.

### End-to-End Spotting

Use:

- Total-Text and ICDAR2015 directly.
- CTW1500 only after decoding `rec` reliably.

Keep per-dataset metrics separate.

## Verification Checklist

Before training, run these checks:

- Count images per split and compare with this file.
- Count parsed instances per split.
- Count ignored and usable instances.
- Draw 20 random polygons per dataset onto images.
- Save 20 recognition crops per dataset and inspect labels.
- Assert no ignored label enters recognition training.
- Assert all polygons stay inside or near image bounds after augmentation.
- Assert train/test splits never mix.
- Log malformed annotation lines with file name and line number.

## Local Summary

This local dataset package is suitable for a strong curved/incidental scene-text project:

- Total-Text gives arbitrary curved polygons and readable transcriptions.
- CTW1500 gives curve-heavy training data with COCO-style JSON and Bezier geometry.
- ICDAR2015 gives fixed-resolution incidental text images with quadrilateral labels.

The most important engineering decision is to build one robust manifest generator that keeps dataset-specific parsing separate but emits one unified schema for training.

