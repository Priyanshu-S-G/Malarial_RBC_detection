# Malaria RBC — Preprocessing & Segmentation Demo

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-demo%20ready-brightgreen)
![Repo size](https://img.shields.io/github/repo-size/Priyanshu-S-G/Malarial_RBC_detection)
![flask](https://img.shields.io/badge/flask-3.x-blue)
![opencv](https://img.shields.io/badge/OpenCV-4.x-green)
![skimage](https://img.shields.io/badge/scikit--image-0.22+-blueviolet)
![license](https://img.shields.io/badge/license-MIT-blue)




> Lightweight demo that demonstrates a **preprocessing-heavy** pipeline to enhance and segment malaria parasites in single-cell crop images.  
> No training or model required, as it is purely deterministic image processing tuned for the Kaggle cell dataset.

---

## TL;DR

- Upload a single RBC crop image via the web UI.  
- The app runs a deterministic preprocessing pipeline (white-balance → CLAHE → HSV purple mask → cleanup → mask overlay) and returns intermediate steps + final overlay.  
- Pipeline is intentionally explainable so the project focuses on **preprocessing** rather than ML training.  
- Known limitation: occasional **false positives** where staining/artifacts mimic parasite color — documented below.

---

## Tech Stack

- Backend: `Flask 3.0.0`
- Image Processing: `OpenCV`, `scikit-image`, `NumPy`
- Frontend: Vanilla `JavaScript`, `HTML5`, `CSS3`
- UI/UX: Gradient design with smooth animations

## Repo layout

Malarial_RBC_detection/  
├─ app.py # Flask app (upload → process endpoints)  
├─ preprocessing.py # Preprocessing pipeline (core logic)  
├─ requirements.txt  
├─ README.md # <-- you are here  
├─ templates/  
│ └─ index.html # Frontend UI  
└─ static/  
└─ .gitkeep  

---

## Quick start (run locally)

1. Clone:
```bash
git clone https://github.com/Priyanshu-S-G/Malarial_RBC_detection.git
cd Malarial_RBC_detection
```

2. Create & activate venv (Windows Git Bash):
```bash
python -m venv venv
source venv/Scripts/activate
```
(or Powershell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Run in Flask dev server:
```bash
python app.py
```

5. Open browser:
```cpp
http://127.0.0.1:5000
```

![demo](static/Animation.gif)

Upload an image → click Process Image → inspect outputs.

---

## API (for automation / demos)

`POST /upload` — multipart form

Field: `file` (single image)
Returns:
```json
{
  "success": true,
  "filename": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
  "preview_url": "data:image/png;base64,<...>"
}
```

Example `curl`:
```bash
curl -X POST -F "file=@/path/to/cell.png" http://127.0.0.1:5000/upload
```

`POST /process` — JSON body
Send:
```json
{ "filename": "the_returned_filename_from_upload.png" }
```

Returns `original` (data-URI) and `outputs` array:
```json
{
  "success": true,
  "original": "data:image/png;base64,<...>",
  "outputs": [
    { "url": "data:image/png;base64,<...>", "label": "CLAHE (L-channel)", "filename": "base_lab_clahe.png" },
    ...
  ]
}
```

Example `curl`:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename":"C100P61ThinF_IMG_20150918_144104_cell_162.png"}' \
  http://127.0.0.1:5000/process
```

---

## Preprocessing pipeline

Implemented in `preprocessing.py`. The pipeline returns a list of `(filename, display_name)` pairs consumed by the UI. Main steps:

1. Resize — normalize input size (MAX_SIDE, default 256).
2. White balance — gray-world to reduce pink staining bias.
3. LAB → CLAHE on L-channel — local contrast enhancement.
4. Bilateral smoothing — edge-preserving denoise.
5. HSV-based purple mask — primary parasite candidate (tuned HSV bounds).
6. Blur + binarize — reduce pixel noise.
7. Morphological cleanup — close/open and remove small blobs (`MIN_OBJECT_AREA`).
8. Contour fill → final mask — filter and fill blobs.
9. Overlay — blend mask onto the resized original.

Parameters to tune are at the top of `preprocessing.py`:
- `MAX_SIDE`, `CLAHE_CLIP`, `BILATERAL_*`, `HSV_LOWER`, `HSV_UPPER`, `MIN_OBJECT_AREA`, `MORPH_KERNEL`.

---

## Tuning advice

- Reduce false positives: increase `MIN_OBJECT_AREA` or tighten HSV saturation lower bound; add a small histogram/intensity check on candidate blob.
- Recover faint parasites: broaden HSV hue or lower saturation threshold (`HSV_LOWER` H down to ~120, S to ~40).
- Large images: increase `MAX_SIDE` but watch latency.

---

## Failure case

The HSV color-thresholding approach isolates parasite-colored regions effectively, but can produce false positives when artefacts or dense stain deposits in uninfected RBCs fall into the same hue/saturation range. This is expected for color-threshold methods.

---

## License & attribution

this project: **MIT License**.

Dataset (used for testing & examples): *Cell Images for Detecting Malaria* — `iarunava` on Kaggle.

---
