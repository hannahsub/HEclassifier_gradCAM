# H&E Classifier with Grad-CAM 
HEclassifier_gradCAM provides an interactive Python Shiny app that extends an existing deep learning-based H&E cell classification system by incorporating Grad-CAM visualisations.

## Features
- Upload or select demo images
- Interactive user feedback + logging
- Dark UI for higher cell visibility
- Supports both CNN pipelines (selectable):
  - **Double Layer CNN**
  - **Multiclass CNN**
- For each pipeline:
  - Cell type predictions
  - Confidence scores
  - Grad-CAM visualisation
  - Blur detection

## Getting Started
```bash
git clone https://github.com/yourusername/HEclassifier_gradCAM.git
cd HEclassifier_gradCAM
pip install -r requirements.txt
shiny run --reload Gradcam.py
```

# **Double Layer CNN** vs **Multiclass CNN**
- Double Layer CNN
    - Step 1: Binary classification – Tumour vs Immune
    - Step 2: Tumour subtype classification (if Tumour)
    - Model: dlcnn_MS_CLAHE.keras
    - Strength: More interpretable tumour pathway
- Multiclass CNN
    - Directly classifies all 5 classes (4 tumour subtypes + 1 immune)
    - Model: multiclass_MS_CLAHE.keras
    - Strength: Higher overall accuracy, stronger on immune detection
    - Both models were trained on CLAHE-preprocessed, mixed-sampled datasets. Users can toggle between both in the app to compare predictions and confidence

## Folder Structure
### Root
- `Gradcam.py` – Main Shiny app with Grad-CAM
- `ORIGINAL.py` – Legacy app version (no Grad-CAM)
- `requirements.txt` – Python dependencies
- `README.md` – This file
- `LICENSE` – License information


### `demo/`
Sample demo images used for testing the app UI and classification functions:
- Clear cell images
- Blurry image
- Tumour Cell image
- Immune Cell image

### `models/`
Contains the deep learning model files and training scripts:
- `dlcnn_MS_CLAHE.keras` – Final double-layer CNN
- `multiclass_MS_CLAHE.keras` – Final multiclass CNN 
- `.zip` copies of the models for upload-friendly backups
- `dlcnn_mixedsampling_models.ipynb` – Trains DL model
- `mccnn_mixedsampling_models.ipynb` – Train MC model

### `image/`
Contains a selection of .zip folders of H&E-stained cell type images to serve as uploadable inputs for the classifier and to demonstrate the app’s functionality alongside the demo images:
- Invasive_Tumor
- Prolif_Invasive_Tumor
- DCIS_1
- DCIS_2
- B_Cells
- CD4+_T_Cells
- CD8+_T_Cells
- Stromal

### `www/`
Web assets used in the Shiny app interface:
- `favicon.png`, `favicon.ico` – Browser tab icon
- `logo.png` – App logo

