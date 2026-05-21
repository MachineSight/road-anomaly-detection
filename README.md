# Road Anomaly Detection

A computer vision project for detecting road anomalies using YOLO-based object detection.

This repository contains:
- Trained YOLO weight files for multiple learning rates.
- Training/evaluation result CSV files and plotting notebooks.
- A simple inference script with a file picker GUI.
- Supporting notebooks for experimentation and visualization.

## Project Structure

```text
Road Anomaly Detection/
├── requirements.txt
├── run_inference.py
├── source-code.ipynb
├── results/
│   ├── plot_predictions.ipynb
│   ├── plot_val_mAP@50.ipynb
│   ├── results_lr_0.00001.csv
│   ├── results_lr_0.0001.csv
│   ├── results_lr_0.001.csv
│   ├── results_lr_0.01.csv
│   ├── results_lr_0.1.csv
│   ├── validation_mAP@0.5.png
│   └── Sample images/
│       └── results/
└── weights/
    ├── lr_0.00001.pt
    ├── lr_0.0001.pt
    ├── lr_0.001.pt
    ├── lr_0.01.pt
    └── lr_0.1.pt
```

## File-by-File Description

### `requirements.txt`
Minimal Python dependencies used in this project:
- `opencv-python` for image processing/color conversion.
- `ultralytics` for YOLO model loading and inference.
- `matplotlib` for plotting and display.

### `run_inference.py`
A desktop-style inference script that:
1. Opens a file picker to choose a `.pt` YOLO weight file.
2. Opens a second file picker to choose an input image (`.jpg/.jpeg/.png`).
3. Runs YOLO prediction on the selected image.
4. Draws detection boxes and displays the result with Matplotlib.

Notes:
- Uses Tkinter dialogs (`filedialog.askopenfilename`) for model/image selection.
- Does not save output to disk by default (`save=False` in prediction call).
- Intended for quick manual testing of trained models.

### `source-code.ipynb`
Primary notebook for project development/experimentation (training, validation, and workflow steps).

### `results/`
Stores experiment outputs and visualization tools.

- `results_lr_*.csv`: Per-learning-rate training logs/metrics exported by training runs.
- `plot_predictions.ipynb`: Notebook for visualizing prediction results.
- `plot_val_mAP@50.ipynb`: Notebook for plotting validation mAP@0.5 trends/comparisons.
- `validation_mAP@0.5.png`: Generated chart for validation mAP at IoU 0.5.
- `Sample images/results/`: Example prediction outputs.

### `weights/`
Trained YOLO model checkpoints (`.pt`) corresponding to different learning rates:
- `lr_0.00001.pt`
- `lr_0.0001.pt`
- `lr_0.001.pt` **(Best performing lr)**
- `lr_0.01.pt`
- `lr_0.1.pt`

These files can be loaded directly by `run_inference.py`.

## Setup

## 1) Create and activate a virtual environment (recommended)

Linux/macOS:

```bash
python3 -m venv .road-anomaly-venv
source .road-anomaly-venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .road-anomaly-venv
.road-anomaly-venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Run inference

```bash
python run_inference.py
```

Then:
1. Select one model from `weights/`.
2. Select an image.
3. View detections in the Matplotlib window.

## How to Interpret the Experiment Artifacts

- The project includes multiple training runs with different learning rates.
- Use files in `results/results_lr_*.csv` to compare performance across runs.
- Use `results/plot_val_mAP@50.ipynb` to visualize and compare mAP@0.5.
- Match the best-performing CSV run with its corresponding checkpoint in `weights/`.

## Typical Workflow

1. Train or experiment in `source-code.ipynb`.
    - The dataset information is contained in this notebook.
2. Review numeric metrics in `results/results_lr_*.csv`.
3. Visualize trends using the plotting notebooks in `results/`.
4. Select the best checkpoint from `weights/`.
5. Run `run_inference.py` for qualitative testing on images.

## License

This project is licensed under the MIT License — see the LICENSE file for details.
