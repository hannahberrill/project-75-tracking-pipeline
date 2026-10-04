# Local BoT-SORT Pipeline

This folder contains the local web interface and the code/assets needed to run the YOLOv4 detection, BoT-SORT tracking, interpolation, optional visualisation, and optional metric-evaluation pipeline. Source videos, detection CSVs, and ground-truth XML files are not included; upload the inputs you need in the browser.

## Requirements

- Python 3.10 or newer
- A local Python virtual environment (you can use one you already have)
- About 25 MB for this bundle; Python packages, especially PyTorch, require additional disk space

If your existing environment has the needed packages, you can try it directly. Otherwise, activate your venv and install the listed requirements:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline_web\server.py
```

### macOS / Linux

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline_web/server.py
```

Open the local URL printed by the server, normally http://127.0.0.1:8765. Keep the terminal open while using the page. Select **Shut down server** in the page or press Ctrl+C in the terminal to stop it.

## Testing

For a quick first test, upload a short video and leave the advanced model paths at their defaults. The bundled YOLOv4-tiny weights run with OpenCV on CPU; CPU inference can be slow. To test tracking without YOLO, upload a detections CSV instead. Ground-truth XML is optional unless **Run tracking metrics** is enabled.

Output files are temporary and available to download from the page during that server session. Stopping the server removes its staged uploads and outputs.

## Existing environments

A venv can be reused if it contains `ultralytics`, `opencv-python`, `numpy`, and `pandas`; `scipy` is needed for metrics. Use the Python executable from that venv to start the server. Do not copy a venv from another machine: environments commonly contain OS- and path-specific binaries.

For a CPU-only environment, install a CPU build of PyTorch using the official PyTorch install selector for your operating system, then install the remaining packages from `requirements.txt`. The default `pip` choice may install a larger CUDA-enabled PyTorch build on some systems.
