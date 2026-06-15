# Crowd Analyzer Pro

Crowd Analyzer Pro is a desktop application for real-time crowd analysis that combines a density-based **MC-CNN** model with **YOLOv8** person detection inside a modern **PyQt6 GUI**. It estimates crowd size, visualizes density as a heatmap, measures motion speed, and classifies the scene into risk levels such as overcrowding and stampede-prone situations.

---

## Features

- **Dual-model fusion**
  - MC-CNN density regression for dense crowds with heavy occlusion.
  - YOLOv8 person detection for sparse or moderately crowded scenes.

- **Automatic cross-model calibration**
  - Uses YOLOv8 counts over the first calibration frames to map MC-CNN raw density outputs into realistic person counts.

- **Adaptive operating modes**
  - **SPARSE**: uses YOLOv8 counts as the primary estimate.
  - **DENSE**: uses MC-CNN density map as the primary estimate.
  - **UNKNOWN**: fuses MC-CNN and YOLOv8 estimates.

- **Risk and safety analytics**
  - Computes average motion speed from frame-to-frame displacement.
  - Flags overcrowding, occupancy percentage, and stampede-prone situations based on configurable thresholds.

- **Modern PyQt6 GUI**
  - Video selection, live visualization, progress bar, and real-time stats panel.
  - Smooth playback via frame buffering.

---

## System Overview

### Input
- User selects a video file (`.mp4`, `.avi`, `.mkv`, `.mov`) from the GUI.
- Frames are read using **OpenCV**.

### Model Loading
- **MC-CNN** is loaded from `crowd_counting.pth`.
- **YOLOv8** is loaded from `yolov8n.pt` or another supported variant.

### Auto-Calibration
During the first calibration frames:

- YOLOv8 detects persons and provides a detection-based count.
- MC-CNN predicts a density map and the raw sum is computed.
- A calibration factor is derived from the ratio between YOLO counts and MC-CNN raw sums.

### Analysis Loop
- Frames are processed in a background **QThread** to keep the GUI responsive.
- YOLOv8 runs on every frame for bounding boxes and speed estimation.
- MC-CNN optionally generates a density heatmap depending on mode.
- Final crowd density comes from YOLOv8, MC-CNN, or a fusion of both.

### Risk Evaluation
- **Overcrowding** is flagged when estimated count exceeds the density threshold.
- **Average speed** is computed from center-point displacement of detections.
- A scene is considered **stampede-prone** when it is overcrowded and the average speed is below or equal to the configured threshold.
- Risk is displayed as **LOW**, **MEDIUM**, or **HIGH**.

---

## Models Used

### MC-CNN
Inspired by the paper *Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (MCNN)*.

- Uses three parallel convolutional columns with different receptive fields.
- Handles scale variation in dense crowds.
- Outputs a single-channel density map.
- Final count estimate is:

```text
sum(density_map) * calibration_factor
```

### YOLOv8
Uses **Ultralytics YOLOv8** for real-time person detection.

- Default model: `yolov8n.pt`
- Only **person class (class 0)** is used.
- Produces bounding boxes and center points.
- Center-point displacement is used to estimate motion speed. Ultralytics documents YOLOv8 as a real-time object detection framework, and YOLO datasets use zero-indexed class numbering. [web:78][web:75]

---

## Modes

### SPARSE
```text
primary_count = YOLO_count
```

Best for scenes where individuals are clearly separated.

### DENSE
```text
primary_count = calibrated_MCNN_count
```

Best for dense or heavily occluded scenes. A heatmap is overlaid on the original frame.

### UNKNOWN
```text
primary_count = (calibrated_MCNN_count + YOLO_count) / 2
```

Best for intermediate crowd conditions.

---

## Risk Logic

### Overcrowding
Overcrowding is determined by comparing `primary_count` to the user-defined **Density Threshold**.

### Occupancy
```text
occupancy = min(100, (primary_count / density_threshold) * 100)
```

### Risk Categories
- **High Risk**: overcrowded and average speed is less than or equal to the speed threshold.
- **Medium Risk**: overcrowded but not stampede-prone.
- **Low Risk**: not overcrowded.

---

## GUI

The PyQt6 interface provides:

### Video Controls
- **Select Video**: choose a video file.
- **Start Analysis**: start processing.
- **Stop**: stop analysis and release resources.

### Settings Panel
- **Mode**: SPARSE / DENSE / UNKNOWN
- **Density Threshold**
- **Speed Threshold**

### Main Display
- Processed video frames
- YOLOv8 bounding boxes
- Optional MC-CNN heatmap
- Live statistics panel showing:
  - Density
  - MCNN and YOLO counts
  - Average speed
  - Occupancy
  - Risk level
  - Overcrowding status
  - Stampede status
  - Calibration factor
  - Current mode

### Logging Area
Displays:
- Device selection
- Model loading
- Calibration progress
- Buffering status
- Playback events

---

## Installation

### Requirements
- Python 3.10+
- PyTorch
- Ultralytics YOLO
- OpenCV
- PyQt6
- NumPy

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m venv venv
```

### Activate Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Typical `requirements.txt`

```txt
torch
torchvision
ultralytics
opencv-python
numpy
PyQt6
```

### Required Model Files
Place these files in the project directory:

- `crowd_counting.pth` — MC-CNN weights
- `yolov8n.pt` — YOLOv8 model file, or update the code to use another YOLOv8 variant

---

## Usage

Run the app:

```bash
python crowd_analyzer.py
```

### Steps
1. Click **Select Video**
2. Choose a crowd video
3. Select **SPARSE**, **DENSE**, or **UNKNOWN**
4. Adjust thresholds if needed
5. Click **Start Analysis**

The app will:

- Load the models
- Auto-calibrate MC-CNN
- Buffer frames for smooth playback
- Display annotated frames with live statistics

Click **Stop** to halt analysis.

---

## Project Structure

```text
.
├── crowd_analyzer.py      # Main GUI and VideoProcessor
├── crowd_counting.pth     # MC-CNN trained weights
├── yolov8n.pt             # YOLOv8 model
├── requirements.txt
└── README.md
```

---

## Datasets and Training

If you trained MC-CNN or fine-tuned YOLOv8 yourself, include details such as:

- MC-CNN training dataset and procedure
- Density map generation method
- YOLOv8 fine-tuning dataset
- Hyperparameters used
- Whether YOLOv8 is pre-trained or custom-trained

---

## Limitations

- MC-CNN calibration depends on YOLOv8 quality during initial frames.
- Speed is measured in **pixels per frame**, not real-world units.
- No explicit multi-object tracker is used.
- Temporary detection indices may reduce motion consistency.

---

## Future Improvements

- Add a tracker such as **DeepSORT** for better speed estimation
- Add ROI-based or zone-based density analysis
- Add support for live RTSP streams
- Save logs and alert history
- Export reports for post-event analysis

---

## References

- MC-CNN crowd counting research
- Ultralytics YOLOv8 documentation
- Crowd density estimation and surveillance analytics literature
General tutorials and surveys on crowd counting and density estimation.
