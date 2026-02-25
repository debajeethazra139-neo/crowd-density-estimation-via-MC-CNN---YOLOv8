Crowd Analyzer Pro: MC-CNN + YOLOv8 Crowd Density and Stampede Risk Monitor
Crowd Analyzer Pro is a desktop application for real-time crowd analysis that combines a density-based MC-CNN model with YOLOv8 person detection inside a modern PyQt6 GUI. It estimates crowd size, visualizes density as a heatmap, measures motion speed, and classifies the scene into risk levels such as overcrowding and stampede-prone.

1. Key Features
Dual-model fusion

MC-CNN density regression for dense crowds with heavy occlusion.

YOLOv8 person detection for sparse or moderately crowded scenes.

Automatic cross-model calibration

Uses YOLOv8 counts over the first N frames to calibrate MC-CNN raw density outputs into realistic person counts.

Adaptive operating modes

SPARSE: use YOLOv8 counts as the primary density estimate.

DENSE: use MC-CNN density map (heatmap overlay) as the primary estimate.

UNKNOWN: fuse both (average of calibrated MC-CNN and YOLOv8) to handle intermediate conditions.

Risk and safety analytics

Computes average motion speed from frame-to-frame displacement of detected persons.

Flags overcrowding, occupancy percentage, and stampede-prone situations based on configurable density and speed thresholds.

Modern PyQt6 GUI

Video selection, live visualization, progress bar, and real-time stats panel.

Smooth playback via frame buffering for a better user experience.

2. System Overview
2.1 Pipeline
Input

User selects a video file (.mp4, .avi, .mkv, .mov) from the GUI file dialog.

Frames are read using OpenCV.

Model loading

MC-CNN is loaded from crowd_counting.pth.

Ultralytics YOLOv8 is loaded from yolov8n.pt (or another YOLOv8 variant).

Auto-calibration (first 60 frames)

For each calibration frame:

YOLOv8 detects persons and provides a detection-based count.

MC-CNN predicts a density map and the raw sum of densities is computed.

The ratio between YOLO counts and MC-CNN raw sums across calibration frames is used to derive a calibration factor, scaled by a constant, to map MC-CNN outputs to approximate person counts.

Analysis loop

Frames are processed in a background QThread (VideoProcessor) to keep the GUI responsive.

YOLOv8 runs on every frame to draw green bounding boxes and compute motion speed.

Depending on the selected mode, MC-CNN may also produce a density map and heatmap overlay.

The final “density” value (crowd count estimate) comes from YOLO, MC-CNN, or their fusion depending on mode.

Risk evaluation

Overcrowding is flagged when estimated count exceeds the user-defined density threshold.

Average speed is computed from center-point displacement of detections between frames.

A scene is considered stampede-prone when it is overcrowded AND the average speed is below or equal to a configurable threshold (indicating a dense, slow-moving or stalled crowd).

Risk is summarized as LOW, MEDIUM, or HIGH, and displayed live.

Visualization

The GUI displays the processed frame with overlays and a live stats panel showing counts, speeds, occupancy, risk labels, and calibration factor.

3. Models Used
3.1 MC-CNN (Multi-Column CNN)
Inspired by “Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (MCNN)”.

Consists of three parallel convolutional columns with different receptive fields to handle varying scales.

Outputs a single-channel density map; integrating this map gives an estimate of the number of people.

In this project:

Input: RGB frame resized to (768, 512), normalized to [0, 1].

Output: density map with the same spatial size (after upsampling if needed).

Final count estimate is sum(density_map) * calibration_factor.

3.2 YOLOv8 (Ultralytics)
Lightweight YOLOv8 model (yolov8n.pt by default) used for real-time person detection.

Only the person class (class 0) is used for detection.

Outputs are bounding boxes, confidences, and class IDs.

In this project:

Each detection yields a bounding box and a center point.

Frame-to-frame displacement of centers is used to estimate per-target speed (in pixels per frame).

The number of detected persons provides a direct count in SPARSE mode, and also supervises the MC-CNN calibration.

4. Risk Logic and Modes
4.1 Modes
SPARSE mode

primary_count = YOLO_count

Best for videos where individuals are well separated and detection is reliable.

DENSE mode

primary_count = calibrated_MCNN_count

MC-CNN density map is visualized as a heatmap blended with the original frame.

Preferred for very dense or heavily occluded crowds where detection may fail.

UNKNOWN mode

primary_count = (calibrated_MCNN_count + YOLO_count) / 2

Uses fusion to hedge between density and detection.

4.2 Overcrowding and Stampede Logic
Overcrowding

Determined by comparing primary_count to the Density Threshold set in the GUI.

Occupancy (%) is computed as:

occupancy = min(100, (primary_count / density_threshold) * 100).

Average speed

Mean of per-person speeds (pixel displacement between frames) for each frame.

If no persons are detected, speed is reported as 0.

Risk categorization

If average speed is very low (e.g., < 1.0), the system reports low or medium risk depending on overcrowding.

If speed is higher, risk is upgraded:

Overcrowded and average speed ≤ speed threshold → High risk, Stampede-prone.

Overcrowded but speed above threshold → Medium risk, not stampede-prone.

Not overcrowded → Low risk, not stampede-prone.

5. GUI and User Controls
The GUI is implemented with PyQt6 and provides:

Video Controls

Select Video: open a file dialog to choose a video.

Start Analysis: launch the VideoProcessor thread to begin processing.

Stop: stop processing and release resources.

Settings Panel

Mode: choose between SPARSE, DENSE, and UNKNOWN.

Density Threshold: numeric threshold for overcrowding detection (default 8.0; adjust based on scene).

Speed Threshold (≤ for STAMPEDE): threshold on average speed used when deciding stampede-prone situations.

Main Display

Large video pane showing the processed frame with:

YOLOv8 person bounding boxes and per-target speeds.

Optional MC-CNN density heatmap overlay (in DENSE/UNKNOWN modes).

Right-hand stats panel with live text for:

Density (current crowd count estimate)

MCNN and YOLO counts

Average speed

Occupancy percentage

Risk level

Overcrowding status

Stampede status

Calibration factor

Current mode

Progress bar indicating current frame vs total frames.

Logging area

Text log showing device selection, model loading, calibration progress, buffering, and playback status.

The video processing runs in a dedicated QThread to keep the GUI responsive, which is a standard pattern in PyQt-based vision applications.

6. Installation
6.1 Requirements
Python 3.10+

PyTorch (with CUDA if GPU acceleration is desired)

Ultralytics YOLO

OpenCV

PyQt6

NumPy

6.2 Setup
bash
git clone https://github.com/<debajeethazra139-neo>/<crowd-density-estimation-via-MC-CNN---YOLOv8>.git
cd <crowd-density-estimation-via-MC-CNN---YOLOv8>

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
Typical requirements.txt entries (adapt as needed):

text
torch
torchvision
ultralytics
opencv-python
numpy
PyQt6
Place your trained MC-CNN weights and YOLO weights in the project directory:

crowd_counting.pth – MC-CNN weights

yolov8n.pt – YOLOv8 model file (or adjust the path in the code)

7. Usage
Run the GUI
bash
python <your_main_file>.py
Steps in the app:

Click Select Video and choose a crowd video.

Choose the Mode (SPARSE / DENSE / UNKNOWN).

Adjust Density Threshold and Speed Threshold if needed.

Click Start Analysis.

The app will:

Load the models.

Auto-calibrate MC-CNN using the first 60 frames.

Buffer 30 frames for smooth playback.

Start displaying annotated frames with live stats.

Click Stop to halt analysis.

8. Project Structure (example)
text
.
├── crowd_analyzer.py         # Main GUI and VideoProcessor (PyQt6 app)
├── crowd_counting.pth        # MC-CNN trained weights
├── yolov8n.pt                # YOLOv8 model file (or other variant)
├── requirements.txt
└── README.md
Adjust this section if you split the code into multiple modules.

9. Datasets and Training (Optional)
If you trained MC-CNN or fine-tuned YOLOv8 yourself, briefly describe:

MC-CNN training dataset and procedure

e.g., ShanghaiTech Part A/B with density map generation via Gaussian kernels.

YOLOv8 training / fine-tuning

Dataset type (COCO-based, custom surveillance data, etc.) and hyperparameters.

If you used pre-trained YOLOv8 (COCO), mention that it is used as-is for the person class.

10. Limitations and Future Work
MC-CNN calibration is derived from YOLOv8 and may be sensitive to detection quality in the initial frames.

Speed is measured in pixels per frame, not physical units; different camera setups may require different thresholds.

No explicit multi-object tracking ID; center-based speed uses temporary indices within a frame.

Potential improvements:

Add a proper multi-object tracker (e.g., DeepSORT) to get more stable speed estimates.

Add ROI-based or zone-based density analysis.

Add support for live RTSP streams and saving risk logs.

11. References
MC-CNN crowd counting papers and implementations.

Ultralytics YOLOv8 documentation and examples.

General tutorials and surveys on crowd counting and density estimation.