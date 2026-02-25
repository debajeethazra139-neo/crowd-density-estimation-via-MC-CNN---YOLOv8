import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import time
from collections import deque


#   MC-CNN MODEL
class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding='same'), nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding='same'), nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding='same'), nn.ReLU(),
        )
        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding='same'), nn.ReLU(),
            nn.Conv2d(40, 20, 5, padding='same'), nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding='same'), nn.ReLU(),
        )
        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding='same'), nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding='same'), nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding='same'), nn.ReLU(),
        )
        self.fusion_layer = nn.Sequential(nn.Conv2d(30, 1, 1, padding=0))

    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fusion_layer(x)
        return x


def preprocess_patch(patch_bgr, target_size=(768, 512)):
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def density_to_heatmap(dmap_np, out_w, out_h):
    if dmap_np.max() > 0:
        norm = dmap_np / (dmap_np.max() + 1e-6)
    else:
        norm = dmap_np
    heat = (norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.resize(heat_color, (out_w, out_h))


def auto_calibrate(mccnn, yolo, cap, device, num_frames=60, conf_thresh=0.10):
    yolo_total = 0.0
    mcnn_raw_total = 0.0
    frames_used = 0
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)

    while frames_used < num_frames:
        ret, frame = cap.read()
        if not ret: break

        results = yolo(frame, conf=conf_thresh, classes=[0], imgsz=960, verbose=False)
        boxes = results[0].boxes if results[0].boxes is not None else []
        yolo_count = len(boxes)
        yolo_total += yolo_count

        inp = preprocess_patch(frame).to(device)
        with torch.no_grad():
            dmap = mccnn(inp)
        mcnn_raw = float(dmap.sum().item())
        mcnn_raw_total += mcnn_raw
        frames_used += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)

    if frames_used == 0 or mcnn_raw_total == 0 or yolo_total == 0:
        return 1.0

    raw_factor = yolo_total / (mcnn_raw_total + 1e-6)
    calib_factor = raw_factor * 15.0
    return calib_factor


# GUI APP
class VideoProcessor(QThread):
    frame_ready = pyqtSignal(np.ndarray, dict)
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    calibration_done = pyqtSignal(float)
    buffering_done = pyqtSignal()

    def __init__(self, video_path, mode, density_thresh, speed_thresh):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.density_thresh = density_thresh
        self.speed_thresh = speed_thresh
        self._stop = False
        self.calibration_factor = 1.0
        self.buffer_frames = []

    def run(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.emit(f" Using device: {device}")

        mccnn = MC_CNN().to(device)
        mccnn.load_state_dict(torch.load("crowd_counting.pth", map_location=device), strict=False)
        mccnn.eval()
        yolo = YOLO("yolov8n.pt")
        self.log.emit("Models loaded")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log.emit(" Cannot open video")
            return

        # AUTO-CALIBRATE
        self.log.emit(" Auto-calibrating MCNN (60 frames)...")
        self.calibration_factor = auto_calibrate(mccnn, yolo, cap, device, num_frames=60)
        self.calibration_done.emit(self.calibration_factor)
        self.log.emit(f"Calibration factor: {self.calibration_factor:.6f}")

        # BUFFERING PHASE (PRELOAD 30 FRAMES)
        self.log.emit(" Buffering 30 frames for smooth playback...")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buffer_count = min(30, total_frames)

        for i in range(buffer_count):
            ret, frame = cap.read()
            if not ret: break
            self.buffer_frames.append(frame)
            self.progress.emit(i + 1, buffer_count)

        self.buffering_done.emit()
        self.log.emit(" Buffering complete! Starting analysis...")

        frame_idx = 0
        prev_centers = {}
        buffer_idx = 0

        while frame_idx < total_frames and not self._stop:
            # Use buffered frames first, then read live
            if buffer_idx < len(self.buffer_frames):
                frame = self.buffer_frames[buffer_idx]
                buffer_idx += 1
            else:
                ret, frame = cap.read()
                if not ret: break

            overlay = frame.copy()
            orig_h, orig_w = frame.shape[:2]
            mcnn_count = 0.0
            total_persons = 0
            avg_speed = 0.0

            #ALWAYS YOLO FOR GREEN BOXES + SPEED
            results = yolo(frame, conf=0.10, classes=[0], imgsz=960, verbose=False)
            boxes = results[0].boxes if results[0].boxes is not None else []
            speeds = []
            new_centers = {}

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_id = idx
                speed = 0.0
                if track_id in prev_centers:
                    px, py = prev_centers[track_id]
                    speed = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2))
                speeds.append(speed)
                new_centers[track_id] = (cx, cy)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"v={speed:.1f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            prev_centers = new_centers
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            total_persons = len(boxes)

            #MODE-BASED DENSITY
            if self.mode == "SPARSE":
                primary_count = float(total_persons)

            elif self.mode == "DENSE":
                inp = preprocess_patch(frame).to(device)
                with torch.no_grad():
                    dmap = mccnn(inp)
                dnp = dmap.squeeze().cpu().numpy()
                mcnn_raw = float(dnp.sum())
                mcnn_count = max(0.0, mcnn_raw * self.calibration_factor)
                heatmap = density_to_heatmap(dnp, orig_w, orig_h)
                overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
                primary_count = mcnn_count

            else:  # UNKNOWN
                inp = preprocess_patch(frame).to(device)
                with torch.no_grad():
                    dmap = mccnn(inp)
                dnp = dmap.squeeze().cpu().numpy()
                mcnn_raw = float(dnp.sum())
                mcnn_count = max(0.0, mcnn_raw * self.calibration_factor)
                heatmap = density_to_heatmap(dnp, orig_w, orig_h)
                overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
                primary_count = (mcnn_count + total_persons) / 2

            #FIXED: SPEED <= THRESHOLD (NOT >=)
            overcrowding = primary_count > self.density_thresh
            stampede_prone = overcrowding and avg_speed <= self.speed_thresh  # 🔥 CHANGED TO <=

            if avg_speed < 1.0:
                risk_text = "LOW RISK" if not overcrowding else "MEDIUM RISK"
                overcrowding_text = "OVER-CROWDING WARNING" if overcrowding else "NOT OVERCROWDED"
                stampede_text = "NOT STAMPEDE PRONE"
            else:
                if stampede_prone:
                    risk_text = "HIGH RISK"
                    overcrowding_text = "OVER-CROWDING WARNING"
                    stampede_text = "STAMPEDE PRONE"
                elif overcrowding:
                    risk_text = "MEDIUM RISK"
                    overcrowding_text = "OVER-CROWDING WARNING"
                    stampede_text = "NOT STAMPEDE PRONE"
                else:
                    risk_text = "LOW RISK"
                    overcrowding_text = "NOT OVERCROWDED"
                    stampede_text = "NOT STAMPEDE PRONE"

            occupancy = min(100.0, (primary_count / self.density_thresh) * 100.0)

            stats = {
                'density': primary_count,
                'mcnn': mcnn_count,
                'yolo': total_persons,
                'speed': avg_speed,
                'occ': occupancy,
                'risk': risk_text,
                'overcrowding': overcrowding_text,
                'stampede': stampede_text,
                'mode': self.mode,
                'calib': self.calibration_factor
            }
            self.frame_ready.emit(overlay, stats)
            self.progress.emit(frame_idx, total_frames)
            frame_idx += 1

        cap.release()


class CrowdApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.processor = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Crowd Analyzer Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2b2b2b, stop:1 #1a1a1a); color: white; }
            QPushButton { background: #4a90e2; border: none; padding: 12px; border-radius: 8px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background: #357abd; }
            QPushButton:pressed { background: #2a6fb3; }
            QGroupBox { font-weight: bold; border: 2px solid #4a90e2; border-radius: 8px; margin: 10px; padding-top: 10px; }
            QLabel { color: white; font-size: 12px; }
            QDoubleSpinBox { background: #555; color: white; border: 1px solid #4a90e2; padding: 5px; }
            QProgressBar { background: #333; border: 1px solid #4a90e2; color: white; text-align: center; }
        """)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Controls
        ctrl = QGroupBox("Video Controls")
        ctrl_layout = QHBoxLayout(ctrl)
        self.select_btn = QPushButton(" Select Video")
        self.select_btn.clicked.connect(self.select_video)
        self.start_btn = QPushButton(" Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.select_btn)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # Settings
        settings = QGroupBox(" Settings")
        s_layout = QGridLayout(settings)
        s_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["SPARSE", "DENSE", "UNKNOWN"])
        s_layout.addWidget(self.mode_cb, 0, 1)

        s_layout.addWidget(QLabel("Density Threshold:"), 1, 0)
        self.dens_sb = QDoubleSpinBox()
        self.dens_sb.setRange(0.01, 999999.0)
        self.dens_sb.setValue(8.0)
        self.dens_sb.setDecimals(6)
        s_layout.addWidget(self.dens_sb, 1, 1)

        s_layout.addWidget(QLabel("Speed Threshold (≤ for STAMPEDE):"), 2, 0)
        self.speed_sb = QDoubleSpinBox()
        self.speed_sb.setRange(0.01, 999.0)
        self.speed_sb.setValue(5.0)
        self.speed_sb.setDecimals(3)
        s_layout.addWidget(self.speed_sb, 2, 1)

        layout.addWidget(settings)

        # Display
        disp_layout = QHBoxLayout()
        self.video_label = QLabel("Select video to start")
        self.video_label.setMinimumSize(900, 500)
        self.video_label.setStyleSheet("border: 2px solid #4a90e2; border-radius: 8px; background: #333;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        stats = QGroupBox(" LIVE WARNINGS & STATS")
        stats_layout = QVBoxLayout(stats)
        self.stats = {
            "Density": QLabel("--"),
            "MCNN/YOLO": QLabel("--"),
            "Speed": QLabel("--"),
            "Occupancy": QLabel("--"),
            "Risk": QLabel("--"),
            "Overcrowding": QLabel("--"),
            "Stampede": QLabel("--"),
            "Calib": QLabel("--"),
            "Mode": QLabel("--")
        }
        for k, v in self.stats.items():
            v.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            stats_layout.addWidget(v)
        self.progress = QProgressBar()
        stats_layout.addWidget(self.progress)

        disp_layout.addWidget(self.video_label)
        disp_layout.addWidget(stats)
        layout.addLayout(disp_layout)

        self.log = QTextEdit()
        self.log.setMaximumHeight(120)
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.log.append(" Ready! Click 'Select Video' to begin.")

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mkv *.mov);;All (*)")
        if path:
            self.video_path = path
            self.log.append(f" Video: {os.path.basename(path)}")
            self.start_btn.setEnabled(True)

    def start_analysis(self):
        if not self.video_path: return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.status_label.setText(" LOADING MODELS...")
        self.processor = VideoProcessor(
            self.video_path,
            self.mode_cb.currentText(),
            self.dens_sb.value(),
            self.speed_sb.value()
        )
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.progress.connect(self.update_progress)
        self.processor.log.connect(self.log.append)
        self.processor.calibration_done.connect(self.on_calibration_done)
        self.processor.buffering_done.connect(self.on_buffering_done)
        self.processor.start()

    def stop_analysis(self):
        if self.processor:
            self.processor._stop = True
            self.processor.wait()

    def on_calibration_done(self, factor):
        self.status_label.setText(" BUFFERING FRAMES...")

    def on_buffering_done(self):
        self.status_label.setText(" PLAYING ANALYSIS...")

    def on_calibration_done(self, factor):
        self.stats["Calib"].setText(f"Calib: {factor:.6f}")

    def update_frame(self, frame, stats):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(900, 500, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        self.stats["Density"].setText(f"Density: {stats['density']:.1f}")
        self.stats["MCNN/YOLO"].setText(f"MCNN: {stats['mcnn']:.1f} / YOLO: {stats['yolo']:.0f}")
        self.stats["Speed"].setText(f"Speed: {stats['speed']:.1f}")
        self.stats["Occupancy"].setText(f"Occupancy: {stats['occ']:.0f}%")
        self.stats["Risk"].setText(f"Risk: {stats['risk']}")
        self.stats["Overcrowding"].setText(f"Overcrowding: {stats['overcrowding']}")
        self.stats["Stampede"].setText(f"Stampede: {stats['stampede']}")
        self.stats["Calib"].setText(f"Calib: {stats['calib']:.6f}")
        self.stats["Mode"].setText(f"Mode: {stats['mode']}")

    def update_progress(self, cur, total):
        self.progress.setMaximum(total)
        self.progress.setValue(cur)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CrowdApp()
    win.show()
    sys.exit(app.exec())
