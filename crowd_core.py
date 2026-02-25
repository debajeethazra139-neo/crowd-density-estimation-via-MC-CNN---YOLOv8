# crowd_core.py
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import os
import time
from collections import deque

# put your MC_CNN class, preprocess_patch, density_to_heatmap, format_time, auto_calibrate here (unchanged)


def run_crowd_analysis(
        video_path,
        user_mode,             # "SPARSE", "DENSE", "UNKNOWN"
        density_threshold,     # float
        speed_threshold,       # float
        yolo_model_path="yolov8n.pt",
        capacity_max=100.0,
        calib_frames=60
):
    """
    This is your main processing function.
    It does everything: loads models, runs calibration, processes video, saves output.
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    mccnn = MC_CNN().to(device)
    mccnn.load_state_dict(torch.load("crowd_counting.pth", map_location=device), strict=False)
    mccnn.eval()
    print("MC-CNN model loaded.")

    yolo = YOLO(yolo_model_path)
    print("YOLOv8 model loaded.")

    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    global CALIBRATION_FACTOR
    CALIBRATION_FACTOR = 1.0  # default
    CALIBRATION_FACTOR = auto_calibrate(mccnn, yolo, cap, device, num_frames=calib_frames)

    output_video_path = "output_density_yolo_gui.mp4"

    codecs = [cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'XVID')]
    out = None
    for codec in codecs:
        out = cv2.VideoWriter(output_video_path, codec, fps, (w, h))
        if out.isOpened():
            break
        out.release()
    if not out or not out.isOpened():
        output_video_path = output_video_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    prev_centers = {}
    frame_idx = 0
    start_time = time.time()
    primary_hist = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        # ---------- MC-CNN ----------
        orig_h, orig_w = frame.shape[:2]
        inp = preprocess_patch(frame).to(device)
        with torch.no_grad():
            dmap = mccnn(inp)
        dnp = dmap.squeeze().cpu().numpy()
        mcnn_raw = float(dnp.sum())
        mcnn_count = max(0.0, mcnn_raw * CALIBRATION_FACTOR)
        heatmap = density_to_heatmap(dnp, orig_w, orig_h)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)

        # ---------- YOLO ----------
        results = yolo(frame, conf=0.10, classes=[0], imgsz=960, verbose=False)
        boxes = results[0].boxes if results[0].boxes is not None else []
        speeds = []
        new_prev_centers = {}
        total_persons = len(boxes)

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track_id = idx
            if track_id in prev_centers:
                px, py = prev_centers[track_id]
                speed = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2))
            else:
                speed = 0.0

            speeds.append(speed)
            new_prev_centers[track_id] = (cx, cy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"v={speed:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        prev_centers = new_prev_centers
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # ---------- choose primary based on user_mode ----------
        if user_mode == "SPARSE":
            primary_count = float(total_persons)
            primary_source = "YOLO"
        elif user_mode == "DENSE":
            primary_count = mcnn_count
            primary_source = "MCNN"
        else:
            primary_count = float(total_persons)
            primary_source = "YOLO+MCNN"

        primary_hist.append(primary_count)
        primary_smooth = sum(primary_hist) / len(primary_hist)

        occupancy = 0.0
        if capacity_max > 0:
            occupancy = min(100.0, (primary_count / capacity_max) * 100.0)

        # ---------- risk + stampede ----------
        if avg_speed < 1.0:
            risk_text = "LOW RISK" if primary_count <= density_threshold else "MEDIUM RISK"
            risk_color = (0, 255, 0) if primary_count <= density_threshold else (0, 255, 255)
            stampede_text = "NOT STAMPEDE PRONE"
            stampede_color = (0, 255, 0)
        else:
            if primary_count > density_threshold and avg_speed >= speed_threshold:
                risk_text = "HIGH RISK"
                risk_color = (0, 0, 255)
                stampede_text = "STAMPede PRONE"
                stampede_color = (0, 0, 255)
            elif primary_count > density_threshold:
                risk_text = "MEDIUM RISK"
                risk_color = (0, 255, 255)
                stampede_text = "NOT STAMPEDE PRONE"
                stampede_color = (0, 255, 0)
            else:
                risk_text = "LOW RISK"
                risk_color = (0, 255, 0)
                stampede_text = "NOT STAMPEDE PRONE"
                stampede_color = (0, 255, 0)

        # ---------- overlays (same as before, just using density_threshold, speed_threshold, etc.) ----------
        # ... (copy your overlay drawing code here, unchanged, using primary_count, mcnn_count, total_persons, avg_speed, etc.)

        elapsed_sec = time.time() - start_time
        time_str = format_time(elapsed_sec)
        cv2.putText(overlay, f"Time: {time_str}", (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame_idx += 1

        out.write(overlay)
        cv2.imshow("Crowd Density and Risk", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", output_video_path)
