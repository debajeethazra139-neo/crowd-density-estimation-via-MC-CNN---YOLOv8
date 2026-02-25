from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")  # small, COCO-pretrained (person class id = 0)

    input_video = r"C:\Users\lekha\Videos\Screen Recordings\Screen Recording 2025-12-02 193345.mp4"
    output_video = "yolo_people_only.mp4"

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps if fps > 0 else 20,
                          (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls != 0:  # only person
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)
        cv2.imshow("YOLOv8 People", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", output_video)

if __name__ == "__main__":
    main()
