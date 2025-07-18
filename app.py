import cv2
import torch
from ultralytics import YOLO
import os
import random
import time
from tqdm import tqdm

# M4 Max GPU
if not torch.backends.mps.is_available():
    print("MPS not available. Falling back to CPU.")
    device = torch.device("cpu")
else:
    print("MPS available. Using M4 Max GPU.")
    device = torch.device("mps")

# Load trainedmodel
model_path = "/Users/anton/PycharmProjects/100WinnerTracking/running_detection/yolov8m_combined/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} does not exist.")
    exit()
model = YOLO(model_path)
model.to(device)

def process_video(input_path, output_dir="runs/track/output", show_video=True, skip_frames=1, target_size=(640, 640), finish_line_start_time=24, valid_winner_time=25, excluded_ids=[97]):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {input_path} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(100)}

    frame_count = 0
    winner_found = False
    winner_id = None

    #edit the position of the finish line
    finish_line_position = int(target_size[1] * 0.60)
    finish_line_start_frame = int(finish_line_start_time * fps)
    valid_winner_frame = int(valid_winner_time * fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame_resized = cv2.resize(frame, target_size)
        frame_resized = cv2.convertScaleAbs(frame_resized, alpha=1.2, beta=10)

        results = model.track(
            source=frame_resized,
            device=device,
            conf=0.5,
            iou=0.7,
            imgsz=target_size[0],
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml"
        )

        annotated_frame = frame_resized.copy()
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                color = colors[int(track_id) % 100]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"ID#{int(track_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)

                if not winner_found and center_y >= finish_line_position and frame_count >= valid_winner_frame and int(track_id) not in excluded_ids:
                    winner_id = int(track_id)
                    winner_found = True
                    print(f"🎉🏅 Winner: Player ID#{winner_id} at frame {frame_count}")

        if frame_count >= finish_line_start_frame or winner_found:
            cv2.line(annotated_frame, (0, finish_line_position), (target_size[0], finish_line_position), (0, 0, 255), 4)
            cv2.putText(annotated_frame, "FINISH", (10, finish_line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if winner_found:
            cv2.putText(annotated_frame, f"Winner: ID#{winner_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        out.write(annotated_frame)

        if show_video:
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracked video saved to {output_path}")

    if winner_found:
        print(f"🎖️ Race winner is Player ID#{winner_id}")
    else:
        print("⚠️ No winner detected.")


if __name__ == "__main__":
    input_path = "/Users/anton/PycharmProjects/100WinnerTracking/kid_race.mp4"
    if not os.path.exists(input_path):
        print(f"Error: Video file {input_path} does not exist.")
    else:
        process_video(input_path, show_video=True, skip_frames=1, target_size=(640, 640))