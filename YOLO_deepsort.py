from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

#Model
model = YOLO(r"") # Model pre-trained/Fine-tuned Path

# Start DeepSort
tracker = DeepSort(max_age=30) # Important to mention that max_age must be fine-tuned to the video lenght and objectives

# Video setting
video_input_path = r"" # Video Input
video_output_path = r".mp4" # Video Output (keep the .mp4)
cap = cv2.VideoCapture(video_input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#Tracker timer
track_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO
    results = model(frame)[0]
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        detections.append(([x1, y1, x2, y2], conf.item(), int(cls.item())))

    # DeepSort tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Box plotting, class and timer
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        track_id = track.track_id

        #Update timers in track_id
        if track_id not in track_times:
            track_times[track_id] = 0
        track_times[track_id] += 1
        time_sec = track_times[track_id] / fps #Convert frames to seconds

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id} | {time_sec:.1f}s", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()