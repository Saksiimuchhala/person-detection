# main script to detect the cattles
import cv2
import os
from ultralytics import YOLO
import time
# Load pre-trained YOLOv8 model
model = YOLO(r"D:\Sakshi muchhala\person detection\yolov8s.pt")  # or yolov8n.pt for faster inference

# Animal classes from COCO
animal_classes = ['bear', 'cat', 'cow', 'dog', 'horse', 'sheep']

# Get class IDs of animal classes
model_names = model.names
animal_class_ids = [cls_id for cls_id, name in model_names.items() if name in animal_classes]

# Input path: can be a single video file or a folder of videos
input_path = r"D:\Sakshi muchhala\person detection\highway.mp4"  # Folder or single file
output_dir = r"D:\Sakshi muchhala\person detection\cattle_output"
os.makedirs(output_dir, exist_ok=True)

# Get list of video files to process
if os.path.isfile(input_path):
    video_files = [input_path]
else:
    video_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'))
    ]

# Process each video
for video_path in video_files:
    cap = cv2.VideoCapture(video_path)

    # Output file setup
    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"8s_{filename}")
    
    # Get the extension of the video file
    ext = os.path.splitext(filename)[1].lower()
    
    # Choose codec based on the file extension
    if ext == '.webm':
        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 for .webm
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # MPEG-4 for .avi
    elif ext == '.mov':
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 for .mov
    elif ext == '.mkv':
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 for .mkv
    elif ext == '.flv':
        fourcc = cv2.VideoWriter_fourcc(*'FLV1')  # Flash Video
    elif ext == '.wmv':
        fourcc = cv2.VideoWriter_fourcc(*'WMV1')  # Windows Media Video
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default for .mp4 and unknowns

    # Set up the output video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {filename}")
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in animal_class_ids:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    cap.release()
    out.release()
    elapsed_time = time.time() - start_time
    print(f"Saved: {output_path} (Processed in {elapsed_time:.2f} seconds)")
    print(f"Saved: {output_path}")

cv2.destroyAllWindows()
print(" All videos processed.")