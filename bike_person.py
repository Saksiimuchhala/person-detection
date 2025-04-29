import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load YOLOv8s model
model = YOLO("yolov8s.pt")

# Define target classes
target_classes = ["person", "bicycle", "motorcycle"]

# Input and output paths
input_path = r"D:\Sakshi muchhala\person detection\test.webm"  # change to folder if needed
output_dir = r"D:\Sakshi muchhala\person detection\output"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate intersection over area (IOA)
def calculate_ioa(box1, box2):
    """
    Calculate the intersection area divided by the area of box1
    box1, box2: [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of box1
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    # Compute the intersection over area (IOA)
    ioa = intersection_area / box1_area
    
    return ioa

# Function to process a single video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    name = os.path.basename(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output path
    output_path = os.path.join(output_dir, f"overlap_{name}")
    if output_path.endswith('.webm'):
        output_path = output_path.replace('.webm', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using YOLOv8
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        
        # Separate person and vehicle detections
        person_boxes = []
        vehicle_boxes = []
        
        # First, collect all detections
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            label = model.names[int(cls)]
            if label in target_classes:
                if label == "person":
                    person_boxes.append((box, label))
                else:  # bicycle or motorcycle
                    vehicle_boxes.append((box, label))
        
        # Check for overlaps and draw bounding boxes
        for vbox, vlabel in vehicle_boxes:
            # Always draw vehicles
            vx1, vy1, vx2, vy2 = vbox.astype(int)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(frame, vlabel, (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for pbox, plabel in person_boxes:
            # Check if person overlaps with any vehicle
            should_draw = True
            for vbox, _ in vehicle_boxes:
                overlap = calculate_ioa(pbox, vbox)
                if overlap >= 0.1:  # 10% threshold
                    should_draw = False
                    break
            
            # Draw person only if no significant overlap with vehicles
            if should_draw:
                px1, py1, px2, py2 = pbox.astype(int)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, plabel, (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Check if input is a file or folder
if os.path.isfile(input_path):
    process_video(input_path)
elif os.path.isdir(input_path):
    for file in os.listdir(input_path):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
            process_video(os.path.join(input_path, file))
else:
    print("Invalid input path.")



# to detect person and bike 
# import cv2
# import os
# from ultralytics import YOLO

# # Load YOLOv8s model
# model = YOLO("yolov8s.pt")

# # Define target classes
# target_classes = ["person", "bicycle", "motorcycle"]

# # Input and output paths
# input_path = r"D:\Sakshi muchhala\person detection\test.webm"  # change to folder if needed
# output_dir = r"D:\Sakshi muchhala\person detection\output"
# os.makedirs(output_dir, exist_ok=True)

# # Function to process a single video
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     name = os.path.basename(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Output path
#     output_path = os.path.join(output_dir, f"bike_{name}")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect objects using YOLOv8
#         results = model.predict(frame, conf=0.5, verbose=False)[0]

#         for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
#             label = model.names[int(cls)]
#             if label in target_classes:
#                 x1, y1, x2, y2 = box.astype(int)
#                 color = (0, 255, 0) if label == "person" else (0, 0, 255)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         out.write(frame)
#         cv2.imshow("Live Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Check if input is a file or folder
# if os.path.isfile(input_path):
#     process_video(input_path)
# elif os.path.isdir(input_path):
#     for file in os.listdir(input_path):
#         if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
#             process_video(os.path.join(input_path, file))
# else:
#     print("Invalid input path.")


