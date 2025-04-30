# import cv2
# import numpy as np
# import os
# from ultralytics import YOLO
# import mediapipe as mp

# # Paths
# INPUT_PATH = r"D:\Sakshi muchhala\person detection\test.webm"
# OUTPUT_PATH = r"D:\Sakshi muchhala\person detection\output"

# # Initialize YOLO and MediaPipe
# model = YOLO("yolov8s.pt")
# mp_pose = mp.solutions.pose
# pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# # Class IDs for YOLO
# PERSON_CLASS = 0
# VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # Bicycle, Car, Motorcycle, Bus, Truck

# def is_overlapping(boxA, boxB):
#     xa1, ya1, xa2, ya2 = boxA
#     xb1, yb1, xb2, yb2 = boxB
#     return not (xa2 < xb1 or xa1 > xb2 or ya2 < yb1 or ya1 > yb2)

# def is_sitting(pose_landmarks):
#     try:
#         hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
#         knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
#         if abs(hip - knee) < 0.1:
#             return True
#     except:
#         pass
#     return False

# def process_video(video_path, output_path):
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)[0]
#         boxes = results.boxes.xyxy.cpu().numpy()
#         classes = results.boxes.cls.cpu().numpy().astype(int)

#         person_boxes = [boxes[i] for i in range(len(boxes)) if classes[i] == PERSON_CLASS]
#         vehicle_boxes = [boxes[i] for i in range(len(boxes)) if classes[i] in VEHICLE_CLASSES]

#         for box in person_boxes:
#             x1, y1, x2, y2 = map(int, box)
#             person_crop = frame[y1:y2, x1:x2]
#             person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
#             results = pose_estimator.process(person_rgb)

#             sitting = is_sitting(results.pose_landmarks) if results.pose_landmarks else False

#             on_vehicle = any(is_overlapping([x1, y1, x2, y2], vbox) for vbox in vehicle_boxes)

#             if on_vehicle and sitting:
#                 continue  # Ignore person sitting on vehicle

#             color = (0, 255, 0)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         writer.write(frame)
#         cv2.imshow("Detection",frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     writer.release()

# # Process all videos in input path
# if os.path.isdir(INPUT_PATH):
#     for file in os.listdir(INPUT_PATH):
#         if file.endswith(('.mp4', '.avi', '.mov', '.webm')):
#             input_file = os.path.join(INPUT_PATH, file)
#             output_file = os.path.join(OUTPUT_PATH, f"processed_{file}")
#             process_video(input_file, output_file)
# else:
#     output_file = os.path.join(OUTPUT_PATH, f"processed_{os.path.basename(INPUT_PATH)}")
#     process_video(INPUT_PATH, output_file)

# print("Processing complete.")


# pedestrian detection
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# from ultralytics import YOLO

# # Load YOLOv8 model
# yolo_model = YOLO("yolov8n.pt")

# # Input and output paths
# input_path = r"D:\Sakshi muchhala\person detection\road_vid.mp4"
# output_path = r"D:\Sakshi muchhala\person detection\output"
# os.makedirs(output_path, exist_ok=True)

# # Vehicle classes
# vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck", "auto rickshaw"]

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# # Helper: check if person is ON vehicle (not just around it)
# def is_person_on_vehicle(person_box, vehicle_boxes):
#     x1, y1, x2, y2 = person_box
#     person_bottom = (x1 + (x2-x1)//2, y2)  # Bottom center point of person
    
#     for vx1, vy1, vx2, vy2 in vehicle_boxes:
#         # Check if bottom of person is inside vehicle box
#         if (vx1 <= person_bottom[0] <= vx2 and 
#             vy1 <= person_bottom[1] <= vy2 + 10):  # +10 pixel tolerance
#             return True
        
#         # Calculate overlap percentage
#         overlap_x1 = max(x1, vx1)
#         overlap_y1 = max(y1, vy1)
#         overlap_x2 = min(x2, vx2)
#         overlap_y2 = min(y2, vy2)
        
#         if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
#             overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
#             person_area = (x2 - x1) * (y2 - y1)
#             # Only consider person ON vehicle if substantial overlap
#             if overlap_area / person_area > 0.5:  # More strict threshold
#                 return True
    
#     return False

# # Process a single video
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     name = os.path.basename(video_path)
#     out_video_path = os.path.join(output_path, f"filtered_{name}")

#     width, height = int(cap.get(3)), int(cap.get(4))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     ext = os.path.splitext(video_path)[-1].lower()

#     # Video writer
#     if ext == ".avi":
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     elif ext == ".mov":
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     elif ext in [".mkv", ".webm"]:
#         fourcc = cv2.VideoWriter_fourcc(*'VP80')
#     else:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
#         if frame_count % 3 != 0:  # Process every 3rd frame for performance
#             out.write(frame)
#             continue

#         vehicle_boxes = []
#         person_boxes = []

#         # Detect objects using YOLOv8
#         detections = yolo_model.predict(frame, conf=0.5, verbose=False)[0]

#         for box, cls in zip(detections.boxes.xyxy.cpu().numpy(), detections.boxes.cls.cpu().numpy()):
#             label = yolo_model.names[int(cls)]
#             box = box.astype(int)
#             if label == "person":
#                 person_boxes.append(box)
#             elif label in vehicle_classes:
#                 vehicle_boxes.append(box)
#                 # No bounding boxes for vehicles

#         # Process each detected person
#         for person_box in person_boxes:
#             x1, y1, x2, y2 = person_box
            
#             # Check if person is ON vehicle (riding/sitting)
#             on_vehicle = is_person_on_vehicle(person_box, vehicle_boxes)
            
#             if not on_vehicle:
#                 # Draw boxes around ALL pedestrians (not on vehicles)
#                 # This includes people near/around vehicles
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, "Pedestrian", (x1, y1-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             # No boxes for people ON vehicles

#         # Write frame and show
#         out.write(frame)
#         cv2.imshow("Filtered Output", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Main: process one video or multiple
# if os.path.isdir(input_path):
#     for file in os.listdir(input_path):
#         if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
#             process_video(os.path.join(input_path, file))
# else:
#     process_video(input_path)




import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Input and output paths
input_path = r"D:\Sakshi muchhala\person detection\test.webm"
output_path = r"D:\Sakshi muchhala\person detection\output"
os.makedirs(output_path, exist_ok=True)

# Vehicle classes
vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck", "auto rickshaw"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Check if person is likely riding (sitting with feet raised)
def is_riding_pose(landmarks, bbox_bottom_y, image_height):
    try:
        # Get landmark Y coordinates in pixels
        l_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
        r_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
        l_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height
        r_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height
        l_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height
        r_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height

        avg_hip_y = (l_hip_y + r_hip_y) / 2
        avg_knee_y = (l_knee_y + r_knee_y) / 2
        avg_ankle_y = (l_ankle_y + r_ankle_y) / 2

        # Criteria for sitting:
        # - Knees close to hips
        # - Ankles significantly higher than bottom of bounding box
        knees_near_hips = abs(avg_hip_y - avg_knee_y) < 60  # pixels
        feet_off_ground = avg_ankle_y < bbox_bottom_y - 40  # at least 40px above ground

        return knees_near_hips and feet_off_ground
    except:
        return False

# Process a single video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    name = os.path.basename(video_path)
    out_path = os.path.join(output_path, f"last_{name}")

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ext = os.path.splitext(video_path)[-1].lower()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        detections = yolo_model.predict(frame, conf=0.5, verbose=False)[0]

        vehicle_boxes = []
        person_boxes = []

        for box, cls in zip(detections.boxes.xyxy.cpu().numpy(), detections.boxes.cls.cpu().numpy()):
            label = yolo_model.names[int(cls)]
            box = box.astype(int)
            if label == "person":
                person_boxes.append(box)
            elif label in vehicle_classes:
                vehicle_boxes.append(box)

        for person_box in person_boxes:
            x1, y1, x2, y2 = person_box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_roi)

            riding = False
            if results.pose_landmarks:
                riding = is_riding_pose(results.pose_landmarks.landmark, y2, height)

            if not riding:
                # Draw only if not riding
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Pedestrian", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Filtered Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main
if os.path.isdir(input_path):
    for file in os.listdir(input_path):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
            process_video(os.path.join(input_path, file))
else:
    process_video(input_path)
