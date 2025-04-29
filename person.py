# -*- coding: utf-8 -*-
# final person detection script
"""
Pedestrian Detection Script
Modified for local execution from Colab version
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Load the YOLOv8 model
model = YOLO(r'D:\Sakshi muchhala\person detection\yolo11n.pt')  # Update with your local model path

# Define input video path - update with your local video path
video_path = r'D:\Sakshi muchhala\person detection\my_video.webm'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output directory if it doesn't exist
output_dir = r'D:\Sakshi muchhala\person detection\output'
os.makedirs(output_dir, exist_ok=True)

# Create output video writer
output_path = os.path.join(output_dir, 'my_vid.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Constants
PERSON_CLASS_ID = 0      # COCO dataset: 0 is person
BICYCLE_CLASS_ID = 1     # COCO dataset: 1 is bicycle
MOTORCYCLE_CLASS_ID = 3  # COCO dataset: 3 is motorcycle
CAR_CLASS_ID = 2         # COCO dataset: 2 is car

# Reduce resolution for faster processing
scale_factor = 0.5  # Process at 50% resolution

# Proximity threshold to consider a person "on" a vehicle (in pixels)
PROXIMITY_THRESHOLD =60 # Adjust based on your video resolution // before it was 50

frame_count = 0
start_time = time.time()

# Create a window to display the output
cv2.namedWindow('Pedestrian Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pedestrian Detection', 1280, 720)

# For performance tracking
display_interval = 30  # Update FPS display every 30 frames
last_update_time = time.time()
last_update_frame = 0
current_fps = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Only process every 3rd frame to speed things up
        if frame_count % 3 != 0:
            continue

        # Resize frame for faster processing
        process_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        # Run YOLOv8 inference
        results = model(process_frame)

        # Scale boxes back to original frame size
        boxes = results[0].boxes.xyxy.cpu().numpy() / scale_factor
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Separate people and vehicles
        people_indices = np.where(classes == PERSON_CLASS_ID)[0]
        vehicle_indices = np.where((classes == BICYCLE_CLASS_ID) | 
                                  (classes == MOTORCYCLE_CLASS_ID) | 
                                  (classes == CAR_CLASS_ID))[0]

        people_boxes = boxes[people_indices] if len(people_indices) > 0 else np.array([])
        people_conf = confidences[people_indices] if len(people_indices) > 0 else np.array([])
        vehicle_boxes = boxes[vehicle_indices] if len(vehicle_indices) > 0 else np.array([])

        # Initialize people status (True = pedestrian, False = on vehicle)
        pedestrian_status = np.ones(len(people_indices), dtype=bool)

        # Check each person if they're close to a vehicle
        for i, person_box in enumerate(people_boxes):
            px1, py1, px2, py2 = person_box
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            person_bottom = ((px1 + px2) / 2, py2)  # Bottom center of person

            # Check distance to all vehicles
            for vehicle_box in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vehicle_box

                # Check if bottom of person is on/near top of vehicle
                if (person_bottom[0] >= vx1 and person_bottom[0] <= vx2 and
                    abs(person_bottom[1] - vy1) < PROXIMITY_THRESHOLD):
                    pedestrian_status[i] = False
                    break

                # Check if person's bounding box overlaps significantly with vehicle
                overlap_x1 = max(px1, vx1)
                overlap_y1 = max(py1, vy1)
                overlap_x2 = min(px2, vx2)
                overlap_y2 = min(py2, vy2)

                # Calculate overlap area
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    person_area = (px2 - px1) * (py2 - py1)

                    # If overlap is significant, consider them on vehicle
                    if overlap_area > 0.2 * person_area:  # 30% overlap threshold
                        pedestrian_status[i] = False
                        break

        # Draw only pedestrians (not on vehicles)
        result_frame = frame.copy()
        for i, box in enumerate(people_boxes):
            if pedestrian_status[i]:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_frame, f"Pedestrian {people_conf[i]:.2f}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(result_frame)

        # Calculate and display FPS
        if frame_count % display_interval == 0:
            current_time = time.time()
            elapsed = current_time - last_update_time
            current_fps = (frame_count - last_update_frame) / elapsed if elapsed > 0 else 0
            last_update_time = current_time
            last_update_frame = frame_count

        # Add processing info to frame
        info_text = f"Frame: {frame_count}, Processing FPS: {current_fps:.2f}"
        cv2.putText(result_frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Pedestrian Detection', result_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Processing interrupted by user")
except Exception as e:
    print(f"Error during processing: {str(e)}")
finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    total_time = time.time() - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average processing speed: {average_fps:.2f} FPS")


# This script is a modified version of the original pedestrian detection script.
# added distance function 
# """
# Pedestrian Detection Script
# Modified for local execution from Colab version
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os
# import time

# # Load the YOLOv8 model
# model = YOLO(r'D:\Sakshi muchhala\person detection\yolo11n.pt')  # Update with your local model path

# # Define input video path - update with your local video path
# video_path = r'D:\Sakshi muchhala\person detection\people_walking.mp4'
# cap = cv2.VideoCapture(video_path)

# # Check if video opened successfully
# if not cap.isOpened():
#     print(f"Error: Could not open video {video_path}")
#     exit()

# # Get video properties for output
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Create output directory if it doesn't exist
# output_dir = r'D:\Sakshi muchhala\person detection\output'
# os.makedirs(output_dir, exist_ok=True)

# # Create output video writer
# output_path = os.path.join(output_dir, 'my_vid.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Constants
# PERSON_CLASS_ID = 0      # COCO dataset: 0 is person
# BICYCLE_CLASS_ID = 1     # COCO dataset: 1 is bicycle
# MOTORCYCLE_CLASS_ID = 3  # COCO dataset: 3 is motorcycle
# CAR_CLASS_ID = 2         # COCO dataset: 2 is car

# # Reduce resolution for faster processing
# scale_factor = 0.5  # Process at 50% resolution

# # Proximity threshold to consider a person "on" a vehicle (in pixels)
# PROXIMITY_THRESHOLD = 60  # Adjust based on your video resolution

# # Confidence thresholds
# PEDESTRIAN_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for pedestrian detection
# VEHICLE_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for vehicle detection

# frame_count = 0
# start_time = time.time()

# # Create a window to display the output
# cv2.namedWindow('Pedestrian Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Pedestrian Detection', 1280, 720)

# # For performance tracking
# display_interval = 30  # Update FPS display every 30 frames
# last_update_time = time.time()
# last_update_frame = 0
# current_fps = 0

# # Improved distance calculation function for better proximity detection
# def calculate_distance(person_center, vehicle_box):
#     vx1, vy1, vx2, vy2 = vehicle_box
#     # Closest point of the vehicle box to the person center
#     closest_x = max(vx1, min(person_center[0], vx2))
#     closest_y = max(vy1, min(person_center[1], vy2))
#     return np.sqrt((closest_x - person_center[0])**2 + (closest_y - person_center[1])**2)

# try:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # Only process every 3rd frame to speed things up
#         if frame_count % 3 != 0:
#             continue

#         # Resize frame for faster processing
#         process_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

#         # Run YOLOv8 inference
#         results = model(process_frame)

#         # Scale boxes back to original frame size
#         boxes = results[0].boxes.xyxy.cpu().numpy() / scale_factor
#         classes = results[0].boxes.cls.cpu().numpy().astype(int)
#         confidences = results[0].boxes.conf.cpu().numpy()

#         # Separate people and vehicles
#         people_indices = np.where((classes == PERSON_CLASS_ID) & (confidences >= PEDESTRIAN_CONFIDENCE_THRESHOLD))[0]
#         vehicle_indices = np.where(((classes == BICYCLE_CLASS_ID) | 
#                                      (classes == MOTORCYCLE_CLASS_ID) | 
#                                      (classes == CAR_CLASS_ID)) & 
#                                     (confidences >= VEHICLE_CONFIDENCE_THRESHOLD))[0]

#         people_boxes = boxes[people_indices] if len(people_indices) > 0 else np.array([])
#         people_conf = confidences[people_indices] if len(people_indices) > 0 else np.array([])
#         vehicle_boxes = boxes[vehicle_indices] if len(vehicle_indices) > 0 else np.array([])

#         # Initialize people status (True = pedestrian, False = on vehicle)
#         pedestrian_status = np.ones(len(people_indices), dtype=bool)

#         # Check each person if they're close to a vehicle
#         for i, person_box in enumerate(people_boxes):
#             px1, py1, px2, py2 = person_box
#             person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
#             person_bottom = ((px1 + px2) / 2, py2)  # Bottom center of person

#             # Check distance to all vehicles using the new distance metric
#             for vehicle_box in vehicle_boxes:
#                 if calculate_distance(person_center, vehicle_box) < PROXIMITY_THRESHOLD:
#                     pedestrian_status[i] = False
#                     break

#             # Check for significant overlap between person and vehicle
#             for vehicle_box in vehicle_boxes:
#                 vx1, vy1, vx2, vy2 = vehicle_box
#                 overlap_x1 = max(px1, vx1)
#                 overlap_y1 = max(py1, vy1)
#                 overlap_x2 = min(px2, vx2)
#                 overlap_y2 = min(py2, vy2)

#                 # Calculate overlap area
#                 if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
#                     overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
#                     person_area = (px2 - px1) * (py2 - py1)

#                     # If overlap is significant, mark as on vehicle
#                     if overlap_area > 0.3 * person_area:  # Adjust threshold if necessary
#                         pedestrian_status[i] = False
#                         break

#         # Draw only pedestrians (not on vehicles)
#         result_frame = frame.copy()
#         for i, box in enumerate(people_boxes):
#             if pedestrian_status[i]:
#                 x1, y1, x2, y2 = box.astype(int)
#                 cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(result_frame, f"Pedestrian {people_conf[i]:.2f}",
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Write frame to output video
#         out.write(result_frame)

#         # Calculate and display FPS
#         if frame_count % display_interval == 0:
#             current_time = time.time()
#             elapsed = current_time - last_update_time
#             current_fps = (frame_count - last_update_frame) / elapsed if elapsed > 0 else 0
#             last_update_time = current_time
#             last_update_frame = frame_count

#         # Add processing info to frame
#         info_text = f"Frame: {frame_count}, Processing FPS: {current_fps:.2f}"
#         cv2.putText(result_frame, info_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Display the frame
#         cv2.imshow('Pedestrian Detection', result_frame)

#         # Break loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("Processing interrupted by user")
# except Exception as e:
#     print(f"Error during processing: {str(e)}")
# finally:
#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
    
#     # Print summary
#     total_time = time.time() - start_time
#     average_fps = frame_count / total_time if total_time > 0 else 0
#     print(f"Video processing complete. Output saved to {output_path}")
#     print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
#     print(f"Average processing speed: {average_fps:.2f} FPS")
