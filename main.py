
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# # import supervision as sv
# from collections import defaultdict
# import time
# import os

# class HighwayPedestrianDetector:
#     def __init__(self, model_path="yolov8s.pt", confidence=0.5):
#         """
#         Initialize the detector with YOLO model
        
#         Args:
#             model_path: Path to YOLOv8 weights
#             confidence: Detection confidence threshold
#         """
#         self.model = YOLO(model_path)
#         self.confidence = confidence
#         self.track_history = defaultdict(lambda: [])
#         self.danger_history = defaultdict(lambda: [])
#         self.danger_score = defaultdict(float)
        
#         # Classes of interest
#         self.vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
#         self.person_class = 0  # person

#     def detect_frame(self, frame):
#         """Process a single frame and show only people NOT on vehicles"""
#         results = self.model.track(frame, persist=True, conf=self.confidence)

#         if results[0].boxes.id is None:
#             return frame

#         annotated_frame = frame.copy()

#         # Extract boxes, classes, and track IDs
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy().astype(int)
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)

#         # Get all vehicle bounding boxes
#         vehicle_boxes = [
#             boxes[i] for i in range(len(classes)) 
#             if classes[i] in self.vehicle_classes
#         ]

#         for i in range(len(boxes)):
#             cls = classes[i]
#             if cls != self.person_class:  # Skip if it's not a person
#                 continue
            
#             x1, y1, x2, y2 = map(int, boxes[i])
#             track_id = track_ids[i]

#             # Check if this person overlaps with any vehicle
#             is_on_vehicle = False
#             for vx1, vy1, vx2, vy2 in vehicle_boxes:
#                 if not (x2 < vx1 or x1 > vx2 or y2 < vy1 or y1 > vy2):
#                     is_on_vehicle = True
#                     break
                
#             # Only show bounding box if person is NOT on a vehicle
#             if not is_on_vehicle:
#                 color = (0, 255, 0)
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
#                 label = f"Person"
#                 cv2.putText(annotated_frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         return annotated_frame




#     def process_video(self, input_path, output_path=None, display=True):
#         """Process a video file or stream of any supported format"""
#         # Check if input is a webcam/stream index
#         if isinstance(input_path, int):
#             cap = cv2.VideoCapture(input_path)
#         else:
#                 # Check if file exists
#             if not os.path.isfile(input_path):
#                 print(f"Error: Input file '{input_path}' not found.")
#                 return
            
#             # Try to open with OpenCV
#             cap = cv2.VideoCapture(input_path)
#             if not cap.isOpened():
#                 print(f"Error: Could not open video file '{input_path}'.")
#                 return
        
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         print(f"Processing video: {width}x{height} at {fps} FPS, {total_frames} frames")
        
#         # Determine output format based on extension
#         if output_path:
#             extension = os.path.splitext(output_path)[1].lower()
            
#             # Set appropriate codec based on extension
#             if extension == '.mp4':
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
#             elif extension == '.avi':
#                 fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI
#             elif extension == '.mov':
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MOV
#             elif extension == '.wmv':
#                 fourcc = cv2.VideoWriter_fourcc(*'WMV2')  # WMV
#             elif extension == '.mkv':
#                 fourcc = cv2.VideoWriter_fourcc(*'X264')  # MKV
#             elif extension == '.webm':
#                 fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM
#             else:
#                 print(f"Warning: Unrecognized output format '{extension}'. Defaulting to MP4.")
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 if not output_path.endswith('.mp4'):
#                     output_path += '.mp4'
            
#             # Create output directory if it doesn't exist
#             output_dir = os.path.dirname(output_path)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
            
#             writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#             if not writer.isOpened():
#                 print(f"Error: Could not create output video '{output_path}'.")
#                 writer = None
#         else:
#             writer = None
        
#         frame_count = 0
#         start_time = time.time()
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             frame_count += 1
#             if frame_count % 100 == 0:
#                 elapsed = time.time() - start_time
#                 frames_per_second = frame_count / elapsed
#                 if total_frames > 0:
#                     percent_complete = (frame_count / total_frames) * 100
#                     print(f"Progress: {frame_count}/{total_frames} frames ({percent_complete:.1f}%) - {frames_per_second:.2f} FPS")
#                 else:
#                     print(f"Processed {frame_count} frames - {frames_per_second:.2f} FPS")
            
#             # Process the frame
#             processed_frame = self.detect_frame(frame)
            
#             if display:
#                 cv2.imshow("Highway Pedestrian Detection", processed_frame)
                
#             if writer:
#                 writer.write(processed_frame)
                
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         # Clean up
#         cap.release()
#         if writer:
#             writer.release()
#         cv2.destroyAllWindows()
        
#         # Print final stats
#         elapsed = time.time() - start_time
#         print(f"Processing complete. {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")


# # Example usage
# if __name__ == "__main__":
#     detector = HighwayPedestrianDetector()
    
#     # Process your specific video file
#     input_file = r"D:\Sakshi muchhala\person detection\test.webm"
#     output_file = r"D:\Sakshi muchhala\person detection\output\my_test.webm"
#     detector.process_video(input_file, output_file)


# using centroid of person and vehicle to check if person is on vehicle or not
import cv2
import os
from ultralytics import YOLO

class HighwayPedestrianDetector:
    def __init__(self, model_path="yolov8s.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        self.person_class = 0  # person

    def is_centroid_inside(self, person_box, vehicle_box):
        px1, py1, px2, py2 = person_box
        cx = int((px1 + px2) / 2)
        cy = int((py1 + py2) / 2)

        vx1, vy1, vx2, vy2 = vehicle_box
        return vx1 <= cx <= vx2 and vy1 <= cy <= vy2

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, conf=self.confidence)

        if results[0].boxes.id is None:
            return frame

        annotated_frame = frame.copy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        vehicle_boxes = [
            boxes[i] for i in range(len(classes)) 
            if classes[i] in self.vehicle_classes
        ]

        for i in range(len(boxes)):
            if classes[i] != self.person_class:
                continue

            person_box = boxes[i]
            x1, y1, x2, y2 = map(int, person_box)
            track_id = track_ids[i]

            centroid_on_vehicle = any(
                self.is_centroid_inside(person_box, v_box) for v_box in vehicle_boxes
            )

            if not centroid_on_vehicle:
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"Person"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame

    def process_video(self, input_path, output_path=None, display=False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{input_path}'.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ext = os.path.splitext(output_path)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext == '.mp4' else cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            writer = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.detect_frame(frame)

            writer.write(processed_frame)
            cv2.imshow("Live Highway Pedestrian Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def process_input(self, input_path, output_folder=None, display=True):
        """Handles both single video file and a folder of videos"""
        if os.path.isfile(input_path):
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"processed_{name}{ext}") if output_folder else None
            self.process_video(input_path, output_path, display=display)

        elif os.path.isdir(input_path):
            supported_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
            for file in os.listdir(input_path):
                if file.lower().endswith(supported_exts):
                    full_input_path = os.path.join(input_path, file)
                    output_path = os.path.join(output_folder, f"processed1_{file}") if output_folder else None
                    self.process_video(full_input_path, output_path, display=display)

        else:
            print("Invalid input path. Please provide a valid video file or folder.")



# -------- Example Usage --------
if __name__ == "__main__":
    input_path = r"D:\Sakshi muchhala\person detection\test.webm"  # Can be a folder or a single video
    output_folder = r"D:\Sakshi muchhala\person detection\output"

    detector = HighwayPedestrianDetector(model_path="yolov8s.pt")
    detector.process_input(input_path, output_folder, display=False)


