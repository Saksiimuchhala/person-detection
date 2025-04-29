import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO

class PedestrianDetector:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pedestrian detector with configuration
        
        Args:
            config (dict): Configuration dictionary containing:
                - model_path: Path to the YOLOv8 model
                - roi: Region of interest (polygon points)
                - min_confidence: Minimum confidence threshold
                - cooldown_seconds: Cooldown between detections
                - fps: Frames per second estimate
                - min_area: Minimum area for pedestrian detection
                - consecutive_frames_required: Number of consecutive frames to confirm pedestrian
        """
        self.config = config
        self.model_path = config.get('model_path', r'D:\Sakshi muchhala\person detection\yolo11n.pt')
        self.roi = config.get('roi', None)
        self.min_confidence = config.get('min_confidence', 0.6)
        self.cooldown_seconds = config.get('cooldown_seconds', 3)
        self.min_area = config.get('min_area', 300)
        self.consecutive_frames_required = config.get('consecutive_frames_required', 3)
        self.model = YOLO(self.model_path)
        self.person_class_id = 0  # Person class ID in COCO dataset (0-indexed)
        
        # Detection tracking
        self.pedestrian_history = {}  # Track pedestrian detections over time
        self.last_detection_time = None
        self.active_pedestrians = set()  # Currently detected pedestrians
        self.pedestrian_counter = {}  # Count consecutive frames for each pedestrian ID
        
    def is_in_roi(self, bbox: List[int]) -> bool:
        """Check if a bounding box is within the region of interest"""
        if self.roi is None:
            return True

        # Extract coordinates from the bbox (assuming bbox format is [x1, y1, x2, y2])
        x1, y1, x2, y2 = bbox

        # Use bottom center point for pedestrian (feet position)
        cx = (x1 + x2) / 2
        cy = y2  # Bottom of bounding box
    
        return cv2.pointPolygonTest(self.roi, (cx, cy), False) >= 0
    
    def is_pedestrian_on_vehicle(self, person_box: List[int], vehicle_boxes: List[List[int]], proximity_threshold: int = 60) -> bool:
        """
        Check if a person might be on a vehicle rather than being a pedestrian

        Args:
            person_box: [x1, y1, x2, y2] of person
            vehicle_boxes: List of vehicle boxes [x1, y1, x2, y2]
            proximity_threshold: Pixel threshold for considering overlap

        Returns:
            bool: True if person appears to be on a vehicle
        """
        # Extract coordinates from the person box
        px1, py1, px2, py2 = person_box
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        person_bottom = ((px1 + px2) / 2, py2)  # Bottom center of person

        for vehicle_box in vehicle_boxes:
            vx1, vy1, vx2, vy2 = vehicle_box

            # Check if bottom of person is on/near top of vehicle
            if (person_bottom[0] >= vx1 and person_bottom[0] <= vx2 and
                abs(person_bottom[1] - vy1) < proximity_threshold):
                return True

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
                if overlap_area > 0.2 * person_area:  # 20% overlap threshold
                    return True

        return False

    def update_pedestrian_tracking(self, tracked_pedestrians: List[List[int]], frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Update pedestrian tracking and detect pedestrians in ROI

        Args:
            tracked_pedestrians: List of tracked pedestrians (x1, y1, x2, y2, id)
            frame: Current frame for visualization

        Returns:
            tuple: (frame_with_visualization, list_of_pedestrian_detections)
        """
        current_pedestrians = set()
        pedestrian_detections = []

        for pedestrian in tracked_pedestrians:
            x1, y1, x2, y2, id = pedestrian
            w, h = x2 - x1, y2 - y1

            # Skip pedestrians with too small area
            if w * h < self.min_area:
                continue

            # Check if pedestrian is in ROI
            if not self.is_in_roi([x1, y1, x2, y2]):
                continue

            # Update tracking info
            if id not in self.pedestrian_history:
                self.pedestrian_history[id] = {
                    'first_seen': datetime.now(),
                    'frames_tracked': 0,
                    'in_roi_frames': 0
                }
                self.pedestrian_counter[id] = 0

            self.pedestrian_history[id]['frames_tracked'] += 1
            self.pedestrian_counter[id] += 1
            current_pedestrians.add(id)

            # Determine pedestrian status based on consecutive frames
            is_confirmed = self.pedestrian_counter.get(id, 0) >= self.consecutive_frames_required

            # Set color based on status
            if is_confirmed:
                color = (0, 0, 255)  # Red for confirmed pedestrians
                if id not in self.active_pedestrians:
                    self.active_pedestrians.add(id)

                    # Create a detection record
                    detection = {
                        'id': id,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) / 2, y2),  # Bottom center point
                        'timestamp': datetime.now(),
                        'class_name': 'Pedestrian'
                    }
                    pedestrian_detections.append(detection)
            else:
                color = (0, 255, 0)  # Green for tracked but not confirmed

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Pedestrian {id}", (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw dot at bottom center (feet)
            foot_point = (int((x1 + x2) / 2), y2)
            cv2.circle(frame, foot_point, 4, color, -1)

            # Add counter to show how many frames detected
            cv2.putText(frame, f"Frames: {self.pedestrian_counter[id]}/{self.consecutive_frames_required}", 
                      (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Check for pedestrians that are no longer visible
        pedestrians_to_remove = []
        for pid in self.active_pedestrians:
            if pid not in current_pedestrians:
                self.pedestrian_counter[pid] = max(0, self.pedestrian_counter[pid] - 1)
                if self.pedestrian_counter[pid] == 0:
                    pedestrians_to_remove.append(pid)

        # Remove inactive pedestrians
        for pid in pedestrians_to_remove:
            if pid in self.active_pedestrians:
                self.active_pedestrians.remove(pid)
            if pid in self.pedestrian_counter:
                del self.pedestrian_counter[pid]

        # Update status info on frame
        if self.active_pedestrians:
            cv2.putText(frame, f"PEDESTRIANS IN AREA: {len(self.active_pedestrians)}", 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, pedestrian_detections

    def detect_pedestrians(self, frame: np.ndarray, vehicle_boxes: Optional[List[List[int]]] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection method to detect pedestrians in the frame

        Args:
            frame: Current video frame
            vehicle_boxes: Optional list of vehicle bounding boxes to filter out people on vehicles

        Returns:
            tuple: (frame_with_visualization, list_of_pedestrian_detections)
        """
        # Only create new events after cooldown period
        current_time = datetime.now()
        if self.last_detection_time:
            time_diff = (current_time - self.last_detection_time).total_seconds()
            if time_diff < self.cooldown_seconds and self.has_active_detection():
                # If in cooldown period, just update visualization
                if self.active_pedestrians:
                    cv2.putText(frame, f"PEDESTRIANS IN AREA: {len(self.active_pedestrians)}", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame, []

        # Run the model to detect objects directly on original frame
        results = self.model(frame, imgsz=288)

        # Get boxes in original frame coordinates
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Separate people and vehicles
        people_indices = np.where(classes == self.person_class_id)[0]
        vehicle_indices = np.where((classes == 1) |  # Bicycle
                                  (classes == 2) |  # Car
                                  (classes == 3))[0]  # Motorcycle

        people_boxes = boxes[people_indices] if len(people_indices) > 0 else np.array([])
        people_conf = confidences[people_indices] if len(people_indices) > 0 else np.array([])

        vehicle_boxes_detected = boxes[vehicle_indices] if len(vehicle_indices) > 0 else np.array([])

        # If vehicle boxes were not provided but we detected them, use our detections
        if vehicle_boxes is None and len(vehicle_boxes_detected) > 0:
            vehicle_boxes = vehicle_boxes_detected

        pedestrian_detections = []

        # Initialize people status (True = pedestrian, False = on vehicle)
        pedestrian_status = np.ones(len(people_indices), dtype=bool) if len(people_indices) > 0 else np.array([])

        # Check each person if they're close to a vehicle
        for i, person_box in enumerate(people_boxes):
            # Skip if confidence is too low
            if people_conf[i] < self.min_confidence:
                pedestrian_status[i] = False
                continue

            # Check area
            px1, py1, px2, py2 = map(int, person_box)
            w, h = px2 - px1, py2 - py1

            if w * h < self.min_area:
                pedestrian_status[i] = False
                continue

            # Check if person is in ROI
            if not self.is_in_roi([px1, py1, px2, py2]):
                pedestrian_status[i] = False
                continue

            # Check if person is on a vehicle
            if vehicle_boxes is not None and len(vehicle_boxes) > 0:
                if self.is_pedestrian_on_vehicle([px1, py1, px2, py2], vehicle_boxes, proximity_threshold=60):
                    pedestrian_status[i] = False
                    continue
                
        # Process pedestrians (people not on vehicles)
        for i, box in enumerate(people_boxes):
            if pedestrian_status[i]:
                x1, y1, x2, y2 = map(int, box)

                # Create a detection record
                detection = {
                    'id': i,  # Temporary ID until tracked
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) / 2, y2),  # Bottom center point
                    'timestamp': current_time,
                    'class_name': 'Pedestrian',
                    'confidence': float(people_conf[i])
                }
                pedestrian_detections.append(detection)

                # Draw directly for immediate feedback
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Pedestrian {people_conf[i]:.2f}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if pedestrian_detections:
            self.last_detection_time = current_time
            cv2.putText(frame, f"PEDESTRIANS DETECTED: {len(pedestrian_detections)}", 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, pedestrian_detections
    
    def has_active_detection(self) -> bool:
        """Check if there are currently active pedestrian detections"""
        return len(self.active_pedestrians) > 0