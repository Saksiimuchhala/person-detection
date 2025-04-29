import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8s model
model = YOLO("yolov8s.pt")

# Define target classes
target_classes = ["person", "bicycle", "motorcycle"]

# Input and output paths
input_path = r"D:\Sakshi muchhala\person detection\people_walking.mp4"  # change to folder if needed
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
    if box1_area == 0.0:  # Avoid division by zero
        return 0.0
    
    ioa = intersection_area / box1_area
    
    return ioa

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    box1, box2: [x1, y1, x2, y2]
    """
    # Find intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Find union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0.0:  # Avoid division by zero
        return 0.0
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# Calculate distance between box centers
def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two boxes
    box1, box2: [x1, y1, x2, y2]
    """
    # Calculate centers
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    
    # Calculate Euclidean distance
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance

# Calculate diagonal size of box (for relative distance measurements)
def box_diagonal(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return np.sqrt(width**2 + height**2)

# Advanced tracker class with state machine
class AdvancedTracker:
    # Person states
    UNKNOWN = 0
    RIDING = 1
    DISMOUNTING = 2
    PEDESTRIAN = 3
    MOUNTING = 4
    
    def __init__(self):
        self.next_id = 0
        self.tracked_objects = {}  # id -> {box, class, frames_since_update, state, etc.}
        self.max_frames_to_keep = 40  # Keep tracking for longer periods
        self.iou_threshold = 0.3  # Threshold for considering it the same object
        
        # Association maps
        self.person_vehicle_map = {}  # person_id -> vehicle_id
        self.vehicle_person_map = {}  # vehicle_id -> list of person_ids
        
        # State transition parameters
        self.overlap_threshold = 0.2  # Threshold for considering person as a rider
        self.dismount_distance_factor = 3.0  # Factor × vehicle diagonal to consider dismounted
        self.dismount_frames_threshold = 15  # Frames of separation to confirm dismounting
        self.remount_frames_threshold = 5  # Frames of overlap to confirm remounting
        
        # Frame counter
        self.frame_count = 0
        
        # Debug info
        self.debug_info = {}
    
    def update(self, detections, frame=None):
        """
        Update the tracker with new detections
        detections: list of (box, class_label)
        frame: current frame (for debugging)
        Returns: list of (box, class_label, object_id, is_riding)
        """
        self.frame_count += 1
        
        # Increment frames_since_update for all objects
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['frames_since_update'] += 1
        
        # Match detections with tracked objects
        matched_detection_ids = []
        unmatched_detections = []
        current_frame_objects = {}  # To store current frame's object IDs by class
        
        for i, detection in enumerate(detections):
            box, label = detection
            matched = False
            
            # Find the best match among tracked objects of the same class
            best_match_id = None
            best_iou = self.iou_threshold
            
            for obj_id, tracked_obj in self.tracked_objects.items():
                if tracked_obj['class'] == label:
                    iou = calculate_iou(box, tracked_obj['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = obj_id
            
            if best_match_id is not None:
                # Update the matched object
                self.tracked_objects[best_match_id]['box'] = box
                self.tracked_objects[best_match_id]['frames_since_update'] = 0
                
                # Store this detection's position and update history
                matched_detection_ids.append((i, best_match_id))
                
                # Update velocity based on position change
                if len(self.tracked_objects[best_match_id]['history']) > 0:
                    prev_box = self.tracked_objects[best_match_id]['history'][-1]
                    curr_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    prev_center = [(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2]
                    vx = curr_center[0] - prev_center[0]
                    vy = curr_center[1] - prev_center[1]
                    self.tracked_objects[best_match_id]['velocity'] = [vx, vy]
                
                # Add to current frame objects
                if label not in current_frame_objects:
                    current_frame_objects[label] = []
                current_frame_objects[label].append(best_match_id)
                
                matched = True
            
            if not matched:
                unmatched_detections.append((i, detection))
        
        # Create new tracked objects for unmatched detections
        for i, (box, label) in unmatched_detections:
            new_id = self.next_id
            self.next_id += 1
            
            self.tracked_objects[new_id] = {
                'box': box,
                'class': label,
                'frames_since_update': 0,
                'history': [],  # Track position history
                'velocity': [0, 0],  # Initial velocity
                'state': self.PEDESTRIAN if label == 'person' else None,
                'last_state_change': self.frame_count,
                'frames_in_state': 0,
                'associated_vehicle': None if label == 'person' else None,
                'associated_persons': [] if label in ['bicycle', 'motorcycle'] else None,
                'separation_frames': 0,  # Frames since separated from vehicle
                'overlap_frames': 0,     # Frames of continuous overlap with vehicle
                'distance_to_vehicle': float('inf')  # Distance to associated vehicle
            }
            
            # Add to matched detection IDs
            matched_detection_ids.append((i, new_id))
            
            # Add to current frame objects
            if label not in current_frame_objects:
                current_frame_objects[label] = []
            current_frame_objects[label].append(new_id)
        
        # Update object history
        for obj_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[obj_id]['frames_since_update'] == 0:
                # Object was detected in this frame, update history
                self.tracked_objects[obj_id]['history'].append(self.tracked_objects[obj_id]['box'])
                if len(self.tracked_objects[obj_id]['history']) > 30:  # Keep 30 frames of history
                    self.tracked_objects[obj_id]['history'].pop(0)
                
                # Increment frames in current state
                self.tracked_objects[obj_id]['frames_in_state'] += 1
        
        # Check for person-vehicle associations in current frame
        if 'person' in current_frame_objects and any(v_class in current_frame_objects for v_class in ['bicycle', 'motorcycle']):
            for person_id in current_frame_objects['person']:
                person_box = self.tracked_objects[person_id]['box']
                person_state = self.tracked_objects[person_id]['state']
                
                # Debug variables
                best_overlap = 0
                best_vehicle_id = None
                
                # Check overlap with any vehicle
                for v_class in ['bicycle', 'motorcycle']:
                    if v_class in current_frame_objects:
                        for vehicle_id in current_frame_objects[v_class]:
                            vehicle_box = self.tracked_objects[vehicle_id]['box']
                            overlap = calculate_ioa(person_box, vehicle_box)
                            
                            # Update debug info
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_vehicle_id = vehicle_id
                            
                            # Calculate distance to vehicle
                            distance = calculate_distance(person_box, vehicle_box)
                            vehicle_size = box_diagonal(vehicle_box)
                            relative_distance = distance / vehicle_size
                            
                            # Store current distance to vehicle
                            self.tracked_objects[person_id]['distance_to_vehicle'] = relative_distance
                            
                            if overlap >= self.overlap_threshold:
                                # This person is overlapping with a vehicle
                                
                                # Update overlap frames counter
                                self.tracked_objects[person_id]['overlap_frames'] += 1
                                
                                # Reset separation frames counter
                                self.tracked_objects[person_id]['separation_frames'] = 0
                                
                                # Check if this is a new association
                                if person_id not in self.person_vehicle_map:
                                    # New association
                                    self.person_vehicle_map[person_id] = vehicle_id
                                    
                                    if vehicle_id not in self.vehicle_person_map:
                                        self.vehicle_person_map[vehicle_id] = []
                                    
                                    if person_id not in self.vehicle_person_map[vehicle_id]:
                                        self.vehicle_person_map[vehicle_id].append(person_id)
                                    
                                    # Set associated vehicle
                                    self.tracked_objects[person_id]['associated_vehicle'] = vehicle_id
                                    
                                    # Add to vehicle's associated persons
                                    if person_id not in self.tracked_objects[vehicle_id]['associated_persons']:
                                        self.tracked_objects[vehicle_id]['associated_persons'].append(person_id)
                                    
                                    # Set state to RIDING
                                    if self.tracked_objects[person_id]['state'] != self.RIDING:
                                        self.tracked_objects[person_id]['state'] = self.RIDING
                                        self.tracked_objects[person_id]['last_state_change'] = self.frame_count
                                        self.tracked_objects[person_id]['frames_in_state'] = 0
                                
                                elif self.person_vehicle_map[person_id] == vehicle_id:
                                    # Existing association, still overlapping
                                    
                                    # If in DISMOUNTING/PEDESTRIAN/MOUNTING state, check if should transition to RIDING
                                    if self.tracked_objects[person_id]['state'] in [self.DISMOUNTING, self.PEDESTRIAN, self.MOUNTING]:
                                        if self.tracked_objects[person_id]['overlap_frames'] >= self.remount_frames_threshold:
                                            # Transition to RIDING
                                            self.tracked_objects[person_id]['state'] = self.RIDING
                                            self.tracked_objects[person_id]['last_state_change'] = self.frame_count
                                            self.tracked_objects[person_id]['frames_in_state'] = 0
                                    
                                elif self.person_vehicle_map[person_id] != vehicle_id:
                                    # Person has switched vehicles
                                    old_vehicle_id = self.person_vehicle_map[person_id]
                                    
                                    # Remove from old vehicle's associated persons
                                    if old_vehicle_id in self.vehicle_person_map and person_id in self.vehicle_person_map[old_vehicle_id]:
                                        self.vehicle_person_map[old_vehicle_id].remove(person_id)
                                    
                                    # Update to new vehicle
                                    self.person_vehicle_map[person_id] = vehicle_id
                                    
                                    if vehicle_id not in self.vehicle_person_map:
                                        self.vehicle_person_map[vehicle_id] = []
                                    
                                    if person_id not in self.vehicle_person_map[vehicle_id]:
                                        self.vehicle_person_map[vehicle_id].append(person_id)
                                    
                                    # Set associated vehicle
                                    self.tracked_objects[person_id]['associated_vehicle'] = vehicle_id
                                    
                                    # Add to vehicle's associated persons
                                    if person_id not in self.tracked_objects[vehicle_id]['associated_persons']:
                                        self.tracked_objects[vehicle_id]['associated_persons'].append(person_id)
                                    
                                    # Set state to RIDING
                                    self.tracked_objects[person_id]['state'] = self.RIDING
                                    self.tracked_objects[person_id]['last_state_change'] = self.frame_count
                                    self.tracked_objects[person_id]['frames_in_state'] = 0
                            
                            else:
                                # Not overlapping with this vehicle
                                # If there's an association, check if they're dismounting
                                if person_id in self.person_vehicle_map and self.person_vehicle_map[person_id] == vehicle_id:
                                    # They're associated but not overlapping
                                    
                                    # Increment separation frames counter
                                    self.tracked_objects[person_id]['separation_frames'] += 1
                                    
                                    # Reset overlap frames counter
                                    self.tracked_objects[person_id]['overlap_frames'] = 0
                                    
                                    # If currently RIDING and separated enough, transition to DISMOUNTING
                                    if self.tracked_objects[person_id]['state'] == self.RIDING:
                                        if relative_distance > self.dismount_distance_factor:
                                            self.tracked_objects[person_id]['state'] = self.DISMOUNTING
                                            self.tracked_objects[person_id]['last_state_change'] = self.frame_count
                                            self.tracked_objects[person_id]['frames_in_state'] = 0
                                    
                                    # If in DISMOUNTING and separated for long enough, transition to PEDESTRIAN
                                    elif self.tracked_objects[person_id]['state'] == self.DISMOUNTING:
                                        if self.tracked_objects[person_id]['separation_frames'] >= self.dismount_frames_threshold:
                                            self.tracked_objects[person_id]['state'] = self.PEDESTRIAN
                                            self.tracked_objects[person_id]['last_state_change'] = self.frame_count
                                            self.tracked_objects[person_id]['frames_in_state'] = 0
                
                # Store debug info
                self.debug_info[person_id] = {
                    'best_overlap': best_overlap,
                    'best_vehicle_id': best_vehicle_id,
                    'state': self.tracked_objects[person_id]['state'],
                    'overlap_frames': self.tracked_objects[person_id]['overlap_frames'],
                    'separation_frames': self.tracked_objects[person_id]['separation_frames'],
                    'distance_to_vehicle': self.tracked_objects[person_id]['distance_to_vehicle']
                }
        
        # Remove objects that haven't been updated for too long
        to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj['frames_since_update'] > self.max_frames_to_keep:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            # Clean up associations
            if self.tracked_objects[obj_id]['class'] == 'person':
                if obj_id in self.person_vehicle_map:
                    vehicle_id = self.person_vehicle_map[obj_id]
                    if vehicle_id in self.vehicle_person_map and obj_id in self.vehicle_person_map[vehicle_id]:
                        self.vehicle_person_map[vehicle_id].remove(obj_id)
                    del self.person_vehicle_map[obj_id]
            
            elif self.tracked_objects[obj_id]['class'] in ['bicycle', 'motorcycle']:
                if obj_id in self.vehicle_person_map:
                    for person_id in self.vehicle_person_map[obj_id]:
                        if person_id in self.person_vehicle_map and self.person_vehicle_map[person_id] == obj_id:
                            del self.person_vehicle_map[person_id]
                    del self.vehicle_person_map[obj_id]
            
            # Remove from tracked objects
            del self.tracked_objects[obj_id]
        
        # Return current tracked objects with state information
        tracked_results = []
        for i, obj_id in matched_detection_ids:
            box, label = detections[i]
            
            # Determine if this person should have a bounding box
            show_box = True
            if label == 'person':
                state = self.tracked_objects[obj_id]['state']
                # Only show box if person is in PEDESTRIAN state or UNKNOWN
                show_box = (state in [self.PEDESTRIAN, self.UNKNOWN])
            
            # Add object to results
            tracked_results.append((box, label, obj_id, show_box, self.tracked_objects[obj_id].get('state', None)))
        
        return tracked_results

# Function to process a single video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    name = os.path.basename(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output path
    output_path = os.path.join(output_dir, f"track_{name}")
    if output_path.endswith('.webm'):
        output_path = output_path.replace('.webm', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracker
    tracker = AdvancedTracker()
    frame_count = 0
    
    # Define state name mapping for visualization
    state_names = {
        0: "UNKNOWN",
        1: "RIDING",
        2: "DISMOUNTING",
        3: "PEDESTRIAN",
        4: "MOUNTING"
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create a copy for drawing
        display_frame = frame.copy()

        # Detect objects using YOLOv8
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        
        # Collect all detections
        detections = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            label = model.names[int(cls)]
            if label in target_classes:
                detections.append((box, label))
        
        # Update tracker with new detections
        tracked_objects = tracker.update(detections, frame)
        
        # Draw tracked objects
        for box, label, obj_id, show_box, state in tracked_objects:
            x1, y1, x2, y2 = box.astype(int)
    
            if label in ["bicycle", "motorcycle"]:
                # Always draw vehicles with label only
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),  # Only show class name
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            elif label == "person":
                # Draw person based on show_box flag
                if show_box:
                    # Show state with different colors
                    if state == tracker.PEDESTRIAN:
                        color = (0, 255, 0)  # Green for pedestrians
                    elif state == tracker.DISMOUNTING:
                        color = (0, 255, 255)  # Yellow for dismounting
                    elif state == tracker.MOUNTING:
                        color = (255, 165, 0)  # Orange for mounting
                    else:
                        color = (255, 0, 0)  # Red for unknown
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Only show the class name for "person"
                    cv2.putText(display_frame, label, (x1, y1 - 10),  # Only class name, no ID or state
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(display_frame)
        cv2.imshow("Vehicle-Person Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     name = os.path.basename(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Output path
#     output_path = os.path.join(output_dir, f"track_{name}")
#     if output_path.endswith('.webm'):
#         output_path = output_path.replace('.webm', '.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Initialize tracker
#     tracker = AdvancedTracker()
#     frame_count = 0
    
#     # Define state name mapping for visualization
#     state_names = {
#         0: "UNKNOWN",
#         1: "RIDING",
#         2: "DISMOUNTING",
#         3: "PEDESTRIAN",
#         4: "MOUNTING"
#     }

#     # Store locked vehicles
#     locked_vehicles = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
        
#         # Create a copy for drawing
#         display_frame = frame.copy()

#         # Detect objects using YOLOv8
#         results = model.predict(frame, conf=0.5, verbose=False)[0]
        
#         # Collect all detections for vehicles and persons
#         vehicles = []
#         persons = []

#         for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
#             label = model.names[int(cls)]
#             if label in target_classes:
#                 if label in ["bicycle", "motorcycle"]:
#                     vehicles.append((box, label))
#                 elif label == "person":
#                     persons.append((box, label))
        
#         # Prioritize vehicle detection and initialize trackers
#         for vehicle_box, vehicle_label in vehicles:
#             x1, y1, x2, y2 = vehicle_box.astype(int)
#             if not any([locked_vehicle['bbox'] == (x1, y1, x2, y2) for locked_vehicle in locked_vehicles]):
#                 # Initialize a tracker for the vehicle
#                 tracker.add(vehicle_box, label=vehicle_label)
#                 locked_vehicles.append({'bbox': (x1, y1, x2, y2), 'label': vehicle_label, 'tracked': True})
            
#             # Draw the locked vehicle
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Track objects (persons and vehicles)
#         tracked_objects = tracker.update(persons + vehicles, frame)

#         # Loop through tracked objects
#         for box, label, obj_id, show_box, state in tracked_objects:
#             x1, y1, x2, y2 = box.astype(int)

#             if label == "person":
#                 # Check if the person is inside a locked vehicle box (rider detection)
#                 for locked_vehicle in locked_vehicles:
#                     vehicle_bbox = locked_vehicle['bbox']
#                     if is_overlap(vehicle_bbox, (x1, y1, x2, y2)):
#                         # Mark as rider (overlap with vehicle detected)
#                         cv2.putText(display_frame, "Rider", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
#                 # Draw person
#                 if show_box:
#                     if state == tracker.PEDESTRIAN:
#                         color = (0, 255, 0)  # Green for pedestrians
#                     elif state == tracker.DISMOUNTING:
#                         color = (0, 255, 255)  # Yellow for dismounting
#                     elif state == tracker.MOUNTING:
#                         color = (255, 165, 0)  # Orange for mounting
#                     else:
#                         color = (255, 0, 0)  # Red for unknown
                    
#                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         out.write(display_frame)
#         cv2.imshow("Vehicle-Person Detection", display_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# def is_overlap(vehicle_bbox, person_bbox, iou_threshold=0.5):
#     """
#     Check if a person's bounding box overlaps with a vehicle's bounding box
#     using IoU (Intersection over Union). If the IoU is greater than the threshold,
#     we consider it as an overlap.
#     """
#     iou = calculate_iou(vehicle_bbox, person_bbox)
#     return iou > iou_threshold


# Check if input is a file or folder
if os.path.isfile(input_path):
    process_video(input_path)
elif os.path.isdir(input_path):
    for file in os.listdir(input_path):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
            process_video(os.path.join(input_path, file))
else:
    print("Invalid input path.")