from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path

class PoolTableMapper:
    def __init__(self):
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.plane_width = 800
        self.plane_height = 400
        self.table_corners = None
        self.holes_2d = []
        self.sides_2d = []
        self.table_2d = None  # Store the 2D mapped table boundary
        
        # Class mapping for table structure model (best.pt)
        self.table_class_names = {
            0: 'pool-table',
            1: 'pool-table-hole', 
            2: 'pool-table-side'
        }
        
        # Class mapping for ball tracking model (weights/best.pt)
        self.ball_class_names = {
            0: '13',
            1: '2',
            2: 'bag',      # We'll filter this out
            3: 'bag0',     # We'll filter this out
            4: 'bag1',     # We'll filter this out
            5: 'bag2',     # We'll filter this out
            6: 'bag3',     # We'll filter this out
            7: 'bag4',     # We'll filter this out
            8: 'bag5',     # We'll filter this out
            9: 'bag6',     # We'll filter this out
            10: 'bal14',   # Typo in original, keeping as is
            11: 'ball0',
            12: 'ball1',
            13: 'ball10',
            14: 'ball11',
            15: 'ball110', # Unusual but keeping as is
            16: 'ball12',
            17: 'ball13',
            18: 'ball14',
            19: 'ball15',
            20: 'ball18',
            21: 'ball2',
            22: 'ball3',
            23: 'ball4',
            24: 'ball5',
            25: 'ball6',
            26: 'ball7',
            27: 'ball8',
            28: 'ball9',
            29: 'flag',
            30: 'rod'
        }
        
        # Filter out bag classes
        self.excluded_classes = {'bag', 'bag0', 'bag1', 'bag2', 'bag3', 'bag4', 'bag5', 'bag6'}
    
    def extract_table_corners_from_detection(self, table_box):
        """Extract four corners from pool-table detection bounding box"""
        x1, y1, x2, y2 = table_box
        corners = np.float32([
            [x1, y1],     # top-left
            [x2, y1],     # top-right  
            [x2, y2],     # bottom-right
            [x1, y2]      # bottom-left
        ])
        return corners
    
    def setup_perspective_transform(self, table_corners):
        """Setup perspective transformation from detected table corners"""
        self.table_corners = table_corners
        
        # Define destination points for 2D plane (standard pool table)
        dst_points = np.float32([
            [50, 50],                                    # top-left with margin
            [self.plane_width-50, 50],                   # top-right
            [self.plane_width-50, self.plane_height-50], # bottom-right  
            [50, self.plane_height-50]                   # bottom-left
        ])
        
        # Calculate transformation matrices
        self.transform_matrix = cv2.getPerspectiveTransform(table_corners, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, table_corners)
        
        print("Perspective transformation setup complete!")
    
    def transform_to_2d_plane(self, frame):
        """Transform frame to 2D top-down view"""
        if self.transform_matrix is None:
            return frame
        return cv2.warpPerspective(frame, self.transform_matrix, (self.plane_width, self.plane_height))
    
    def transform_detections_to_2d(self, detections, update_static_table=False):
        """Transform all detections to 2D plane coordinates"""
        if self.transform_matrix is None:
            return detections
        
        # Only clear and update table structure if explicitly requested
        if update_static_table:
            self.holes_2d = []
            self.sides_2d = []
            self.table_2d = None
        
        transformed_detections = []
        
        for detection in detections:
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            box = detection['box']
            detection_type = detection.get('type', 'unknown')  # Preserve the type field
            
            # Get center point and corners of bounding box
            x1, y1, x2, y2 = box
            center = np.array([[[(x1 + x2) / 2, (y1 + y2) / 2]]], dtype=np.float32)
            corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            
            # Transform to 2D plane
            center_2d = cv2.perspectiveTransform(center, self.transform_matrix)[0][0]
            corners_2d = cv2.perspectiveTransform(corners, self.transform_matrix)[0]
            
            # Calculate new bounding box in 2D plane
            x_coords = corners_2d[:, 0]
            y_coords = corners_2d[:, 1]
            new_box = [
                np.min(x_coords), np.min(y_coords),
                np.max(x_coords), np.max(y_coords)
            ]
            
            transformed_detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'box': new_box,
                'center_2d': center_2d,
                'corners_2d': corners_2d,
                'original_box': box,
                'type': detection_type  # Include the type field
            }
            
            # Copy track_id if it exists (for balls)
            if 'track_id' in detection:
                transformed_detection['track_id'] = detection['track_id']
            
            transformed_detections.append(transformed_detection)
            
            # Store table structure components in 2D only if updating static table
            if update_static_table:
                if class_name == 'pool-table':
                    self.table_2d = {
                        'box': new_box,
                        'center': center_2d.tolist(),
                        'corners': corners_2d.tolist(),
                        'confidence': confidence
                    }
                elif class_name == 'pool-table-hole':
                    self.holes_2d.append({
                        'box': new_box,
                        'center': center_2d.tolist(),
                        'corners': corners_2d.tolist(),
                        'confidence': confidence
                    })
                elif class_name == 'pool-table-side':
                    self.sides_2d.append({
                        'box': new_box,
                        'center': center_2d.tolist(),
                        'corners': corners_2d.tolist(),
                        'confidence': confidence
                    })
        
        return transformed_detections

class PoolTableTracker:
    def __init__(self, table_model_path="best.pt", ball_model_path="weights/best.pt"):
        self.table_model = YOLO(table_model_path)  # For table structure
        self.ball_model = YOLO(ball_model_path)    # For ball tracking
        self.mapper = PoolTableMapper()
        self.frame_detections = []  # Store detections history
        self.ball_tracks = {}       # Store individual ball tracking history
        
    def process_table_detections(self, results):
        """Process YOLO detection results for table structure"""
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, class_id, conf in zip(boxes, classes, confidences):
                class_name = self.mapper.table_class_names.get(class_id, f'class_{class_id}')
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'box': box.tolist(),
                    'type': 'table'
                })
        
        return detections
    
    def process_ball_detections(self, results):
        """Process YOLO detection results for balls and other objects"""
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Get tracking IDs if available
            track_ids = None
            if hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for i, (box, class_id, conf) in enumerate(zip(boxes, classes, confidences)):
                class_name = self.mapper.ball_class_names.get(class_id, f'class_{class_id}')
                
                # Skip bag classes
                if class_name in self.mapper.excluded_classes:
                    continue
                
                track_id = track_ids[i] if track_ids is not None else None
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'box': box.tolist(),
                    'track_id': track_id,
                    'type': 'ball_object'
                })
        
        return detections
    
    def update_ball_tracks(self, ball_detections, frame_count):
        """Update ball tracking history"""
        for detection in ball_detections:
            if detection['track_id'] is not None:
                track_id = detection['track_id']
                
                if track_id not in self.ball_tracks:
                    self.ball_tracks[track_id] = {
                        'class_name': detection['class_name'],
                        'positions': []
                    }
                
                # Calculate center position
                box = detection['box']
                center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                
                self.ball_tracks[track_id]['positions'].append({
                    'frame': frame_count,
                    'center': center,
                    'box': box,
                    'confidence': detection['confidence']
                })
                
                # Keep only last 50 positions for trail
                if len(self.ball_tracks[track_id]['positions']) > 50:
                    self.ball_tracks[track_id]['positions'].pop(0)
    
    def find_pool_table_detection(self, detections):
        """Find the pool-table detection with highest confidence"""
        table_detections = [d for d in detections if d['class_name'] == 'pool-table']
        if table_detections:
            return max(table_detections, key=lambda x: x['confidence'])
        return None
    
    def track_video(self, video_path, setup_from_first_frame=True):
        """Process video using both models - table structure and ball tracking"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print("Using dual models:")
        print("  - Table model: detecting pool-table structure")
        print("  - Ball model: tracking individual balls, flags, and rods")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = self.mapper.plane_width * 2  # Side by side view
        out_height = max(480, self.mapper.plane_height)
        out = cv2.VideoWriter('dual_model_tracking.mp4', fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        transform_setup = False
        table_structure_setup = False  # Track if static table structure is set up
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run both models
            table_results = self.table_model(frame, verbose=False)
            ball_results = self.ball_model.track(frame, persist=True, verbose=False)
            
            # Process detections from both models
            table_detections = self.process_table_detections(table_results)
            ball_detections = self.process_ball_detections(ball_results)
            
            # Update ball tracking
            self.update_ball_tracks(ball_detections, frame_count)
            
            # Setup transformation from first frame if not done
            if not transform_setup and setup_from_first_frame:
                table_detection = self.find_pool_table_detection(table_detections)
                if table_detection:
                    table_corners = self.mapper.extract_table_corners_from_detection(table_detection['box'])
                    self.mapper.setup_perspective_transform(table_corners)
                    transform_setup = True
                    print(f"Table mapping setup from frame {frame_count}")
            
            # Setup static table structure (only once after transform is ready)
            if transform_setup and not table_structure_setup and len(table_detections) > 0:
                # Transform table structure detections and update static storage
                self.mapper.transform_detections_to_2d(table_detections, update_static_table=True)
                table_structure_setup = True
                print(f"Static table structure mapped: Table={'✓' if self.mapper.table_2d else '✗'}, Holes={len(self.mapper.holes_2d)}, Sides={len(self.mapper.sides_2d)}")
            
            # Combine all detections
            all_detections = table_detections + ball_detections
            
            # Transform detections to 2D if setup is complete (but don't update static table)
            transformed_detections = []
            if transform_setup:
                transformed_detections = self.mapper.transform_detections_to_2d(all_detections, update_static_table=False)
            
            # Store frame data
            frame_data = {
                'frame': frame_count,
                'table_detections': table_detections,
                'ball_detections': ball_detections,
                'detections_2d': transformed_detections
            }
            self.frame_detections.append(frame_data)
            
            # Create visualizations
            original_annotated = self.draw_original_frame(frame, all_detections)
            plane_2d = self.draw_2d_plane(transformed_detections, frame_count)
            
            # Create side-by-side display
            if original_annotated.shape[0] != plane_2d.shape[0]:
                original_resized = cv2.resize(original_annotated, 
                                            (int(original_annotated.shape[1] * plane_2d.shape[0] / original_annotated.shape[0]), 
                                             plane_2d.shape[0]))
            else:
                original_resized = original_annotated
            
            display_frame = np.hstack([original_resized, plane_2d])
            
            # Save and display
            out.write(display_frame)
            
            # Display frame
            cv2.imshow('Dual Model Pool Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested early termination.")
                break
            
            # Progress indicator
            if frame_count % 30 == 0:  # Every second at 30fps
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                
                # Show current detections info
                table_counts = {}
                ball_counts = {}
                
                for det in table_detections:
                    table_counts[det['class_name']] = table_counts.get(det['class_name'], 0) + 1
                
                for det in ball_detections:
                    ball_counts[det['class_name']] = ball_counts.get(det['class_name'], 0) + 1
                
                if table_counts:
                    print(f"Table structure - {', '.join([f'{k}: {v}' for k, v in table_counts.items()])}")
                if ball_counts:
                    print(f"Ball objects - {', '.join([f'{k}: {v}' for k, v in ball_counts.items()])}")
                    print(f"Active ball tracks: {len(self.ball_tracks)}")
                
                if transform_setup and transformed_detections:
                    print(f"2D mapping active - {len(transformed_detections)} objects mapped to 2D plane")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Output video saved as: 'dual_model_tracking.mp4'")
        print(f"Transform setup: {'✓ Success' if transform_setup else '✗ Failed - no pool table detected'}")
        print(f"Total ball tracks: {len(self.ball_tracks)}")
        
        self.save_detection_data()
    
    def draw_original_frame(self, frame, detections):
        """Draw detections on original frame for both table and ball objects"""
        annotated_frame = frame.copy()
        
        # Color mapping for different object types
        table_colors = {
            'pool-table': (0, 255, 0),      # Green
            'pool-table-hole': (0, 0, 255),  # Red  
            'pool-table-side': (255, 0, 0)   # Blue
        }
        
        ball_colors = {
            'flag': (255, 255, 0),    # Yellow
            'rod': (128, 0, 128),     # Purple
            # Ball colors - different colors for different balls
            'ball0': (255, 255, 255), # White (cue ball)
            'ball1': (255, 255, 0),   # Yellow/Orange (1-ball)
            'ball2': (0, 0, 255),     # Blue (2-ball)
            'ball3': (255, 0, 0),     # Red (3-ball)
            'ball4': (128, 0, 128),   # Purple (4-ball)
            'ball5': (255, 165, 0),   # Orange (5-ball)
            'ball6': (0, 128, 0),     # Green (6-ball)
            'ball7': (128, 0, 0),     # Maroon (7-ball)
            'ball8': (0, 0, 0),       # Black (8-ball)
            'ball9': (255, 255, 0),   # Yellow stripe (9-ball)
        }
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            box = detection['box']
            detection_type = detection.get('type', 'unknown')
            
            x1, y1, x2, y2 = [int(x) for x in box]
            
            # Choose color based on detection type and class
            if detection_type == 'table':
                color = table_colors.get(class_name, (255, 255, 255))
                thickness = 2
            else:  # ball_object
                color = ball_colors.get(class_name, (0, 255, 255))  # Cyan for unknown balls
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
            
            # Draw label with track ID if available
            label = f"{class_name}: {confidence:.2f}"
            if 'track_id' in detection and detection['track_id'] is not None:
                label = f"ID{detection['track_id']} {class_name}: {confidence:.2f}"
            
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw trail for tracked balls
            if 'track_id' in detection and detection['track_id'] is not None:
                track_id = detection['track_id']
                if track_id in self.ball_tracks:
                    positions = self.ball_tracks[track_id]['positions']
                    if len(positions) > 1:
                        # Draw trail (last 10 positions)
                        trail_positions = positions[-10:]
                        for i in range(1, len(trail_positions)):
                            pt1 = tuple([int(x) for x in trail_positions[i-1]['center']])
                            pt2 = tuple([int(x) for x in trail_positions[i]['center']])
                            # Fading trail effect
                            alpha = i / len(trail_positions)
                            trail_color = tuple([int(c * alpha) for c in color])
                            cv2.line(annotated_frame, pt1, pt2, trail_color, 1)
        
        # Draw title and info
        cv2.putText(annotated_frame, "Original View - Dual Model Tracking", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Count objects by type
        table_objects = len([d for d in detections if d.get('type') == 'table'])
        ball_objects = len([d for d in detections if d.get('type') == 'ball_object'])
        cv2.putText(annotated_frame, f"Table: {table_objects}, Balls/Objects: {ball_objects}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def draw_2d_plane(self, detections_2d, frame_count):
        """Draw 2D plane view with static table structure and dynamic ball positions"""
        plane = np.zeros((self.mapper.plane_height, self.mapper.plane_width, 3), dtype=np.uint8)
        
        # Draw a dark green background for the table playing area
        cv2.rectangle(plane, (0, 0), (self.mapper.plane_width, self.mapper.plane_height), (0, 50, 0), -1)
        
        # Color mapping for different object types
        table_colors = {
            'pool-table': (0, 255, 0),      # Green
            'pool-table-hole': (0, 0, 255),  # Red
            'pool-table-side': (255, 0, 0)   # Blue
        }
        
        ball_colors = {
            'flag': (255, 255, 0),    # Yellow
            'rod': (128, 0, 128),     # Purple
            'ball0': (255, 255, 255), # White (cue ball)
            'ball1': (255, 255, 0),   # Yellow
            'ball2': (0, 0, 255),     # Blue
            'ball3': (255, 0, 0),     # Red
            'ball4': (128, 0, 128),   # Purple
            'ball5': (255, 165, 0),   # Orange
            'ball6': (0, 128, 0),     # Green
            'ball7': (128, 0, 0),     # Maroon
            'ball8': (0, 0, 0),       # Black
            'ball9': (255, 255, 0),   # Yellow stripe
        }
        
        # ===== DRAW STATIC TABLE STRUCTURE FIRST =====
        
        # Draw pool table boundary (from static storage)
        if self.mapper.table_2d is not None:
            box = self.mapper.table_2d['box']
            x1, y1, x2, y2 = [int(x) for x in box]
            x1 = max(0, min(x1, self.mapper.plane_width-1))
            y1 = max(0, min(y1, self.mapper.plane_height-1))
            x2 = max(0, min(x2, self.mapper.plane_width-1))
            y2 = max(0, min(y2, self.mapper.plane_height-1))
            
            # Draw table outline with thicker border
            cv2.rectangle(plane, (x1, y1), (x2, y2), table_colors['pool-table'], 3)
            # Fill with semi-transparent green
            overlay = plane.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 80, 0), -1)
            cv2.addWeighted(overlay, 0.3, plane, 0.7, 0, plane)
        
        # Draw sides (from static storage)
        for side in self.mapper.sides_2d:
            box = side['box']
            x1, y1, x2, y2 = [int(x) for x in box]
            x1 = max(0, min(x1, self.mapper.plane_width-1))
            y1 = max(0, min(y1, self.mapper.plane_height-1))
            x2 = max(0, min(x2, self.mapper.plane_width-1))
            y2 = max(0, min(y2, self.mapper.plane_height-1))
            
            # Draw sides as filled rectangles with border
            cv2.rectangle(plane, (x1, y1), (x2, y2), table_colors['pool-table-side'], -1)  # Filled
            cv2.rectangle(plane, (x1, y1), (x2, y2), (200, 200, 200), 1)  # Border
        
        # Draw holes (from static storage)
        for hole in self.mapper.holes_2d:
            center = hole['center']
            center_x = max(0, min(int(center[0]), self.mapper.plane_width-1))
            center_y = max(0, min(int(center[1]), self.mapper.plane_height-1))
            
            # Draw holes as larger filled circles with border
            cv2.circle(plane, (center_x, center_y), 15, (0, 0, 0), -1)  # Black hole
            cv2.circle(plane, (center_x, center_y), 15, table_colors['pool-table-hole'], 2)  # Red border
        
        # ===== DRAW DYNAMIC BALL TRAILS =====
        for track_id, track_data in self.ball_tracks.items():
            if len(track_data['positions']) < 2:
                continue
                
            class_name = track_data['class_name']
            color = ball_colors.get(class_name, (0, 255, 255))
            
            # Transform trail positions to 2D
            trail_positions_2d = []
            for pos_data in track_data['positions'][-20:]:  # Last 20 positions
                original_center = pos_data['center']
                # Transform to 2D
                if self.mapper.transform_matrix is not None:
                    center_array = np.array([[original_center]], dtype=np.float32)
                    center_2d = cv2.perspectiveTransform(center_array, self.mapper.transform_matrix)[0][0]
                    
                    # Check bounds
                    if 0 <= center_2d[0] < self.mapper.plane_width and 0 <= center_2d[1] < self.mapper.plane_height:
                        trail_positions_2d.append(center_2d)
            
            # Draw trail
            for i in range(1, len(trail_positions_2d)):
                pt1 = tuple([int(x) for x in trail_positions_2d[i-1]])
                pt2 = tuple([int(x) for x in trail_positions_2d[i]])
                # Fading trail effect
                alpha = i / len(trail_positions_2d)
                trail_color = tuple([int(c * alpha * 0.7) for c in color])  # Dimmer for trails
                cv2.line(plane, pt1, pt2, trail_color, 2)
        
        # ===== DRAW CURRENT DYNAMIC DETECTIONS (BALLS, FLAGS, RODS) =====
        for detection in detections_2d:
            class_name = detection['class_name']
            detection_type = detection.get('type', 'unknown')
            
            # Skip table structure detections since they're already drawn statically
            if detection_type == 'table':
                continue
            
            confidence = detection['confidence']
            box = detection['box']
            center_2d = detection['center_2d']
            
            x1, y1, x2, y2 = [int(x) for x in box]
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, self.mapper.plane_width-1))
            y1 = max(0, min(y1, self.mapper.plane_height-1))
            x2 = max(0, min(x2, self.mapper.plane_width-1))
            y2 = max(0, min(y2, self.mapper.plane_height-1))
            
            center_x = max(0, min(int(center_2d[0]), self.mapper.plane_width-1))
            center_y = max(0, min(int(center_2d[1]), self.mapper.plane_height-1))
            
            # Draw ball objects only (not table structure)
            color = ball_colors.get(class_name, (0, 255, 255))
            
            if 'ball' in class_name or class_name in ['13', '2', 'bal14']:
                # Draw balls as circles
                cv2.circle(plane, (center_x, center_y), 10, color, -1)
                cv2.circle(plane, (center_x, center_y), 10, (255, 255, 255), 2)
                
                # Draw ball number/name
                text = class_name.replace('ball', '')  # Remove 'ball' prefix for cleaner display
                if text == class_name:  # If no 'ball' prefix, show first few characters
                    text = class_name[:3]
                cv2.putText(plane, text, (center_x-8, center_y+3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                           
            elif class_name == 'flag':
                # Draw flag as triangle
                pts = np.array([[center_x-8, center_y-8], [center_x+8, center_y], 
                               [center_x-8, center_y+8]], np.int32)
                cv2.fillPoly(plane, [pts], color)
                cv2.polylines(plane, [pts], True, (255, 255, 255), 1)
                
            elif class_name == 'rod':
                # Draw rod as rectangle
                cv2.rectangle(plane, (x1, y1), (x2, y2), color, 2)
                cv2.putText(plane, "ROD", (center_x-15, center_y+3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add title and frame info
        cv2.putText(plane, "2D Pool Table Mapping", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(plane, f"Frame: {frame_count}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Count only dynamic ball/object detections (skip table structure)
        ball_counts = {}
        
        for detection in detections_2d:
            class_name = detection['class_name']
            detection_type = detection.get('type', 'unknown')
            
            if detection_type != 'table':  # Only count non-table objects
                ball_counts[class_name] = ball_counts.get(class_name, 0) + 1
        
        y_offset = 70
        
        # Display static table structure info
        cv2.putText(plane, "Static Table Structure:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 15
        
        table_info = [
            (f"  Table: {'✓' if self.mapper.table_2d else '✗'}", table_colors['pool-table']),
            (f"  Holes: {len(self.mapper.holes_2d)}", table_colors['pool-table-hole']),
            (f"  Sides: {len(self.mapper.sides_2d)}", table_colors['pool-table-side'])
        ]
        
        for text, color in table_info:
            cv2.putText(plane, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_offset += 15
        
        y_offset += 5
        
        # Display dynamic ball/object counts
        if ball_counts:
            cv2.putText(plane, "Dynamic Balls & Objects:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 15
            for class_name, count in ball_counts.items():
                color = ball_colors.get(class_name, (0, 255, 255))
                cv2.putText(plane, f"  {class_name}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                y_offset += 12
        
        # Display active track count
        cv2.putText(plane, f"Active tracks: {len(self.ball_tracks)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return plane
    
    def save_detection_data(self):
        """Save all detection data to JSON file"""
        output_data = {
            'video_info': {
                'total_frames': len(self.frame_detections),
                'plane_dimensions': [self.mapper.plane_width, self.mapper.plane_height]
            },
            'table_structure_2d': {
                'table': self.mapper.table_2d,
                'holes': self.mapper.holes_2d,
                'sides': self.mapper.sides_2d
            },
            'ball_tracks': self.ball_tracks,
            'frames': self.frame_detections
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        output_data = convert_numpy(output_data)
        
        with open('pool_table_detection_data.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("Detection data saved to 'pool_table_detection_data.json'")
        print(f"Table structure mapped: Table={'✓' if self.mapper.table_2d else '✗'}, Holes={len(self.mapper.holes_2d)}, Sides={len(self.mapper.sides_2d)}")

# Main execution
if __name__ == "__main__":
    # Initialize tracker with both models
    tracker = PoolTableTracker(
        table_model_path="best.pt",      # Table structure detection
        ball_model_path="weights/best.pt"  # Ball tracking
    )
    
    print("🎱 Dual Model Pool Ball Tracking System")
    print("=" * 50)
    print("Table Model: best.pt (pool-table, holes, sides)")
    print("Ball Model: weights/best.pt (individual balls, flags, rods)")
    print("Output: dual_model_tracking.mp4")
    print("=" * 50)

    tracker.track_video("tiktok4.mp4", setup_from_first_frame=True)
