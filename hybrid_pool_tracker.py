"""
Hybrid Pool Table Ball Tracker
Combines color segmentation and YOLO detection for robust ball tracking.
Uses both methods and merges results for better detection.
"""

import cv2
import numpy as np
import sys
from ultralytics import YOLO

class PoolTableSelector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.corners = []
        self.image = None
        self.display_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select corners"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                # Scale coordinates back to original resolution
                original_x = int(x / self.scale_factor) if hasattr(self, 'scale_factor') else x
                original_y = int(y / self.scale_factor) if hasattr(self, 'scale_factor') else y
                
                self.corners.append([original_x, original_y])
                print(f"Corner {len(self.corners)}/4 selected at: ({original_x}, {original_y}) [original resolution]")
                
                # Draw circle at selected point (using display coordinates)
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
                # Label based on order
                labels = ['TL', 'TR', 'BL', 'BR']
                cv2.putText(self.display_image, labels[len(self.corners)-1], 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Draw lines between corners
                if len(self.corners) > 1:
                    cv2.line(self.display_image, 
                            tuple(self.corners[-2]), 
                            tuple(self.corners[-1]), 
                            (0, 255, 0), 2)
                
                # Close the rectangle if 4 corners selected
                if len(self.corners) == 4:
                    cv2.line(self.display_image, 
                            tuple(self.corners[3]), 
                            tuple(self.corners[0]), 
                            (0, 255, 0), 2)
                    print("\n4 corners selected! Press 'c' to continue or 'r' to reset.")
                
                cv2.imshow('Select Pool Table Corners', self.display_image)
    
    def select_corners(self):
        """Display first frame and allow user to select corners"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read first frame")
            return None
        
        # Store original frame size
        self.original_height, self.original_width = frame.shape[:2]
        
        # Scale down image if too large for display
        max_display_height = 900
        max_display_width = 1600
        
        if self.original_height > max_display_height or self.original_width > max_display_width:
            scale = min(max_display_height / self.original_height, max_display_width / self.original_width)
            new_width = int(self.original_width * scale)
            new_height = int(self.original_height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.scale_factor = scale
            print(f"Video resolution: {self.original_width}x{self.original_height}")
            print(f"Display resolution: {new_width}x{new_height} (scaled to fit screen)")
        else:
            self.scale_factor = 1.0
        
        self.image = frame.copy()
        self.display_image = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Pool Table Corners', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Pool Table Corners', self.mouse_callback)
        
        # Instructions
        print("\n" + "="*60)
        print("HYBRID POOL TABLE CORNER SELECTION")
        print("="*60)
        print("\nInstructions:")
        print("Click on the 4 corners of the pool table in THIS ORDER:")
        print("  1. TOP-LEFT corner")
        print("  2. TOP-RIGHT corner")
        print("  3. BOTTOM-LEFT corner")
        print("  4. BOTTOM-RIGHT corner")
        print("\nControls:")
        print("  'c' - Continue to processing (after 4 corners)")
        print("  'r' - Reset and start over")
        print("  'q' - Quit")
        print("="*60 + "\n")
        
        cv2.imshow('Select Pool Table Corners', self.display_image)
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r') or key == ord('R'):
                print("Resetting corners...")
                self.corners = []
                self.display_image = self.image.copy()
                cv2.imshow('Select Pool Table Corners', self.display_image)
            
            elif key == ord('c') or key == ord('C'):
                if len(self.corners) == 4:
                    print("Continuing to video processing...")
                    cv2.destroyAllWindows()
                    return np.array(self.corners, dtype=np.float32)
                else:
                    print(f"Please select 4 corners (currently {len(self.corners)}/4)")
            
            elif key != 255:
                print(f"Key pressed: {key} (Press 'c' to continue, 'r' to reset, 'q' to quit)")
        
        cv2.destroyAllWindows()
        return None


class HybridPoolBallTracker:
    def __init__(self, video_path, corners, yolo_confidence=0.5):
        self.video_path = video_path
        self.corners = corners
        self.width = 280  # Output width
        self.height = 560  # Output height
        self.yolo_confidence = yolo_confidence  # YOLO confidence threshold
        
        # Load YOLO model
        print(f"Loading YOLO model with confidence threshold: {self.yolo_confidence}")
        self.model = YOLO("weights/best.pt")
        
        # Define destination points for perspective transform
        self.dst_points = np.array([
            [0, 0],              # TOP-LEFT
            [self.width, 0],     # TOP-RIGHT
            [0, self.height],    # BOTTOM-LEFT
            [self.width, self.height]  # BOTTOM-RIGHT
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        self.matrix = cv2.getPerspectiveTransform(self.corners, self.dst_points)
    
    def create_table(self):
        """Creates a 2D snooker table image with markings"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = [0, 180, 10]  # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create semi-circle at the top (baulk line)
        cv2.circle(img, (int(self.width/2), int(self.height/5)), 
                   int((self.width/3)/2), (50, 255, 50), -1)
        
        # Delete bottom half of circle
        img[int(self.height/5):self.height, 0:self.width] = [0, 180, 10]
        
        # Draw the baulk line
        cv2.line(img, (0, int(self.height/5)), 
                 (self.width, int(self.height/5)), (50, 255, 50), 2)
        
        return img
    
    def draw_holes(self, input_img):
        """Draws borders and holes on 2D snooker table"""
        color = (190, 190, 190)
        color2 = (120, 120, 120)
        color3 = (200, 140, 0)
        
        img = input_img.copy()
        
        # Draw borders
        cv2.line(img, (0, 0), (self.width, 0), color3, 3)
        cv2.line(img, (0, self.height), (self.width, self.height), color3, 3)
        cv2.line(img, (0, 0), (0, self.height), color3, 3)
        cv2.line(img, (self.width, 0), (self.width, self.height), color3, 3)
        
        # Draw corner pockets
        cv2.circle(img, (0, 0), 11, color, -1)
        cv2.circle(img, (self.width, 0), 11, color, -1)
        cv2.circle(img, (0, self.height), 11, color, -1)
        cv2.circle(img, (self.width, self.height), 11, color, -1)
        
        # Draw middle pockets
        cv2.circle(img, (self.width, int(self.height/2)), 8, color, -1)
        cv2.circle(img, (0, int(self.height/2)), 8, color, -1)
        
        # Inner circles for depth effect
        cv2.circle(img, (0, 0), 9, color2, -1)
        cv2.circle(img, (self.width, 0), 9, color2, -1)
        cv2.circle(img, (0, self.height), 9, color2, -1)
        cv2.circle(img, (self.width, self.height), 9, color2, -1)
        cv2.circle(img, (self.width, int(self.height/2)), 6, color2, -1)
        cv2.circle(img, (0, int(self.height/2)), 6, color2, -1)
        
        return img
        
    def get_perspective_transform(self, frame):
        """Apply perspective transformation to frame"""
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height))
    
    def detect_balls_yolo(self, frame):
        """
        Detect balls using YOLO model.
        Returns list of ball dictionaries with center, bbox, confidence, and class name.
        """
        balls = []
        
        # Run YOLO detection with confidence threshold
        results = self.model.track(frame, persist=True, verbose=False, conf=self.yolo_confidence)
        
        if len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                names = result.names
                
                # Get detection data
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    class_idx = int(cls[i])
                    ball_name = names.get(class_idx, f"class_{class_idx}")
                    confidence = float(conf[i])
                    
                    # Calculate center
                    cX = int((x1 + x2) / 2)
                    cY = int((y1 + y2) / 2)
                    
                    # Calculate area for consistency with color detection
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    balls.append({
                        'center': (cX, cY),
                        'bbox': (int(x1), int(y1), int(width), int(height)),
                        'area': area,
                        'confidence': confidence,
                        'class_name': ball_name,
                        'source': 'yolo'
                    })
        
        return balls
    
    def detect_balls_color(self, frame):
        """Detect balls on the table using color segmentation"""
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        
        # Convert BGR to RGB first
        blur_RGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to HSV
        hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)
        
        # HSV colors of the snooker table
        #pool-game2
        # lower_green = np.array([48, 109, 91])
        # upper_green = np.array([55, 226, 189])
        
        #trafck
        lower_green = np.array([56, 144, 144])
        upper_green = np.array([63, 253, 234])
        
        
        # Create mask for green table
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological closing to fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Invert mask to get non-table objects (balls)
        _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []
        
        # Filter contours
        min_s = 90
        max_s = 358
        alpha = 3.445
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Get bounding rectangle
            rot_rect = cv2.minAreaRect(contour)
            w = rot_rect[1][0]
            h = rot_rect[1][1]
            
            # Check aspect ratio
            if h > 0 and w > 0:
                if (h * alpha < w) or (w * alpha < h):
                    continue
            
            # Filter by area
            if (area < min_s) or (area > max_s):
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                
                balls.append({
                    'center': (cX, cY),
                    'contour': contour,
                    'area': area,
                    'bbox': cv2.boundingRect(contour),
                    'source': 'color'
                })
        
        return balls, mask_inv, mask, mask_closing
    
    def merge_detections(self, yolo_balls, color_balls, distance_threshold=15):
        """
        Merge YOLO and color segmentation detections.
        Removes duplicates based on proximity and keeps the best detection.
        """
        merged_balls = []
        used_color_indices = set()
        
        # First, add all YOLO detections (they are generally more reliable)
        for yolo_ball in yolo_balls:
            merged_balls.append(yolo_ball)
        
        # Then, add color detections that don't overlap with YOLO detections
        for i, color_ball in enumerate(color_balls):
            color_center = color_ball['center']
            
            # Check if this color detection is close to any YOLO detection
            is_duplicate = False
            for yolo_ball in yolo_balls:
                yolo_center = yolo_ball['center']
                distance = np.sqrt((color_center[0] - yolo_center[0])**2 + 
                                 (color_center[1] - yolo_center[1])**2)
                
                if distance < distance_threshold:
                    is_duplicate = True
                    break
            
            # If not a duplicate, add it to merged results
            if not is_duplicate:
                merged_balls.append(color_ball)
        
        return merged_balls
    
    def draw_balls(self, balls, frame):
        """Draw balls on the table with their colors"""
        K = np.ones((3, 3), np.uint8)
        
        final = self.create_table()
        mask = np.zeros((self.height, self.width), np.uint8)
        
        # Convert frame to RGB for color extraction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for ball in balls:
            cX, cY = ball['center']
            source = ball.get('source', 'unknown')
            
            # Extract color differently based on detection source
            if source == 'yolo':
                # For YOLO detections, sample color from bbox area
                x, y, w, h = ball['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Make sure bbox is within frame bounds
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                w = min(w, self.width - x)
                h = min(h, self.height - y)
                
                if w > 0 and h > 0:
                    # Sample color from center region of bbox
                    center_x = x + w // 2
                    center_y = y + h // 2
                    sample_size = min(w, h) // 3
                    
                    x1 = max(0, center_x - sample_size)
                    y1 = max(0, center_y - sample_size)
                    x2 = min(self.width, center_x + sample_size)
                    y2 = min(self.height, center_y + sample_size)
                    
                    roi = frame_rgb[y1:y2, x1:x2]
                    if roi.size > 0:
                        mean_color = cv2.mean(roi)
                        color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
                    else:
                        color = (255, 255, 255)  # White fallback
                else:
                    color = (255, 255, 255)  # White fallback
                    
            else:  # color detection
                # For color detections, use contour mask
                contour = ball['contour']
                mask[...] = 0
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mask_eroded = cv2.erode(mask, K, iterations=3)
                
                mean_color = cv2.mean(frame_rgb, mask_eroded)
                color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
            
            # Draw ball
            radius = 8
            cv2.circle(final, (cX, cY), radius, color, -1)
            
            # Black border around ball
            cv2.circle(final, (cX, cY), radius, 0, 1)
            
            # Light reflection
            cv2.circle(final, (cX - 2, cY - 2), 2, (255, 255, 255), -1)
            
            # Add small indicator for detection source (optional debug)
            # if source == 'yolo':
            #     cv2.circle(final, (cX, cY), radius + 2, (255, 0, 0), 1)  # Blue ring for YOLO
            # else:
            #     cv2.circle(final, (cX, cY), radius + 2, (0, 255, 255), 1)  # Yellow ring for color
        
        return final
    
    def process_video(self, output_path=None):
        """Process the entire video using hybrid detection"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video with HYBRID detection:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        print(f"  YOLO Confidence: {self.yolo_confidence}")
        
        # Video writer for output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
            print(f"  Saving to: {output_path}")
        
        frame_count = 0
        
        print("\nProcessing... Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Apply perspective transformation
            warped = self.get_perspective_transform(frame)
            
            # Detect balls using YOLO
            yolo_balls = self.detect_balls_yolo(warped)
            
            # Detect balls using color segmentation
            color_balls, mask_inv, mask_table, mask_closing = self.detect_balls_color(warped)
            
            # Merge both detection methods
            merged_balls = self.merge_detections(yolo_balls, color_balls)
            
            # Draw table with balls
            top_view = self.draw_balls(merged_balls, warped)
            top_view = self.draw_holes(top_view)
            
            # Count detections by source
            yolo_count = sum(1 for b in merged_balls if b.get('source') == 'yolo')
            color_count = sum(1 for b in merged_balls if b.get('source') == 'color')
            
            # Add text overlay
            cv2.putText(top_view, f"Total Balls: {len(merged_balls)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            cv2.putText(top_view, f"YOLO: {yolo_count} | Color: {color_count}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            cv2.putText(top_view, f"Frame: {frame_count}/{total_frames}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            cv2.putText(top_view, f"Conf: {self.yolo_confidence}", 
                       (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1)
            
            # Create debug view with all detections
            debug_view = warped.copy()
            
            # Draw YOLO detections in blue
            for ball in yolo_balls:
                x, y, w, h = ball['bbox']
                cv2.rectangle(debug_view, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                cv2.circle(debug_view, ball['center'], 3, (255, 0, 0), -1)
                cv2.putText(debug_view, f"Y:{ball['confidence']:.2f}", 
                           (ball['center'][0]+5, ball['center'][1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw color detections in yellow
            for ball in color_balls:
                cv2.drawContours(debug_view, [ball['contour']], -1, (0, 255, 255), 2)
                cv2.circle(debug_view, ball['center'], 3, (0, 255, 255), -1)
                cv2.putText(debug_view, f"C:{int(ball['area'])}", 
                           (ball['center'][0]+5, ball['center'][1]+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw merged detections in green
            for ball in merged_balls:
                cv2.circle(debug_view, ball['center'], 5, (0, 255, 0), 2)
            
            # Write to output video
            if output_path:
                out.write(top_view)
            
            # Create visual masks for debugging
            mask_inv_visual = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            
            # Resize for display
            max_display_height = 800
            max_display_width = 800
            
            def resize_for_display(img, max_h=max_display_height, max_w=max_display_width):
                h, w = img.shape[:2]
                if h > max_h or w > max_w:
                    scale = min(max_h / h, max_w / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return img
            
            # Display windows
            cv2.imshow('1. Hybrid Tracking - Final Output', resize_for_display(top_view))
            cv2.imshow('2. Debug View (Blue=YOLO, Yellow=Color, Green=Merged)', resize_for_display(debug_view))
            cv2.imshow('3. Original Warped Frame', resize_for_display(warped))
            cv2.imshow('4. Color Mask (White = Detected Objects)', resize_for_display(mask_inv_visual))
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Total: {len(merged_balls)} (YOLO: {yolo_count}, Color: {color_count})")
            
            # Check for quit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nStopped by user")
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete! Processed {frame_count} frames")
        if output_path:
            print(f"Output saved to: {output_path}")


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python hybrid_pool_tracker.py <video_path> [output_path] [yolo_confidence]")
        print("\nArguments:")
        print("  video_path      - Path to input video")
        print("  output_path     - (Optional) Path to save output video")
        print("  yolo_confidence - (Optional) YOLO confidence threshold (default: 0.25)")
        print("\nExamples:")
        print("  python hybrid_pool_tracker.py input.mp4")
        print("  python hybrid_pool_tracker.py input.mp4 output.mp4")
        print("  python hybrid_pool_tracker.py input.mp4 output.mp4 0.5")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    yolo_confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    
    # Validate confidence threshold
    if not 0.0 <= yolo_confidence <= 1.0:
        print(f"Error: YOLO confidence must be between 0.0 and 1.0 (got {yolo_confidence})")
        sys.exit(1)
    
    # Step 1: Select corners interactively
    print("Step 1: Selecting pool table corners...")
    selector = PoolTableSelector(video_path)
    corners = selector.select_corners()
    
    if corners is None:
        print("Corner selection cancelled or failed.")
        sys.exit(1)
    
    print("\nSelected corners:")
    for i, corner in enumerate(corners):
        print(f"  Corner {i+1}: ({corner[0]:.0f}, {corner[1]:.0f})")
    
    # Step 2: Process video with hybrid detection
    print(f"\nStep 2: Processing video with HYBRID detection (YOLO + Color Segmentation)...")
    tracker = HybridPoolBallTracker(video_path, corners, yolo_confidence=yolo_confidence)
    tracker.process_video(output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
