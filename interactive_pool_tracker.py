"""
Interactive Pool Table Ball Tracker
Allows user to mark the 4 corners of the pool table on the first frame,
then applies perspective transformation and processes the video.
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
                self.corners.append([x, y])
                print(f"Corner {len(self.corners)}/4 selected at: ({x}, {y})")
                
                # Draw circle at selected point
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
        
        self.image = frame.copy()
        self.display_image = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Pool Table Corners')
        cv2.setMouseCallback('Select Pool Table Corners', self.mouse_callback)
        
        # Instructions
        print("\n" + "="*60)
        print("INTERACTIVE POOL TABLE CORNER SELECTION")
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
            key = cv2.waitKey(20) & 0xFF  # Increased wait time for better key detection
            
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
            
            elif key != 255:  # If any other key is pressed (for debugging)
                print(f"Key pressed: {key} (Press 'c' to continue, 'r' to reset, 'q' to quit)")
        
        cv2.destroyAllWindows()
        return None


class PoolBallTracker:
    def __init__(self, video_path, corners):
        self.video_path = video_path
        self.corners = corners
        self.width = 280  # Output width (matching notebook)
        self.height = 560  # Output height (matching notebook)
        self.model = YOLO("weights/best.pt")
        
        # Define destination points for perspective transform
        # Matching notebook: [0,0], [width,0], [0,height], [width,height]
        # Order: TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT
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
        # Create green table (matching notebook - BGR then convert to RGB)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = [0, 180, 10]  # BGR format (notebook uses this)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create semi-circle at the top (baulk line)
        # The line is at height/5 and circle radius is (width/3)/2
        cv2.circle(img, (int(self.width/2), int(self.height/5)), 
                   int((self.width/3)/2), (50, 255, 50), -1)
        
        # Delete bottom half of circle by coloring green
        img[int(self.height/5):self.height, 0:self.width] = [0, 180, 10]
        
        # Draw the baulk line
        cv2.line(img, (0, int(self.height/5)), 
                 (self.width, int(self.height/5)), (50, 255, 50), 2)
        
        return img
    
    def draw_holes(self, input_img):
        """Draws borders and holes on 2D snooker table"""
        color = (190, 190, 190)  # Gray color for outer circle
        color2 = (120, 120, 120)  # Darker gray for inner circle
        color3 = (200, 140, 0)  # Border color (brown)
        
        img = input_img.copy()
        
        # Draw borders
        cv2.line(img, (0, 0), (self.width, 0), color3, 3)  # Top
        cv2.line(img, (0, self.height), (self.width, self.height), color3, 3)  # Bottom
        cv2.line(img, (0, 0), (0, self.height), color3, 3)  # Left
        cv2.line(img, (self.width, 0), (self.width, self.height), color3, 3)  # Right
        
        # Draw corner pockets (larger)
        cv2.circle(img, (0, 0), 11, color, -1)  # Top left
        cv2.circle(img, (self.width, 0), 11, color, -1)  # Top right
        cv2.circle(img, (0, self.height), 11, color, -1)  # Bottom left
        cv2.circle(img, (self.width, self.height), 11, color, -1)  # Bottom right
        
        # Draw middle pockets (smaller)
        cv2.circle(img, (self.width, int(self.height/2)), 8, color, -1)  # Mid right
        cv2.circle(img, (0, int(self.height/2)), 8, color, -1)  # Mid left
        
        # Draw smaller inner circles for depth effect
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
    
    def detect_balls_from_yolo(self, frame):
        """
        Detect balls using YOLO model and print detected ball names and bounding boxes.
        """
        detect = self.model.track(frame, persist=True, verbose=False)
        # detect.boxes: ultralytics.engine.results.Boxes object
        # detect.names: dict mapping class indices to ball names

        if hasattr(detect, 'boxes') and detect.boxes is not None:
            boxes = detect.boxes
            names = detect.names
            # boxes.xyxy: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            # boxes.cls: (N,) array of class indices
            # boxes.conf: (N,) array of confidences

            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf

            print(f"YOLO detected {len(xyxy)} balls:")
            for i in range(len(xyxy)):
                class_idx = int(cls[i])
                ball_name = names.get(class_idx, f"class_{class_idx}")
                bbox = xyxy[i]
                confidence = conf[i] if conf is not None else None
                print(f"  Ball: {ball_name}, BBox: {bbox}, Confidence: {confidence:.2f}" if confidence is not None else f"  Ball: {ball_name}, BBox: {bbox}")
        else:
            print("No YOLO ball detections found.")
    
    def detect_balls(self, frame):
        """Detect balls on the table using color segmentation"""
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        
        # Convert BGR to RGB first (matching notebook)
        blur_RGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to HSV (matching notebook)
        hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)
        
        # HSV colors of the snooker table (EXACT notebook values)
        # lower_green = np.array([60, 200, 150])
        # upper_green = np.array([70, 255, 240])
        # lower_green = np.array([32, 42, 40])
        # upper_green = np.array([85, 255, 255])
        
        
        lower_green = np.array([56, 144, 144])
        upper_green = np.array([63, 253, 234])
        
        # overhead
        
        # lower_green = np.array([48, 39, 58])
        # upper_green = np.array([77, 255, 255])
        
        # lower_green = np.array([68, 15, 67])
        # upper_green = np.array([82, 239, 154])

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
        
        # Filter contours (matching notebook's filter_ctrs)
        min_s = 90
        max_s = 358
        alpha = 3.445
        
        
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Get bounding rectangle
            rot_rect = cv2.minAreaRect(contour)
            w = rot_rect[1][0]
            h = rot_rect[1][1]
            
            # Check aspect ratio first (matching notebook)
            if h > 0 and w > 0:
                if (h * alpha < w) or (w * alpha < h):
                    continue  # Skip non-circular shapes
            
            # Filter by area (matching notebook)
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
                    'bbox': cv2.boundingRect(contour)
                })
        
        # Return both mask_inv and intermediate masks for debugging
        return balls, mask_inv, mask, mask_closing
    
    def draw_balls(self, balls, frame):
        """Draw balls on the table with their colors (matching notebook style)"""
        K = np.ones((3, 3), np.uint8)
        
        final = self.create_table()
        mask = np.zeros((self.height, self.width), np.uint8)
        
        # Convert frame to RGB for color extraction (matching notebook)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for ball in balls:
            contour = ball['contour']
            cX, cY = ball['center']
            
            # Find color average inside contour (matching notebook)
            mask[...] = 0
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask_eroded = cv2.erode(mask, K, iterations=3)
            
            # Get mean color from RGB frame (matching notebook)
            mean_color = cv2.mean(frame_rgb, mask_eroded)
            color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
            
            # Draw ball (matching notebook - radius=8)
            radius = 8
            cv2.circle(final, (cX, cY), radius, color, -1)
            
            # Black border around ball (matching notebook - thickness=1)
            cv2.circle(final, (cX, cY), radius, 0, 1)
            
            # Light reflection (matching notebook)
            cv2.circle(final, (cX - 2, cY - 2), 2, (255, 255, 255), -1)
        
        return final
    
    def process_video(self, output_path=None):
        """Process the entire video"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
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
            
            # Detect balls (now returns 4 values)
            balls, mask_inv, mask_table, mask_closing = self.detect_balls(warped)
            # self.detect_balls_from_yolo(frame)
            # Draw table with balls (notebook style)
            top_view = self.draw_balls(balls, warped)
            top_view = self.draw_holes(top_view)
            
            # Add text overlay
            cv2.putText(top_view, f"Balls: {len(balls)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(top_view, f"Frame: {frame_count}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Create debug view with detected contours
            debug_view = warped.copy()
            for ball in balls:
                cv2.drawContours(debug_view, [ball['contour']], -1, (0, 255, 0), 2)
                cv2.circle(debug_view, ball['center'], 3, (0, 0, 255), -1)
                # Show area for debugging
                x, y = ball['center']
                cv2.putText(debug_view, f"{int(ball['area'])}", 
                           (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Write to output video
            if output_path:
                out.write(top_view)
            
            # Create visual masks for debugging (convert to BGR for display)
            mask_inv_visual = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            mask_table_visual = cv2.cvtColor(mask_table, cv2.COLOR_GRAY2BGR)
            mask_closing_visual = cv2.cvtColor(mask_closing, cv2.COLOR_GRAY2BGR)
            
            # Display multiple windows for debugging
            cv2.imshow('1. Pool Ball Tracking - 2D Top View', top_view)
            cv2.imshow('2. Warped View with Detections', debug_view)
            cv2.imshow('3. Original Warped Frame', warped)
            cv2.imshow('4. Table Mask (White = Table)', mask_table_visual)
            cv2.imshow('5. Table Mask After Closing', mask_closing_visual)
            cv2.imshow('6. Inverted Mask (White = Objects)', mask_inv_visual)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Balls detected: {len(balls)}")
                if len(balls) == 0:
                    print("  WARNING: No balls detected! Check if table color is correct.")
            
            # Check for quit (increased wait time for better key detection)
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
        print("Usage: python interactive_pool_tracker.py <video_path> [output_path]")
        print("\nExample:")
        print("  python interactive_pool_tracker.py input.mp4")
        print("  python interactive_pool_tracker.py input.mp4 output.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
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
    
    # Step 2: Process video with selected corners
    print("\nStep 2: Processing video with perspective transformation...")
    tracker = PoolBallTracker(video_path, corners)
    tracker.process_video(output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
