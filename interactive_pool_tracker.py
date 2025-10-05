"""
Interactive Pool Table Ball Tracker
Allows user to mark the 4 corners of the pool table on the first frame,
then applies perspective transformation and processes the video.
"""

import cv2
import numpy as np
import sys

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
                cv2.putText(self.display_image, str(len(self.corners)), 
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
        print("1. Click on the 4 corners of the pool table in order")
        print("   (e.g., top-left, top-right, bottom-right, bottom-left)")
        print("2. Press 'c' to continue after selecting 4 corners")
        print("3. Press 'r' to reset and start over")
        print("4. Press 'q' to quit")
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
        self.width = 800  # Output width
        self.height = 400  # Output height
        
        # Define destination points for perspective transform
        self.dst_points = np.array([
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        self.matrix = cv2.getPerspectiveTransform(self.corners, self.dst_points)
        
    def get_perspective_transform(self, frame):
        """Apply perspective transformation to frame"""
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height))
    
    def detect_balls(self, frame):
        """Detect balls on the table using color segmentation"""
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        # Define range for green table (adjust these values as needed)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green table
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological closing to fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Invert mask to get non-table objects (balls)
        mask_inv = cv2.bitwise_not(mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []
        
        # Filter contours to find balls
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these values based on your video)
            if 50 < area < 2000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (balls should be roughly circular)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if 0.7 < aspect_ratio < 1.3:  # Roughly square/circular
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        
                        balls.append({
                            'center': (cX, cY),
                            'contour': contour,
                            'area': area,
                            'bbox': (x, y, w, h)
                        })
        
        return balls, mask
    
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
            
            # Detect balls
            balls, mask = self.detect_balls(warped)
            
            # Draw detected balls
            result = warped.copy()
            for ball in balls:
                # Draw contour
                cv2.drawContours(result, [ball['contour']], -1, (0, 255, 0), 2)
                
                # Draw center
                cv2.circle(result, ball['center'], 5, (0, 0, 255), -1)
                
                # Draw bounding box
                x, y, w, h = ball['bbox']
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Add text overlay
            cv2.putText(result, f"Balls detected: {len(balls)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Frame: {frame_count}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Write to output video
            if output_path:
                out.write(result)
            
            # Display
            cv2.imshow('Pool Ball Tracking', result)
            cv2.imshow('Table Mask', mask)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
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
