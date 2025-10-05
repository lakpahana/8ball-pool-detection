"""
HSV Color Range Tuner for Pool Table Detection
Use this tool to find the correct HSV range for your table color
"""

import cv2
import numpy as np
import sys

def nothing(x):
    pass

def tune_hsv(video_path):
    """Interactive HSV tuner with trackbars"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Resize frame if too large
    max_width = 800
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Create window and trackbars
    cv2.namedWindow('HSV Tuner')
    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Result')
    
    # Create trackbars for HSV range
    # Default values for green table
    cv2.createTrackbar('H Min', 'HSV Tuner', 35, 179, nothing)
    cv2.createTrackbar('H Max', 'HSV Tuner', 85, 179, nothing)
    cv2.createTrackbar('S Min', 'HSV Tuner', 40, 255, nothing)
    cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'HSV Tuner', 40, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)
    
    print("\n" + "="*60)
    print("HSV TUNER - Find the right color range for your table")
    print("="*60)
    print("\nInstructions:")
    print("1. Adjust the trackbars until the MASK shows only the table")
    print("2. White in the mask = detected table")
    print("3. Try to get the entire table white, nothing else")
    print("4. Press 's' to save the values")
    print("5. Press 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        # Get current trackbar positions
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')
        
        # Convert to HSV
        blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        blur_RGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)
        
        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create result image
        result = cv2.bitwise_and(frame, frame, mask=mask_closing)
        
        # Add text with current values
        info = frame.copy()
        cv2.putText(info, f"H: [{h_min}, {h_max}]", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info, f"S: [{s_min}, {s_max}]", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info, f"V: [{v_min}, {v_max}]", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info, "Press 's' to save, 'q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display images
        cv2.imshow('HSV Tuner', info)
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask_closing)
        cv2.imshow('Result', result)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting without saving...")
            break
        
        elif key == ord('s') or key == ord('S'):
            print("\n" + "="*60)
            print("SAVE THESE VALUES:")
            print("="*60)
            print(f"\nIn your code, change these lines:")
            print(f"lower_green = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_green = np.array([{h_max}, {s_max}, {v_max}])")
            print("\n" + "="*60)
            print("\nPress any key to continue tuning or 'q' to quit...")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("Usage: python hsv_tuner.py <video_path>")
        print("\nExample:")
        print("  python hsv_tuner.py overhead.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    tune_hsv(video_path)

if __name__ == "__main__":
    main()
