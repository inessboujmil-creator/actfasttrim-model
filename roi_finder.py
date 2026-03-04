
import cv2
import numpy as np

# Global variables to store the ROI coordinates
roi_start_point = (-1, -1)
roi_end_point = (-1, -1)
drawing = False
frame_copy = None
roi_rect = None

def draw_roi(event, x, y, flags, param):
    """Mouse callback function to draw a rectangle and select the ROI."""
    global roi_start_point, roi_end_point, drawing, frame_copy, roi_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end_point = (x, y)
        # Final rectangle
        cv2.rectangle(frame_copy, roi_start_point, roi_end_point, (0, 255, 0), 2)
        # Ensure x1 < x2 and y1 < y2
        x1, y1 = roi_start_point
        x2, y2 = roi_end_point
        roi_rect = (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))


def main():
    """Main function to run the ROI and threshold finder."""
    global frame_copy, roi_rect

    video_path = input("\n>>> Please enter the full path to one of your video files and press Enter: ")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: Could not open video file: {video_path}")
        print("Please check the path and try again.")
        return

    # Read a frame from 5 seconds into the video
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("\nERROR: Could not read a frame from the video.")
        print("The video file might be corrupted or in an unsupported format.")
        return
        
    frame_copy = frame.copy()
    window_name = "ROI Selector - Draw a box around the timestamp, then press 'c'"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_roi)

    print("\n--- INSTRUCTIONS ---")
    print("1. A window named 'ROI Selector' has appeared.")
    print("2. Use your mouse to draw a tight box around the timestamp (e.g., 16:20:42).")
    print("3. Once you are happy with the box, press the 'c' key on your keyboard to confirm.")
    print("4. A second window named 'Processed' will appear with a slider.")
    print("--------------------")

    while True:
        # Show the frame with the rectangle
        display_frame = frame_copy.copy()
        if drawing:
            cv2.rectangle(display_frame, roi_start_point, roi_end_point, (0, 0, 255), 2)
        elif roi_rect:
            cv2.rectangle(display_frame, (roi_rect[0], roi_rect[1]), (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if roi_rect is None:
                print("\nWARNING: You did not select a region. Please draw a box first.")
            else:
                break
        elif key == 27: # Escape key
            cv2.destroyAllWindows()
            print("\nExited without selecting a region.")
            return
            
    cv2.destroyAllWindows()
    
    x, y, w, h = roi_rect
    # Convert to the format needed by the main script: [y_start, y_end, x_start, x_end]
    final_roi_coords = [y, y + h, x, x + w]

    print("\n--- THRESHOLD FINDER ---")
    print("1. A new window named 'Processed' is now open.")
    print("2. Adjust the 'Threshold' slider until the timestamp text is sharp, black, and isolated.")
    print("3. The goal is a pure black text on a pure white background.")
    print("4. Press the 'q' key on your keyboard when you find the best value.")
    print("------------------------")
    
    proc_window = "Processed"
    cv2.namedWindow(proc_window)
    cv2.createTrackbar("Threshold", proc_window, 150, 255, lambda x: None)
    
    while True:
        threshold_val = cv2.getTrackbarPos("Threshold", proc_window)
        
        # Apply the exact same processing as the main script
        roi_frame = frame[final_roi_coords[0]:final_roi_coords[1], final_roi_coords[2]:final_roi_coords[3]]
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, threshold_val, 255, cv2.THRESH_BINARY)
        inverted_frame = cv2.bitwise_not(thresh_frame)
        
        # Scale up the small processed window so it's readable
        scale_factor = 5
        h, w = inverted_frame.shape
        inverted_frame_resized = cv2.resize(inverted_frame, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(proc_window, inverted_frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    final_threshold = cv2.getTrackbarPos("Threshold", proc_window)
    cv2.destroyAllWindows()

    print("\n\n--- SUCCESS! ---")
    print("The script can now be fixed. Please copy the following output and paste it in the chat.")
    print("\n----- COPY BELOW THIS LINE -----")
    print(f""roi": {final_roi_coords},")
    print(f""threshold": {final_threshold}")
    print("----- COPY ABOVE THIS LINE -----\n")


if __name__ == "__main__":
    main()
