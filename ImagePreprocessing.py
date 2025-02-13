import cv2

def preprocess_frame(frame, width, height, lower_hsv, upper_hsv):
    """
    Resize, apply color filtering, and process the input frame for lane detection.
    Args:
        frame: Original input frame (BGR).
        width: Target width for resizing.
        height: Target height for resizing.
        lower_hsv: Lower HSV bound (lower color filter).
        upper_hsv: Upper HSV bound (upper color filter).
    Returns:
        processed_frame: Preprocessed color image after color filtering.
    """
    # Resize the frame to the target dimensions
    resized_frame = cv2.resize(frame, (width, height))

    # Convert the resized frame to HSV color space
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the HSV bounds (detect specific color range)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    # Bitwise AND the original frame with the mask to apply the color filter
    filtered_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

    # Apply edge detection for lane detection (Canny edge detection)
    edges = cv2.Canny(filtered_frame, 50, 150)  # You can adjust these threshold values

    # Optional: You can combine the edges with the original frame to highlight lanes
    return cv2.bitwise_or(filtered_frame, edges)