import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_and_crop_plate(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Create a copy for drawing
    img_display = img.copy()
    
    # 2. Preprocessing
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter: Removes noise while keeping edges sharp (better than Gaussian for plates)
    # d=11, sigmaColor=17, sigmaSpace=17 are standard starting points
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # 3. Canny Edge Detection
    # Thresholds are empirical; Egyptian plates often have high contrast (Blue/White)
    edged = cv2.Canny(filtered, 30, 200)

    # 4. Find Contours based on Edges
    # RETR_TREE retrieves all contours and reconstructs a full hierarchy
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first) to speed up finding the main plate
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    screen_cnt = None
    plate_roi = None

    for c in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # We are looking for a rectangle (4 points)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Egyptian plates usually have an aspect ratio between 2 and 5
            # (European style is long, older style is closer to 2:1)
            if 2.0 <= aspect_ratio <= 4.0:
                screen_cnt = approx
                plate_roi = gray[y:y+h, x:x+w]
                
                # --- HARRIS CORNER DETECTION (Validation Step) ---
                # We use Harris here to confirm the region has "text-like" features.
                # A blank rectangle (like a window) won't have many corners inside.
                
                harris_dst = cv2.cornerHarris(plate_roi, blockSize=2, ksize=3, k=0.04)
                
                # Threshold for an optimal value, it may vary depending on the image.
                # We count strong corners.
                harris_corners_count = np.sum(harris_dst > 0.01 * harris_dst.max())
                
                # Heuristic: A plate should have a reasonable density of corners due to text
                if harris_corners_count > 20: 
                    # Draw the contour on the original image
                    cv2.drawContours(img_display, [screen_cnt], -1, (0, 255, 0), 3)
                    
                    # Crop the actual color plate
                    cropped_plate = img[y:y+h, x:x+w]
                    
                    # Visualizing Harris on the crop for your requirement
                    # Dilate to mark the corners clearly
                    harris_vis = cropped_plate.copy()
                    harris_dst_norm = cv2.normalize(harris_dst, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # Mark corners in Red
                    harris_vis[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]

                    # Plotting
                    plot_results(img, edged, harris_vis, cropped_plate)
                    return cropped_plate
                
    print("No valid Egyptian license plate found.")
    return None

def plot_results(original, edges, harris_vis, crop):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 2)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Harris Corners (Text Features)")
    plt.imshow(cv2.cvtColor(harris_vis, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 4)
    plt.title("Final Crop")
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()

# --- USAGE ---
# Replace 'car_image.jpg' with your actual file path (e.g., the Alfa Romeo image you uploaded)
# process_and_crop_plate('blue car.jpeg')