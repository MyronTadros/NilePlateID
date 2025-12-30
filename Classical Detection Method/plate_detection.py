import cv2
import numpy as np
from image_transforms import four_point_transform
from image_preprocessing import automatic_brightness_and_contrast


# plates have a blue stripe at the top which makes them easier to find
def detect(img_rgb):
    img = img_rgb.copy()
    h = img_rgb.shape[0]  # image height
    w = img_rgb.shape[1]  # image width
    
    # convert to HSV color space (better for color detection)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    # detect blue color - plates have blue stripe on top
    # these values work for detecting blue
    low_blue = np.array([100, 150, 50])
    high_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    blue = cv2.bitwise_and(blue_mask, blue_mask, mask=blue_mask)

    # save step for debugging (will be saved for each detected plate later)
    cv2.imwrite("temp/steps/1_blue_color_detection.png", blue)
    
    # use morphology to fill small gaps in the detection
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("temp/steps/2_closing_morphology.png", closing)
    
    # find all contours in the blue regions
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # this list will hold all the detected plates
    crops = []

    # loop through each contour and check if it could be a plate
    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)

        # check dimensions - plates are roughly 2:1 ratio
        # also filter out very small detections
        if height*6 > width > 2 * height and height > 0.1 * width and width * height > h * w * 0.0001:
            print("Found candidate at x=", x, " y=", y, " w=", width, " h=", height)

            # crop the top part where the blue stripe should be
            crop_top = img_rgb[y:y + round(height/3), x:x+width]
            crop_top = crop_top.astype('uint8')

            # check how much blue is actually in the top part
            try:
                hsv2 = cv2.cvtColor(crop_top, cv2.COLOR_BGR2HSV)
                low = np.array([100,150,50])
                high = np.array([130,255,255])
                mask = cv2.inRange(hsv2, low, high)
                blue_sum = mask.sum()
            except:
                blue_sum = 0  # if something goes wrong assume no blue

            # if we found enough blue pixels
            if blue_sum > 200:
                print("  Blue found:", blue_sum)

                # now check for white/gray color below the blue stripe
                # the actual plate text is on white/yellow background
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                # crop the part below blue stripe
                blue_h = round(height/4)
                crop_white = img_rgb[y+blue_h:y+height, x:x+width]
                crop_white = crop_white.astype('uint8')

                # convert to hsv for color detection
                hsv3 = cv2.cvtColor(crop_white, cv2.COLOR_BGR2HSV)
                
                # detect white pixels
                low_white = np.array([0, 0, 180])
                high_white = np.array([180, 60, 255])
                white_mask = cv2.inRange(hsv3, low_white, high_white)

                # detect gray pixels (some plates look grayish)
                low_gray = np.array([0, 0, 150])
                high_gray = np.array([180, 100, 255])
                gray_mask = cv2.inRange(hsv3, low_gray, high_gray)
                
                # also check for bright pixels in general
                white_mask_bright = cv2.inRange(hsv3, np.array([0, 0, 180]), np.array([180, 255, 255]))
                
                # combine all the masks together
                white_mask = cv2.bitwise_or(white_mask, gray_mask)
                white_mask = cv2.bitwise_or(white_mask, white_mask_bright)

                white_sum = white_mask.sum()
                threshold = 1500  # lowered from 5000 to be more permissive with different lighting conditions
                print("  White/Gray:", white_sum, "threshold:", threshold)

                # if we found enough white/gray pixels
                if white_sum > threshold:
                    print("  White found:", white_sum)

                    # create gray crop for further processing
                    crop_gray = gray[y:y + height, x:x + width]
                    crop_gray = crop_gray.astype('uint8')

                    # find all white regions and get their bounding box
                    try:
                        white_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    except:
                        white_cnts = []

                    if white_cnts:
                        # find the bounding box of all white contours combined
                        xs = []
                        ys = []
                        xws = []
                        yhs = []
                        for wc in white_cnts:
                            xx, yy, ww, hh = cv2.boundingRect(wc)
                            xs.append(xx)
                            ys.append(yy)
                            xws.append(xx + ww)
                            yhs.append(yy + hh)

                        wx_min = min(xs)
                        wy_min = min(ys)
                        wx_max = max(xws)
                        wy_max = max(yhs)

                        # convert back to original image coordinates
                        wx_min_g = x + wx_min
                        wy_min_g = y + blue_h + wy_min
                        wx_max_g = x + wx_max
                        wy_max_g = y + blue_h + wy_max

                        # make sure we dont go outside image boundaries
                        x_min = max(0, min(x, int(wx_min_g)))
                        y_min = max(0, min(y, int(wy_min_g)))
                        x_max = min(w - 1, max(x + width, int(wx_max_g)))
                        y_max = min(h - 1, max(y + height, int(wy_max_g)))

                        # fix aspect ratio to match plate dimensions (35cm x 17cm)
                        plate_ratio = 35.0 / 17.0
                        cur_w = x_max - x_min
                        cur_h = y_max - y_min
                        if cur_h <= 0:
                            cur_h = 1
                        if cur_w <= 0:
                            cur_w = 1

                        # adjust width or height to match proper ratio
                        if (cur_w / float(cur_h)) < plate_ratio:
                            # width is too small, increase it
                            new_w = int(plate_ratio * cur_h)
                            dw = new_w - cur_w
                            x_min = max(0, x_min - dw // 2)
                            x_max = min(w - 1, x_max + (dw - dw // 2))
                        else:
                            # height is too small, increase it
                            new_h = int(cur_w / plate_ratio)
                            dh = new_h - cur_h
                            y_min = max(0, y_min - dh // 2)
                            y_max = min(h - 1, y_max + (dh - dh // 2))

                        # shift down a bit to include full plate
                        shift = int(0.3 * (y_max - y_min))
                        y_min_shifted = y_min + shift
                        y_max_shifted = y_max + shift
                        if y_max_shifted > h - 1:
                            over = y_max_shifted - (h - 1)
                            y_min_shifted = max(0, y_min_shifted - over)
                            y_max_shifted = h - 1

                        # create the points for perspective transform
                        pts_full = np.array([[x_min, y_min_shifted], [x_max, y_min_shifted], [x_max, y_max_shifted], [x_min, y_max_shifted]], dtype="float32")
                        warped_full = four_point_transform(img, pts_full)
                    else:
                        # if we couldn't find white contours, just use the original contour
                        rect_full = cv2.minAreaRect(cnt)
                        box_full = cv2.boxPoints(rect_full)
                        box_full = np.int32(box_full)
                        bx = box_full[:, 0]
                        by = box_full[:, 1]
                        x_min = max(0, int(bx.min()))
                        y_min = max(0, int(by.min()))
                        x_max = min(w - 1, int(bx.max()))
                        y_max = min(h - 1, int(by.max()))

                        # adjust aspect ratio again
                        plate_ratio = 35.0 / 17.0
                        cur_w = x_max - x_min
                        cur_h = y_max - y_min
                        if cur_h <= 0:
                            cur_h = 1
                        if cur_w <= 0:
                            cur_w = 1

                        if (cur_w / float(cur_h)) < plate_ratio:
                            new_w = int(plate_ratio * cur_h)
                            dw = new_w - cur_w
                            x_min = max(0, x_min - dw // 2)
                            x_max = min(w - 1, x_max + (dw - dw // 2))
                        else:
                            new_h = int(cur_w / plate_ratio)
                            dh = new_h - cur_h
                            y_min = max(0, y_min - dh // 2)
                            y_max = min(h - 1, y_max + (dh - dh // 2))

                        # small shift down
                        shift = int(0.05 * (y_max - y_min))
                        y_min_shifted = y_min + shift
                        y_max_shifted = y_max + shift
                        if y_max_shifted > h - 1:
                            over = y_max_shifted - (h - 1)
                            y_min_shifted = max(0, y_min_shifted - over)
                            y_max_shifted = h - 1

                        pts_full = np.array([[x_min, y_min_shifted], [x_max, y_min_shifted], [x_max, y_max_shifted], [x_min, y_max_shifted]], dtype="float32")
                        warped_full = four_point_transform(img, pts_full)

                    # add the warped plate to our crops list
                    crops.append(warped_full)

                    # now lets check if we actually have characters in this region
                    # this helps filter out false positives
                    th = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    contours2, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # count how many character-like regions we found
                    chars = 0
                    for c in contours2:
                        area2 = cv2.contourArea(c)
                        x2, y2, w2, h2 = cv2.boundingRect(c)
                        # characters should be taller than wide and not too big
                        if w2 * h2 > height * width * 0.01 and h2 > w2 and area2 < height * width * 0.9:
                            chars = chars + 1
                    
                    print("  Chars found:", chars)

                    img_rgb = cv2.putText(img_rgb, 'LP', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    box = np.int32(pts_full)
                    cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)

    return img_rgb, crops

# process the detected plate to prepare it for OCR
def process(src, plate_num=1):
    # save original detected plate
    cv2.imwrite(f"temp/steps/plate{plate_num}_3_detected_plate.png", src)
    
    # adjust brightness and contrast to make text clearer
    adjusted, a, b = automatic_brightness_and_contrast(src)
    cv2.imwrite(f"temp/steps/plate{plate_num}_4_Brigthness_contrast_adjustment.png", adjusted)
    
    # convert to grayscale (OCR works better on grayscale)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"temp/steps/plate{plate_num}_5_gray.png", gray)
    
    # apply threshold to get black and white image
    # OTSU method automatically finds best threshold value
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"temp/steps/plate{plate_num}_6_threshold.png", th)
    
    return th
