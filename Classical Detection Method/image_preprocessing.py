import cv2
import numpy as np


# adjust brightness and contrast automatically (helps with dark images)
def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    # convert to gray first
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # calculate cumulative distribution
    acc = []
    acc.append(float(hist[0]))
    for index in range(1, hist_size):
        acc.append(acc[index -1] + float(hist[index]))

    # find the clip values
    maximum = acc[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # find minimum gray level
    minimum_gray = 0
    while acc[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # find maximum gray level
    maximum_gray = hist_size -1
    while acc[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # apply the transformation
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result, alpha, beta
