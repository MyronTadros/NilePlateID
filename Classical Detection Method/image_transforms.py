import cv2
import numpy as np


# function to sort points in correct order
def order_points(pts):
    # we need to make sure the 4 corner points are in the right order
    # for perspective transform to work properly
    result = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum of x+y
    # the bottom-right has the largest sum
    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top left
    result[2] = pts[np.argmax(s)]  # bottom right

    # top-right has smallest diff between y-x
    # bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(diff)]  # top right
    result[3] = pts[np.argmax(diff)]  # bottom left

    return result


# transform a region to rectangular shape (makes it easier to read)
def four_point_transform(image, pts):
    # first order the points correctly
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # calculate the width of the new image
    # we take the max of top and bottom widths
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # calculate the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # these are the destination points for our transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # do the perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
