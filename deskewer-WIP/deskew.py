import os

import cv2
import numpy as np


def order_points(pts):
    """Sort points based on their x and y coordinates."""
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)  # each entry in s is the sum of the x any y components of a point
    rect[0] = pts[np.argmin(s)]  # top-left will have the smallest combined x,y value
    rect[2] = pts[np.argmax(s)]  # bottom-right will have the largest combined x,y value

    diff = np.diff(pts, axis=1)  # each entry in diff is the y-x value of a point
    rect[1] = pts[np.argmin(diff)]  # top-right will have the most negative y-x value
    rect[3] = pts[np.argmax(diff)]  # bottom-left will have the most positive y-x value

    return rect


def four_point_transform(image, pts):
    """Perform perspective transformation with an added margin."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # top-left, top-right, bottom-right, bottom-left

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # length of bottom of quadrilateral
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # length of top of quadrilateral
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # height of right side of quadrilateral
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # height of left side of quadrilateral
    maxHeight = max(int(heightA), int(heightB))

    # shift the quadrilateral to the top left of the image
    min_y = min(y for x, y in rect)
    min_x = min(x for x, y in rect)
    for i in range(4):
        rect[i][0] -= min_x
        rect[i][1] -= min_y

    # generate the coordinates of quadrangle vertices in the destination image
    dst = np.array([
        [0, 0],  # top-left
        [maxWidth, 0],  # top-right
        [maxWidth, maxHeight],  # bottom-right
        [0, maxHeight]], dtype="float32")  # bottom-left

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    return warped


def debug_draw_and_save_contour(image, contour, save_path):
    image = image.copy()

    # draw the contour
    cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)

    # draw the points
    for (x, y) in contour.reshape(-1, 2):
        cv2.circle(image, (x, y), 10, (255, 0, 0), 3)

    # get and draw the minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)

    return rect[2]  # angle of minAreaRect rectangle


def find_largest_quadrilateral(image):
    """Automatically detect the largest box in the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    # edged = cv2.Canny(blurred, 50, 150)

    binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # sort contours by area (largest first)

    debug_draw_and_save_contour(image, contours[0], 'deskewer-WIP/test01.png')

    largest_contour = contours[0]

    perimeter = cv2.arcLength(largest_contour, True)
    for relaxation_factor in np.linspace(0, 0.05, 50):
        approx = cv2.approxPolyDP(largest_contour, relaxation_factor * perimeter, True)
        print(f'len(approx): {len(approx)}')

        if len(approx) == 4:
            debug_draw_and_save_contour(image, approx, 'deskewer-WIP/test02.png')
            return approx.reshape(4, 2)
    print('faild to find quadrilateral contour! using minAreaRect instead!')

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box


def transform_image(image_path, output_path):
    image = cv2.imread(image_path)

    refPts = find_largest_quadrilateral(image)

    warped = four_point_transform(image, np.array(refPts))
    cv2.imwrite(output_path, warped)


if __name__ == '__main__':
    input_image = './input.png'
    # input_image = 'ParachuteData/pdf-pages-as-images/T-11 LAT (SEPT 2022)-064.png'
    output_image = './output.png'
    transform_image(input_image, output_image)

    # for fname in os.listdir('pdf-pages-as-images'):
    #     fpath = os.path.join('pdf-pages-as-images', fname)
    #     output_fpath = os.path.join('output-images', fname)
    #     print(fname)
    #     rectify_image(fpath, output_fpath)
