import os

import cv2
import numpy as np


# set the minium proportion of the page area that must be taken up by a quadrilateral to use it for deskewing
# if the quadrilateral takes up less then this fraction of the page area, it is likely that the page
# does not contain a grid
minAreaProportion = 1/20


def debug_draw_contour_and_save_image(image, contour, save_path):
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


def preprocess_image(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # convert to binary
    binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)
    
    return binary


def find_largest_quadrilateral(image):
    # preprocess the image
    binary = preprocess_image(image)
    
    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by area (largest first)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    def areaOfConvexHull(c):
        return cv2.contourArea(cv2.convexHull(c))
    contours = sorted(contours, key=areaOfConvexHull, reverse=True)
    # select largest contour
    largest_contour = contours[0]

    largest_contour = cv2.convexHull(largest_contour)

    debug_draw_contour_and_save_image(image, largest_contour, 'deskewer-WIP/test01.png')

    # try to approximate the contour with polygons using "cv2.approxPolyDP" (check OpenCV docs for more info)
    # we try to relax the epsilon parameter until we find a quadrilateral approximation of the contour
    # if we fail to find a quadrilateral approximation, we will simply use a minmum area rectangle to approximate it
    epsilon_fraction_of_max_dimension = 1/20
    max_epsilon = max([image.shape[0], image.shape[1]])*epsilon_fraction_of_max_dimension
    for epsilon in np.linspace(0, max_epsilon, 100):
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        print(f'len(approx): {len(approx)}')

        # if we found an approximation of the contour with only 4 sides, we have our quadrilateral
        if len(approx) == 4:
            debug_draw_contour_and_save_image(image, approx, 'deskewer-WIP/test02.png')
            return approx.reshape(4, 2)
        
        # if we've reduced to our approximation to a polygon with less then 4 points, we can stop looking
        # NOTE: this isn't needed, but it saves on computation by not checking values that won't work
        if len(approx) < 4:
            break
    
    # if we failed to find a (non-rectangular) quadrilateral contour, we can just use a minium area rectangle
    print('faild to find quadrilateral contour approximation! using minAreaRect instead!')
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    print(box)
    return box


def order_quadrilateral_points(pts):
    """Sort points based on their x and y coordinates."""
    quadrilateral = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1) # each entry in s is the sum of the x any y components of a point
    quadrilateral[0] = pts[np.argmin(s)] # top-left will have the smallest combined x,y value
    quadrilateral[2] = pts[np.argmax(s)] # bottom-right will have the largest combined x,y value

    diff = np.diff(pts, axis=1) # each entry in diff is the y-x value of a point
    quadrilateral[1] = pts[np.argmin(diff)] # top-right will have the most negative y-x value
    quadrilateral[3] = pts[np.argmax(diff)] # bottom-left will have the most positive y-x value

    return quadrilateral


def four_point_transform(image, quadrilateral):
    (tl, tr, br, bl) = quadrilateral # top-left, top-right, bottom-right, bottom-left

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) # length of bottom of quadrilateral
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)) # length of top of quadrilateral
    maxWidth = int(max(widthA, widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) # height of right side of quadrilateral
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # height of left side of quadrilateral
    maxHeight = int(max(heightA, heightB))

    # shift the quadrilateral to the top left of the image
    # NOTE: this step is required so that the part of the page above the quadrilateral doesn't get cropped off
    min_y = min(y for x, y in quadrilateral)
    min_x = min(x for x, y in quadrilateral)
    for i in range(4):
        quadrilateral[i][0] -= min_x
        quadrilateral[i][1] -= min_y
    
    # generate the coordinates of quadrangle vertices in the destination image
    dst = np.array([
        [0, 0], #top-left
        [maxWidth, 0], #top-right
        [maxWidth, maxHeight], #bottom-right
        [0, maxHeight]], dtype="float32") #bottom-left
    
    # generate transformation matrix and perform transform
    M = cv2.getPerspectiveTransform(quadrilateral, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), borderValue=(255, 255, 0))
    return warped


def deskew_image(image_path, output_path):
    image = cv2.imread(image_path)

    pts = find_largest_quadrilateral(image)
    quadrilateral = order_quadrilateral_points(pts)
    
    # calculate approximate area of quadrilateral
    (tl, tr, br, bl) = quadrilateral # top-left, top-right, bottom-right, bottom-left
    height = br[1] - tl[1] # approximation of height of quadrilateral
    width = br[0] - tl[0] # approximation of width of quadrilateral
    area = height*width
    
    # claculate minimum required area of quadrilateral
    minArea = image.shape[0]*image.shape[1]*minAreaProportion

    if area < minArea:
        print('Was unable to find large enough quadrilateral to confidently deskew image!'\
              ' Saving original image as deskewed iamge!')
        cv2.imwrite(output_path, image)
    else:
        warped = four_point_transform(image, quadrilateral)
        cv2.imwrite(output_path, warped)


def main():
    skewed_images_path = './ParachuteData/pdf-pages-as-images'
    deskewed_images_path = './deskewer-WIP/output-images'

    for skewed_image_filename in os.listdir(skewed_images_path):
        skewed_image_path = os.path.join(skewed_images_path, skewed_image_filename)
        deskewed_image_path = os.path.join(deskewed_images_path, skewed_image_filename)
        print(skewed_image_filename)
        deskew_image(skewed_image_path, deskewed_image_path)


if __name__ == '__main__':
    input_image = 'deskewer-WIP/input1.png'
    input_image = 'ParachuteData/pdf-pages-as-images/T-11 LAT (SEPT 2022)-030.png' # 53
    # input_image = 'ParachuteData/pdf-pages-as-images/T-11 W911QY-19-D-0046 LOT 45_09282023-008.png'
    output_image = 'deskewer-WIP/output.png'
    deskew_image(input_image, output_image)

    main()
