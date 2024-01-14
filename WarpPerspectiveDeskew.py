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


def extract_cell_edges_from_image(image):
    # NOTE: this is similar, but not exactly the same as the function defined in ConvertImagesToXLSX

    def convert_image_to_binary(image):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # convert to binary
        binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)

        return binary

    binary = convert_image_to_binary(image)
    # cv2.imwrite('test01.png', binary)

    # detect horizontal edges
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal_edges = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cv2.imwrite('test02.png', horizontal_edges)

    # detect vertical edges
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_edges = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cv2.imwrite('test03.png', vertical_edges)

    # combine edges
    all_edges = vertical_edges | horizontal_edges
    # cv2.imwrite('./deskewer-WIP/processing-test04.png', all_edges)

    return all_edges


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


def find_largest_quadrilateral(image):
    absolute_top_bottom_slope_difference_threshold = 0.01

    # preprocess the image
    cell_edges = extract_cell_edges_from_image(image)
    # cv2.imwrite('deskewer-WIP/test00.png', cell_edges)
    
    # find contours
    contours, _ = cv2.findContours(cell_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # replace contours with their convex hull
    contours = [cv2.convexHull(c) for c in contours]

    # sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # loop through contours from largest to smallest. on each one we try to approximate the contour with polygons
    # using "cv2.approxPolyDP" (check OpenCV docs for more info).
    # we relax the epsilon parameter until we find a quadrilateral approximation of the contour or skip past a
    # quadrilateral to a fewer sided shape (in which case we try again with the next largest contour).
    # if we find a quadrilateral, we need to check if it's roughly a parallelogram.
    # if it is, we can return it. if it's not, we can just try again with the next largest contour
    fraction_epsilon_is_of_max_dimension = 1/50
    max_epsilon = max([image.shape[0], image.shape[1]])*fraction_epsilon_is_of_max_dimension
    for contour in contours[:5]:
        for epsilon in np.linspace(0, max_epsilon, 100):
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # print(f'len(approx): {len(approx)}')
            # if we found an approximation of the contour with only 4 sides, we have a quadrilateral
            # if we've reduced to our approximation to a polygon with less then 4 points, we can stop looking
            if len(approx) <= 4:
                break
        
        # if the approximation has less then 4 sides, try the next contour
        if len(approx) < 4:
            continue

        # name the points of the quadrilateral
        tl, tr, br, bl = order_quadrilateral_points(approx.reshape(4, 2)) # top-left, top-right, bottom-right, bottom-left
        
        # calculate the slopes of the sides
        slope_top = (tl[1]-tr[1])/(tl[0]-tr[0])
        slope_bottom = (bl[1]-br[1])/(bl[0]-br[0])
        # slope_left = (tl[1]-bl[1])/(tl[0]-bl[0])
        # slope_right = (tr[1]-br[1])/(tr[0]-br[0])
        # print(f'slope_top: {slope_top}, slope_bottom: {slope_bottom}, slope_left: {slope_left}, slope_right: {slope_right}')

        # if the absolute difference of the top slopes exceeds our threshold, keep looking (at the next largest contour)
        if np.abs(slope_top - slope_bottom) > absolute_top_bottom_slope_difference_threshold:
            continue
        
        # debug_draw_contour_and_save_image(image, approx, 'deskewer-WIP/test02.png')
        return approx.reshape(4, 2)
    
    # if we failed to find a parallel quadrilateral contour within the first few (as specified by num_contours_to_check),
    # we can just return the original image
    raise Exception("wasn't sure we would ever hit this point!\
                    we should handle this case by just doing nothing to the image")


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


def warp_perspective_deskew(image_path, output_path):
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
    skewed_images_path = './ParachuteData/pdf-pages-as-images-preprocessed'
    deskewed_images_path = './ParachuteData/pdf-pages-as-images-preprocessed-deskewed'

    for skewed_image_filename in os.listdir(skewed_images_path):
        skewed_image_path = os.path.join(skewed_images_path, skewed_image_filename)
        deskewed_image_path = os.path.join(deskewed_images_path, skewed_image_filename)
        print(skewed_image_filename)
        warp_perspective_deskew(skewed_image_path, deskewed_image_path)


if __name__ == '__main__':
    # # input_image = 'deskewer-WIP/i1.png'
    # input_image = 'ParachuteData/pdf-pages-as-images-preprocessed/T-11 LAT (SEPT 2022)-064.png'
    # # input_image = 'ParachuteData/pdf-pages-as-images-preprocessed/T-11 W911QY-19-D-0046 LOT 45_09282023-008.png'
    # output_image = 'deskewer-WIP/output.png'
    # warp_perspective_deskew(input_image, output_image)

    main()
