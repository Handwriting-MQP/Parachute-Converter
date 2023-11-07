import os

import cv2
import numpy as np


# doc info
# graphics: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
# examples: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html


# example of how MORPH_OPEN works:
# k = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
# a = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
# b = cv2.erode(binary, k)
# b = cv2.dilate(b, k)
# print(np.all(a == b)) # this will return True



max_image_dimension = 4000


def basic_image_preprocessing(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # convert to binary
    binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)
    
    return binary


def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    # scale the image to have a constant max dimension
    # for more info: https://ai.stackexchange.com/questions/24311/why-do-we-resize-images-before-using-them-for-object-detection
    scale_factor = max_image_dimension/max(image.shape)
    new_shape = (int(scale_factor*image.shape[1]), int(scale_factor*image.shape[0]))
    image = cv2.resize(image, new_shape)
    cv2.imwrite('deskewer-WIP/test00.png', image)

    binary = basic_image_preprocessing(image)
    cv2.imwrite('deskewer-WIP/test01.png', binary)

    # detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cv2.imwrite('deskewer-WIP/test02.png', horizontal_lines)

    # detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cv2.imwrite('deskewer-WIP/test03.png', vertical_lines)

    # combine lines
    all_lines = vertical_lines | horizontal_lines
    cv2.imwrite('deskewer-WIP/test04.png', all_lines)

    # run kernel over image to close up gaps
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    all_lines_2 = cv2.morphologyEx(all_lines, cv2.MORPH_CLOSE, cross_kernel)
    cv2.imwrite('deskewer-WIP/test05.png', all_lines_2)

    # ones_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # all_lines_3 = cv2.morphologyEx(all_lines_2, cv2.MORPH_CLOSE, ones_kernel)
    # cv2.imwrite('deskewer-WIP/test05.png', all_lines_2)

    # kernel = np.ones((2,2), np.uint8)
    # d = cv2.morphologyEx(all_lines, cv2.MORPH_CLOSE, kernel, iterations=5)

    # cv2.imwrite('deskewer-WIP/test04.png', d)
    
    # found_lines = np.zeros(all_lines.shape, dtype='uint8')
    # lines = cv2.HoughLinesP(all_lines, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=50)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(found_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.imwrite('deskewer-WIP/test04.png', found_lines)

    # tmp2 = np.zeros(d.shape, dtype='uint8')
    # lines = cv2.HoughLinesP(tmp, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=1000)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(tmp2, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.imwrite('deskewer-WIP/test08.png', tmp2)


if __name__ == '__main__':
    # input_image = 'ParachuteData/pdf-pages-as-images/T-11 LAT (SEPT 2022)-030.png'
    input_image = 'ParachuteData/pdf-pages-as-images/T-11 W911QY-19-D-0046 LOT 45_09282023-002.png'

    output_image = 'deskewer-WIP/output.png'
    process_image(input_image, output_image)
