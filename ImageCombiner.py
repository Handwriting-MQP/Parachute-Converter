#!/usr/local/bin/python3
import math
import random

import cv2 as cv
import numpy as np
import os


def getMNIST(digit):
    if not 0 <= digit <= 9:
        raise Exception("Digit must be between 0 - 9 inclusive")
    # Get the list of all files and directories
    # in the root directory
    path = f'MNIST/trainingSet/{digit}'
    dir_list = os.listdir(path)
    url = path + "/" + random.choice(dir_list)

    image = cv.imread(url)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = 255 * (gray > 200).astype(np.uint8)  # To invert the text to white
    # cv.imshow("Cropped", gray)  # Show it
    # cv.waitKey(0)
    coords = cv.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
    rect = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    return (~rect)


"""
# define a function for horizontally
# concatenating images of different
# heights
def hconcat_resize(img_list,
                   interpolation
                   =cv.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv.resize(img,
                                (int(img.shape[1] * h_min / img.shape[0]),
                                 h_min), interpolation
                                =interpolation)
                      for img in img_list]

    # return final image
    return cv.hconcat(im_list_resize)


# define a function for vertically
# concatenating images of different
# widths
def vconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv.resize(img,
                                (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv.vconcat(im_list_resize)



# define a function for concatenating
# images of different sizes in
# vertical and horizontal tiles
def concat_tile_resize(list_2d,
                       interpolation=cv.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h,
                                 interpolation=cv.INTER_CUBIC)
                  for list_h in list_2d]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv.INTER_CUBIC)

"""


def horizontal_concat(img_1, img_2):
    print(f'Before {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
    if img_1.shape[0] > img_2.shape[0]:
        scale = img_1.shape[0] / img_2.shape[0]
        width = int(img_2.shape[1] * scale)
        dim = (width, img_1.shape[0])
        img_2 = cv.resize(img_2, dim, interpolation=cv.INTER_AREA)
    elif img_2.shape[0] > img_1.shape[0]:
        scale = img_2.shape[0] / img_1.shape[0]
        width = int(img_1.shape[1] * scale)
        dim = (width, img_2.shape[0])
        img_1 = cv.resize(img_1, dim, interpolation=cv.INTER_AREA)
    img_1 = cv.copyMakeBorder(img_1, 0, 0, 0, 3, cv.BORDER_CONSTANT,
                              value=[255, 255, 255])
    return cv.hconcat([img_1, img_2])


def vertical_concat(img_1, img_2):
    print(f'Before {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
    if img_1.shape[0] > img_2.shape[0]:
        print(1)
        img_1, img_2 = vertical_resize(img_1, img_2)
    elif img_2.shape[0] > img_1.shape[0]:
        print(2)
        img_2, img_1 = vertical_resize(img_2, img_1)
    else:
        if img_1.shape[1] > img_2.shape[1]:
            print(1)
            img_2 = horizontal_resize(img_1, img_2)
        elif img_2.shape[1] > img_1.shape[1]:
            print(2)
            img_1 = horizontal_resize(img_2, img_1)

    print(f'After {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
    img_1 = cv.copyMakeBorder(img_1, 0, 1, 0, 0, cv.BORDER_CONSTANT,
                              value=[255, 255, 255])
    img_2 = cv.copyMakeBorder(img_2, 1, 0, 0, 0, cv.BORDER_CONSTANT,
                              value=[255, 255, 255])
    slash = fraction_slash()

    if slash.shape[1] > img_1.shape[1]:
        img_1 = horizontal_resize(slash, img_1)
        img_2 = horizontal_resize(slash, img_2)
    elif slash.shape[1] < img_1.shape[1]:
        slash = horizontal_resize(img_1, slash)

    return cv.vconcat([img_1, slash, img_2])


def vertical_resize(bigger, smaller):
    scale = bigger.shape[0] / smaller.shape[0]
    print(f'scale {scale}')
    width = int(smaller.shape[1] * scale)
    height = int(smaller.shape[0] * scale)
    dim = (width, height)
    new_smaller = cv.resize(smaller, dim, interpolation=cv.INTER_AREA)
    print(f'New smaller {new_smaller.shape[0]} {new_smaller.shape[1]}')

    if bigger.shape[1] > new_smaller.shape[1]:
        print(1)
        new_smaller = horizontal_resize(bigger, new_smaller)
    elif new_smaller.shape[1] > bigger.shape[1]:
        print(2)
        bigger = horizontal_resize(new_smaller, bigger)
    return bigger, new_smaller


def horizontal_resize(bigger, smaller):
    diff = (bigger.shape[1] - smaller.shape[1]) / 2
    new_smaller = cv.copyMakeBorder(smaller, 0, 0, math.ceil(diff), math.floor(diff), cv.BORDER_CONSTANT,
                                    value=[255, 255, 255])
    return new_smaller


def fraction_slash():
    line = getMNIST(1)
    image = cv.rotate(line, cv.ROTATE_90_CLOCKWISE)
    return image
    """
    test = 255 * (image > 200).astype(np.uint8)
    mask = cv.inRange(image, np.array([0,0,0]), np.array([100,100,100]))
    print(test)
    print(mask)
    print(f'Fraction {image.shape[0]} {image.shape[1]}\nMask {mask.shape[0]} {mask.shape[1]}')

    test = cv.bitwise_and(~image,~image, mask=mask)
    cv.imwrite("img_slash.png", image)
    """

if __name__ == '__main__':
    fraction_slash()
    img1 = getMNIST(random.randint(0, 9))

    img2 = getMNIST(random.randint(0, 9))

    img3 = getMNIST(random.randint(0, 9))

    img4 = getMNIST(random.randint(0, 9))

    fraction_type = 1  # random.getrandbits(1)
    print(f'bool{fraction_type}')
    if fraction_type:
        # function calling
        whole_number = horizontal_concat(img1, img2)
        whole_number = cv.copyMakeBorder(whole_number, 7, 7, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        fraction = vertical_concat(img3, img4)
        fraction = cv.copyMakeBorder(fraction, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        im_tile_resize = horizontal_concat(whole_number, fraction)
        # im_tile_resize = 255 * (im_tile_resize > 200).astype(np.uint8)  # To darken numbers
        cv.imwrite("img_inv_1.png", im_tile_resize)

    # TODO Implement whole number above fraction
    else:
        # function calling
        whole_number = horizontal_concat(img1, img2)
        whole_number = cv.copyMakeBorder(whole_number, 7, 7, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        fraction = horizontal_concat(img3, img4)
        fraction = cv.copyMakeBorder(fraction, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        im_tile_resize = vertical_concat(whole_number, fraction)

        cv.imwrite("img_inv_0.png", im_tile_resize)
