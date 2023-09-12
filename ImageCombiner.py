#!/usr/local/bin/python3
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
    gray = 1 * (gray > 200).astype(np.uint8)  # To invert the text to white

    coords = cv.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
    rect = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    return(~rect)


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


if __name__ == '__main__':

    img1 = getMNIST(random.randint(0,9))
    img2 = getMNIST(random.randint(0,9))
    img3 = getMNIST(random.randint(0,9))
    img4 = getMNIST(random.randint(0,9))

    # function calling
    whole_number = concat_tile_resize([[img1, img2]])
    fraction = concat_tile_resize([[img3], [img4]])
    im_tile_resize = concat_tile_resize([[whole_number, fraction]])
    cv.imwrite("img_inv.png", im_tile_resize)
