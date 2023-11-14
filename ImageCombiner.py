#!/usr/local/bin/python3
import json
import math
import random

import cv2 as cv
import numpy as np
import os

from sympy import fraction

# TODO: add tqdm progress bar (report fraction type on bar?)
# TODO: make sure you have all proper directories for synthetic data so "cv.imwrite()" doesn't fail silently


# Gets an MNIST digit in the range 0 to 9.
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


# Enlarges the smaller image to the big one to get the same height, by adding whitespace.
# Then, cv2.hconcat() them together (which requires exactly same height).
def horizontal_concat(img_1, img_2, fraction):
    # print(f'Before {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
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
    # img_1 = cv.copyMakeBorder(img_1, 0, 0, 0, 3, cv.BORDER_CONSTANT,
    #                           value=[255, 255, 255])

    if fraction:
        slash = vert_fraction_slash()
        scale = 1.5
        # print(f'scale {scale}')
        width = int(slash.shape[1] * scale)
        height = int(slash.shape[0] * scale)
        dim = (width, height)
        slash = cv.resize(slash, dim, interpolation=cv.INTER_AREA)

        diff1 = (slash.shape[0] - img_1.shape[0]) / 2
        img_1 = cv.copyMakeBorder(img_1, math.ceil(diff1), math.floor(diff1), 0, 0, cv.BORDER_CONSTANT,
                                  value=[255, 255, 255])
        diff2 = (slash.shape[0] - img_2.shape[0]) / 2
        img_2 = cv.copyMakeBorder(img_2, math.ceil(diff2), math.floor(diff2), 0, 0, cv.BORDER_CONSTANT,
                                  value=[255, 255, 255])

        # if slash.shape[0] > img_1.shape[0]:
        #     img_1 = vertical_resize(img_1, slash)[1]
        #     img_2 = vertical_resize(img_2, slash)[1]
        # elif slash.shape[0] < img_1.shape[0]:
        #     slash = vertical_resize(img_1, slash)[1]

        # vertical_resize(bigger, smaller):
        # img_1 = cv.copyMakeBorder(img_1, 0, 1, 0, 0, cv.BORDER_CONSTANT,
        #                           value=[255, 255, 255])
        # img_2 = cv.copyMakeBorder(img_2, 1, 0, 0, 0, cv.BORDER_CONSTANT,
        #                           value=[255, 255, 255])

        return cv.hconcat([img_1, slash, img_2])
    else:
        return cv.hconcat([img_1, img_2])


# Vertical analog to custom horizontal_concat() function. See horizontal_concat()
# description.
def vertical_concat(img_1, img_2, fraction):
    # print(f'Before {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
    if img_1.shape[0] > img_2.shape[0]:
        # print(1)
        img_1, img_2 = vertical_resize(img_1, img_2)
    elif img_2.shape[0] > img_1.shape[0]:
        # print(2)
        img_2, img_1 = vertical_resize(img_2, img_1)
    else:
        if img_1.shape[1] > img_2.shape[1]:
            # print(1)
            img_2 = horizontal_resize(img_1, img_2)
        elif img_2.shape[1] > img_1.shape[1]:
            # print(2)
            img_1 = horizontal_resize(img_2, img_1)

    # print(f'After {img_1.shape[0]} {img_1.shape[1]}\n{img_2.shape[0]} {img_2.shape[1]}')
    img_1 = cv.copyMakeBorder(img_1, 0, 1, 0, 0, cv.BORDER_CONSTANT,
                              value=[255, 255, 255])
    img_2 = cv.copyMakeBorder(img_2, 1, 0, 0, 0, cv.BORDER_CONSTANT,
                              value=[255, 255, 255])
    if fraction:
        slash = horiz_fraction_slash()

        if slash.shape[1] > img_1.shape[1]:
            img_1 = horizontal_resize(slash, img_1)
            img_2 = horizontal_resize(slash, img_2)
        elif slash.shape[1] < img_1.shape[1]:
            slash = horizontal_resize(img_1, slash)

        return cv.vconcat([img_1, slash, img_2])
    else:
        return cv.vconcat([img_1, img_2])


def vertical_resize(bigger, smaller):
    scale = bigger.shape[0] / smaller.shape[0]
    # print(f'scale {scale}')
    width = int(smaller.shape[1] * scale)
    height = int(smaller.shape[0] * scale)
    dim = (width, height)
    new_smaller = cv.resize(smaller, dim, interpolation=cv.INTER_AREA)
    # print(f'New smaller {new_smaller.shape[0]} {new_smaller.shape[1]}')

    if bigger.shape[1] > new_smaller.shape[1]:
        # print(1)
        new_smaller = horizontal_resize(bigger, new_smaller)
    elif new_smaller.shape[1] > bigger.shape[1]:
        # print(2)
        bigger = horizontal_resize(new_smaller, bigger)
    return bigger, new_smaller


def horizontal_resize(bigger, smaller):
    diff = (bigger.shape[1] - smaller.shape[1]) / 2
    new_smaller = cv.copyMakeBorder(smaller, 0, 0, math.ceil(diff), math.floor(diff), cv.BORDER_CONSTANT,
                                    value=[255, 255, 255])
    return new_smaller


def horiz_fraction_slash():
    line = getMNIST(1)
    image = cv.rotate(line, cv.ROTATE_90_CLOCKWISE)
    return image


def vert_fraction_slash():
    line = getMNIST(1)
    # image = cv.rotate(line, cv.ROTATE_90_CLOCKWISE)
    return line
    """
    test = 255 * (image > 200).astype(np.uint8)
    mask = cv.inRange(image, np.array([0,0,0]), np.array([100,100,100]))
    print(test)
    print(mask)
    print(f'Fraction {image.shape[0]} {image.shape[1]}\nMask {mask.shape[0]} {mask.shape[1]}')

    test = cv.bitwise_and(~image,~image, mask=mask)
    cv.imwrite("img_slash.png", image)
    """


def create_image(file_name):
    image_file = f'{synthetic_directory}/images/{file_name}'

    whole_digit1 = random.randint(1, 9)
    whole_image1 = getMNIST(whole_digit1)
    # print(f'whole number a: {number1}')

    whole_digit2 = random.randint(0, 9)
    whole_image2 = getMNIST(whole_digit2)

    whole_digit3 = random.randint(0, 9)
    whole_image3 = getMNIST(whole_digit3)
    # print(f'whole number b: {number2}')

    #Determine how many digits make up the fraction
    fraction_digit_count = random.randint(2, 4)
    if fraction_digit_count == 2:
        #The denominator is a power of 2, so it is either 2, 4, or 8
        fraction_digit2 = np.random.choice([2, 4, 8])
        fraction_image2 = getMNIST(fraction_digit2)
        #Numerator is always odd, so pick a random odd number between 1 and the denominator
        fraction_digit1 = random.randrange(1, fraction_digit2, 2)
        fraction_image1 = getMNIST(fraction_digit1)
    # print(f'frac a: {number3}')
    # print(f'frac b: {number4}')
    elif fraction_digit_count == 3:
        fraction_digit1 = np.random.choice([1, 3, 5, 7, 9])
        fraction_image1 = getMNIST(fraction_digit1)
        #Two digit denominator is always 16
        fraction_denom1 = 1
        fraction_denom_im1 = getMNIST(fraction_denom1)

        fraction_denom2 = 6
        fraction_denom_im2 = getMNIST(fraction_denom2)
        fraction_digit2 = f'{fraction_denom1}{fraction_denom2}'
        fraction_image2 = horizontal_concat(fraction_denom_im1, fraction_denom_im2, False)
    else:
        fraction_num1 = 1
        fraction_num_im1 = getMNIST(fraction_num1)

        fraction_num2 = np.random.choice([1, 3, 5])
        fraction_num_im2 = getMNIST(fraction_num2)
        fraction_image1 = horizontal_concat(fraction_num_im1, fraction_num_im2, False)
        fraction_digit1 = 10 + fraction_num2

        fraction_denom1 = 1
        fraction_denom_im1 = getMNIST(fraction_denom1)

        fraction_denom2 = 6
        fraction_denom_im2 = getMNIST(fraction_denom2)
        fraction_digit2 = 16
        fraction_image2 = horizontal_concat(fraction_denom_im1, fraction_denom_im2, False)

    # this randint() determines the case of the mixed number to be generated
    # each case is explained before its elif block
    # ******randint() inclusive at both ends******
    fraction_type = random.randint(0, 3)
    #print(f'fraction type: {fraction_type}')

    whole_digit_count = random.randint(1, 3)

    if whole_digit_count == 1:
        text_num = str(whole_digit1)
        whole_number = whole_image1
    elif whole_digit_count == 2:
        text_num = f'{whole_digit1}{whole_digit2}'
        whole_number = horizontal_concat(whole_image1, whole_image2, False)
    else:
        text_num = f'{whole_digit1}{whole_digit2}{whole_digit3}'
        number = horizontal_concat(whole_image1, whole_image2, False)
        whole_number = horizontal_concat(number, whole_image3, False)

    # In this case, the whole number is above the fraction.
    # The fraction is vertical.
    if fraction_type == 0:

        # this line shifts the whole number left by an amount in [1, 40].
        whole_number = cv.copyMakeBorder(whole_number, 7, 7, 1, random.randint(1, 40), cv.BORDER_CONSTANT,
                                         value=[255, 255, 255])

        fraction = vertical_concat(fraction_image1, fraction_image2, True)
        fraction = cv.copyMakeBorder(fraction, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        im_tile_resize = vertical_concat(whole_number, fraction, False)

        cv.imwrite(image_file, im_tile_resize)
        text = f'{text_num} {fraction_digit1}/{fraction_digit2}'

    # Whole number image is attached to the fraction at the left.
    # The fraction is vertical (numerator above denominator).
    elif fraction_type == 1:

        whole_number = cv.copyMakeBorder(whole_number, 7, random.randint(7, 25), 1, 1, cv.BORDER_CONSTANT,
                                         value=[255, 255, 255])
        fraction = vertical_concat(fraction_image1, fraction_image2, True)
        fraction = cv.copyMakeBorder(fraction, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        im_tile_resize = horizontal_concat(whole_number, fraction, False)
        # im_tile_resize = 255 * (im_tile_resize > 200).astype(np.uint8)  # To darken numbers
        cv.imwrite(image_file, im_tile_resize)
        text = f'{text_num} {fraction_digit1}/{fraction_digit2}'

    # The whole number has whitespace to the right and is above the fraction.
    # The fraction is horizontal.
    elif fraction_type == 2:

        whole_number = cv.copyMakeBorder(whole_number, 7, 7, 1, random.randint(1, 40), cv.BORDER_CONSTANT,
                                         value=[255, 255, 255])
        fraction = horizontal_concat(fraction_image1, fraction_image2, True)
        fraction = cv.copyMakeBorder(fraction, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        im_tile_resize = vertical_concat(whole_number, fraction, False)

        cv.imwrite(image_file, im_tile_resize)
        text = f'{text_num} {fraction_digit1}/{fraction_digit2}'
    else:
        cv.imwrite(image_file, whole_number)
        text = text_num
    print(text)
    labeled_json.append({'filename': file_name, 'text': text})


labeled_json = []
synthetic_directory = "SyntheticMixedNumberData"
if __name__ == '__main__':
    for i in range(1000):
        create_image(f'{i}.jpg')
    labeled_data = json.dumps(labeled_json, indent=4)

    # Writing to sample.json
    with open(f'{synthetic_directory}/labels.json', "w") as outfile:
        outfile.write(labeled_data)
