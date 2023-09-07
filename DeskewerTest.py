import os

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew

input_folder = 'SampleDocument_PNG'
output_folder = 'DeskewedImages'


def convert_image(file_name):
    image = io.imread(input_folder + '/' + file_name)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True, cval=1) * 255
    io.imsave(output_folder + '/' + file_name, rotated.astype(np.uint8))


if __name__ == '__main__':
    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)

    skewed_files = os.listdir('SampleDocument_PNG')

    for file_name in skewed_files:
        convert_image(file_name)
