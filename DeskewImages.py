import os

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew

from tqdm import tqdm


# NOTE: this is not doing a real "deskew". instead this is rotating the image to try to line up edges.
def deskew_image(skewed_image_path, deskewed_image_path):
    image = io.imread(skewed_image_path)

    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True, cval=1)*255

    io.imsave(deskewed_image_path, rotated.astype(np.uint8))


if __name__ == '__main__':
    skewed_images_path = './ParachuteData/pdf-pages-as-images'
    deskewed_images_path = './ParachuteData/pdf-pages-as-images-deskewed'

    for skewed_image_filename in tqdm(os.listdir(skewed_images_path)):
        skewed_image_path = os.path.join(skewed_images_path, skewed_image_filename)
        deskewed_image_path = os.path.join(deskewed_images_path, skewed_image_filename)
        deskew_image(skewed_image_path, deskewed_image_path)
