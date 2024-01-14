import os

import numpy as np
import cv2

from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew

from tqdm import tqdm


max_image_dimension = 4000


def resize_image(image):
    scale_factor = max_image_dimension/max(image.shape)
    new_shape = (int(scale_factor*image.shape[1]), int(scale_factor*image.shape[0]))
    return cv2.resize(image, new_shape)


# NOTE: this is not doing a real "deskew". instead this is rotating the image to try to get straight lines.
def deskew_image(image):
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True, cval=1)*255
    return rotated.astype(np.uint8)


def preprocess_image(image):
    image = resize_image(image)
    image = deskew_image(image)
    return image


def main():
    raw_images_path = './ParachuteData/pdf-pages-as-images'
    processed_images_path = './ParachuteData/pdf-pages-as-images-preprocessed'

    for image_filename in tqdm(os.listdir(raw_images_path)):
        raw_image_path = os.path.join(raw_images_path, image_filename)
        processed_image_path = os.path.join(processed_images_path, image_filename)

        image = cv2.imread(raw_image_path)
        image = preprocess_image(image)
        cv2.imwrite(processed_image_path, image)


if __name__ == '__main__':
    main()
