import json
import math
import random
import cv2 as cv
import numpy as np
import os

# Path to the MNIST dataset directory
MNIST_DATASET_PATH = 'MNIST/trainingSet'

# Directory to save synthetic data
SYNTHETIC_DATA_DIR = "SyntheticMixedNumberDataWithBoxLines"

# JSON structure for storing labels
labeled_data_list = []


def get_mnist_digit_image(digit):
    """
    Retrieves a random image of the specified MNIST digit.

    :param digit: Integer, the digit image to retrieve (0-9).
    :return: Image array of the digit.
    """
    if not 0 <= digit <= 9:
        raise ValueError("Digit must be between 0 and 9 inclusive")

    digit_path = f'{MNIST_DATASET_PATH}/{digit}'
    image_files = os.listdir(digit_path)
    image_url = os.path.join(digit_path, random.choice(image_files))

    image = cv.imread(image_url)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Invert the text to white
    binary_image = 255 * (gray_image > 200).astype(np.uint8)
    # Find all non-zero points (text)
    coords = cv.findNonZero(binary_image)
    # Find minimum spanning bounding box
    x, y, w, h = cv.boundingRect(coords)
    # Crop the image on the original image
    cropped_image = image[y:y + h, x:x + w]
    return ~cropped_image


def concatenate_horizontally(img_left, img_right, make_fraction=False):
    """
    Concatenates two images horizontally with an optional fraction bar.

    :param img_left: Image array, the left image to concatenate.
    :param img_right: Image array, the right image to concatenate.
    :param make_fraction: Boolean, whether to add a fraction bar between images.
    :return: Image array of the concatenated images.
    """
    # Scale images to the same height if needed
    img_left, img_right = scale_to_same_height(img_left, img_right)

    if make_fraction:
        slash_image = get_fraction_slash(vertical=True)
        img_left, img_right, slash_image = equalize_heights_for_fraction(img_left, img_right, slash_image)
        return cv.hconcat([img_left, slash_image, img_right])
    else:
        return cv.hconcat([img_left, img_right])


def concatenate_vertically(img_top, img_bottom, make_fraction=False):
    """
    Concatenates two images vertically with an optional fraction bar.

    :param img_top: Image array, the top image to concatenate.
    :param img_bottom: Image array, the bottom image to concatenate.
    :param make_fraction: Boolean, whether to add a fraction bar between images.
    :return: Image array of the concatenated images.
    """
    # Scale images to the same width if needed
    img_top, img_bottom = scale_to_same_width(img_top, img_bottom)

    if make_fraction:
        slash_image = get_fraction_slash(vertical=False)
        img_top, img_bottom, slash_image = equalize_widths_for_fraction(img_top, img_bottom, slash_image)
        return cv.vconcat([img_top, slash_image, img_bottom])
    else:
        return cv.vconcat([img_top, img_bottom])


def get_fraction_slash(vertical=True):
    """
    Creates an image of a fraction slash.

    :param vertical: Boolean, whether the slash is vertical or horizontal.
    :return: Image array of the slash.
    """
    line_image = get_mnist_digit_image(1)
    if vertical:
        return line_image
    else:
        return cv.rotate(line_image, cv.ROTATE_90_CLOCKWISE)


def equalize_heights_for_fraction(img_left, img_right, slash_image):
    """
    Equalizes the heights of two images and a fraction slash for proper concatenation.

    :param img_left: Image array, the left image.
    :param img_right: Image array, the right image.
    :param slash_image: Image array, the fraction slash image.
    :return: Tuple of image arrays with equalized heights.
    """
    max_height = max(img_left.shape[0], img_right.shape[0], slash_image.shape[0])
    img_left = add_border_to_match_height(img_left, max_height)
    img_right = add_border_to_match_height(img_right, max_height)
    slash_image = add_border_to_match_height(slash_image, max_height)
    return img_left, img_right, slash_image

def equalize_widths_for_fraction(img_top, img_bottom, slash_image):
    """
    Equalizes the widths of two images and a fraction slash for proper concatenation.

    :param img_top: Image array, the top image.
    :param img_bottom: Image array, the bottom image.
    :param slash_image: Image array, the fraction slash image.
    :return: Tuple of image arrays with equalized widths.
    """
    max_width = max(img_top.shape[1], img_bottom.shape[1], slash_image.shape[1])
    img_top = add_border_to_match_width(img_top, max_width)
    img_bottom = add_border_to_match_width(img_bottom, max_width)
    slash_image = add_border_to_match_width(slash_image, max_width)
    return img_top, img_bottom, slash_image


def scale_to_same_height(img1, img2):
    """
    Scales two images to the same height.

    :param img1: Image array, the first image.
    :param img2: Image array, the second image.
    :return: Tuple of image arrays with equalized heights.
    """
    if img1.shape[0] != img2.shape[0]:
        if img1.shape[0] > img2.shape[0]:
            img2 = resize_image_by_height(img2, img1.shape[0])
        else:
            img1 = resize_image_by_height(img1, img2.shape[0])
    return img1, img2


def scale_to_same_width(img1, img2):
    """
    Scales two images to the same width.

    :param img1: Image array, the first image.
    :param img2: Image array, the second image.
    :return: Tuple of image arrays with equalized widths.
    """
    if img1.shape[1] != img2.shape[1]:
        if img1.shape[1] > img2.shape[1]:
            img2 = resize_image_by_width(img2, img1.shape[1])
        else:
            img1 = resize_image_by_width(img1, img2.shape[1])
    return img1, img2


def add_border_to_match_height(image, target_height):
    """
    Adds a border to an image to match a given height.

    :param image: Image array to add border to.
    :param target_height: Integer, the target height for the image.
    :return: Image array with added borders.
    """
    top_border = (target_height - image.shape[0]) // 2
    bottom_border = target_height - image.shape[0] - top_border
    return cv.copyMakeBorder(image, top_border, bottom_border, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])

def add_border_to_match_width(image, target_width):
    """
    Adds a border to an image to match a given width.

    :param image: Image array to add border to.
    :param target_width: Integer, the target width for the image.
    :return: Image array with added borders.
    """
    left_border = (target_width - image.shape[1]) // 2
    right_border = target_width - image.shape[1] - left_border
    return cv.copyMakeBorder(image, 0, 0, left_border, right_border, cv.BORDER_CONSTANT, value=[255, 255, 255])

def resize_image_by_height(image, target_height):
    """
    Resizes an image to a target height while keeping the aspect ratio.

    :param image: Image array to resize.
    :param target_height: Integer, the target height for the image.
    :return: Image array resized to the target height.
    """
    scaling_factor = target_height / image.shape[0]
    new_width = int(image.shape[1] * scaling_factor)
    return cv.resize(image, (new_width, target_height), interpolation=cv.INTER_AREA)


def resize_image_by_width(image, target_width):
    """
    Resizes an image to a target width while keeping the aspect ratio.

    :param image: Image array to resize.
    :param target_width: Integer, the target width for the image.
    :return: Image array resized to the target width.
    """
    scaling_factor = target_width / image.shape[1]
    new_height = int(image.shape[0] * scaling_factor)
    return cv.resize(image, (target_width, new_height), interpolation=cv.INTER_AREA)


def create_synthetic_image(file_name, add_box_lines=0):
    """
    Creates a synthetic image of a mixed number with an optional box line.

    :param file_name: String, the name of the file to save the image as.
    :param add_box_lines: Boolean, whether to add box lines to the image.
    :return: None, but saves the image to a file and prints the number.
    """
    # Generate random whole number and fraction parts
    whole_number, whole_number_text = generate_whole_number()
    fraction, fraction_text = generate_fraction()

    # Decide the layout type of the mixed number
    layout_type = random.randint(0, 3)

    # Generate image based on layout type

    # Layout Type 0: Vertical arrangement of whole number and fraction
    if layout_type == 0:
        # Add whitespace to the whole number for vertical alignment
        whole_number = add_whitespace_to_image(whole_number, vertical=True)
        # Create a fraction by vertically concatenating the numerator and denominator
        fraction = concatenate_vertically(fraction[0], fraction[1], make_fraction=True)
        # Combine the whole number and fraction images vertically
        mixed_number_image = concatenate_vertically(whole_number, fraction, make_fraction=False)
        # Form the textual representation of the mixed number
        mixed_number_text = f'{whole_number_text} {fraction_text}'

    # Layout Type 1: Horizontal arrangement of whole number and fraction
    elif layout_type == 1:
        # Add whitespace to the whole number for horizontal alignment
        whole_number = add_whitespace_to_image(whole_number, vertical=False)
        # Create a fraction by vertically concatenating the numerator and denominator
        fraction = concatenate_vertically(fraction[0], fraction[1], make_fraction=True)
        # Combine the whole number and fraction images horizontally
        mixed_number_image = concatenate_horizontally(whole_number, fraction, make_fraction=False)
        # Form the textual representation of the mixed number
        mixed_number_text = f'{whole_number_text} {fraction_text}'

    # Layout Type 2: Whole number above a horizontally arranged fraction
    elif layout_type == 2:
        # Add whitespace to the whole number for vertical alignment
        whole_number = add_whitespace_to_image(whole_number, vertical=True)
        # Create a fraction by horizontally concatenating the numerator and denominator
        fraction = concatenate_horizontally(fraction[0], fraction[1], make_fraction=True)
        # Combine the whole number and fraction images vertically
        mixed_number_image = concatenate_vertically(whole_number, fraction, make_fraction=False)
        # Form the textual representation of the mixed number
        mixed_number_text = f'{whole_number_text} {fraction_text}'

    # Layout Type 3: Image consists only of the whole number
    elif layout_type == 3:
        # The mixed number image is simply the whole number image
        mixed_number_image = whole_number
        # The textual representation is just the whole number text
        mixed_number_text = whole_number_text

    # # Default case: Layout Type not recognized
    # else:
    #     # Create a fraction by vertically concatenating the numerator and denominator
    #     fraction = concatenate_vertically(fraction[0], fraction[1], make_fraction=True)
    #     # The mixed number image is just the fraction image
    #     mixed_number_image = fraction
    #     # The textual representation is just the fraction text
    #     mixed_number_text = fraction_text

    # Optionally add box lines
    for i in range(add_box_lines):
        add_box_line_to_image(mixed_number_image)

    # Save the image and append the label to the data list
    cv.imwrite(os.path.join(SYNTHETIC_DATA_DIR, 'images', file_name), mixed_number_image)
    print("file_name: " + file_name + ", "
          + "text: " + mixed_number_text + ", "
          + "layout_type: " + str(layout_type))
    labeled_data_list.append({'filename': file_name, 'text': mixed_number_text, 'layout_type': str(layout_type)})


def add_whitespace_to_image(image, vertical=True, amount=None):
    """
    Adds whitespace to an image either vertically or horizontally.

    :param image: Image array to add whitespace to.
    :param vertical: Boolean, whether to add vertical whitespace.
    :param amount: Integer, the amount of whitespace to add. If None, a random amount is used.
    :return: Image array with added whitespace.
    """
    if amount is None:
        amount = random.randint(1, 40)
    if vertical:
        return cv.copyMakeBorder(image, 7, 7, 1, amount, cv.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        return cv.copyMakeBorder(image, 7, amount, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])


def add_box_line_to_image(image):
    """
    Adds a random box line to an image on one of its sides.

    :param image: Image array to add the box line to.
    :return: None, but modifies the image in place.
    """
    side = random.randint(1, 4)
    color = (0, 0, 0)
    thickness = 1
    width, height = image.shape[1], image.shape[0]

    if side == 1:  # top
        cv.line(image, (0, 5), (width, 5), color, thickness)
    elif side == 2:  # left
        cv.line(image, (5, 0), (5, height), color, thickness)
    elif side == 3:  # bottom
        cv.line(image, (0, height - 5), (width, height - 5), color, thickness)
    elif side == 4:  # right
        cv.line(image, (width - 5, 0), (width - 5, height), color, thickness)


def generate_whole_number():
    """
    Generates a random whole number image and its corresponding text.

    :return: Tuple (Image array of the whole number, text representation of the number).
    """
    digits_count = random.randint(1, 3)
    whole_number_images = []
    whole_number_text_parts = []

    for _ in range(digits_count):
        digit = random.randint(0, 9)
        whole_number_images.append(get_mnist_digit_image(digit))
        whole_number_text_parts.append(str(digit))

    whole_number_image = concatenate_images_list(whole_number_images)
    whole_number_text = ''.join(whole_number_text_parts)

    return whole_number_image, whole_number_text


def generate_fraction():
    """
    Generates a random fraction image and its corresponding text.

    :return: Tuple (Tuple of numerator and denominator images, text representation of the fraction).
    """
    numerator, numerator_image = generate_fraction_part(odd_only=True)
    denominator, denominator_image = generate_fraction_part(odd_only=False, power_of_two=True)
    return (numerator_image, denominator_image), f'{numerator}/{denominator}'


def generate_fraction_part(odd_only=False, power_of_two=False):
    """
    Generates a random fraction part image and its corresponding number.

    :param odd_only: Boolean, whether to generate only odd numbers.
    :param power_of_two: Boolean, whether to generate a power of two number.
    :return: Tuple (Integer representation of the fraction part, Image array of the fraction part).
    """
    if power_of_two:
        number = random.choice([2, 4, 8, 16])
    elif odd_only:
        number = random.choice([1, 3, 5, 7, 9])
    else:
        number = random.randint(1, 9)

    if number < 10:
        return number, get_mnist_digit_image(number)
    else:
        digit_images = [get_mnist_digit_image(int(digit)) for digit in str(number)]
        return number, concatenate_images_list(digit_images)


def concatenate_images_list(images_list):
    """
    Concatenates a list of image arrays horizontally.

    :param images_list: List of image arrays to concatenate.
    :return: Image array of the concatenated images.
    """
    if not images_list:
        raise ValueError("images_list cannot be empty")

    concatenated_image = images_list[0]
    for image in images_list[1:]:
        concatenated_image = concatenate_horizontally(concatenated_image, image, make_fraction=False)
    return concatenated_image


if __name__ == '__main__':
    # Ensure the synthetic data directory exists
    if not os.path.exists(SYNTHETIC_DATA_DIR):
        os.makedirs(SYNTHETIC_DATA_DIR)
    if not os.path.exists(os.path.join(SYNTHETIC_DATA_DIR, 'images')):
        os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, 'images'))

    # Number of synthetic images to generate
    number_of_images = 1000

    for i in range(number_of_images):
        file_name = f'image_{i}.png'
        # Randomly decide to add box lines
        add_box_lines = random.choice([True, False])
        create_synthetic_image(file_name, add_box_lines=random.randint(0,4))

    # Saving the labeled data list in a JSON file
    with open(os.path.join(SYNTHETIC_DATA_DIR, 'labels.json'), 'w') as json_file:
        json.dump(labeled_data_list, json_file, indent=4)

    print(f"Generated {number_of_images} synthetic mixed number images.")
