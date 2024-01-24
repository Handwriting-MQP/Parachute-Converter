import os
import random
import json

from tqdm import tqdm
import cv2

# Path to the MNIST dataset directory
MNIST_DATASET_PATH = '../MNIST/trainingSet'


def get_random_cropped_mnist_digit_image(digit):
    """
    Retrieves a random image of the specified MNIST digit cropped to the minimum bounding box and color inverted.

    :param digit: Integer, the digit image to retrieve (0-9).
    :return: Image array of the digit.
    """
    if not 0 <= digit <= 9:
        raise ValueError("Digit must be between 0 and 9 inclusive")

    # get a random image of the specified digit
    digit_path = f'{MNIST_DATASET_PATH}/{digit}'
    image_files = os.listdir(digit_path)
    image_path = os.path.join(digit_path, random.choice(image_files))

    # read in the random digit image
    image = cv2.imread(image_path)

    # convert to grayscale and threshold to binary
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # find the minimum bounding box of non-zero pixels
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(binary_image))

    # crop the original image using the bounding box on the binary image
    cropped_image = image[y:y + h, x:x + w]

    # return the cropped image with inverted the colors (so it has a white background)
    return -cropped_image + 255


def get_mnist_fraction_slash():
    digit_dir_path = os.path.join(f'{MNIST_DATASET_PATH}', '1')
    ones_allowlist = [
        'img_2.jpg',
        'img_12.jpg',
        'img_15.jpg',
        'img_35.jpg',
        'img_37.jpg',
        'img_41.jpg',
        'img_52.jpg',
        'img_59.jpg',
        'img_68.jpg',
        'img_77.jpg',
        'img_79.jpg',
        'img_96.jpg'
    ]

    image_path = os.path.join(digit_dir_path, random.choice(ones_allowlist))

    # read in the random digit image
    image = cv2.imread(image_path)

    # convert to grayscale and threshold to binary
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # find the minimum bounding box of non-zero pixels
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(binary_image))

    # crop the original image using the bounding box on the binary image
    cropped_image = image[y:y + h, x:x + w]

    # return the cropped image with inverted the colors (so it has a white background)
    return -cropped_image + 255


# ----------------------------------------------------------------------------------------------------

def concatenate_images_horizontally(images):
    """Concatenates a list of images horizontally by padding them all to be the same height."""

    def pad_image_to_match_height(image, target_height):
        if target_height < image.shape[0]:
            raise ValueError("target_height must be greater than or equal to the image height")

        # calculate the amount of border to add to the top and bottom
        top_border = (target_height - image.shape[0]) // 2
        bottom_border = target_height - image.shape[0] - top_border

        # add the border to the image
        return cv2.copyMakeBorder(image, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # pick the max height of all images
    max_height = max([image.shape[0] for image in images])

    # add borders to all images to make them the same height
    padded_images = [pad_image_to_match_height(image, max_height) for image in images]

    return cv2.hconcat(padded_images)


def concatenate_images_vertically(images):
    """Concatenates a list of images horizontally by padding them all to be the same height."""

    def pad_image_to_match_width(image, target_width):
        if target_width < image.shape[1]:
            raise ValueError("target_width must be greater than or equal to the image width")

        # calculate the amount of border to add to the left and right
        left_border = (target_width - image.shape[1]) // 2
        right_border = target_width - image.shape[1] - left_border

        # add the border to the image
        return cv2.copyMakeBorder(image, 0, 0, left_border, right_border, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # pick the max width of all images
    max_width = max([image.shape[1] for image in images])

    # add borders to all images to make them the same height
    padded_images = [pad_image_to_match_width(image, max_width) for image in images]

    return cv2.vconcat(padded_images)


# ----------------------------------------------------------------------------------------------------

def generate_image_from_whole_number(whole_number):
    whole_number_str = str(whole_number)

    digit_images = []
    for i in range(len(whole_number_str)):
        # get a random image of the digit
        digit = int(whole_number_str[i])
        digit_image = get_random_cropped_mnist_digit_image(digit)

        # add a random amount of whitespace to the right of the digit if it is not the last digit
        if i != len(whole_number_str) - 1:
            right_border = random.randint(0, 7)  # NOTE: the max of 7 pixels of whitespace is somewhat arbitrary
            digit_image = cv2.copyMakeBorder(digit_image, 0, 0, 0, right_border, cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])

        digit_images.append(digit_image)

    return concatenate_images_horizontally(digit_images)


def generate_random_whole_number_image():
    # randomly decide how many digits the natural number will have
    num_digits = random.randint(1, 3)

    digits = []
    for i in range(num_digits):
        # first digit cannot be 0
        if i == 0:
            digit = random.randint(1, 9)
        else:
            digit = random.randint(0, 9)
        # add the digit to the list
        digits.append(digit)

    # convert the list of digits to a string and then to an number
    number = int(''.join([str(digit) for digit in digits]))

    # return an image of the number and the number
    return generate_image_from_whole_number(number), number


def generate_random_fraction_image(vertical_orientation):
    """
    Generates a random fraction image and its corresponding text.

    :return: Tuple (numerator image, denominator images, text representation of the fraction).
    """
    # pick a random denominator from 2, 4, 8, 16
    denominator = random.choice([2, 4, 8, 16])
    denominator_image = generate_image_from_whole_number(denominator)

    # pick a random odd number for the numerator (to avoid fraction simplification)
    numerator = random.randrange(1, denominator, 2)
    numerator_image = generate_image_from_whole_number(numerator)

    # pick a random mnsit "1" to use as the slash and add some whitespace on the sides
    slash_image = get_mnist_fraction_slash()
    slash_image = cv2.copyMakeBorder(slash_image, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # concatenate the images horizontally or vertically to form the fraction image
    if vertical_orientation is True:
        # rotated the slash image 90 degrees clockwise
        slash_image = cv2.rotate(slash_image, cv2.ROTATE_90_CLOCKWISE)
        fraction_image = concatenate_images_vertically([numerator_image, slash_image, denominator_image])
    else:
        fraction_image = concatenate_images_horizontally([numerator_image, slash_image, denominator_image])

    return fraction_image, f'{numerator}/{denominator}'


# ----------------------------------------------------------------------------------------------------

def generate_synthetic_image(num_cell_lines=0):
    """
    Creates a synthetic image of a mixed number with optional box lines.

    :param num_box_lines: Integer, the number of box lines to add to the image.
    :return: None, but saves the image to a file and prints the number.
    """
    # generate random whole number and fraction images
    whole_number, whole_number_text = generate_random_whole_number_image()

    # Decide the layout type of the mixed number
    layout_type = random.randint(1, 7)

    # Layout Type 1: whole number above a horizontally arranged fraction
    # Layout Type 2: whole number above a vertically arranged fraction
    if layout_type in (1, 2):
        # Add whitespace to the whole number for vertical alignment
        bottom_padding = random.randint(0, 10)  # NOTE: these values are somewhat arbitrary
        whole_number = cv2.copyMakeBorder(whole_number, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
        # generate a random fraction image and its corresponding text in a horizontal or vertical orientation
        if layout_type == 1:
            fraction_image, fraction_text = generate_random_fraction_image(vertical_orientation=False)
        elif layout_type == 2:
            fraction_image, fraction_text = generate_random_fraction_image(vertical_orientation=True)
        # Combine the whole number and fraction images vertically
        mixed_number_image = concatenate_images_vertically([whole_number, fraction_image])
        # Form the textual representation of the mixed number
        mixed_number_text = f'{whole_number_text} {fraction_text}'

    # Layout Type 3: whole number next to a horizontally arranged fraction
    # layout type 4: whole number next to a vertically arranged fraction
    elif layout_type in (3, 4):
        # Add whitespace to the whole number for vertical alignment
        right_padding = random.randint(10, 20)  # NOTE: these values are somewhat arbitrary
        whole_number = cv2.copyMakeBorder(whole_number, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
        # generate a random fraction image and its corresponding text in a horizontal or vertical orientation
        if layout_type == 3:
            fraction_image, fraction_text = generate_random_fraction_image(vertical_orientation=False)
        elif layout_type == 4:
            fraction_image, fraction_text = generate_random_fraction_image(vertical_orientation=True)
        # Combine the whole number and fraction images horizontally
        mixed_number_image = concatenate_images_horizontally([whole_number, fraction_image])
        # Form the textual representation of the mixed number
        mixed_number_text = f'{whole_number_text} {fraction_text}'

    # Layout Type 5: Image consists only of the whole number
    elif layout_type == 5:
        mixed_number_image = whole_number
        mixed_number_text = whole_number_text

    # Layout Type 6: Image consists of just a horizontally arranged fraction
    elif layout_type == 6:
        mixed_number_image, mixed_number_text = generate_random_fraction_image(vertical_orientation=False)

    # Layout Type 7: Image consists of just a vertically arranged fraction
    elif layout_type == 7:
        mixed_number_image, mixed_number_text = generate_random_fraction_image(vertical_orientation=True)

    # add cell lines to the image
    possible_sides = [1, 2, 3, 4]
    for _ in range(num_cell_lines):
        # pick a side and remove it from the list of possible sides
        side = random.choice(possible_sides)
        possible_sides.remove(side)

        color = (0, 0, 0)
        thickness = 1
        width, height = mixed_number_image.shape[1], mixed_number_image.shape[0]

        # pick a random distance to place a line from the edge of the image
        distance_from_edge = random.randint(0, 5)

        if side == 1:  # top
            cv2.line(mixed_number_image, (0, distance_from_edge), (width, distance_from_edge), color, thickness)
        elif side == 2:  # left
            cv2.line(mixed_number_image, (distance_from_edge, 0), (distance_from_edge, height), color, thickness)
        elif side == 3:  # bottom
            cv2.line(mixed_number_image, (0, height - distance_from_edge), (width, height - distance_from_edge), color,
                     thickness)
        elif side == 4:  # right
            cv2.line(mixed_number_image, (width - distance_from_edge, 0), (width - distance_from_edge, height), color,
                     thickness)

    return mixed_number_image, mixed_number_text, layout_type


def main():
    # Directory to save synthetic data
    synthetic_data_directory = "SyntheticMixedNumberData"

    # Number of synthetic images to generate
    number_of_images = 1000

    # JSON structure for storing labels
    synthetic_data_list = []

    # Ensure the synthetic data directory exists
    if not os.path.exists(synthetic_data_directory):
        os.makedirs(synthetic_data_directory)
    if not os.path.exists(os.path.join(synthetic_data_directory, 'images')):
        os.makedirs(os.path.join(synthetic_data_directory, 'images'))

    for i in tqdm(range(number_of_images)):
        # generate a synthetic image
        mixed_number_image, mixed_number_text, layout_type = generate_synthetic_image(
            num_cell_lines=random.randint(0, 4))

        # save the image
        file_name = f'image_{i}.png'
        file_path = os.path.join(synthetic_data_directory, 'images', file_name)
        cv2.imwrite(file_path, mixed_number_image)

        # add the image to the list of labeled data
        synthetic_data_list.append({'filename': file_name, 'text': mixed_number_text, 'layout_type': layout_type})

    # save the labeled data list in a JSON file
    with open(os.path.join(synthetic_data_directory, 'labels.json'), 'w') as f:
        json.dump(synthetic_data_list, f, indent=4)

    print(f"Generated {number_of_images} synthetic mixed number images.")


if __name__ == '__main__':
    # image, text, _ = generate_synthetic_image()
    # cv2.imwrite('test1.png', image)
    # print(text)

    main()
