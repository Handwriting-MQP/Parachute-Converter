import os
import random

import numpy as np
import cv2

from trdg.generators import GeneratorFromStrings

from ImageCombiner11_28Prototype import generate_synthetic_image

#----------------------------------------------------------------------------------------------------

minimum_word_length = 3
maximum_word_length = 12

IAM_data_base_path = './IAM-data/'

# read in lines from words.txt
with open(os.path.join(IAM_data_base_path, 'words.txt'), 'r') as f:
    words_lines = f.readlines()

# process the lines into a list of simple dictionaries
IAM_words = []
for i, line in enumerate(words_lines):
    clean_line = line.strip()

    # skip comment lines in file
    if clean_line[0] == '#':
        continue

    # split the line into its various pieces
    line_elements = clean_line.split(' ')

    # don't deal with the few odd cases
    # NOTE: there are only 36 out of 115337 cases where this occurs
    if len(line_elements) != 9:
        continue

    word_id, segmented_correctly, graylevel, x, y, w, h, tag, word = line_elements

    # we don't need to keep incorrectly segmented words
    if segmented_correctly != 'ok':
        continue

    # we don't need to keep words that are too short or too long
    if len(word) < minimum_word_length or len(word) > maximum_word_length:
        continue

    IAM_words.append({'word_id': word_id,
                      'segmented_correctly': segmented_correctly,
                      'graylevel': int(graylevel),
                      'bounding_box': (int(x), int(y), int(w), int(h)),
                      'tag': tag,
                      'word': word})

#----------------------------------------------------------------------------------------------------

def generate_blank_data(height=30, width=80, central_brightness=200):
    """generate a blank image (with noise) of the given size and brightness"""

    shape = (height, width)

    base_image = np.ones(shape, dtype=np.uint8)*central_brightness
    noise = np.int8(np.random.normal(0, 5, shape))

    image = base_image + noise

    return image


def generate_printed_text(text, height=30):
    """generate a printed image of the given text and of the given height"""

    # NOTE: GeneratorFromStrings has a number of paramaters we can mess with if we would like
    generator = GeneratorFromStrings([text], size=height, fonts=['./fonts/Roboto-Regular.ttf'],
                                     skewing_angle=3, random_skew=True)
    image = next(generator)[0]
    gray_image = np.array(image)[:, :, 0]

    return gray_image


def generate_written_text():
    """generate a written image of a word from the IAM dataset"""

    def get_IAM_word_image(word_id):
        """given a valid IAM word id, calculate its local path and return the image as a numpy array"""

        # split the word ID into parts
        id1, id2, id3, id4 = word_id.split('-')

        # use parts of the ID to generate the path to the file
        image_path = os.path.join(IAM_data_base_path, 'words', id1, id1 + '-' + id2, word_id + '.png')

        # check if the image path exists (because imread() will fail silently)
        if os.path.exists(image_path) is False:
            raise Exception('IAM word path does not exist')
        
        image = cv2.imread(image_path)

        # convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # because the IAM dataset has segmented letters with a white background, we need replace the white background with
        # noise that has a central value of "background_value"
        background_value = np.median(image)*0.99 # this background value is rather arbitrary. it just looks good to me.
        image[image == 255] = np.clip(np.random.normal(background_value, size=image[image == 255].shape), 0, 255).astype(np.uint8)

        return image
    
    # select a word a random
    IAM_word = random.choice(IAM_words)

    # get the image for the given word
    word_image = get_IAM_word_image(IAM_word['word_id'])

    # return the word and the image
    return IAM_word['word'], word_image


def generate_written_fraction(text):
    # make call to image combiner to generate an appropraite image
    mixed_number_image, mixed_number_text = generate_synthetic_image()
    return mixed_number_image, mixed_number_text
