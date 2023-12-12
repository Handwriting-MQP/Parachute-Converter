import os
import random
import json

import numpy as np
import cv2

from trdg.generators import GeneratorFromStrings

# import generate_synthetic_image from parent directory (a bit of trickery is needed to do this)
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from GenerateSyntheticMixedNumberData import generate_synthetic_image

#----------------------------------------------------------------------------------------------------

minimum_word_length = 3
maximum_word_length = 12

IAM_data_base_path = 'CellClassifier/IAM-data/'

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

# read in lines from most_common_words.txt
most_common_words = []
with open('./CellClassifier/1000-most-common-words.txt', 'r') as f:
    word_lines = f.readlines()

# process the lines into a list of words
for line in word_lines:
    word = line.strip()

    # we don't need to keep words that are too short or too long
    if len(word) < minimum_word_length or len(word) > maximum_word_length:
        continue
    
    most_common_words.append(word)

#----------------------------------------------------------------------------------------------------

def generate_blank_data(height=30, width=80, central_brightness=200, noise_standard_deviation=5):
    """generate a blank image (with noise) of the given size and brightness"""

    shape = (height, width)

    base_image = np.ones(shape)*central_brightness
    noise = np.random.normal(0, noise_standard_deviation, shape)

    image = np.clip(base_image + noise, 0, 255).astype(np.uint8)

    return image


def generate_printed_text(text, height=30):
    """generate a printed image of the given text and of the given height"""

    # NOTE: GeneratorFromStrings has a number of paramaters we can mess with if we would like
    generator = GeneratorFromStrings([text], size=height, fonts=['./CellClassifier/fonts/Roboto-Regular.ttf'],
                                     skewing_angle=3, random_skew=True)
    # pull out one channel of the image (they're all the same value)
    image = next(generator)[0]
    gray_image = np.array(image, dtype=np.uint8)[:, :, 0]
    
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
    return word_image, IAM_word['word']


def generate_written_fraction():
    # make call to image combiner to generate an appropraite image
    mixed_number_image, mixed_number_text, layout_type = generate_synthetic_image()
    return mixed_number_image, mixed_number_text


def main():
    image_output_folder = './CellClassifier/SyntheticCellData/images'

    # create output folder if it doesn't exist
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
    
    # number of synthetic images to generate
    number_of_images = 100
    
    # generate a bunch of synthetic images of each type
    data_list = []
    for i in range(number_of_images):
        # generate a blank cell image
        image = generate_blank_data(height=random.randint(30, 50),
                                    width=random.randint(80, 100),
                                    central_brightness=random.randint(200, 250))
        fpath = os.path.join(image_output_folder, str(i) + '_blank.png')
        cv2.imwrite(fpath, image)
        data_list.append({'path': fpath, 'label': 'blank'})

        # generate a printed cell image
        image = generate_printed_text(random.choice(most_common_words), height=random.randint(30, 50))
        fpath = os.path.join(image_output_folder, str(i) + '_printed.png')
        cv2.imwrite(fpath, image)
        data_list.append({'path': fpath, 'label': 'printed'})

        # generate a written cell image
        image, word = generate_written_text()
        fpath = os.path.join(image_output_folder, str(i) + '_written.png')
        cv2.imwrite(fpath, image)
        data_list.append({'path': fpath, 'label': 'written'})

        # generate a fraction cell image
        image, mixed_number = generate_written_fraction()
        fpath = os.path.join(image_output_folder, str(i) + '_fraction.png')
        cv2.imwrite(fpath, image)
        data_list.append({'path': fpath, 'label': 'fraction'})
    
    # save the labels to a json file
    with open('./CellClassifier/SyntheticCellData/labels.json', 'w') as f:
        json.dump(data_list, f, indent=4)


if __name__ == '__main__':
    print('started')
    main()
    print('done')
