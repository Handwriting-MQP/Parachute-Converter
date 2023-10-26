import os

import cv2
import numpy
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import pytesseract


# set some global paramaters
min_box_width = 150
min_box_height = 70
max_area = 1000000


def preprocess_image(image):
    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply GaussianBlur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)

    # edges = cv2.Canny(blurred_image, 50, 150)
    # cv2.imwrite(output_path, edges)
    # NOTE: Canny does a really good job finding edges!

    # use adaptive thresholding to convert the image to binary
    binary_image = cv2.adaptiveThreshold(blurred_image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                         thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    # NOTE: tuning "blockSize" may be a source for improvment

    return binary_image


def rotate_image_using_aspect_ratio(image, filtered_contours):
    threshold_aspect_raio = 1
    # NOTE: from a few tests, it seems like the average aspect ratio of normal pages is around 5,
    #       while rotated pages are around 0.6

    # calculate mean aspect ratio
    aspect_ratios = []
    for x, y, w, h in filtered_contours:
        aspect_ratios.append(w/h)
    mean_aspect_ratio = sum(aspect_ratios)/len(aspect_ratios)
    
    # if we think the image is rotate, rotate both the image and the contour points
    if mean_aspect_ratio <= threshold_aspect_raio:
        # rotate points on contour
        for i in range(len(filtered_contours)):
            c = filtered_contours[i]
            x, y, w, h = c[0], c[1], c[2], c[3]
            c[0], c[1], c[2], c[3] = image.shape[0] - (y + h), x, h, w
        
        # rotate image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    return image, filtered_contours


def generate_image_with_rectangle_overlay(image, filtered_contours, output_path):
    for x, y, w, h in filtered_contours:
        # add a green rectangle to the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    
    # save image with green rectangles
    cv2.imwrite(output_path, image)


def generate_csv_with_detected_text(image, filtered_contours, csv_path):
    # set parameters
    vertical_tolerance = 60

    data = {}
    # data has the following form:
    # {y1: {x1: text, x2: text, ...},
    #  y2: {x1: text, x2: text, ...},
    #  y3: ...}
    # here x and y are the corners of contours

    for x, y, w, h in tqdm(filtered_contours):
        # crop the image to within the bounding box
        cropped_image = image[y:y + h, x:x + w]
        # use a model to detect the text within the box
        detected_text = pytesseract.image_to_string(cropped_image)

        # search if the y value of the current box already exists in "rows" (within a tolerance)
        found_suitable_y_value = False
        for current_row_y in data.keys():
            if current_row_y - vertical_tolerance < y < current_row_y + vertical_tolerance:
                data[current_row_y][x] = detected_text
                found_suitable_y_value = True
                break
        
        # if we didn't find the y value in rows already, we need to add it
        if found_suitable_y_value is False:
            data[y] = {x: detected_text}
    
    # convert data into a 2d array of rows
    rows = []
    for current_row_y in sorted(data.keys()):
        # add a new row to "rows"
        rows.append([])

        # populate new row
        for currnet_row_x in sorted(data[current_row_y].keys()):
            text = data[current_row_y][currnet_row_x]
            rows[-1].append(text)
    
    # pad each row in "rows" to be the same length
    if len(rows) != 0: # this line just catches the case where rows is empty
        max_num_cols = max([len(row) for row in rows])
        for i in range(len(rows)):
            rows[i] += ['']*(max_num_cols - len(rows[i]))
    
    # convert "rows" from a 2d array to a dataframe
    df = pd.DataFrame(numpy.array(rows, dtype='object'))
    df.to_csv(csv_path, index=False)


def detect_rectangles(image_path, output_path, csv_path):
    # load the image
    image = cv2.imread(image_path)
    # check if the image didn't load correctly (most likely because the path didn't exist)
    if image is None:
        print(f'Unable to load image: {image_path}')
        return
    
    # pre-process image for contour detection
    processed_image = preprocess_image(image)

    # find contours in the binary image
    contours, _ = cv2.findContours(processed_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # filter contours to keep only those within size thresholds
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h <= max_area and w >= min_box_width and h >= min_box_height:
            filtered_contours.append([x, y, w, h])
    
    image, filtered_contours = rotate_image_using_aspect_ratio(image, filtered_contours)
    
    generate_image_with_rectangle_overlay(image, filtered_contours, output_path)
    generate_csv_with_detected_text(image, filtered_contours, csv_path)


def main():
    input_folder = './ParachuteData/pdf-pages-as-images-deskewed/T-11 LAT (SEPT 2022)'
    output_folder = './ParachuteData/RectangleDetectorOutputImages'
    csv_folder = './ParachuteData/RectangleDetectorOutputCSVs'

    # check if input folder exists
    if not os.path.exists(input_folder):
        print(f'Input folder {input_folder} does not exist!')
        return

    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # loop through all images in the input folder
    for image_file in os.listdir(input_folder):
        # skip non-image files
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            print(f'"{image_file}" is not an image file!')
            continue

        # get paths
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        csv_filename = os.path.basename(image_file).split('.')[0] + '.csv'
        csv_path = os.path.join(csv_folder, csv_filename)

        print(f'{input_path} started')
        detect_rectangles(input_path, output_path, csv_path)
        print(f'{input_path} done')

        # break # TODO: remove this when done testing!


if __name__ == "__main__":
    # NOTE: if tesseract isn't already installed, you can install it here: https://github.com/UB-Mannheim/tesseract/wiki
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    main()

