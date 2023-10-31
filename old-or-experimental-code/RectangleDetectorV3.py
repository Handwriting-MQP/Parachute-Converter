import cv2
import os
import csv
from itertools import zip_longest

import numpy
import pandas as pd
import pytesseract

min_box_side_length = 30
max_area = 100000

def addToRow(image, x, y, w, h, row):
    cropped = image[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped)

    for current_y in row.keys():
        if current_y - 40 < y < current_y + 40:
            list = row[current_y]
            list.append(text)
            row.update({current_y:list})
            return row

    row.update({y: [text]})
    return row


def detect_rectangles(csv_num, image_path, output_path, min_area=min_box_side_length * min_box_side_length):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to load image: {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to convert the image to binary
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    csv_path = "RectangleCSV"
    row = {}
    # Filter contours based on area and draw bounding rectangles around them

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        #print(f'y={y}')
        if max_area > area > min_area and w >= min_box_side_length and h >= min_box_side_length / 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            row = addToRow(image, x, y, w, h, row)

    rows = list(row.keys())
    rows.sort()
    text_val = []
    max_val = -1000

    for i in rows:
        max_val = max(max_val, len(row[i]))
        print(row[i])
        text_val.append(row[i])
    data = [row + [''] * (max_val - len(row)) for row in text_val]
    #print(text_val)

    # Save the output image
    cv2.imwrite(output_path, image)
    num = numpy.array(data, dtype="object")
    df = pd.DataFrame(num)
    # Suppose we have two lists of different sizes

    df.to_csv(f'{csv_path}/{csv_num}.csv')



def main():
    input_folder = 'DeskewedImages'
    output_folder = 'RectangleDetectorOutput'

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i = 0
    # Loop through all images in the input folder
    for image_file in os.listdir(input_folder):
        i = i + 1
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, image_file)
            output_path = os.path.join(output_folder, image_file)
            detect_rectangles(i, input_path, output_path)


if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    main()
