import cv2
import os

import numpy
import pandas as pd
import pytesseract
import xlsxwriter

min_box_side_length = 30
max_area = 100000

columns = []


def addToRow(image, x, y, w, h, row):
    cropped = image[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped)
    update_columns(x)

    for current_y in row.keys():
        if current_y - 10 <= y <= current_y + 10:
            dict_x = row[current_y]
            dict_x.update({x: [text, w]})
            row.update({current_y: dict_x})
            return row

    row.update({y: {x: [text, w]}})
    return row


def update_columns(x):
    for val in columns:
        if val - 10 <= x <= val + 10:
            return
    columns.append(x)
    columns.sort()


def get_closest_val(x):
    for i, val in enumerate(columns):
        if val - 10 <= x <= val + 10:
            return i
    return i + 1


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
        # print(f'y={y}')
        if max_area > area > min_area and w >= min_box_side_length and h >= min_box_side_length / 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            row = addToRow(image, x, y, w, h, row)
    print(columns)

    y_vals = list(row.keys())
    y_vals.sort()
    text_val = []
    max_val = -1000
    workbook = xlsxwriter.Workbook(f'{csv_num}.xlsx')
    worksheet = workbook.add_worksheet()
    ## FOR VISUALIZATION ##
    merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "yellow",
        }
    )
    for i, dict_x in enumerate(y_vals):
        x_row = []
        x_items = row[dict_x]
        x_keys = list(x_items.keys())
        x_keys.sort()
        for item in x_keys:
            text = x_items[item][0]
            w = x_items[item][1]
            low_index = get_closest_val(item)
            high_index = get_closest_val(item + w)
            # worksheet.write(i, low_index, text)
            try:
                if high_index-low_index == 1:
                    worksheet.write(i, low_index, text)
                else:
                    worksheet.merge_range(first_row=i, first_col=low_index, last_row=i, last_col=high_index - 1,
                                          data=text, cell_format = merge_format)
            except:
                print(f"An exception occurred{i},{low_index},{high_index}")
            for inbetween in range(high_index - low_index):
                x_row.append(text)
        text_val.append(x_row)
    workbook.close()

    for row_text in text_val:
        max_val = max(max_val, len(row_text))

    data = [row + [''] * (max_val - len(row)) for row in text_val]
    # print(text_val)
    for row in data:
        print(len(row))
        print(row)
    # Save the output image
    cv2.imwrite(output_path, image)
    num = numpy.array(data, dtype="object")

    df = pd.DataFrame(num)
    print(df)
    # Suppose we have two lists of different sizes

    df.to_csv(f'{csv_path}/{csv_num}.csv')
    raise Exception('This is an exception')


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
    i = -1
    input_path = "img_4.png"
    # input_path = "ParachuteData/pdf-pages-as-images/T-11 LAT (SEPT 2022)-014.png"
    output_path = "coords_with_boxes.jpg"
    detect_rectangles(i, input_path, output_path)
    main()
