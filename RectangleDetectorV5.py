import os

import cv2
import numpy
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import pytesseract


# set some global paramaters
# TODO: play with tuning these!
min_box_width = 150
min_box_height = 30
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
                                         thresholdType=cv2.THRESH_BINARY_INV, blockSize=21, C=2)
    # NOTE: tuning "blockSize" may be a source for improvment

    return binary_image


def rotate_image_using_aspect_ratio(image, filtered_rectangles):
    threshold_aspect_raio = 1
    # NOTE: from a few tests, it seems like the average aspect ratio of normal pages is around 5,
    #       while rotated pages are around 0.6

    # calculate mean aspect ratio
    aspect_ratios = [w/h for x, y, w, h in filtered_rectangles]
    mean_aspect_ratio = sum(aspect_ratios)/len(aspect_ratios)

    # if we think the image is rotate, rotate both the image and the rectangle points
    if mean_aspect_ratio <= threshold_aspect_raio:
        # rotate points on rectangle
        for i in range(len(filtered_rectangles)):
            c = filtered_rectangles[i]
            x, y, w, h = c[0], c[1], c[2], c[3]
            c[0], c[1], c[2], c[3] = image.shape[0] - (y + h), x, h, w
        
        # rotate image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    return image, filtered_rectangles


def generate_image_with_rectangle_overlays(image, filtered_rectangles, image_output_path):
    for x, y, w, h in filtered_rectangles:
        # add a green rectangle to the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        # add little circles to the corners of the rectangle
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=2)
        cv2.circle(image, (x + w, y), radius=5, color=(0, 0, 255), thickness=2)
        cv2.circle(image, (x, y + h), radius=5, color=(0, 0, 255), thickness=2)
        cv2.circle(image, (x + w, y + h), radius=5, color=(0, 0, 255), thickness=2)
    
    # save image with green rectangles
    cv2.imwrite(image_output_path, image)


def generate_csv_with_detected_text(image, filtered_rectangles, csv_output_path):
    # set parameters
    vertical_tolerance = 60

    data = {}
    # data has the following form:
    # {y1: {x1: text, x2: text, ...},
    #  y2: {x1: text, x2: text, ...},
    #  y3: ...}
    # here x and y are the corners of contours

    for x, y, w, h in tqdm(filtered_rectangles):
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
    df.to_csv(csv_output_path, index=False)


# this is Harsh's idea!
def generate_csv_with_detected_text_2(image, filtered_rectangles, csv_path):
    vertical_tolerance = 60
    horizontal_tolerance = 60 # was 10

    column_x_values = []
    rows = {}

    # check if the current x value is alreay close to a column value, if it isn't add it
    def update_column_x_values(new_x):
        for val in column_x_values:
            if val - horizontal_tolerance <= new_x <= val + horizontal_tolerance:
                return
        column_x_values.append(x)
    
    # check if there is already a row with a y value close to one that already exists, if there is, add to that row
    def addToRows(image, x, y, w, h):
        cropped = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)

        for current_y in rows.keys():
            if current_y - vertical_tolerance <= y <= current_y + vertical_tolerance:
                rows[current_y][x] = [text, w]
                return
        rows.update({y: {x: [text, w]}})
    

    def get_closest_val(x):
        for i, val in enumerate(column_x_values):
            if val - horizontal_tolerance <= x <= val + horizontal_tolerance:
                return i
        return i + 1
    

    for x, y, w, h in tqdm(filtered_rectangles):
        update_column_x_values(x)
        addToRows(image, x, y, w, h)
    
    column_x_values.sort()
    

    row_y_vals = sorted(list(rows.keys()))
    text_val = []

    for row_y_val in row_y_vals:
        row = []

        row_items = rows[row_y_val]
        x_keys = sorted(list(row_items.keys()))
        for row_x in x_keys:
            text = row_items[row_x][0]
            w = row_items[row_x][1]
            low_index = get_closest_val(row_x)
            high_index = get_closest_val(row_x + w)
            for _ in range(high_index - low_index):
                row.append(text)
        
        text_val.append(row)
    
    max_val = -1000
    for row_text in text_val:
        max_val = max(max_val, len(row_text))

    data = [row + [''] * (max_val - len(row)) for row in text_val]
    # print(text_val)
    for row in data:
        print(len(row))
        print(row)
    
    num = numpy.array(data, dtype="object")

    df = pd.DataFrame(num)
    df.to_csv(csv_path, index=False)


def detect_rectangles(image_input_path, image_output_path, csv_output_path):
    # load the image
    image = cv2.imread(image_input_path)
    # check if the image didn't load correctly (most likely because the path didn't exist)
    if image is None:
        print(f'Unable to load image: {image_input_path}')
        return
    
    # TODO: scale the image to have a constant max dimension
    # for more info: https://ai.stackexchange.com/questions/24311/why-do-we-resize-images-before-using-them-for-object-detection
    
    # pre-process image for contour detection
    processed_image = preprocess_image(image)

    # FOR DEBUG
    cv2.imwrite('./ParachuteData/RectangleDetectorOutputImages/p-test.png', processed_image)

    # find contours in the binary image
    contours, _ = cv2.findContours(processed_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # filter contours to keep only those within size thresholds
    filtered_rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h <= max_area and w >= min_box_width and h >= min_box_height:
            filtered_rectangles.append([x, y, w, h])
    
    image, filtered_rectangles = rotate_image_using_aspect_ratio(image, filtered_rectangles)
    
    generate_image_with_rectangle_overlays(image, filtered_rectangles, image_output_path)
    generate_csv_with_detected_text_2(image, filtered_rectangles, csv_output_path)


def main():
    image_input_folder = './ParachuteData/pdf-pages-as-images-deskewed'
    image_output_folder = './ParachuteData/RectangleDetectorOutputImages'
    csv_output_folder = './ParachuteData/RectangleDetectorOutputCSVs'

    # check if input folder exists
    if not os.path.exists(image_input_folder):
        print(f'Input folder {image_input_folder} does not exist!')
        return

    # create output folder if it doesn't exist
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
    
    # create output CSV folder if it doesn't exist
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)
    
    # loop through all images in the input folder
    for image_file in os.listdir(image_input_folder):
        # skip non-image files
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            print(f'"{image_file}" is not an image file!')
            continue

        # get paths
        image_input_path = os.path.join(image_input_folder, image_file)
        image_output_path = os.path.join(image_output_folder, image_file)
        csv_filename = os.path.basename(image_file).split('.')[0] + '.csv'
        csv_output_path = os.path.join(csv_output_folder, csv_filename)

        print(f'{image_input_path} started')
        detect_rectangles(image_input_path, image_output_path, csv_output_path)
        print(f'{image_input_path} done')


if __name__ == "__main__":
    # NOTE: if tesseract isn't already installed, you can install it here: https://github.com/UB-Mannheim/tesseract/wiki
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # main()

    image_input_path = './ParachuteData/pdf-pages-as-images-deskewed/T-11 W911QY-19-D-0046 LOT 45_09282023-031.png'
    image_output_path = './ParachuteData/RectangleDetectorOutputImages/test.png'
    csv_output_path = './ParachuteData/RectangleDetectorOutputCSVs/test.csv'
    detect_rectangles(image_input_path, image_output_path, csv_output_path)
