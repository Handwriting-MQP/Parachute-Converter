import os

import cv2
import numpy as np

import pytesseract
import xlsxwriter
from tqdm import tqdm


# TODO: add this in
# def extract_image_bounded_by_contour(image, contour):
#     # get the bounding rectangle for the contour
#     x, y, w, h = cv2.boundingRect(contour)

#     # overlay the filled in contour in green, then crop to the bounding rectangle
#     cropped_green = image.copy()
#     cv2.drawContours(cropped_green, [contour], 0, color=(0, 255, 0), thickness=cv2.FILLED)
#     cropped_green = cropped_green[y:y + h, x:x + w]

#     # crop the bounding rectangle from the original (non-green) image
#     cropped = image.copy()[y:y + h, x:x + w]

#     # set the non-green areas of the corpped image to be white
#     cropped[cropped_green != (0, 255, 0)] = 255

#     return cropped


def convert_image_to_binary(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # convert to binary
    binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)
    
    return binary


def extract_cell_lines_from_image(image):
    # doc info
    # graphics: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
    # examples: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    # example of how MORPH_OPEN works:
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    # a = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    # b = cv2.erode(binary, k)
    # b = cv2.dilate(b, k)
    # print(np.all(a == b)) # this will return True

    binary = convert_image_to_binary(image)
    # cv2.imwrite('test01.png', binary)

    # detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cv2.imwrite('test02.png', horizontal_lines)

    # detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cv2.imwrite('test03.png', vertical_lines)

    # combine lines
    all_lines = vertical_lines | horizontal_lines
    # cv2.imwrite('test04.png', all_lines)

    # run kernel over image to close up gaps
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    all_lines_2 = cv2.morphologyEx(all_lines, cv2.MORPH_CLOSE, cross_kernel)
    # cv2.imwrite('test05.png', all_lines_2)

    return all_lines_2


def rotate_image_using_aspect_ratio(image, filtered_rectangles):
    # set a threshold for figuring out if we need to rotate an image
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


def generate_xlsx_with_detected_text(image, filtered_rectangles, xlsx_path):
    vertical_tolernce = 50
    horizontal_tolerance = 50

    columns = [] # stores the x values each column starts at
    rows = [] # stores the y values each row starts at
    data_tuples = []

    def update_columns(x):
        for val in columns:
            if val - horizontal_tolerance <= x <= val + horizontal_tolerance:
                return
        columns.append(x)
        columns.sort()

    def update_rows(y):
        for val in rows:
            if val - vertical_tolernce <= y <= val + vertical_tolernce:
                return
        rows.append(y)
        rows.sort()

    def get_index_of_closest_xval(x):
        for i, val in enumerate(columns):
            if val - horizontal_tolerance <= x <= val + horizontal_tolerance:
                return i
        return len(columns) # TODO: remember you changed this

    def get_index_of_closest_yval(y):
        for i, val in enumerate(rows):
            if val - vertical_tolernce <= y <= val + vertical_tolernce:
                return i
        return len(rows) # TODO: remember you changed this
    

    for x, y, w, h in tqdm(filtered_rectangles):
        cropped = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        # text = 'blah'
        update_columns(x)
        update_rows(y)

        # calculate the median pixel color
        median_pixel_value = np.median(cropped.reshape((-1, 3)), axis=0, overwrite_input=False).astype('uint8')
        s = f'{hex(median_pixel_value[1])}{hex(median_pixel_value[2])}{hex(median_pixel_value[0])}'
        median_color_str = s.replace('0x', '')
        
        data_tuples.append((x, y, w, h, text, median_color_str))
    
    print(f'column start x values: {list(enumerate(columns))}')
    print(f'row start y values: {list(enumerate(rows))}')

    workbook = xlsxwriter.Workbook(xlsx_path)
    worksheet = workbook.add_worksheet()

    for x, y, w, h, text, median_color_str in data_tuples:
        low_xindex = get_index_of_closest_xval(x)
        high_xindex = get_index_of_closest_xval(x + w)
        low_yindex = get_index_of_closest_yval(y)
        high_yindex = get_index_of_closest_yval(y + h)

        cell_format = workbook.add_format({
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": f'#{median_color_str}',
        })

        # try to add the cell to excel
        if high_xindex - low_xindex == 1 and high_yindex - low_yindex == 1:
            worksheet.write(low_yindex + 1, low_xindex + 1, text, cell_format)
        else:
            try:
                worksheet.merge_range(first_row=low_yindex + 1, last_row=high_yindex,
                                      first_col=low_xindex + 1, last_col=high_xindex,
                                      data=text, cell_format=cell_format)
            except xlsxwriter.exceptions.OverlappingRange as e:
                print(e)
                # print(f'An exception occurred in "merge_range" x: {x}, y: {y}, w: {w}, h: {h}, text: {repr(text)}')
    
    workbook.close()


def detect_rectangles(image_input_path, image_output_path, xlsx_output_path):
    # set box size paramaters
    min_box_width = 75
    min_box_height = 15
    max_area = 1000000

    # load the image
    image = cv2.imread(image_input_path)
    # check if the image didn't load correctly (most likely because the path didn't exist)
    if image is None:
        print(f'Unable to load image: {image_input_path}')
        return
    
    # extract lines for contour detection
    cell_lines = extract_cell_lines_from_image(image)

    # find contours in the binary image
    contours, _ = cv2.findContours(cell_lines, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # produce listing of bounding rectangles within size thresholds
    filtered_rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h <= max_area and w >= min_box_width and h >= min_box_height:
            filtered_rectangles.append([x, y, w, h])
    
    image, filtered_rectangles = rotate_image_using_aspect_ratio(image, filtered_rectangles)
    
    generate_image_with_rectangle_overlays(image, filtered_rectangles, image_output_path)
    generate_xlsx_with_detected_text(image, filtered_rectangles, xlsx_output_path)


def main():
    image_input_folder = './ParachuteData/pdf-pages-as-images-preprocessed'
    output_folder = './RectangleDetectorOutput'

    # check if input folder exists
    if not os.path.exists(image_input_folder):
        print(f'Input folder {image_input_folder} does not exist!')
        return

    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # loop through all images in the input folder
    for image_filename in os.listdir(image_input_folder):
        # generate file paths
        image_input_path = os.path.join(image_input_folder, image_filename)
        image_output_path = os.path.join(output_folder, image_filename)
        xlsx_filename = os.path.basename(image_filename).split('.')[0] + '.xlsx'
        xlsx_output_path = os.path.join(output_folder, xlsx_filename)

        print(f'{image_input_path} started')
        detect_rectangles(image_input_path, image_output_path, xlsx_output_path)


if __name__ == "__main__":
    # NOTE: if tesseract isn't already installed, you can install it here: https://github.com/UB-Mannheim/tesseract/wiki
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # main()

    image_input_path = './ParachuteData/pdf-pages-as-images-preprocessed/T-11 LAT (SEPT 2022)-019.png'
    image_output_path = './RectangleDetectorOutput/test.png'
    xlsx_output_path = './RectangleDetectorOutput/test.xlsx'
    detect_rectangles(image_input_path, image_output_path, xlsx_output_path)
