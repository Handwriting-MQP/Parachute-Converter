import os

import cv2
import numpy as np

import pytesseract
import xlsxwriter
from tqdm import tqdm
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from datasets import load_dataset
import torch

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')


def extract_image_bounded_by_contour(full_image, contour):
    """
    Extracts a sub-image from a larger image, bound by a specified contour.

    Parameters:
    full_image (numpy.ndarray): The original image from which a sub-image is to be extracted.
    contour (numpy.ndarray): A contour defining the boundary of the sub-image.

    Returns:
    numpy.ndarray: A cropped image which is the portion of the full_image inside the contour.
    """

    # replace the contour with it's convex hull
    # Definition of convex hull: the smallest convex shape that completely encloses the contour.
    # NOTE: this helps fix cases where the contour contains points inside of it's convex hull
    #       this can happen when text ends up touching cell lines
    contour = cv2.convexHull(contour)

    # get the bounding rectangle for the contour
    # Definition of bounding rectangle:
    #   the smallest rectangle that can completely enclose the contour.
    x, y, w, h = cv2.boundingRect(contour)

    # generate a mask for the contour on the image
    # 'contour_mask' is an image the same size as full_image with the contour filled
    # with white, and outside the contour in black.
    contour_mask = cv2.drawContours(np.zeros_like(full_image, dtype='uint8'), [contour], 0, color=(255, 255, 255),
                                    thickness=cv2.FILLED) != (255, 255, 255)

    # crop this mask to only toe bounding rectangle of the contour
    cropped_contour_mask = contour_mask[y:y + h, x:x + w]
    # cv2.imshow("Cropped Contour Mask", cropped_contour_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # crop the bounding rectangle from the original image
    cropped = full_image.copy()[y:y + h, x:x + w]

    # set the areas outside of the contour in the cropped image (that is, the bounding rectangle) to be white
    cropped[cropped_contour_mask] = 255

    return cropped


def extract_cell_lines_from_image(image):
    """
    Extracts horizontal and vertical lines from an image, typically used for identifying cell boundaries in tables.

    Parameters:
    image (numpy.ndarray): The image from which cell lines are to be extracted.

    Returns:
    numpy.ndarray: A binary image with cell lines highlighted.
    """

    def convert_image_to_binary(image):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # convert to binary
        binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)

        return binary

    # documentation on morphological transformations:
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


def determine_if_image_should_be_rotated(filtered_contours):
    """
    Determines if an image should be rotated based on the aspect ratio of the detected contours.

    Parameters:
    filtered_contours (list of numpy.ndarray): Contours filtered from the image.

    Returns:
    bool: True if the image should be rotated, False otherwise.
    """
    # set a threshold for figuring out if we need to rotate an image
    threshold_aspect_raio = 1
    # NOTE: from a few tests, it seems like the average aspect ratio of normal pages is around 5,
    #       while rotated pages are around 0.6

    filtered_rectangles = [cv2.boundingRect(contour) for contour in filtered_contours]

    # calculate mean aspect ratio
    aspect_ratios = [w / h for x, y, w, h in filtered_rectangles]
    mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)

    # if we think the image is rotate, rotate both the image and the rectangle points
    if mean_aspect_ratio <= threshold_aspect_raio:
        return True
    return False


def generate_image_with_rectangle_overlays(image, filtered_contours, image_output_path):
    """
    Overlays rectangles on an image at positions defined by contours and saves the modified image.

    Parameters:
    image (numpy.ndarray): The original image to be modified.
    filtered_contours (list of numpy.ndarray): Contours to be used for drawing rectangles.
    image_output_path (str): Path to save the modified image.

    """
    # copy the image so we aren't writing on the image we passed into the function
    image = image.copy()

    filtered_rectangles = [cv2.boundingRect(contour) for contour in filtered_contours]

    for x, y, w, h in filtered_rectangles:
        # add a green rectangle to the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        # add little circles to the corners of the rectangle
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=5)
        cv2.circle(image, (x + w, y), radius=5, color=(255, 0, 0), thickness=5)
        cv2.circle(image, (x, y + h), radius=5, color=(255, 0, 255), thickness=5)
        cv2.circle(image, (x + w, y + h), radius=5, color=(0, 165, 255), thickness=5)

    # save image with green rectangles
    cv2.imwrite(image_output_path, image)


def do_OCR_on_cell(cropped_image):
    """
    Performs Optical Character Recognition (OCR) on a cropped image.

    Parameters:
    cropped_image (numpy.ndarray): The cropped image on which OCR is to be performed.

    Returns:
    str: The text extracted from the cropped image.
    """
    pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text
    return text


def generate_xlsx_with_detected_text(image, filtered_contours, xlsx_path):
    """
    Generates an Excel file with text detected in various sections of an image.

    Parameters:
    image (numpy.ndarray): The original image from which text is to be extracted.
    filtered_contours (list of numpy.ndarray): Contours used to define sections for text extraction.
    xlsx_path (str): Path to save the generated Excel file.

    """
    vertical_tolernce = 50
    horizontal_tolerance = 50

    columns = []  # stores the x values each column starts at
    rows = []  # stores the y values each row starts at
    data_tuples = []  # stores data in the form (x, y, w, h, text, median_color_str)

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
        return len(columns)

    def get_index_of_closest_yval(y):
        for i, val in enumerate(rows):
            if val - vertical_tolernce <= y <= val + vertical_tolernce:
                return i
        return len(rows)

    # generate  data tuples from filtered contours
    counter = 0
    for contour in tqdm(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)

        # update rows and cols
        update_columns(x)
        update_rows(y)

        # extract cell from image
        cropped = image[y:y + h, x:x + w]

        # TODO: remove this debugging code at some point
        # cropped_new = extract_image_bounded_by_contour(image, contour)
        # cv2.imwrite(f'./tmp/{counter:03}-new.png', cropped_new)
        # cv2.imwrite(f'./tmp/{counter:03}-old.png', cropped)
        # counter += 1

        # generate text for cell
        text = do_OCR_on_cell(cropped)

        # calculate the median pixel color
        median_pixel_value = np.median(cropped.reshape((-1, 3)), axis=0, overwrite_input=False).astype('uint8')
        s = f'{hex(median_pixel_value[1])}{hex(median_pixel_value[2])}{hex(median_pixel_value[0])}'
        median_color_str = s.replace('0x', '')

        data_tuples.append((x, y, w, h, text, median_color_str))

    print(f'column start x values: {list(enumerate(columns))}')
    print(f'row start y values: {list(enumerate(rows))}')

    # generate XLSX file
    workbook = xlsxwriter.Workbook(xlsx_path)
    worksheet = workbook.add_worksheet()

    # update column/row width/height in spreadsheet
    column_px_to_width_ratio = 1 / 20
    row_px_to_width_ratio = 1 / 4

    for index in range(1, len(columns)):
        width_px = columns[index] - columns[index - 1]
        worksheet.set_column(first_col=index, last_col=index, width=width_px * column_px_to_width_ratio)

    for index in range(1, len(rows)):
        height_px = rows[index] - rows[index - 1]
        worksheet.set_row(row=index, height=height_px * row_px_to_width_ratio)

    # generate XLSX file from data tuples
    for x, y, w, h, text, median_color_str in data_tuples:
        # clean up the text
        text = text.strip()

        low_xindex = get_index_of_closest_xval(x)
        high_xindex = get_index_of_closest_xval(x + w)
        low_yindex = get_index_of_closest_yval(y)
        high_yindex = get_index_of_closest_yval(y + h)

        # set cell formatting
        cell_format = workbook.add_format({
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
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


def find_filtered_contours(binary_image):
    """
    Finds contours in a binary image that meet certain size criteria.

    Parameters:
    binary_image (numpy.ndarray): The binary image from which contours are to be found.

    Returns:
    list of numpy.ndarray: A list of contours filtered based on size criteria.
    """
    # set box size paramaters
    min_box_width = 75
    min_box_height = 15
    max_area = 1000000

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # produce listing of filtered contours within size thresholds
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h <= max_area and w >= min_box_width and h >= min_box_height:
            filtered_contours.append(contour)

    # TODO: remove contours inside of other contours

    return filtered_contours


def process_image(image_input_path, image_output_path, xlsx_output_path):
    """
    Processes an image to detect text regions, performs OCR, and generates an Excel file with the extracted data.

    Parameters:
    image_input_path (str): Path to the input image.
    image_output_path (str): Path to save the image with drawn rectangles.
    xlsx_output_path (str): Path to save the generated Excel file.

    """
    # load the image
    image = cv2.imread(image_input_path)
    # check if the image didn't load correctly (most likely because the path didn't exist)
    if image is None:
        print(f'Unable to load image: {image_input_path}')
        return

    # extract lines for contour detection
    cell_lines = extract_cell_lines_from_image(image)
    filtered_contours = find_filtered_contours(cell_lines)

    # determine if the image should be rotated
    # if it should, we'll need to recalculate the contours
    if determine_if_image_should_be_rotated(filtered_contours) is True:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cell_lines = extract_cell_lines_from_image(image)
        filtered_contours = find_filtered_contours(cell_lines)

    generate_image_with_rectangle_overlays(image, filtered_contours, image_output_path)
    generate_xlsx_with_detected_text(image, filtered_contours, xlsx_output_path)


def main():
    """
    Main function to process images in a specified input folder and generate outputs.
    """
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
        process_image(image_input_path, image_output_path, xlsx_output_path)


if __name__ == "__main__":
    # NOTE: if tesseract isn't already installed, you can install it here: https://github.com/UB-Mannheim/tesseract/wiki
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # main()

    image_input_path = 'ParachuteData/pdf-pages-as-images-preprocessed/T-11 W911QY-19-D-0046 LOT 45_09282023-001.png'
    # image_input_path = './ParachuteData/pdf-pages-as-images-preprocessed/T-11 LAT (SEPT 2022)-001.png'
    image_output_path = './RectangleDetectorOutput/test.png'
    xlsx_output_path = './RectangleDetectorOutput/test.xlsx'
    process_image(image_input_path, image_output_path, xlsx_output_path)
