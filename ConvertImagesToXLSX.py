print('started running ConvertImagesToXLSX. imports may take a moment to load.')

import os
import sys  # for sys.stdout in tqdm

import cv2
import numpy as np

import xlsxwriter
from tqdm import tqdm

import pytesseract  # backup OCR library (for debugging)
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, ViTForImageClassification
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


# -------------------------------------------------- load models --------------------------------------------------
print('started loading models')

print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'models will be running on device: {device}')

textProcessor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
textModel = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)

# uses the same processor as textModel
fractionModel = VisionEncoderDecoderModel.from_pretrained("./Models/FractionModel", local_files_only=True).to(device)

classifier_labels = ['fraction', 'written', 'printed', 'blank']
classifierProcessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
classifierModel = ViTForImageClassification.from_pretrained(
    './Models/ClassifierModel',
    num_labels=len(classifier_labels),
    id2label={str(i): c for i, c in enumerate(classifier_labels)},
    label2id={c: str(i) for i, c in enumerate(classifier_labels)}
).to(device)
print('finished loading models')


def extract_cell_edges_from_image(image):
    """
    Extracts horizontal and vertical cell edges from an image

    Parameters:
    image (numpy.ndarray): The image from which cell edges are to be extracted.

    Returns:
    numpy.ndarray: A binary image with cell edges highlighted.
    """

    def convert_image_to_binary(image):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # convert to binary image
        binary = cv2.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)

        return binary

    binary = convert_image_to_binary(image)
    # cv2.imwrite('tmp/test01.png', binary)

    # detect horizontal edges
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal_edges = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cv2.imwrite('tmp/test02.png', horizontal_edges)

    # detect vertical edges
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_edges = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cv2.imwrite('tmp/test03.png', vertical_edges)

    # combine edges
    all_edges = vertical_edges | horizontal_edges
    # cv2.imwrite('tmp/test04.png', all_edges)

    # run kernel over image to close up gaps
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    all_edges_2 = cv2.morphologyEx(all_edges, cv2.MORPH_CLOSE, ellipse_kernel)
    # cv2.imwrite('tmp/test05.png', all_edges_2)

    return all_edges_2


def find_cell_contours(image):
    """find cell contours in an image and filter them"""
    overlap_fraction_threshold = 0.9

    # extract edges for contour detection
    cell_edges_image = extract_cell_edges_from_image(image)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(cell_edges_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    cell_contours = []

    # we want to only keep contours if they have a parent but not a grandparent
    # that is, they are one level down from the outermost level
    for contour, hierarchy_info in zip(contours, hierarchy[0]):
        next_contour, previous_contour, first_child_contour, parent_contour = hierarchy_info
        if parent_contour != -1 and hierarchy[0][parent_contour][3] == -1:
            cell_contours.append(contour)

    # remove any contours that have bounding rectangles that mostly overlap the bounding rectangle of any other contour
    # for each contour, check if its bounding rectangle is mostly within the bounding rectangles of any other contour
    # if it is, we should remove it!
    indices_to_remove = []
    for i, contour1 in enumerate(cell_contours):
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        for j, contour2 in enumerate(cell_contours):
            # ignore that a contour will overlap itself
            if i == j:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(contour2)

            # get edges of overlapping area
            left = max(x1, x2)
            right = min(x1 + w1, x2 + w2)
            top = max(y1, y2)
            bottom = min(y1 + h1, y2 + h2)

            # check for the case of no overlap
            if left >= right or top >= bottom:
                continue

            # calculate overlapping area
            overlap_area = (right - left) * (bottom - top)

            # check if the overlapping area is more then a certain fraction of the area of the contour being chcked
            if overlap_area / (w1 * h1) > overlap_fraction_threshold:
                indices_to_remove.append(i)

    # remove the offending contours
    cell_contours = [contour for i, contour in enumerate(cell_contours) if i not in indices_to_remove]

    return cell_contours


def determine_if_page_should_be_rotated(cell_contours):
    """
    Determines if an image should be rotated based on the aspect ratio of the detected contours.

    Parameters:
    cell_contours (list of numpy.ndarray): Contours filtered from the image.

    Returns:
    bool: True if the image should be rotated, False otherwise.
    """
    # set a threshold for figuring out if we need to rotate an image
    threshold_aspect_raio = 1
    # NOTE: from a few tests, it seems like the average aspect ratio of normal pages is around 5,
    #       while rotated pages are around 0.6

    # check if there are any contours
    if len(cell_contours) == 0:
        raise ValueError('cell_contours must have at least one contour')

    filtered_rectangles = [cv2.boundingRect(contour) for contour in cell_contours]

    # calculate mean aspect ratio
    aspect_ratios = [w / h for x, y, w, h in filtered_rectangles]
    mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)

    # return True if the mean aspect ratio is less then the threshold
    return mean_aspect_ratio <= threshold_aspect_raio


def find_word_contours_in_cell(cell_image):
    """generate a list of contours around words in a cell"""

    # convert to grayscale
    gray_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    # perform OTSU threshold
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # set the edges of the binary image to be 0
    # this removes cell lines that are part of the image border
    zero_border_size = 10  # NOTE: this can safely be at least as large as the kernal size used in the next step
    binary_image[:zero_border_size, :] = 0
    binary_image[-zero_border_size:, :] = 0
    binary_image[:, :zero_border_size] = 0
    binary_image[:, -zero_border_size:] = 0

    # dialte the image using a rectangular kernel
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_image = cv2.dilate(binary_image, rect_kernel, iterations=2)

    # finding contours
    word_contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return word_contours


def generate_image_with_rectangle_overlays(image, cell_contours, image_output_path):
    """
    Overlays rectangles on an image at positions defined by contours and saves the modified image.

    Parameters:
    image (numpy.ndarray): The original image to be modified.
    cell_contours (list of numpy.ndarray): Contours to be used for drawing rectangles.
    image_output_path (str): Path to save the modified image.

    """
    # copy the image so we aren't writing on the image we passed into the function
    image = image.copy()

    for contour in cell_contours:
        # get the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # add a green rectangle to the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # add little circles to the corners of the rectangle
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=5)
        cv2.circle(image, (x + w, y), radius=5, color=(255, 0, 0), thickness=5)
        cv2.circle(image, (x, y + h), radius=5, color=(255, 0, 255), thickness=5)
        cv2.circle(image, (x + w, y + h), radius=5, color=(0, 165, 255), thickness=5)

        # draw word contours
        cell_image = image[y:y + h, x:x + w]
        word_contours = find_word_contours_in_cell(cell_image)
        for word_contour in word_contours:
            cv2.drawContours(image, [word_contour + np.array([x, y])], -1, (0, 0, 255), thickness=1)

    cv2.imwrite(image_output_path, image)


def do_OCR_on_word_group(word_group_image, use_tesseract=False):
    def getClassifierPrediction(image):
        inputs = classifierProcessor(images=image, return_tensors="pt").to(device)
        outputs = classifierModel(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        options = {
            0: 'fraction',
            1: 'written',
            2: 'printed',
            3: 'blank',

        }
        # print(predicted_class_idx)
        # print("Predicted class:", options[predicted_class_idx])
        return options[predicted_class_idx]
    
    if use_tesseract:
        return pytesseract.image_to_string(word_group_image).strip()
    
    cellType = getClassifierPrediction(word_group_image)
    if cellType == 'typed' or 'written':
        pixel_values = textProcessor(images=word_group_image, return_tensors="pt").to(device).pixel_values
        generated_ids = textModel.generate(pixel_values)
        generated_text = textProcessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif cellType == 'fraction':
        pixel_values = textProcessor(images=word_group_image, return_tensors="pt").to(device).pixel_values
        generated_ids = fractionModel.generate(pixel_values)
        generated_text = textProcessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_text = ''
    
    return generated_text


def do_OCR_on_cell(cell_image):
    # sort contours by the sum of their x and y coordinates of their center
    def contour_sort_key(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2) + (y + h / 2)
    
    word_contours = find_word_contours_in_cell(cell_image)
    word_contours = sorted(word_contours, key=contour_sort_key)

    cell_text = []
    for word_contour in word_contours:
        x, y, w, h = cv2.boundingRect(word_contour)

        # crop out the word group from the cell
        word_group_image = cell_image[y:y + h, x:x + w]

        # do OCR on the cropped word group
        word_group_text = do_OCR_on_word_group(word_group_image)

        cell_text.append(word_group_text)

    return ' '.join(cell_text)


def generate_xlsx_with_detected_text(image, cell_contours, xlsx_path, display_debug_info=False):
    """
    Generates an Excel file with text detected in various sections of an image.

    Parameters:
    image (numpy.ndarray): The original image from which text is to be extracted.
    cell_contours (list of numpy.ndarray): Contours used to define sections for text extraction.
    xlsx_path (str): Path to save the generated Excel file.

    """

    def add_value_to_list_with_tolerance(new_val, lst, tolerance):
        for v in lst:
            if v - tolerance <= new_val <= v + tolerance:
                return
        lst.append(new_val)
        lst.sort()

    def get_index_of_closest_val(target_val, lst, tolerance):
        for i, v in enumerate(lst):
            if v - tolerance <= target_val <= v + tolerance:
                return i
        raise ValueError(f'No value in list is within tolerance of target value: {target_val}')

    # TODO: think about tuning these values in some way
    # NOTE: a value of 50 was too big! that is, we skipped over small cells that were close together!
    vertical_tolernce = 30
    horizontal_tolerance = 30

    column_edges = []  # stores the x values for each vertical edge
    row_edges = []  # stores the y values for each horizontal edges
    data_tuples = []  # stores data in the form (x, y, w, h, cell_text, median_color_str)

    # generate data tuples from filtered contours
    tqdm_output = sys.stdout if display_debug_info else open(os.devnull, 'w')
    for cell_contour in tqdm(cell_contours, file=tqdm_output):
        # get the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(cell_contour)

        # update rows and cols
        add_value_to_list_with_tolerance(x, column_edges, horizontal_tolerance)
        add_value_to_list_with_tolerance(x + w, column_edges, horizontal_tolerance)
        add_value_to_list_with_tolerance(y, row_edges, vertical_tolernce)
        add_value_to_list_with_tolerance(y + h, row_edges, vertical_tolernce)

        # extract cell from image
        cell_image = image[y:y + h, x:x + w]

        # generate text for cell
        cell_text = do_OCR_on_cell(cell_image)

        # calculate the median pixel color
        median_pixel_value = np.median(cell_image.reshape((-1, 3)), axis=0, overwrite_input=False).astype('uint8')
        s = f'{hex(median_pixel_value[1])}{hex(median_pixel_value[2])}{hex(median_pixel_value[0])}'
        median_color_str = s.replace('0x', '')

        # add cell data to data tuples
        data_tuples.append((x, y, w, h, cell_text, median_color_str))

    # print(f'column_edges: {list(enumerate(column_edges))}')
    # print(f'row_edges: {list(enumerate(row_edges))}')

    # generate XLSX file
    workbook = xlsxwriter.Workbook(xlsx_path)
    worksheet = workbook.add_worksheet()

    # update column/row width/height in spreadsheet
    column_px_to_width_ratio = 1 / 20
    row_px_to_width_ratio = 1 / 4
    # set column widths
    for index in range(len(column_edges) - 1):
        width_px = column_edges[index + 1] - column_edges[index]
        worksheet.set_column(first_col=index, last_col=index, width=width_px * column_px_to_width_ratio)
    # set row heights
    for index in range(len(row_edges) - 1):
        height_px = row_edges[index + 1] - row_edges[index]
        worksheet.set_row(row=index, height=height_px * row_px_to_width_ratio)

    # populate XLSX file from data tuples
    for x, y, w, h, text, median_color_str in data_tuples:
        # get the index of the cell in the spreadsheet
        column_left_edge_index = get_index_of_closest_val(x, column_edges, horizontal_tolerance)
        column_right_edge_index = get_index_of_closest_val(x + w, column_edges, horizontal_tolerance)
        row_top_edge_index = get_index_of_closest_val(y, row_edges, vertical_tolernce)
        row_bottom_edge_index = get_index_of_closest_val(y + h, row_edges, vertical_tolernce)

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
        if column_right_edge_index - column_left_edge_index == 1 and row_bottom_edge_index - row_top_edge_index == 1:
            worksheet.write(row_top_edge_index, column_left_edge_index, text, cell_format)
        else:
            try:
                worksheet.merge_range(first_row=row_top_edge_index, last_row=row_bottom_edge_index - 1,
                                      first_col=column_left_edge_index, last_col=column_right_edge_index - 1,
                                      data=text, cell_format=cell_format)
            except xlsxwriter.exceptions.OverlappingRange as e:
                if display_debug_info:
                    print(e)
                    print(f'\tx: {x}, y: {y}, w: {w}, h: {h}, text: {repr(text)}')
                    print(f'\tcolumn_left_edge_index: {column_left_edge_index},\
                            column_right_edge_index: {column_right_edge_index},\
                            row_top_edge_index: {row_top_edge_index},\
                            row_bottom_edge_index: {row_bottom_edge_index}')

    workbook.close()
    print(f'Finished generating an Excel for: {xlsx_path}')


def convert_image_to_xlsx(image_input_path, image_output_path, xlsx_output_path, display_debug_info=False):
    """
    Processes an image to detect text regions, performs OCR, and generates an Excel file with the extracted data.

    Parameters:
    image_input_path (str): Path to the input image.
    image_output_path (str): Path to save the image with drawn rectangles.
    xlsx_output_path (str): Path to save the generated Excel file.

    """
    # set the minimum number of detected cells needed to produce an XLSX file
    # NOTE: we need this becase some pages don't include any cells and we shouldn't try to process those
    min_num_cells_threshold = 20

    # load the image
    image = cv2.imread(image_input_path)
    # check if the image didn't load correctly (most likely because the path didn't exist)
    if image is None:
        print(f'Unable to load image: {image_input_path}')
        return

    # get cell contours
    cell_contours = find_cell_contours(image)

    # determine if the image should be rotated
    # if it should, we'll need to recalculate the contours
    if len(cell_contours) > 0 and determine_if_page_should_be_rotated(cell_contours) is True:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cell_contours = find_cell_contours(image)

    generate_image_with_rectangle_overlays(image, cell_contours, image_output_path)

    # if there are less then the minium number of contours, don't try to produce an XLSX file
    if len(cell_contours) < min_num_cells_threshold:
        print('there are less then the minium number of contours')
        # make a placeholder file to have a clear record of what happened
        with open(xlsx_output_path + '.txt', 'w') as f:
            f.write("this file is a placeholder. we didn't think this page had cells on it.")
    else:
        generate_xlsx_with_detected_text(image, cell_contours, xlsx_output_path, display_debug_info)


def main():
    """
    Main function to process images in a specified input folder and generate outputs.
    """
    image_input_folder = './ParachuteData/pdf-pages-as-images-preprocessed-deskewed'
    output_folder = './XLSXOutput'

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
        convert_image_to_xlsx(image_input_path, image_output_path, xlsx_output_path)


if __name__ == "__main__":
    # NOTE: if tesseract isn't already installed, you can install it here: https://github.com/UB-Mannheim/tesseract/wiki
    # pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # main()

    # image_input_path = 'ParachuteData/pdf-pages-as-images-preprocessed-deskewed/T-11 W911QY-19-D-0046 LOT 45_09282023-014.png'
    image_input_path = './ParachuteData/pdf-pages-as-images-preprocessed-deskewed/T-11 LAT (SEPT 2022)-020.png'
    image_output_path = './XLSXOutput/test.png'
    xlsx_output_path = './XLSXOutput/test.xlsx'
    convert_image_to_xlsx(image_input_path, image_output_path, xlsx_output_path, display_debug_info=True)
