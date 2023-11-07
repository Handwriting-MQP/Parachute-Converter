import cv2
import os

import pytesseract
import xlsxwriter

min_box_side_length = 30
max_area = 100000

columns = []
rows = []
values = []


def addToRow(image, x, y, w, h):
    cropped = image[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped)
    update_columns(x)
    update_rows(y)
    values.append((x, y, w, h, text))


def update_columns(x):
    for val in columns:
        if val - 10 <= x <= val + 10:
            return
    columns.append(x)
    columns.sort()


def update_rows(y):
    for val in rows:
        if val - 10 <= y <= val + 10:
            return
    rows.append(y)
    rows.sort()


def get_closest_xval(x):
    for i, val in enumerate(columns):
        if val - 10 <= x <= val + 10:
            return i
    return i + 1


def get_closest_yval(y):
    for i, val in enumerate(rows):
        if val - 10 <= y <= val + 10:
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
    # Filter contours based on area and draw bounding rectangles around them

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # print(f'y={y}')
        if max_area > area > min_area and w >= min_box_side_length and h >= min_box_side_length / 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            addToRow(image, x, y, w, h)
    cv2.imwrite("RectangleDetectorV3Out.png", image)
    print(columns)
    print(rows)
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

    for cell in values:
        x,y,w,h,text = cell
        low_xindex = get_closest_xval(x)
        high_xindex = get_closest_xval(x + w)
        low_yindex = get_closest_yval(y)
        high_yindex = get_closest_yval(y + h)
        try:
            if high_xindex - low_xindex == 1 and high_yindex - low_yindex == 1:
                worksheet.write(low_yindex, low_xindex, text)
            else:
                worksheet.merge_range(first_row=low_yindex, first_col=low_xindex, last_row=high_yindex-1,
                                      last_col=high_xindex - 1, data=text, cell_format=merge_format)
        except:
            print(f"An exception occurred{cell}")

    workbook.close()
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
