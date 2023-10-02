import cv2
import os


def detect_rectangles(image_path, output_path, min_area=48*48):
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
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)


def main():
    input_folder = 'SampleDocument_PNG'
    output_folder = 'RectangleDetectorOutput'

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, image_file)
            output_path = os.path.join(output_folder, image_file)
            detect_rectangles(input_path, output_path)


if __name__ == "__main__":
    main()
