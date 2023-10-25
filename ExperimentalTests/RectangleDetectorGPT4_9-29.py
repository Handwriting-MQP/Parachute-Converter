import os

import cv2


def recursive_rectangle_detector(img, original, level=0):
    max_level = 10
    if level > max_level:
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Draw the rectangle on the original image
            cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)

            # Crop the image to the detected rectangle
            x, y, w, h = cv2.boundingRect(approx)
            cropped_img = img[y:y + h, x:x + w]

            # Recursively detect rectangles in the cropped image
            recursive_rectangle_detector(cropped_img, original, level + 1)


def rectangle_detector(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Unable to load image: {image_path}")
            continue

        original = img.copy()
        recursive_rectangle_detector(img, original)

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, original)


def main():
    input_folder = 'SampleDocument_PNG'
    output_folder = 'RectangleDetectorOutput'

    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist, creating it.")
        os.makedirs(output_folder)

    rectangle_detector(input_folder, output_folder)


if __name__ == "__main__":
    main()
