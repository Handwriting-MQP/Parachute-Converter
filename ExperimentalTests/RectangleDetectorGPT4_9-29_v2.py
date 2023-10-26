import cv2
import os

def recursive_rectangle_detector(img, level=0, offsetX=0, offsetY=0):
    max_level = 10
    if level > max_level:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Adjusting the coordinates of the detected rectangle
            adjusted_approx = approx + [offsetX, offsetY]
            rectangles.append(adjusted_approx)

            x, y, w, h = cv2.boundingRect(approx)
            cropped_img = img[y:y + h, x:x + w]

            # Adding the offsets for the recursive call
            rectangles += recursive_rectangle_detector(cropped_img, level + 1, offsetX + x, offsetY + y)

    return rectangles

def main():
    input_folder = 'SampleDocument_PNG'
    output_folder = 'RectangleDetectorOutput'

    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist, creating it.")
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Unable to load image: {image_path}")
            continue

        rectangles = recursive_rectangle_detector(img)

        for rect in rectangles:
            cv2.drawContours(img, [rect], -1, (0, 255, 0), 2)

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()
