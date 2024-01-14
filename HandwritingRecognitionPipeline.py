import shutil
import sys
import tempfile
import os

from SplitPDFsIntoImages import split_pdf_into_images
from ConvertImagesToXLSX import process_image
from WarpPerspectiveDeskew import warp_perspective_deskew
from PreprocessImages import preprocess_image

# TODO: add in preprocess_image from PreprocessImages.py + add in warp_perspective_deskew from WarpPerspectiveDeskew.py

def print_usage_and_exit():
    print("Usage: python HandwritingRecognitionPipeline.py <directory/containing/data/pdfs>")
    sys.exit()

if __name__ == '__main__':

    pipeline_results_dir = 'PipelineResults'
    if os.path.exists(pipeline_results_dir):
        shutil.rmtree(pipeline_results_dir)
    os.makedirs(pipeline_results_dir)

    # Make sure that the input command line argument is good, that is,
    # it is a directory containing only the pdfs intended to be processed.

    if len(sys.argv) < 2:
        print_usage_and_exit()

    pdfs_dir = sys.argv[1]
    data_pdfs = []
    if os.path.isdir(pdfs_dir):
        for pdf in os.listdir(pdfs_dir):
            if ".pdf" not in pdf:
                print_usage_and_exit()
            else:
                data_pdfs.append(pdf)
    else:
        print_usage_and_exit()

    # Split the pdfs into images in temporary files based on the pdf name.
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        print("Created directory " + tmp_dir_name)
        # A list of just the names of each of the temporary directories,
        # which contain the images corresponding to each pdf.
        temporary_pdf_image_dirs = []
        for pdf in data_pdfs:
            cur_pdf_dir = pdf.replace(".pdf", "")
            temporary_pdf_image_dirs.append(cur_pdf_dir)
            output_path_for_cur_pdfs_images = os.path.join(tmp_dir_name, cur_pdf_dir)

            os.makedirs(output_path_for_cur_pdfs_images, exist_ok=True)
            path_to_cur_pdf = os.path.join(pdfs_dir, pdf)
            # Output to one of the temporary directories.
            split_pdf_into_images(path_to_cur_pdf, output_path_for_cur_pdfs_images)

        for temporary_pdf_image_dir in temporary_pdf_image_dirs:
            cur_images_out_dir = os.path.join(pipeline_results_dir, temporary_pdf_image_dir, 'Images')
            if os.path.exists(cur_images_out_dir):
                shutil.rmtree(cur_images_out_dir)
            os.makedirs(cur_images_out_dir)

            cur_xlsxs_out_dir = os.path.join(pipeline_results_dir, temporary_pdf_image_dir, 'Excels')
            if os.path.exists(cur_xlsxs_out_dir):
                shutil.rmtree(cur_xlsxs_out_dir)
            os.makedirs(cur_xlsxs_out_dir)

            full_current_temp_path = os.path.join(tmp_dir_name, temporary_pdf_image_dir)
            for image in os.listdir(full_current_temp_path):
                current_image_input_path = os.path.join(full_current_temp_path, image)
                current_image_output_path = os.path.join(cur_images_out_dir, image)
                current_xlsx_output_path = os.path.join(cur_xlsxs_out_dir, image).replace(".png", ".xlsx")
                process_image(current_image_input_path, current_image_output_path, current_xlsx_output_path)