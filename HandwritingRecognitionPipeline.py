import queue
import shutil
import sys
import tempfile
import os
import threading
from tkinter import filedialog, scrolledtext
import tkinter as tk

import cv2

from SplitPDFsIntoImages import split_pdf_into_images
from PreprocessImages import preprocess_image
from WarpPerspectiveDeskew import warp_perspective_deskew
from ConvertImagesToXLSX import convert_image_to_xlsx

def print_usage_and_exit():
    print("Data directory should contain only pdfs.")
    sys.exit()

# A function to update the GUI every 100 milliseconds.
def update_gui_from_queue(root, gui_queue):
    while not gui_queue.empty():
        message = gui_queue.get_nowait()
        # Update your GUI here, e.g., insert message into a text widget
        print(message)  # or your mechanism to update the GUI
    root.after(100, update_gui_from_queue, root, gui_queue)  # Reschedule

# Start a thread to bear the load of the pipeline which is not the "main thread",
# as tkinter requires use of the main thread.
def start_processing_thread(pdfs_dir, gui_queue):
    threading.Thread(target=process_handwriting_data, args=(pdfs_dir, gui_queue), daemon=True).start()

# A function to prompt the user to select a folder, then start processing that folder.
def select_folder(gui_queue):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        start_processing_thread(folder_selected, gui_queue)

# An object to send text as a result of calls to print() to the gui instead of stdout.
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)  # Scroll to the end
        self.widget.config(state=tk.DISABLED)
    def flush(self):
        pass

# The function that processes the data directory.
def process_handwriting_data(pdfs_dir, gui_queue):

    gui_queue.put("Processing started for: " + pdfs_dir)

    # Get the path to the parent directory of the data directory.
    parent_dir = os.path.dirname(pdfs_dir)

    # Create an empty results folder in the parent directory of the data directory.
    pipeline_results_dir = os.path.join(parent_dir, 'PipelineResults')
    if os.path.exists(pipeline_results_dir):
        shutil.rmtree(pipeline_results_dir)
    os.makedirs(pipeline_results_dir)

    # Make sure that the data directory is in fact a directory, and make sure that each item in
    # the data directory is a PDF (by making sure it has ".pdf" in its name).
    # If so, append the filename of the PDF to "data_pdfs".
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
        print("Created top-level temporary directory: " + tmp_dir_name)
        # A list of just the names of each of the temporary directories,
        # which contain the images corresponding to each PDF.
        temporary_pdf_image_dirs = []
        for pdf in data_pdfs:
            cur_pdf_dir = pdf.replace(".pdf", "")
            temporary_pdf_image_dirs.append(cur_pdf_dir)
            print("Created temporary directory for: " + cur_pdf_dir)

            # Here, "tmp_dir_name" is the top-level temporary directory, whereas
            # "cur_pdf_dir" is the temporary directory for the particular data PDF.
            output_path_for_cur_pdfs_images = os.path.join(tmp_dir_name, cur_pdf_dir)

            os.makedirs(output_path_for_cur_pdfs_images, exist_ok=True)
            path_to_cur_pdf = os.path.join(pdfs_dir, pdf)
            # Output to one of the temporary directories.
            split_pdf_into_images(path_to_cur_pdf, output_path_for_cur_pdfs_images)

        for temporary_pdf_image_dir in temporary_pdf_image_dirs:

            # These first two code blocks make the results folders for each data PDF.

            # After the run is finished, "Images" contains the images of each page of a data PDF,
            # after "preprocess_image" and "warp_perspective_deskew" have been called on them.
            cur_images_out_dir = os.path.join(pipeline_results_dir, temporary_pdf_image_dir, 'Images')
            if os.path.exists(cur_images_out_dir):
                shutil.rmtree(cur_images_out_dir)
            os.makedirs(cur_images_out_dir)

            # These Excels are the final result.
            cur_xlsxs_out_dir = os.path.join(pipeline_results_dir, temporary_pdf_image_dir, 'Excels')
            if os.path.exists(cur_xlsxs_out_dir):
                shutil.rmtree(cur_xlsxs_out_dir)
            os.makedirs(cur_xlsxs_out_dir)

            # This is the path to the temporary directory for the data PDF with name (with ".pdf"
            # removed) "temporary_pdf_image_dir".
            cur_data_pdf_temp_dir_path = os.path.join(tmp_dir_name, temporary_pdf_image_dir)

            # Processing location/ THE PIPELINE
            for image in os.listdir(cur_data_pdf_temp_dir_path):
                # Getting paths
                current_image_input_path = os.path.join(cur_data_pdf_temp_dir_path, image)
                current_image_output_path = os.path.join(cur_images_out_dir, image)
                current_xlsx_output_path = os.path.join(cur_xlsxs_out_dir, image).replace(".png", ".xlsx")

                # calling PreprocessImages.preprocess_image
                image = cv2.imread(current_image_input_path)
                image = preprocess_image(image)

                # Calling WarpPerspectiveDeskew.warp_perspective_deskew
                image = warp_perspective_deskew(image)

                # save the now cleaned up image
                cv2.imwrite(current_image_input_path, image)

                # This is where the machine learning models are used.
                convert_image_to_xlsx(current_image_input_path, current_image_output_path, current_xlsx_output_path)

    gui_queue.put("Processing completed for: " + pdfs_dir)

def main():
    # Create the main window
    root = tk.Tk()
    gui_queue = queue.Queue()
    root.title("Handwriting Recognition Pipeline")

    # Create a scrolled text widget for console output
    console = scrolledtext.ScrolledText(root, state='disabled', height=10)
    console.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Redirect stdout
    sys.stdout = TextRedirector(console)

    # The first call to the function which calls itself to keep the GUI updated.
    root.after(100, update_gui_from_queue, root, gui_queue)

    # Create a button to open the folder selection window.
    select_folder_button = tk.Button(root, text="Select Folder With PDFs", command=lambda: select_folder(gui_queue))
    select_folder_button.pack(pady=20)

    # Run the application
    root.mainloop()

if __name__ == '__main__':
    main()