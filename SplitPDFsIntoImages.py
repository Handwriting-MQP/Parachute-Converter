from pdf2image import convert_from_path
import os
import itertools


def split_pdf_into_images(pdf_filepath, image_dir):
    for i in itertools.count(start=1):
        # convert the i'th page to an image
        pages = convert_from_path(pdf_filepath, dpi=500, first_page=i, last_page=i)
        
        # break if we hit the end of the PDF
        if len(pages) == 0:
            break
        
        # generate the proper filepath and save the pdf
        image_filename = os.path.basename(pdf_filepath).split('.')[0] + f'-{i:03}' + '.png'
        image_filepath = os.path.join(image_dir, image_filename)
        pages[0].save(image_filepath, 'png')

        print(f'    saved page {i} to temp dir')


if __name__ == '__main__':
    # you can change this path to process different PDFs
    pdf_filepath = './ParachuteData/PDFs/T-11 W911QY-19-D-0046 LOT 45_09282023.pdf'

    # set where you'd like to save the PDF pages as images
    image_dir = './ParachuteData/pdf-pages-as-images'

    split_pdf_into_images(pdf_filepath, image_dir)
