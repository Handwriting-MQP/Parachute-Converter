To generate synthetic cell data (for training a cell classifier model), you will need to download "words.zip" from Google Drive. Once you do, unzip it to "./CellClassifier/IAM-data/words"

main script pipeline
- run SplitPDFsIntoImages.py to break a PDF up into image files
- run PreprocessImages.py to do a basic rotation deskew and resize the images
- run WarpPerspectiveDeskew.py to do a perspective deskew on the images
- run RectangleDetector.py (bad name) to do OCR on the images
    - first it finds cell edges
    - then it does OCR on each cell

other scripts
- GenerateSyntheticMixedNumberData.py does what is says on the tin
- TrainMixedNumberTrOCR.py does what is says on the tin
