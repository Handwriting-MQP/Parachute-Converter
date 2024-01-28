downloading data (for training cell classifier model)
- download "words.zip" from Google Drive in the folder "Handwriting MQP/Code and Data/Large Data Files"
- unzip it to "./TrainCellClassifierModel/IAM-data/words"

downloading models
- Both the classifier and fraction models are stored in the Handwriting MQP folder. Go to the Code and Data folder,
  and in there is a folder called "Models"
- Download this entire folder and copy it to the root project folder. So it should be ${project_folder_name}/Models/FractionModel
  for example
- These folders are too big for Github so should be downloaded this way

main script pipeline
- run SplitPDFsIntoImages.py to break a PDF up into image files
- run PreprocessImages.py to do a basic rotation deskew and resize the images
- run WarpPerspectiveDeskew.py to do a perspective deskew on the images
- run ConvertImagesToXLSX to do OCR on the images
    - first it finds cell edges
    - then it does OCR on each cell
