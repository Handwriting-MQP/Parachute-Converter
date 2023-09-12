from PIL import Image
from pix2tex.cli import LatexOCR

url = 'Screenshot 2023-09-09 233011.png'

img = Image.open(url)
model = LatexOCR()
print(model(img))