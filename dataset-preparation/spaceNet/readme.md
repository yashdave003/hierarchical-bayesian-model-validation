SpaceNet Data Cleaning Process:
1. Run the image_extraction.ipynb notebook, change the file locations to local locations.
2. Copy the cleaned images into the rawdata folder if needed, run the spaceNet_normalized_xxx.ipynb file.

image_extraction.ipynb: random extraction of a 400x400 cropped image with no black border
 spaceNet_normalized_xxx.ipynb: Fourier/Wavelet transformation with pixel level jitter + batch normaliztion based on tiles.