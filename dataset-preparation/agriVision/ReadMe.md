SpaceNet Data Cleaning Process:
1. Run the randomizer.ipynb notebook if presented with raw data files, the notebook randomzied the entire dataset into batches of 1500 images.
2. Copy the cleaned batches of images into the rawdata folder if needed, run the transform_agriVision.ipynb file.

randomizer.ipynb: The notebook randomzied the entire dataset into batches of 1500 images.
transform_agriVision.ipynb: Fourier/Wavelet transformation with pixel level jitter + batch normaliztion based on tiles.