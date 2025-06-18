2D MRI EDA Summary

1. Orientations:
- Axial: top to bottom
- Coronal: left to right
- Sagittal: front to back


2. Fill Holes and Mask - mask_images_cleaned.ipynb

- We want to fit our model to only the data inside the brain, and not the noise from the background of the MRI scan.

- We needed to seprate the data values of the brain scan from the background. The background wasn't a complete 0 grayscale value, and there were 0 grayscale values within the brain portion on some MRIs, so to only remove the data from the outside, we used a two-step process of filling holes, then masking. 

- Filling holes would allow values inside the brain to remain, and masking afterwards turned the values outside of the brain to Nan values since they were not internally contained.

- Since the sagittal and coronal views followed through the the left boundary of the image, we filled the left border for both when adding the Nan values, so fill holes would include pockets on the left edge.


3. Transformation

- We changed the image values to float32, so that we could normalize the images before running them through the wavelet transformation.

- We then ran KS tests on each pair of directions for each layer.


4. Combining Orientations 

- The three orientations were significantly different and we would lose information by combining them. The KS statistics were significantly high and the p-values were significantly low. 

- We created histogram plots overlapping the PDFs and CDFs of each direction (horizontal, vertical, diagonal) at each layer. While the directions did show more similarities as the layers increased, there were still clear differences in the shapes of the distributions.

- Horizontal and vertical are more similar to each other, but it makes sense to keep them separate as the brain also has consistent directionality. This was our finding for all layers of the axial, coronal, and sagittal orientations.





