2D MRI EDA Summary

1. Orientations:
- Axial: top to bottom
- Coronal: left to right
- Sagittal: front to back

- Although they look similar, we decided to treat each orientation seperately and we are trying to model them distictly. We are making the assumption model will be more accurate for each distinct orientation than one model for all three.


2. Fill Holes and Mask - mask_images_cleaned.ipynb

- We want to fit our model to only the data inside the brain, and not the noise from the background of the MRI scan. Since these are generated images of MRI and not actual MRIs, the background noise is even less informational, and we want to focus on just the brain.

- We needed to seprate the data values of the brain scan from the background. The background wasn't a complete 0 grayscale value, and there were 0 grayscale values within the brain portion on some MRIs, so to only remove the data from the outside, we used a two-step process of filling holes, then masking. 

- Filling holes would allow values inside the brain to remain, and masking afterwards turned the values outside of the brain to Nan values since they were not internally contained.

- Since the sagittal and coronal views followed through the the left boundary of the image, we filled the left border for both when adding the Nan values, so fill holes would include pockets on the left edge.


3. Transformation 

- After the mask, we jittered the images to introduce some noise, but did this step after the mask so we wouldn't add random noise to the background. We added uniform random noise so that the variability of our data increased and our histograms would be more informational.

- We changed the image values to float32, so that we could normalize the images before running them through the wavelet transformation. We normalized the images after the mask so that it wouldn't include all of the 0 values from the background.


4. Combining Directions - Horizontal (H), Vertical (V), Diagonal (D)

-  We ran KS tests on each pair of directions (HV, VD, DH) for each layer for each orientation (axial, sagittal, coronal).

- We concluded the three orientations were significantly different and we would lose information by combining them. The KS statistics were significantly high and the p-values were significantly low. 

- We created histogram plots overlapping the PDFs and CDFs of each direction (horizontal, vertical, diagonal) at each layer. While the directions did show more similarities as the layers increased, there were still clear differences in the shapes of the distributions.

- Horizontal and vertical are more similar to each other compared to diagonal, but it makes sense to keep them separate as the brain also has consistent directionality. This was our finding for all layers of the axial, coronal, and sagittal orientations.
