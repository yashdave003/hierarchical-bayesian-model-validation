Audio Data Cleaning Process:

1. Run the preprocessing_template, decide whether or not you want to pitch normalize (Speech data) or if you want to trim silence
2. Choose a transform to run (cwt, erblet, fft, stft)
3. Run KS distribution comparison (?)
4. Run subsampling algorithm
5. Run testing pipeline in results-audio

In each step of the pipeline, make sure to adjust the variables as necessary and to follow the file structure outline in the files