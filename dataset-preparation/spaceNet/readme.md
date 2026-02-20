README (SpaceNet crop + jitter + normalization)

This script reads raw SpaceNet RGB tiles from ROOT_DIR/raw-data/spaceNet/uncleaned/, crops a 400×400 window centered on the non-black region, then applies per-batch (timestamp-based) channel mean/std normalization with small uniform jitter. Processed outputs are saved as compressed RGB float32 .npz files to ROOT_DIR/raw-data/spaceNet/full-spaceNet-cleaned-jitter/, preserving the original SpaceNet filename stem.

Downstream: set RAW_DATA_SUFFIX = "spaceNet-cleaned-jitter" and load the results using existing npz_opener() for Fourier conversion, frequency-map updates, and dataframe/pickle generation.