README (COCO indoor/outdoor → crop → jitter/normalize → merged .npz)

This script reads the uncropped COCO dataset from ROOT_DIR/raw-data/coco/val2017/ using annotations in ROOT_DIR/raw-data/coco/annotations/instances_val2017.json, classifies images as indoor vs outdoor (exclusive only), then applies a random 256×256 crop before computing RGB statistics. It then applies optional jitter and per-split normalization, and saves processed outputs as RGB float32 .npz files into ROOT_DIR/raw-data/coco/full-coco-indoor-cropped/.

Downstream: set RAW_DATA_SUFFIX="coco-indoor-cropped" and point pipelines to raw-data/coco/full-coco-indoor-cropped/; existing npz_opener() works unchanged for Fourier/wavelet transforms and frequency-map generation.