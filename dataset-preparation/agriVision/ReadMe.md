README (AgriVision batching + jitter/normalization)

This script reads raw images from ROOT_DIR/raw-data/agriVision/uncleaned/, randomly shuffles them (seeded), splits them into 10 batches of 1500 images, and writes each batch to a folder named batch{idx}-agriVision-RGB-cleaned-jitter/ inside ROOT_DIR/raw-data/agriVision/. Each output folder contains compressed .npz files storing a float32 RGB image array under the key "image" after applying per-batch noise jitter and per-channel normalization.

Downstream: these batch folders are designed to be discovered by transformation pipelines via RAW_DATA_SUFFIX="agriVision-RGB-cleaned-jitter"; Set BATCH_NUM = 0..9 and use npz_opener() to load the saved RGB arrays.