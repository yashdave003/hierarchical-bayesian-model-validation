README (SegmentAnything crop + deblur-avoid + jitter)

This script reads raw images from ROOT_DIR/raw-data/segmentAnything/uncleaned/, and for each image attempts to take a 512×512 random crop that avoids blurred regions (blur detected via low Laplacian variance on 64×64 blocks). If a valid crop is found, it optionally adds small uniform jitter in (-0.5, 0.5) per pixel/channel, clips back to uint8, and saves the cropped result as an image file.

Outputs are written to ROOT_DIR/raw-data/segmentAnything/segmentAnything-croppedDeblurred/ using the same filenames/extensions as the originals; images that are too small or cannot produce a valid crop within 100 attempts are skipped.

Downstream usage: set DATASET="segmentAnything" and RAW_DATA_SUFFIX="segmentAnything-croppedDeblurred", then point pipelines to batch_dir = os.path.join(ROOT_DIR, "raw-data", "segmentAnything", RAW_DATA_SUFFIX)