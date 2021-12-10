## Convert Pascal VOC format to COCO

The [voc2coco.py](voc2coco.py) contains the script to convert the Pascal VOC XML
format to COCO JSON. The dataset annotation XMLs should be stored under `annotations`
directory and the images in the `images` directory. The `test_ids.txt` (or a
text file for any other split) containing a sequence of images names to be
included in the split without the extension.

### Usage

```
usage: voc2coco.py [-h] [--ann_dir ANN_DIR] [--ann_ids ANN_IDS] [--ann_paths_list ANN_PATHS_LIST] [--labels LABELS] [--output OUTPUT] [--ext EXT]

This script support converting voc format xmls to coco format json

optional arguments:
  -h, --help            show this help message and exit
  --ann_dir ANN_DIR     path to annotation files directory. It is not need when use --ann_paths_list
  --ann_ids ANN_IDS     path to annotation files ids list. It is not need when use --ann_paths_list
  --ann_paths_list ANN_PATHS_LIST
                        path of annotation paths list. It is not need when use --ann_dir and --ann_ids
  --labels LABELS       path to label list.
  --output OUTPUT       path to output json file
  --ext EXT             additional extension of annotation file
```

The below command is an example to run the converter:

```sh
python tools/voc2coco.py \
    --ann_dir data/annotations/ \
    --output data/annotations/test.json \
    --ann_ids test_ids.txt \
    --labels labels.txt \
    --ext xml
```

## Correct COCO Dataset

The [coco_corrector.py](coco_corrector.py) contains the script to correct the
COCO dataset to use relative image paths. This should not be required now with
the final release of the dataset

### Usage

The below command runs this script:

```sh
cd tools
python coco_corrector.py
```