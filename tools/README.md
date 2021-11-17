## Convert Pascal VOC format to COCO

The [voc2coco.py](voc2coco.py) contains the script to convert the Pascal VOC XML
format to COCO JSON. The dataset annotation XMLs should be stored under `annotations`
directory and the images in the `images` directory. The `test_ids.txt` (or a
text file for any other split) containing a sequence of images names to be
included in the split without the extension.

The below command runs the converter:

```sh
cd tools
python voc2coco.py \
    --ann_dir annotations/ \
    --output coco/test.json \
    --ann_ids test_ids.txt \
    --labels labels.txt \
    --ext xml
```

## Correct COCO Dataset

The [coco_corrector.py](coco_corrector.py) conatins the script to correct the
COCO dataset to use relative image paths.

The below command runs this script:

```sh
cd tools
python coco_corrector.py
```