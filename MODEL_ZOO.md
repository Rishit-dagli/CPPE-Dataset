# Model Zoo

This is an exhaustive list of pre-trained models trained on the CPPE-5
(Medical Personal Protective) Equipment dataset that are currently available and
you could start using any of these models very easily. We also include
tensorbaord.dev logs for each model as well as a link to the PyTorch model
(`.pth`) and a TensorFlow SavedModel (`.tar.gz` containing the SavedModel
directory) files.

#### Notes:

- All the models can also be trained from scratch using the configs provided in the `configs` or `baselines` directory.
- The metrics are the same as the COCO object detection metrics, more information about this could be found in the original paper.
- The PyTorch models use the channels first format. The TensorFlow models use the channels last format.
- The tb.dev dashboards represent the tensorbaord logs for each model training.

## Baseline Models

This section contains the baseline models that are trained on the CPPE-5 dataset
. More information about how these are trained could be found in the original
paper and the config files.

|   Method    | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sub>S</sub><sup>box</sup> | AP<sub>M</sub><sup>box</sup> | AP<sub>L</sub><sup>box</sup> | Configs | TensorBoard.dev | PyTorch model | TensorFlow model |
|:-----------:|:--------------------------:|:---------------------------------------:|:---------------------------------------:|:----------------------------------------:|:----------------------------------------:|:----------------------------------------:|:-------:|:------:|:-------:|:------:|
|     SSD     |           29.50            |                  57.0                   |                  24.9                   |                   32.1                   |                   23.1                   |                   34.6                   | [config](baselines/ssd.config) | [tb.dev](https://tensorboard.dev/experiment/2EimzQz9Q4GCJjYsyo1MKQ/) | [bucket]() | [bucket](https://storage.googleapis.com/cppe-5/trained_models/ssd/tf_ssd.tar.gz) |
|    YOLO     |            38.5            |                  79.4                   |                  35.3                   |                   23.1                   |                   28.4                   |                   49.0                   | [config](baselines/yolov3_d53_mstrain-608_273e_coco.py) | [tb.dev](https://tensorboard.dev/experiment/5JrpU22hRnOOOXCLKvxFyQ) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/yolo/yolov3_d53_608_273e-2942d1ca.pth) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/yolo/yolo.tar.gz) |
| Faster RCNN |            44.0            |                  73.8                   |                  47.8                   |                   30.0                   |                   34.7                   |                   52.5                   | [config](baselines/faster_rcnn_r101_fpn_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/20XQ37HgQUyMJuOlbqmVDQ/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/faster_rcnn/faster_rcnn_r101_fpn_2x_coco-77efa99b.pth) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/faster_rcnn/faster_rcnn.tar.gz) |
