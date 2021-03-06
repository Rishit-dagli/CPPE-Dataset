# Model Zoo

This is an exhaustive list of pre-trained models trained on the CPPE-5
(Medical Personal Protective Equipment) dataset that are currently available and
you could start using any of these models very easily. We also include
tensorbaord.dev logs for each model as well as a link to the PyTorch model
(`.pth`) and a TensorFlow SavedModel (`.tar.gz` containing the SavedModel
directory) files.

#### Notes:

- All the models can also be trained from scratch using the configs provided in the `configs` or `baselines` directory.
- The metrics are the same as the COCO object detection metrics, more information about this could be found in the original paper.
- The PyTorch models use the channels first format. The TensorFlow models use the channels last format.
- The tb.dev dashboards represent the tensorbaord logs for each model training.
- FPS benchmark for models here could be found in the original paper, which are measured on 1 Tesla V100 GPU.
- These models are trained on either a cluster of TPUs or on multiple Tesla A100 GPUs.
- More information about model complexity and size can be found in the original paper..

## Baseline Models

This section contains the baseline models that are trained on the CPPE-5 dataset
. More information about how these are trained could be found in the original
paper and the config files.

|   Method    | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sub>S</sub><sup>box</sup> | AP<sub>M</sub><sup>box</sup> | AP<sub>L</sub><sup>box</sup> | Configs | TensorBoard.dev | PyTorch model | TensorFlow model |
|:-----------:|:--------------------------:|:---------------------------------------:|:---------------------------------------:|:----------------------------------------:|:----------------------------------------:|:----------------------------------------:|:-------:|:------:|:-------:|:------:|
|     SSD     |           29.50            |                  57.0                   |                  24.9                   |                   32.1                   |                   23.1                   |                   34.6                   | [config](baselines/ssd.config) | [tb.dev](https://tensorboard.dev/experiment/2EimzQz9Q4GCJjYsyo1MKQ/) | [bucket]() | [bucket](https://storage.googleapis.com/cppe-5/trained_models/ssd/tf_ssd.tar.gz) |
|    YOLO     |            38.5            |                  79.4                   |                  35.3                   |                   23.1                   |                   28.4                   |                   49.0                   | [config](baselines/yolov3_d53_mstrain-608_273e_coco.py) | [tb.dev](https://tensorboard.dev/experiment/5JrpU22hRnOOOXCLKvxFyQ) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/yolo/yolov3_d53_608_273e-2942d1ca.pth) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/yolo/yolo.tar.gz) |
| Faster RCNN |            44.0            |                  73.8                   |                  47.8                   |                   30.0                   |                   34.7                   |                   52.5                   | [config](baselines/faster_rcnn_r101_fpn_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/20XQ37HgQUyMJuOlbqmVDQ/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/faster_rcnn/faster_rcnn_r101_fpn_2x_coco-77efa99b.pth) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/faster_rcnn/faster_rcnn.tar.gz) |

## SOTA Models

This section contains the SOTA models that are trained on the CPPE-5 dataset
. More information about how these are trained could be found in the original
paper and the config files.

|           Method           | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sub>S</sub><sup>box</sup> | AP<sub>M</sub><sup>box</sup> | AP<sub>L</sub><sup>box</sup> | Configs | TensorBoard.dev                                                      | PyTorch model                                                                                                                                  | TensorFlow model                                                                               |
|:--------------------------:|:----------:|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------------:|:------------:|:----------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
|         RepPoints          |    43.0    |        75.9       |        40.1       |       27.3       |       36.7       |       48.0       | [config](configs/reppoints_moment_r50_fpn_gn_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/Co6JQVe1RDmxgbMx4gD0Qg/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/reppoints/reppoints_moment_r50_fpn_gn_2x_coco-18beef36.pth)                      |                                                -                                               |
|        Sparse RCNN         |    44.0    |        69.6       |        44.6       |       30.0       |       30.6       |       54.7       | [config](configs/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/se3w7zQ7SlyE6T8q59P79w/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.pth) |                                                -                                               |
|            FCOS            |    44.4    |        79.5       |        45.9       |       36.7       |       39.2       |       51.7       | [config](configs/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/O343s1kRQIKTqs508jESDA/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-031dc428.pth)           | [bucket](https://storage.googleapis.com/cppe-5/trained_models/fcos/tf_fcos.tar.gz)             |
|         Grid RCNN          |    47.5    |        77.9       |        50.6       |       43.4       |       37.2       |       54.4       | [config](configs/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/fgGkJ4IBSZmDQj1QEKgXqA/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco-65319c19.pth)                 |                                                -                                               |
|      Deformable DETR       |    48.0    |        76.9       |        52.8       |       36.4       |       35.2       |       53.9       | [config](configs/deformable_detr_refine_r50_16x2_50e_coco.py) | [tb.dev](https://tensorboard.dev/experiment/uq80boznQY2iJVhWSXAKTw/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/deformable_detr/deformable_detr_refine_r50_16x2_50e-d36a2db1.pth)                |                                                -                                               |
|            FSAF            |    49.2    |        84.7       |        48.2       |       45.3       |       39.6       |       56.7       | [config](configs/fsaf_x101_64x4d_fpn_1x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/jUa0QjFJQZe68o4vbP194Q/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/fsaf/fsaf_x101_64x4d_fpn_1x_coco-7284d216.pth)                                   | [bucket](https://storage.googleapis.com/cppe-5/trained_models/fsaf/tf_fsaf.tar.gz)             |
| Localization Distillation  |    50.9    |        76.5       |        58.8       |       45.8       |       43.0       |       59.4       | [config](configs/ld_r50_gflv1_r101_fpn_coco_1x.py) | [tb.dev](https://tensorboard.dev/experiment/UMGK5cbATVSDZM5DKN1QAA/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/ld/ld_r50_gflv1_r101_fpn_coco_1x-e12b2422.pth)                                   |                                                -                                               |
|        VarifocalNet        |    51.0    |        82.6       |        56.7       |       39.0       |       42.1       |       58.8       | [config](configs/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/bE7LlxNLRU2nGanjxEs2rg/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco-8d841df9.pth)                  |                                                -                                               |
|           RegNet           |    51.3    |        85.3       |        51.8       |       35.7       |       41.1       |       60.5       | [config](configs/faster_rcnn_regnetx-3.2GF_fpn_2x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/eYyj3lwcR5O3XDbuyFZ81Q/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/regnet/regnet-4GF-987ef260.pth)                                                  | [bucket](https://storage.googleapis.com/cppe-5/trained_models/regnet/regnet.tar.gz)            |
|        Double Heads        |    52.0    |        87.3       |        55.2       |       38.6       |       41.0       |       60.8       | [config](configs/dh_faster_rcnn_r50_fpn_1x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/cLMEyMJEQPqWXWeW4XpRkA/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/double_heads/dh_faster_rcnn_r50_fpn_1x_coco-b10cef7a.pth)                        |                                                -                                               |
|            DCN             |    51.6    |        87.1       |        55.9       |       36.3       |       41.4       |       61.3       | [config](configs/faster_rcnn_r50_fpn_mdpool_1x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/GWTGBFo5TruxPlazzkIpXQ/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco-1d85638a.pth)                             |                                                -                                               |
|     Empirical Attention    |    52.5    |        86.5       |        54.1       |       38.7       |       43.4       |       61.0       | [config](configs/) | [tb.dev](https://tensorboard.dev/experiment/56OgPsWLTWe1jhAV1i00iw/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco-f69549ae.pth) |                                                -                                               |
|         TridentNet         |    52.9    |        85.1       |        58.3       |       42.6       |       41.3       |       62.6       | [config](configs/tridentnet_r50_caffe_mstrain_3x_coco.py) | [tb.dev](https://tensorboard.dev/experiment/9O0MAFnlRMWWezz1TbLYGQ/) | [bucket](https://storage.googleapis.com/cppe-5/trained_models/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco-eb569217.pth)                    | [bucket](https://storage.googleapis.com/cppe-5/trained_models/tridentnet/tf_tridentnet.tar.gz) |

## Model Complexity

In this section we proide a comparision between model complexities for the aformentioned models.

|          Method           |      AP<sup>box</sup>      | #Params  |   FLOPs   | FPS  |
|:-------------------------:|:--------------------------:|:--------:|:---------:|:----:|
|            SSD            |            29.5            | 64.34 M  | 103.216 G | 25.6 |
|           YOLO            |            38.5            | 61.55 M  | 193.93 G  | 48.1 |
|         RepPoints         |            43.0            |  36.6 M  | 189.83 G  | 18.8 |
|        Faster RCNN        |            44.0            | 60.14 M  | 282.75 G  | 15.6 |
|        Sparse RCNN        |            44.0            | 124.99 M | 241.53 G  | 21.7 |
|           FCOS            |            44.4            |  50.8 M  | 272.93 G  | 9.7  |
|         Grid RCNN         |            47.5            | 121.98 M | 553.44 G  | 7.7  |
|      Deformable DETR      |            48.0            |  40.5 M  | 195.47 G  | 18.8 |
|           FSAF            |            49.2            | 93.75 M  | 435.88 G  | 5.6  |
| Localization Distillation |            50.9            | 32.05 M  | 204.71 G  | 19.5 |
|       VarifocalNet        |            51.0            | 53.54 M  | 180.05 G  | 4.8  |
|          RegNet           |            51.3            |  31.5 M  | 183.29 G  | 18.2 |
|       Double Heads        |            52.0            | 148.7 M  | 220.05 G  | 9.5  |
|            DCN            |            51.6            | 148.71 M | 219.97 G  | 16,6 |
|    Empirical Attention    |            52.5            | 47.63 M  | 185.83 G  | 12.7 |
|        TridentNet         |            52.9            |  32.8 M  | 822.13 G  | 4.2  |
