norm_cfg = dict(type="BN", requires_grad=False)

model = dict(
    type="TridentFasterRCNN",
    backbone=dict(
        type="TridentResNet",
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[16],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="TridentRoIHead",
        shared_head=dict(
            type="ResLayer",
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style="caffe",
            norm_cfg=dict(type="BN", requires_grad=False),
            norm_eval=True,
        ),
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16],
        ),
        bbox_head=dict(
            type="BBoxHead",
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            num_classes=5,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        num_branch=3,
        test_branch_idx=1,
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=500,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
        ),
    ),
)

dataset_type = "COCODataset"

data_root = "data/coco/"

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[
            (1333, 640),
            (1333, 672),
            (1333, 704),
            (1333, 736),
            (1333, 768),
            (1333, 800),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Normalize",
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        ann_file="data/coco/train.json",
        img_prefix="data/images/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="Resize",
                img_scale=[
                    (1333, 640),
                    (1333, 672),
                    (1333, 704),
                    (1333, 736),
                    (1333, 768),
                    (1333, 800),
                ],
                multiscale_mode="value",
                keep_ratio=True,
            ),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
        classes=("Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"),
    ),
    val=dict(
        type="CocoDataset",
        ann_file="data/coco/test.json",
        img_prefix="data/images/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=("Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"),
    ),
    test=dict(
        type="CocoDataset",
        ann_file="data/coco/test.json",
        img_prefix="data/images/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=("Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"),
    ),
)

evaluation = dict(interval=1, metric="bbox")

optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[28, 34]
)

runner = dict(type="EpochBasedRunner", max_epochs=36)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50, hooks=[dict(type="TensorboardLoggerHook"), dict(type="TextLoggerHook")]
)

custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")

log_level = "INFO"

load_from = None

resume_from = None

workflow = [("train", 1)]

classes = ("Coverall", "Face_Shield", "Gloves", "Goggles", "Mask")
