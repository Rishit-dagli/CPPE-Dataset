model = dict(
    type="FasterRCNN",
    backbone=dict(
        type="RegNet",
        arch="regnetx_3.2gf",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://regnetx_3.2gf"),
    ),
    neck=dict(
        type="FPN", in_channels=[96, 192, 432, 1008], out_channels=256, num_outs=5
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
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
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
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
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
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

img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Normalize",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
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
                std=[57.375, 57.12, 58.395],
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
            dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
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
                        std=[57.375, 57.12, 58.395],
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
                        std=[57.375, 57.12, 58.395],
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

optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=5e-05)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[16, 22]
)

runner = dict(type="EpochBasedRunner", max_epochs=24)

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
