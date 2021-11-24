dataset_type = "COCODataset"

data_root = "data/coco/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                {
                    "type": "Resize",
                    "img_scale": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "multiscale_mode": "value",
                    "keep_ratio": True,
                }
            ],
            [
                {
                    "type": "Resize",
                    "img_scale": [(400, 1333), (500, 1333), (600, 1333)],
                    "multiscale_mode": "value",
                    "keep_ratio": True,
                },
                {
                    "type": "RandomCrop",
                    "crop_type": "absolute_range",
                    "crop_size": (384, 600),
                    "allow_negative_crop": True,
                },
                {
                    "type": "Resize",
                    "img_scale": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "multiscale_mode": "value",
                    "override": True,
                    "keep_ratio": True,
                },
            ],
        ],
    ),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
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
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
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
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="AutoAugment",
                policies=[
                    [
                        {
                            "type": "Resize",
                            "img_scale": [
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            "multiscale_mode": "value",
                            "keep_ratio": True,
                        }
                    ],
                    [
                        {
                            "type": "Resize",
                            "img_scale": [(400, 1333), (500, 1333), (600, 1333)],
                            "multiscale_mode": "value",
                            "keep_ratio": True,
                        },
                        {
                            "type": "RandomCrop",
                            "crop_type": "absolute_range",
                            "crop_size": (384, 600),
                            "allow_negative_crop": True,
                        },
                        {
                            "type": "Resize",
                            "img_scale": [
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            "multiscale_mode": "value",
                            "override": True,
                            "keep_ratio": True,
                        },
                    ],
                ],
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
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
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
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
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
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
optimizer = dict(type="AdamW", lr=2.5e-05, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[27, 33]
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

num_stages = 6

num_proposals = 300

model = dict(
    type="SparseRCNN",
    backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_input",
        num_outs=4,
    ),
    rpn_head=dict(
        type="EmbeddingRPNHead", num_proposals=300, proposal_feature_channel=256
    ),
    roi_head=dict(
        type="SparseRoIHead",
        num_stages=6,
        stage_loss_weights=[1, 1, 1, 1, 1, 1],
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
            dict(
                type="DIIHead",
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type="ReLU", inplace=True),
                dynamic_conv_cfg=dict(
                    type="DynamicConv",
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type="ReLU", inplace=True),
                    norm_cfg=dict(type="LN"),
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0],
                ),
            ),
        ],
    ),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
            dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                pos_weight=1,
            ),
        ],
    ),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=300)),
)

min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)

classes = ("Coverall", "Face_Shield", "Gloves", "Goggles", "Mask")
