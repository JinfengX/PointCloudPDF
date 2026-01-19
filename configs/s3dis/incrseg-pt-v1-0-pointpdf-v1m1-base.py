_base_ = ["../_base_/incrseg_runtime.py"]
# misc custom setting
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

unknown_label = [5, 9]
incr_label_remap = {5: 13, 9: 14}
incr_label_select = [5, 9]

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=13,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

model_hooks = dict(
    type="ModelHook",
    hook_config={
        "backbone": "forward_output",
    },
    exclude_clone={"backbone": "forward_output"},
)

# recognizer settings
# recognizer = dict(
#     type="PointPdf-v2m1",
#     recognizer_model=dict(
#         type="PTV1-Recognizer-Light",
#         in_channels=6,
#     ),
#     num_classes=13,
#     grid_size=0.4,
#     warmup_epoch=10,
#     start_search_epoch=51,
#     dim_feat=13,
#     unknown_label=unknown_label,
#     criteria=[dict(type="CrossEntropyLoss", loss_weight=0.2, ignore_index=-1)],
# )

incremental_learner = dict(
    type="PointPdf-incr-v1m1",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=13 + len(incr_label_remap),
    ),
    eval_criteria=[dict(type="CrossEntropyLoss", loss_weight=1, ignore_index=-1)],
)


# scheduler settings
epoch = 300
eval_epoch = 300
optimizer = dict(type="SGD", lr=0.5, momentum=0.9, weight_decay=0.0001)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis"

hooks = [
    dict(type="IncrSegCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="IncrSegEvaluator"),
    dict(
        type="IncrSegCheckpointSaver",
        save_freq=5,
        tracked_best_metrics=[
            "mIoU_known",
            "mIoU_incr",
            "mIoU_remap",
        ],
        tracked_epoch=60,
    ),
]

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.005),
            dict(type="HueSaturationTranslation", hue_max=0.5, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ShufflePoint"),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor", mode="zeroOne"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="RemapLabel", remap_dict=incr_label_remap),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "segment_known",
                    "segment_incr",
                    "segment_incr_remap",
                ),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="CenterShift", apply_z=False),
            dict(type="SphereCrop", point_max=800000, mode="center"),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor", mode="zeroOne"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="RemapLabel", remap_dict=incr_label_remap),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "segment_known",
                    "segment_incr",
                    "segment_incr_remap",
                ),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[dict(type="PositiveShift"), dict(type="NormalizeColor", mode="zeroOne")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="PositiveShift"),
                # dict(type="CenterShift", apply_z=False),
                # dict(type="RemapLabel", remap_dict=incr_label_remap),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                # [dict(type="RandomScale", scale=[0.9, 0.9])],
                # [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                # [dict(type="RandomScale", scale=[1.05, 1.05])],
                # [dict(type="RandomScale", scale=[1.1, 1.1])],
                # [
                #     dict(type="RandomScale", scale=[0.9, 0.9]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                # [
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [
                #     dict(type="RandomScale", scale=[1.1, 1.1]),
                #     dict(type="RandomFlip", p=1),
                # ],
            ],
        ),
    ),
)
