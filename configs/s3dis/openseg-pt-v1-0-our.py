_base_ = ["../_base_/openseg_runtime.py"]
# misc custom setting
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

unknown_label = [5, 9]


model_hooks = dict(
    type="ModelHook",
    register_module_name=[
        "enc1",
        "enc2",
        "enc3",
        "enc4",
        "enc5",
        "dec5.1",  # the last layer of dec sequence
        "dec4.1",
        "dec3.1",
        "dec2.1",
        "dec1.1",
    ],
    hook_and_action=["forward_getOutput"] * 10,
)


# model settings
model = dict(
    type="OpenSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=13,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# recognizer settings
recognizer = dict(
    type="PseudoLabeler",
    recognizer=dict(type="PointTransformer-Recognizer"),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    loss_weight=0.01,
    step_loss_weight=False,
    num_classes=13,
    start_epoch=61,
    use_existing_nn=False,
    kp_ball_radius=0.04 * 2.5,
    kp_max_neighbor=34,
    condition_from="msp",
    beta = 1.5,
    seed_from="ml",
    seed_range = 0.01,
    num_seed = 20,
    slide_window=True,
    adaptive_radius=False,
)


# scheduler settings
epoch = 3000
optimizer = dict(type="SGD", lr=0.5, momentum=0.9, weight_decay=0.0001)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis"

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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_known"),
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_known"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=False,
    ),
    # test=dict(
    #     type=dataset_type,
    #     split="Area_5",
    #     data_root=data_root,
    #     transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
    #     test_mode=True,
    #     unknown_label=unknown_label,
    #     test_cfg=dict(
    #         voxelize=dict(
    #             type="GridSample",
    #             grid_size=0.04,
    #             hash_type="fnv",
    #             mode="test",
    #             keys=("coord", "color"),
    #             return_grid_coord=True,
    #         ),
    #         crop=None,
    #         post_transform=[
    #             dict(type="CenterShift", apply_z=False),
    #             dict(type="ToTensor"),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "index"),
    #                 feat_keys=("coord", "color"),
    #             ),
    #         ],
    #         aug_transform=[
    #             [dict(type="RandomScale", scale=[0.9, 0.9])],
    #             [dict(type="RandomScale", scale=[0.95, 0.95])],
    #             [dict(type="RandomScale", scale=[1, 1])],
    #             [dict(type="RandomScale", scale=[1.05, 1.05])],
    #             [dict(type="RandomScale", scale=[1.1, 1.1])],
    #             [
    #                 dict(type="RandomScale", scale=[0.9, 0.9]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
    #             [
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[1.1, 1.1]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #         ],
    #     ),
    # ),
)
