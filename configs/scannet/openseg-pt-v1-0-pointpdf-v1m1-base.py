_base_ = ["../_base_/openseg_runtime.py"]

# misc custom setting
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

unknown_label = [4, 7, 14, 16]

model_hooks = dict(
    type="ModelHook",
    hook_config={
        "backbone.enc1":"forward_output",
        "backbone.enc2":"forward_output",
        "backbone.enc3":"forward_output",
        "backbone.enc4":"forward_output",
        "backbone.enc5":"forward_output",
        "backbone.dec5.1":"forward_output",  # the last layer of dec sequence
        "backbone.dec4.1":"forward_output",
        "backbone.dec3.1":"forward_output",
        "backbone.dec2.1":"forward_output",
        "backbone.dec1.1":"forward_output",
        "backbone":"forward_output",
        }
    exclude_clone={"backbone":"forward_output"}
)

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=9,
        num_classes=20,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# recognizer settings
recognizer = dict(
    type="PointPdf-v1m1",
    recognizer=dict(type="PointTransformer-Recognizer"),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    loss_weight=0.04,
    step_loss_weight=False,
    num_classes=20,
    start_epoch=61,
    use_existing_nn=False,
    kp_ball_radius=0.02 * 5,
    kp_max_neighbor=64,
    condition_from="msp",
    beta=1.5,
    seed_from="ml",
    seed_range=0.15,
    num_seed=100,
    slide_window=True,
    adaptive_radius=False,
)

# scheduler settings
epoch = 900
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "segment_known"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "segment_known"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    # test=dict(
    #     type=dataset_type,
    #     split="val",
    #     data_root=data_root,
    #     transform=[
    #         dict(type="CenterShift", apply_z=True),
    #         dict(type="NormalizeColor"),
    #     ],
    #     test_mode=True,
    #     test_cfg=dict(
    #         voxelize=dict(
    #             type="GridSample",
    #             grid_size=0.02,
    #             hash_type="fnv",
    #             mode="test",
    #             keys=("coord", "color", "normal"),
    #         ),
    #         crop=None,
    #         post_transform=[
    #             dict(type="CenterShift", apply_z=False),
    #             dict(type="ToTensor"),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "index"),
    #                 feat_keys=("coord", "color", "normal"),
    #             ),
    #         ],
    #         aug_transform=[
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[0],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 )
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 )
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 )
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[3 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 )
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[0],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[3 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[0],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[1],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #             ],
    #             [
    #                 dict(
    #                     type="RandomRotateTargetAngle",
    #                     angle=[3 / 2],
    #                     axis="z",
    #                     center=[0, 0, 0],
    #                     p=1,
    #                 ),
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #             ],
    #             [dict(type="RandomFlip", p=1)],
    #         ],
    #     ),
    # ),
)
