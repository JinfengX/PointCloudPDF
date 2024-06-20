_base_ = ["../_base_/openseg_runtime.py"]

# misc custom setting
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True
find_unused_parameters = True

# Open world setting
unknown_label = [4, 7, 14, 16]

# model settings
model = dict(
    type="OpenSegmentor",
    backbone=dict(
        type="ST-v1m1",
        downsample_scale=4,
        depths=[3, 3, 9, 3, 3],
        channels=[48, 96, 192, 384, 384],
        num_heads=[3, 6, 12, 24, 24],
        window_size=[0.1, 0.2, 0.4, 0.8, 1.6],
        up_k=3,
        grid_sizes=[0.02, 0.04, 0.08, 0.16, 0.32],
        quant_sizes=[0.005, 0.01, 0.02, 0.04, 0.08],
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.3,
        num_layers=5,
        concat_xyz=True,
        num_classes=20,
        ratio=0.25,
        k=16,
        prev_grid_size=0.02,
        sigma=1.0,
        stem_transformer=False,
        kp_ball_radius=0.02 * 2.5,
        kp_max_neighbor=34,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

model_hooks = dict(
    type="ModelHook",
    register_module_name=[
        "upsamples.0",
        "upsamples.1",
        "upsamples.2",
        "upsamples.3",
    ],
    hook_and_action=[["forward_getInput", "forward_getOutput"]] * 4,
)

# recognizer settings
# recognizer settings
recognizer = dict(
    type="PseudoLabeler",
    recognizer=dict(
        type="ST-v1m1-Recognizer",
        up_k=3,
        channels=[48, 96, 192, 384, 384],
        num_layers=5,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    loss_weight=0.008,
    step_loss_weight=False,
    num_classes=20,
    start_epoch=61,
    kp_ball_radius=0.02 * 5,
    kp_max_neighbor=64,
    condition_from="msp",
    beta=2,
    seed_from="ml",
    seed_range=0.15,
    num_seed=150,
    slide_window=True,
    adaptive_radius=False,
)

# scheduler settings
epoch = 600
eval_epoch = 100
param_dicts = [dict(keyword="blocks", lr=0.006 * 0.1)]
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="MultiStepWithWarmupLR",
    milestones=[0.6, 0.8],
    gamma=0.1,
    warmup_rate=0.05,
    warmup_scale=1e-6,
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
            # dict(type="CenterShift", apply_z=True),
            # dict(
            # type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ShufflePoint"),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor", mode="zeroOne"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment_known"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=800000, mode="center"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor", mode="zeroOne"),
            dict(type="MaskLabel", mask_label=unknown_label),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "segment_known"),
                feat_keys=("coord", "color"),
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
    #             keys=("coord", "color"),
    #         ),
    #         crop=None,
    #         post_transform=[
    #             dict(type="CenterShift", apply_z=False),
    #             dict(type="ToTensor"),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "index"),
    #                 feat_keys=("coord", "color"),
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
