_base_ = ["../_base_/openseg_runtime.py"]
# misc custom setting
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True
find_unused_parameters = True

unknown_label = [5, 9]

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="ST-v1m1",
        downsample_scale=8,
        depths=[2, 2, 6, 2],
        channels=[48, 96, 192, 384],
        num_heads=[3, 6, 12, 24],
        window_size=[0.16, 0.32, 0.64, 1.28],
        up_k=3,
        grid_sizes=[0.04, 0.08, 0.16, 0.32],
        quant_sizes=[0.01, 0.02, 0.04, 0.08],
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.3,
        num_layers=4,
        concat_xyz=True,
        num_classes=13,
        ratio=0.25,
        k=16,
        prev_grid_size=0.04,
        sigma=1.0,
        stem_transformer=True,
        kp_ball_radius=0.04 * 2.5,
        kp_max_neighbor=34,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

model_hooks = dict(
    type="ModelHook",
    register_module_name={
        "backbone.upsamples.0": ["forward_input", "forward_output"],
        "backbone.upsamples.1": ["forward_input", "forward_output"],
        "backbone.upsamples.2": ["forward_input", "forward_output"],
        "backbone.upsamples.3": ["forward_input", "forward_output"],
        "backbone": "forward_output",
    },
    exclude_clone={"backbone": "forward_output"},
)

# recognizer settings
recognizer = dict(
    type="PointPdf-v1m1",
    recognizer=dict(
        type="ST-v1m1-Recognizer",
        up_k=3,
        channels=[48, 96, 192, 384],
        num_layers=4,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    loss_weight=0.008,
    step_loss_weight=False,
    num_classes=20,
    start_epoch=61,
    kp_ball_radius=0.04 * 5,
    kp_max_neighbor=34,
    condition_from="msp",
    beta=2,
    seed_from="ml",
    seed_range=0.05,
    num_seed=50,
    slide_window=True,
    adaptive_radius=False,
)


# scheduler settings
epoch = 3000
param_dicts = [dict(keyword="blocks", lr=0.006 * 0.1)]
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.01)
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
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.005),
            # dict(type="HueSaturationTranslation", hue_max=0.5, saturation_max=0.2),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
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
