base_ckpt = None  # path to base model weight
resume = False  # whether to resume incremental training process from base weight
incr_ckpt = None  # path to incremental model weight
incr_resume = False  # whether to resume incremental training process
load_base_weight_to_incr_learner = True # whether to load base model weight to incremental model
base_weight_process_func = "reserve_matched" # method to adapt base model weights for incremental model loading

evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = 2024  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch

sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
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
        tracked_epoch=float("inf"),
    ),
    # dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="IncrSegTrainer")

# Tester
test = dict(type="IncrSegTester", verbose=True)
