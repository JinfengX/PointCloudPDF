weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
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
    dict(type="OpensegCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="OpenSegEvaluator"),
    dict(type="OpensegCheckpointSaver", save_freq=10),
    # dict(type="PreciseEvaluator", test_last=False),
    
]
model_hooks = dict() # [ ] update

# Trainer
train = dict(type="OpensegTrainer")

# Tester
# test = dict(type="OpenSegTester", verbose=True)
