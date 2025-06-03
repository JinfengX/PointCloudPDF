from pointcept.utils.registry import Registry

INCREMENTALLEARNER = Registry("incremental_learner")


def build_incremental_learner(cfg):
    """Build incremental learner."""
    return INCREMENTALLEARNER.build(cfg)
