from pointcept.utils.registry import Registry

RECOGNIZER = Registry("recognizer")


def build_recognizer(cfg):
    """Build recognizers."""
    return RECOGNIZER.build(cfg)
