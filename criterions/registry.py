import torch.nn as nn


CRITERION_REGISTRY = {}


def register_criterion(name):
    def decorator(fn):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Criterion `{name}` is already registered")
        CRITERION_REGISTRY[name] = fn
        return fn
    return decorator


def build_criterion(cfg) -> nn.Module:
    name = cfg.CRITERION.NAME
    if name not in CRITERION_REGISTRY:
        raise KeyError(f"Criterion `{name}` not found in registry. "
                       f"Available: {list(CRITERION_REGISTRY.keys())}")
    return CRITERION_REGISTRY[name](cfg=cfg)
