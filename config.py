import os
import torch
import yaml
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = None

""" Bert """
_C.MODEL.BERT = CN()
_C.MODEL.BERT.ADD_POOLING = False
_C.MODEL.BERT.REDUCE_DIM = False
_C.MODEL.BERT.DIM = None

""" Transformer (Math) """
_C.MODEL.MATH_ENC = CN()
_C.MODEL.MATH_ENC.TOK_EMB = None
_C.MODEL.MATH_ENC.VOCAB_SIZE = None
_C.MODEL.MATH_ENC.DIM = 768
_C.MODEL.MATH_ENC.N_LAYERS = 8
_C.MODEL.MATH_ENC.N_HEADS = 12
_C.MODEL.MATH_ENC.N_KV_HEADS = 12
_C.MODEL.MATH_ENC.MULTIPLE_OF = 256
_C.MODEL.MATH_ENC.FFN_DIM_MULTIPLIER = None
_C.MODEL.MATH_ENC.NORM_EPS = 1e-5
_C.MODEL.MATH_ENC.THETA = 10000
_C.MODEL.MATH_ENC.MAX_SEQ_LEN = 256


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "saved_models"
_C.CKPT.BEST = _C.CKPT.DIR + "/best.ckpt"  # not used
_C.CKPT.LAST = _C.CKPT.DIR + "/last.ckpt"

""" Bert """
_C.CKPT.BERT = CN()
_C.CKPT.BERT.CFG = "pretrain/bert_cfg.json"
_C.CKPT.BERT.PRETRAIN = "pretrain/bert.pt"
_C.CKPT.BERT.TOKENIZER = "pretrain/bert-math-tokenizer"

""" Math Encoder """
_C.CKPT.MATH_ENC = CN()
# _C.CKPT.MATH_ENC.PRETRAIN = _C.CKPT.PRETRAIN_DIR + "/math-enc.ckpt"
_C.CKPT.MATH_ENC.PRETRAIN = None


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.NAME = None
_C.OPTIM.BASE_LR = 1e-4
_C.OPTIM.WARMUP_LR = 1e-7
_C.OPTIM.MIN_LR = 1e-6

""" SGD """
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0.90
_C.OPTIM.SGD.WEIGHT_DECAY = 0.05
_C.OPTIM.SGD.NESTEROV = True

""" AdamW """
_C.OPTIM.ADAMW = CN()
_C.OPTIM.ADAMW.BETAS = (0.9, 0.999)
_C.OPTIM.ADAMW.EPS = 1e-8
_C.OPTIM.ADAMW.WEIGHT_DECAY = 1e-2


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
_C.LRS = CN()
_C.LRS.NAME = None

""" CosineLRScheduler """
# set learning rate scheduler parameters in training
""" LinearLRScheduler """
# set learning rate scheduler parameters in training
""" StepLRScheduler """
_C.LRS.STEP_LRS = CN()
_C.LRS.STEP_LRS.DECAY_RATE = 0.1


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()
_C.CRITERION.NAME = None

""" InfoNCE """
_C.CRITERION.INFONCE = CN()
_C.CRITERION.INFONCE.TEMPERATURE = 0.1
_C.CRITERION.INFONCE.REDUCTION = "mean"

""" MaxSim """
_C.CRITERION.MAXSIM = CN()
_C.CRITERION.MAXSIM.TEMPERATURE = 0.1
_C.CRITERION.MAXSIM.REDUCTION = "mean"


# -----------------------------------------------------------------------------
# Postprocess
# -----------------------------------------------------------------------------
_C.POSTPROCESS = CN()
_C.POSTPROCESS.NAME = None


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = None
_C.DATA.ARQMATH = None
_C.DATA.MATH = None
_C.DATA.N_EXPRS = None


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 64
_C.LOADER.TRAIN.SHUFFLE = False
_C.LOADER.TRAIN.NUM_WORKERS = 1
_C.LOADER.TRAIN.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Hyperparams
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
LOG_LEVEL = LogLevel.INFO


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

""" Training """
# epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.DECAY_EPOCHS = 3
_C.TRAIN.MAX_NORM = 1.0
_C.TRAIN.N_ITER_PER_EPOCH = 312500
_C.TRAIN.WARMUP_EPOCHS = 0.016
_C.TRAIN.N_EPOCHS = 3
_C.TRAIN.SAVE_N_ITERS = 1000
_C.TRAIN.STATS_FILEPATH = "stats.json"


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f'[INFO] merge config from `{cfg_file}`')
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    _update_config_from_file(config, args.dataset)

    # cfg.defrost()
    # if args.opts:
    #     cfg.merge_from_list(args.opts)

    # cfg.freeze()


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
