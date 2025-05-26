import torch
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

""" Transformer """
_C.MODEL.TX = CN()
_C.MODEL.TX.VOCAB_SIZE = 10000
_C.MODEL.TX.DIM = 512
_C.MODEL.TX.N_LAYERS = 6
_C.MODEL.TX.N_HEADS = 8
_C.MODEL.TX.N_KV_HEADS = 8
_C.MODEL.TX.BASE = 10000
_C.MODEL.TX.MAX_SEQ_LEN = 256
_C.MODEL.TX.MULTIPLE_OF = 256
_C.MODEL.TX.FFN_DIM_MULTIPLIER = None
_C.MODEL.TX.NORM_EPS = 1e-5


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "models_avgpool"

""" Transformer """
_C.CKPT.TX = CN()
_C.CKPT.TX.BEST = _C.CKPT.DIR + "/tx_best.ckpt"
_C.CKPT.TX.LAST = _C.CKPT.DIR + "/tx_last.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()

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

""" CosineLRScheduler """
# set learning rate scheduler parameters in training
""" LinearLRScheduler """
# set learning rate scheduler parameters in training
""" StepLRScheduler """
# set learning rate scheduler parameters in training


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()

""" InfoNCE """
_C.CRITERION.INFONCE = CN()
_C.CRITERION.INFONCE.TEMPERATURE = 0.1
_C.CRITERION.INFONCE.REDUCTION = "mean"

""" MaxSim """
_C.CRITERION.MAXSIM = CN()
_C.CRITERION.MAXSIM.TEMPERATURE = 0.1
_C.CRITERION.MAXSIM.REDUCTION = "mean"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" Formulas """
_C.DATA.DATA_DIR = "data"
_C.DATA.VOCAB_FILE = _C.DATA.DATA_DIR + "/vocabs.txt"
# _C.DATA.FORMULA_FILE = _C.DATA.DATA_DIR + "/formulas.txt"
_C.DATA.FORMULA_FILE = "/projects/illinois/eng/ece/kani/user/suyuan2/data/train_set_small.txt"
_C.DATA.VAL_FILE = _C.DATA.DATA_DIR + "/exprs_val.txt"


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 96
_C.LOADER.TRAIN.SHUFFLE = False
_C.LOADER.TRAIN.NUM_WORKERS = 1
_C.LOADER.TRAIN.PIN_MEMORY = True

""" Val DataLoader """
_C.LOADER.VAL = CN()
_C.LOADER.VAL.BATCH_SIZE = 256
_C.LOADER.VAL.SHUFFLE = False
_C.LOADER.VAL.NUM_WORKERS = 1
_C.LOADER.VAL.PIN_MEMORY = True


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

""" Optimizer """
_C.TRAIN.OPTIM = CN()
_C.TRAIN.OPTIM.NAME = "adamw"
_C.TRAIN.OPTIM.BASE_LR = 1e-4
_C.TRAIN.OPTIM.WARMUP_LR = 1e-7
_C.TRAIN.OPTIM.MIN_LR = 1e-6

""" LR Scheduler """
_C.TRAIN.LRS = CN()
_C.TRAIN.LRS.NAME = "cosine"
# epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LRS.DECAY_EPOCHS = 5
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LRS.DECAY_RATE = 0.1

""" Training """
_C.TRAIN.MAX_NORM = 5.0
_C.TRAIN.N_ITER_PER_EPOCH = 13013
_C.TRAIN.WARMUP_EPOCHS = 2
_C.TRAIN.N_EPOCHS = 20
_C.TRAIN.SAVE_N_ITERS = 500
_C.TRAIN.STATS_FILEPATH = "stats.json"


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
