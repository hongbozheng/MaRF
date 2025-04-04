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
_C.MODEL.TX.MAX_SEQ_LEN = 2048
_C.MODEL.TX.MULTIPLE_OF = 256
_C.MODEL.TX.FFN_DIM_MULTIPLIER = None
_C.MODEL.TX.NORM_EPS = 1e-5


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "models"

""" Mamba """
_C.CKPT.TX = CN()
_C.CKPT.TX.BEST = _C.CKPT.DIR + "/tx_best.ckpt"
_C.CKPT.TX.LAST = _C.CKPT.DIR + "/tx_last.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()

""" AdamW """
_C.OPTIM.ADAMW = CN()
_C.OPTIM.ADAMW.LR = 1e-4
_C.OPTIM.ADAMW.WEIGHT_DECAY = 1e-2


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
_C.LRS = CN()

""" CosineAnnealingWarmRestarts """
_C.LRS.CAWR = CN()
_C.LRS.CAWR.T_0 = 10
_C.LRS.CAWR.T_MULT = 2
_C.LRS.CAWR.ETA_MIN = 1e-8
_C.LRS.CAWR.LAST_EPOCH = -1

""" CosineAnnealingLR """
_C.LRS.CALR = CN()
_C.LRS.CALR.T_MAX = 50
_C.LRS.CALR.ETA_MIN = 1e-8
_C.LRS.CALR.LAST_EPOCH = -1


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()

""" InfoNCE """
_C.CRITERION.INFONCE = CN()
_C.CRITERION.INFONCE.TEMPERATURE = 0.1
_C.CRITERION.INFONCE.REDUCTION = "mean"

""" SimCSE """
_C.CRITERION.SIMCSE = CN()
_C.CRITERION.SIMCSE.TEMPERATURE = 0.1
_C.CRITERION.SIMCSE.REDUCTION = "mean"

""" Contrastive Loss """
_C.CRITERION.CL = CN()
_C.CRITERION.CL.MARGIN = 1.0
_C.CRITERION.CL.REDUCTION = "mean"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" Formulas """
_C.DATA.DATA_DIR = "data"
_C.DATA.VOCAB_FILE = _C.DATA.DATA_DIR + "/vocabs.txt"
_C.DATA.FORMULA_FILE = _C.DATA.DATA_DIR + "/formulas.txt"
_C.DATA.TRAIN_FILE = _C.DATA.DATA_DIR + "/expr_pairs.txt"
_C.DATA.VAL_FILE = _C.DATA.DATA_DIR + "/exprs_val.txt"


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 4
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

""" Training """
_C.TRAIN.N_EPOCHS = 50
_C.TRAIN.MAX_NORM = 4.0


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """
START = 25.0
END = 75.0
N = 3
TOL = 1e-10
SECS = 10


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
