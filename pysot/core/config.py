from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamkpn_hg_all_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Keypoint Target

__C.TRAIN.EXEMPLAR_SIZE = 128

__C.TRAIN.SEARCH_SIZE = 256

__C.TRAIN.BASE_SIZE = 16

__C.TRAIN.STRIDE = 4 

__C.TRAIN.OUTPUT_SIZE = 49

__C.TRAIN.STACK = 1

__C.TRAIN.FOCAL_NEG = 1.0

__C.TRAIN.OPTI = 'adam'

__C.TRAIN.EVAL = False

__C.TRAIN.DIF_STD = False

__C.TRAIN.CROP_TEMPLATE = True

__C.TRAIN.INTER_SUPER = False

__C.TRAIN.OFFSETS = True

__C.TRAIN.NORMWH = False

__C.TRAIN.SAMEOFF = False

__C.TRAIN.RESUME = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0

__C.TRAIN.HM_WEIGHT = 1.0

__C.TRAIN.OF_WEIGHT = 1.0

__C.TRAIN.WH_WEIGHT = 0.1

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'MOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.VAL_ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.VAL_ANNO = 'training_dataset/vid/val.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000     # repeat until reach NUM_USE
__C.DATASET.VID.VAL_NUM_USE = 10000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/crop/train_new.json'
__C.DATASET.YOUTUBEBB.VAL_ROOT = 'training_dataset/yt_bb/crop/crop511'
__C.DATASET.YOUTUBEBB.VAL_ANNO = 'training_dataset/yt_bb/crop/val_new.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1         # use all not repeat
__C.DATASET.YOUTUBEBB.VAL_NUM_USE = 20000  # use all not repeat

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = 'training_dataset/lasot/crop511'
__C.DATASET.LASOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 300
__C.DATASET.LASOT.NUM_USE = -1  # use all not repeat

__C.DATASET.TRANET = CN()
__C.DATASET.TRANET.ROOT = 'training_dataset/trackingnet/crop511'
__C.DATASET.TRANET.ANNO = 'training_dataset/trackingnet/train.json'
__C.DATASET.TRANET.FRAME_RANGE = 100
__C.DATASET.TRANET.NUM_USE = -1  # use all not repeat

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = 'training_dataset/got10k/crop511'
__C.DATASET.GOT10K.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 100
__C.DATASET.GOT10K.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.VAL_ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.VAL_ANNO = 'training_dataset/coco/val2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1
__C.DATASET.COCO.VAL_NUM_USE = 5000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.VAL_ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.VAL_ANNO = 'training_dataset/det/val.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1
__C.DATASET.DET.VAL_NUM_USE = 15000

__C.DATASET.VIDEOS_PER_EPOCH = 600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'HourglassNet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']
# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# KPN options
# ------------------------------------------------------------------------ #
__C.KPN = CN()

# RPN type
__C.KPN.TYPE = 'DepthwiseKPN'

__C.KPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamKPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

__C.TRACK.TH = 0.15

__C.TRACK.OFFSETS = False

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

