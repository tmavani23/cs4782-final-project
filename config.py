import os
import torch

# paths
DATA_DIR       = "data"
IMAGE_DIR      = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE  = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = "checkpoints"

# preprocessing
IMAGE_SIZE         = 224
MAX_CAPTION_LENGTH = 52
MIN_WORD_FREQ      = 5

# train/val/test split fractions
TRAIN_FRAC = 0.75
VAL_FRAC   = 0.125

# special tokens
PAD_TOKEN   = "<pad>"
START_TOKEN = "<start>"
END_TOKEN   = "<end>"
UNK_TOKEN   = "<unk>"
PAD_IDX     = 0
START_IDX   = 1
END_IDX     = 2
UNK_IDX     = 3

# model dimensions (from paper, Flickr8k)
ENCODER_DIM   = 512
ATTENTION_DIM = 512
EMBED_DIM     = 512
DECODER_DIM   = 512
DROPOUT       = 0.5

# training
BATCH_SIZE  = 32
EPOCHS      = 120
LR          = 4e-4
GRAD_CLIP   = 5.0
ALPHA_C     = 1.0   # doubly stochastic attention regularizer weight
PATIENCE    = 20    # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
