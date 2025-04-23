# Preprocessing configuration






#  Model configurations



# Training configuration
LR = 0.001
WEIGHT_DECAY = 3e-5
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE_M = 10
ALPHA_M = 0.5
BETA_M = 0.5
K_FOLD = 5
DATASET_DIR = "dataset/processed_data"


## Scheduler parameters
MODE='min'
FACTOR=0.1
PATIENCE_S=2
THRESHOLD=1e-4
THRESHOLD_MODE='rel'
COOLDOWN=0
MIN_LR=0
EPS=1e-7
VERBOSE=True

## Tversky hyperparameters
ALPHA_T = 0.5 
BETA_T  = 0.5   


INN_CHANNEL = 2
OUT_CHANNEL = 1

