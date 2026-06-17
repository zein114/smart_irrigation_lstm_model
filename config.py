import os

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "iotsensordata.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

NUM_FEATURES = 8 

# Fenêtre temporelle augmentée pour mieux capturer les patterns
TIME_STEPS = 20  
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

TOLERANCE_PERCENT = 5.0 