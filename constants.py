# constants.py

# Audio processing constants
FRAME_SIZE = 1024
HOP_LENGTH = 512
DURATION = 7  # in seconds
SAMPLE_RATE = 16000
MONO = True

# Training constants
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Directory paths
CLEAN_SPECTROGRAMS_SAVE_DIR = "../input/spectrograms/clean/"
NOISY_SPECTROGRAMS_SAVE_DIR = "../input/spectrograms/noise/"
MIN_MAX_VALUES_SAVE_DIR = "../input/"
CLEAN_SPEECH_DIR = "../dataset/clean_speech"
NOISY_SPEECH_DIR = "../dataset/noisy_speech"

# Model constants
MODEL_INPUT_DIR = "input/spectrograms/"
MODEL_MIN_MAX_PATH = "input/min_max_values.pkl"

CURRENT_MODEL_NAME = "UNetPhase"

MIN_MAX_NORMALIZER_MIN = -1
MIN_MAX_NORMALIZER_MAX = 1

DATASET_SIZE = 2069