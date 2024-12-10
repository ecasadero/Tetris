# config.py

import torch

# Hyperparameters for DQN
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200000
TARGET_UPDATE = 10
NUM_EPISODES = 500
MEMORY_CAPACITY = 100000
LEARNING_RATE = 1e-4

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Environment configuration
RENDER_MODE = True  # Set True to render the game
SAVE_MODEL = True
MODEL_SAVE_PATH = 'tetris_dqn_model.pth'

# Other configurations (if any)