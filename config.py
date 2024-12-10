# config.py

import torch

# Hyperparameters for DQN
BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 40000
TARGET_UPDATE = 10
NUM_EPISODES = 20000
MEMORY_CAPACITY = 1000000
LEARNING_RATE = 1e-4

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Environment configuration
RENDER_MODE = True  # Set True to render the game
SAVE_MODEL = True
MODEL_SAVE_PATH = 'tetris_dqn_model1.pth'

