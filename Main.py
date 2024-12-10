# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import deque
from tetrisEnvironment import TetrisEnv, start_new_run, end_run, save_episode_reward
import config  # Import config file
from dqnAgent import DQN, ReplayMemory, select_action, optimize_model

import torch
import torch.optim as optim
import numpy as np
from tetrisEnvironment import TetrisEnv, start_new_run, end_run, save_episode_reward
import config  # Import your config file
from dqnAgent import DQN, ReplayMemory, select_action, optimize_model

# Set the device for computations
device = torch.device(config.DEVICE)

def main():
    # Start a new run
    run_id = start_new_run()

    # Initialize the environment
    env = TetrisEnv(render_mode=config.RENDER_MODE, run_id=run_id)
    state_size = len(env.get_state())
    action_size = len(env.action_space)

    # Initialize the policy and target networks
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Set up the optimizer and replay memory
    optimizer = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    memory = ReplayMemory(config.MEMORY_CAPACITY)

    steps_done = 0

    for episode in range(config.NUM_EPISODES):
        # Reset the environment and get initial state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        pieces_placed = 0  # Track number of pieces placed in the game

        while not done:
            # Select action using epsilon-greedy policy
            action = select_action(state, policy_net, steps_done)
            steps_done += 1

            # Perform action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            pieces_placed += 1  # Increment pieces placed

            # Convert reward to tensor
            reward = torch.tensor([reward], dtype=torch.float32).to(device)

            # Convert next_state to tensor if not done
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                next_state = None

            # Store transition in memory
            memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state if next_state is not None else torch.zeros_like(state)

            # Perform optimization
            optimize_model(policy_net, target_net, memory, optimizer)

        # Update the target network periodically
        if episode % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save reward for the episode, including pieces placed
        save_episode_reward(run_id, episode, total_reward, pieces_placed)
        print(f"Episode {episode} finished with reward {total_reward}, pieces placed: {pieces_placed}")



    # Save the trained model if required
    if config.SAVE_MODEL:
        torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)

    # End the current run
    end_run(run_id)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
