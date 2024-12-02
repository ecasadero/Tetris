# dqnAgent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import math
import config  # Import hyperparameters and configurations

# Set the device for computations
device = torch.device(config.DEVICE)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_size)

    def forward(self, x):
        # Forward pass through the network
        x = x.float()  # Ensure the input is float32
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Save a transition to memory
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a batch of transitions
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state, policy_net, steps_done):
    # Epsilon-greedy action selection
    eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * \
        math.exp(-1. * steps_done / config.EPS_DECAY)
    if random.random() < eps_threshold:
        # Explore: select a random action
        action_size = policy_net.out.out_features  # Get number of actions
        return random.randrange(action_size)
    else:
        # Exploit: select the action with max Q-value
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).item()


def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < config.BATCH_SIZE:
        return
    transitions = memory.sample(config.BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    # Convert batches to tensors
    batch_state = torch.cat(batch_state).to(device)  # Shape: [BATCH_SIZE, state_size]
    batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(device)  # Shape: [BATCH_SIZE, 1]
    batch_reward = torch.cat(batch_reward).to(device)  # Shape: [BATCH_SIZE]
    batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)  # Shape: [BATCH_SIZE]

    # Compute Q(s_t, a)
    state_action_values = policy_net(batch_state).gather(1, batch_action).squeeze(1)  # Shape: [BATCH_SIZE]

    # Compute V(s_{t+1}) for all next states
    non_final_mask = (batch_done == 0)  # Shape: [BATCH_SIZE]
    non_final_next_states = torch.cat([s for s, d in zip(batch_next_state, batch_done) if d == 0]).to(device)  # Shape: [N, state_size]

    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)  # Shape: [BATCH_SIZE]
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]  # Shape: [N]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.GAMMA) + batch_reward  # Shape: [BATCH_SIZE]

    # Compute the loss
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clamp gradients to prevent excessive updates
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()