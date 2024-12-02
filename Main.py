# main.py

import torch
import math
from tetris_env import TetrisEnv
from dqn_agent import DQN, ReplayMemory, optimize_model

# Hyperparameters and configurations
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10
NUM_EPISODES = 1000  # Set the number of episodes for training
MEMORY_CAPACITY = 100000

def main():
    # Initialize the environment
    env = TetrisEnv()
    state_size = len(env.get_state())
    action_size = len(env.action_space)

    # Initialize the policy and target networks
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Set up the optimizer and replay memory
    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(MEMORY_CAPACITY)

    steps_done = 0

    for episode in range(NUM_EPISODES):
        state = env.reset()
        state = torch.tensor([state], dtype=torch.float32)
        total_reward = 0

        while True:
            # Select action using epsilon-greedy policy
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if random.random() < eps_threshold:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1).item()

            # Perform action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32)

            if not done:
                next_state = torch.tensor([next_state], dtype=torch.float32)
            else:
                next_state = None

            # Store transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform optimization
            if len(memory) > BATCH_SIZE:
                optimize_model(policy_net, target_net, memory, optimizer)

            if done:
                print(f"Episode {episode} finished with reward {total_reward}")
                break

        # Update the target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    main()
