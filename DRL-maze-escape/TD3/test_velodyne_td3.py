import time

import numpy as np
import torch

from velodyne_env import GazeboEnv

import random

from test_common import TD3



# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 42  # Random seed number
max_ep = 2000  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()
random_action = []
random_count = 0

# Begin the testing loop
while True:
    action = network.get_action(np.array(state))
    expl_noise = 0.1
    max_action = 1.0
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    if (np.random.uniform(0, 1) > 0.85
        and min(state[4:-8]) < 0.4
        and random_count < 1):
        random_count = random.randint(8, 15)
    if random_count > 0:
        action[0] = -2
        action[1] = np.random.uniform(-1, 1)
        random_count -= 1
    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    momentum = np.random.uniform(0, 0.5)
    a_in = [(action[0] + 1) * 1.5, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # On termination of episode
    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
