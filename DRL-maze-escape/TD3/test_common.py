

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fcn_1 = nn.Linear(state_dim, 256)
        self.fcn_2 = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.fcn_1(s))
        a = self.tanh(self.fcn_2(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device='cpu')

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device='cpu')
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )