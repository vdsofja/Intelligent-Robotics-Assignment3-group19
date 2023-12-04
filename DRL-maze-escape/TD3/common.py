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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fcn_1 = nn.Linear(state_dim, 256)
        self.fcn_2_s = nn.Linear(256, 256)
        self.fcn_2_a = nn.Linear(action_dim, 256)
        self.fcn_3 = nn.Linear(256, 1)

        self.fcn_4 = nn.Linear(state_dim, 256)
        self.fcn_5_s = nn.Linear(256, 256)
        self.fcn_5_a = nn.Linear(action_dim, 256)
        self.fcn_6 = nn.Linear(256, 1)

    def forward(self, s, a):
        s1 = F.relu(self.fcn_1(s))
        self.fcn_2_s(s1)
        self.fcn_2_a(a)
        s11 = torch.mm(s1, self.fcn_2_s.weight.data.t())
        s12 = torch.mm(a, self.fcn_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.fcn_2_a.bias.data)
        q1 = self.fcn_3(s1)

        s2 = F.relu(self.fcn_4(s))
        self.fcn_5_s(s2)
        self.fcn_5_a(a)
        s21 = torch.mm(s2, self.fcn_5_s.weight.data.t())
        s22 = torch.mm(a, self.fcn_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.fcn_5_a.bias.data)
        q2 = self.fcn_6(s2)
        return q1, q2