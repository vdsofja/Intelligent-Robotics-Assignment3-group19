"""
数据结构，用于实现经验回放
"""
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()
        self.buffer.append((s, a, r, t, s2))

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, sample_size)

        s_batch = np.array([sample[0] for sample in samples])
        a_batch = np.array([sample[1] for sample in samples])
        r_batch = np.array([sample[2] for sample in samples]).reshape(-1, 1)
        t_batch = np.array([sample[3] for sample in samples]).reshape(-1, 1)
        s2_batch = np.array([sample[4] for sample in samples])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
