import numpy as np


class NextClipping:

    def __init__(self):
        self.gama = 0
        # 初始向量
        self.state_clip = np.array([0.5, 1, 2])
        self.n = np.random.rand(6) / 20

    def set_froward(self):
        p = np.array([
            [1 - self.n[0] - self.n[1], self.n[0], self.n[1]],
            [self.n[2], 1 - self.n[2] - self.n[3], self.n[3]],
            [self.n[4], self.n[5], 1 - self.n[4] - self.n[5]]
        ])
        return np.dot(self.state_clip, p)

    def get_clip_parameter(self):
        return self.set_froward()
