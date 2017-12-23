import numpy as np
from environment import Environment


# TODO - Deep RL Algorithm
class Agent:

    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .99
    __EPSILON = .1

    def __init__(self, env):
        self.__env = env
        self.Q = {}  # Each Q Value is 2D, 8x8 Matrix.

    def __get_best_action(self, state):
        mini_Q = np.zeros(3, dtype=np.float32)

        mini_Q[0] = np.average(self.Q[state][:, 0:2])  # LEFT
        mini_Q[1] = np.average(self.Q[state][:, 2:6])  # FORWARD
        mini_Q[2] = np.average(self.Q[state][:, 6:8])  # RIGHT

        return np.argmax(mini_Q)

    def __learn(self):
        pass

