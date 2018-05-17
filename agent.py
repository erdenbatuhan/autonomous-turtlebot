import numpy as np
from random import random
from model import Model


class Agent:

    EPSILON = 0.05

    def __init__(self, connector, server):
        self.connector = connector
        self.server = server

        self.model = Model(name=None, input_size=1, output_size=4, hidden_size=24, num_layers=2,
                           max_memory=512, learning_rate=0.01, discount_factor=0.95)

    def load_models(self):
        self.model.load_model()

    def save_models(self):
        self.model.save_model()

    def get_random_action(self):
        epsilon_multiplier = 1. if random() < 0.5 else 2.  # Dice rolled

        if np.random.rand() <= (self.EPSILON * epsilon_multiplier):
            return np.random.randint(0, self.model.output_size, size=1)[0]

        return None

    def get_next_action(self, state):
        random_action = self.get_random_action()
        actions = np.add(self.model.models[0].predict(state)[0], self.model.models[1].predict(state)[0])

        if random_action is not None:
            return random_action, True

        return np.argmax(actions), False

    def experience_replay(self, batch_size=32):
        model_id = 0 if random() < 0.5 else 1  # Dice rolled
        len_memory = len(self.model.memory)

        inputs = np.zeros((min(len_memory, batch_size), self.model.input_size))
        targets = np.zeros((inputs.shape[0], self.model.output_size))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, terminal = self.model.memory.get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = self.model.models[model_id].predict(state)[0]

            if terminal:
                targets[i, action] = reward
            else:
                # Double Q-Learning Algorithm
                Q1 = self.model.models[model_id].predict(next_state)[0]
                Q2 = self.model.models[1 - model_id].predict(next_state)[0]

                targets[i, action] = reward + self.model.discount_factor * Q2[np.argmax(Q1)]

        return self.model.models[model_id].train_on_batch(inputs, targets)

    def train(self):
        self.load_models()

        state = self.server.receive_data()
        step, loss, terminal = 0, 0., False

        while True:
            step += 1

            action, is_random = self.get_next_action(state)
            self.connector.send_data(int(action))

            next_state, reward, terminal = self.server.receive_data()
            self.model.memory.remember_experience((state, action, reward, next_state, terminal))

            loss += self.experience_replay()
            self.report(step, loss, state, action, is_random)

            state_prev = state
            state = next_state

            if step > 0 and step % 100 == 0:
                self.save_models()

            while terminal:
                if state_prev >= 0:
                    self.connector.send_data(0)
                else:
                    self.connector.send_data(3)

                state, _, terminal = self.server.receive_data()

            #self.connector.send_data(4)
            #_, _, _ = self.server.receive_data()

    @staticmethod
    def report(step, loss, state, action, is_random):
        print("Step {} | Loss {:.2f} | State {} | Act {} | Random Act {}".
              format(step, loss, state, action, is_random))

