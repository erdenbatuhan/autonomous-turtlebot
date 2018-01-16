import numpy as np
from random import random
from model import Model
from matplotlib import pyplot as plt


class Agent:

    EPSILON = 0.05

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

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

    def train(self, epoch, max_episode_length):
        self.load_models()
        results = []

        for episode in range(epoch):
            state = self.__server.receive_data()
            step, loss, terminal = 0, 0., False

            while True:
                step += 1

                if step > max_episode_length or terminal:
                    print("Episode {}'s Report -> State {} | Terminal {}".format(episode, state, terminal))
                    self.__connector.send_data(-1)  # Reset base

                    break

                action, is_random = self.get_next_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal = self.__server.receive_data()
                self.model.memory.remember_experience((state, action, reward, next_state, terminal))

                loss += self.experience_replay()
                self.report(step, episode, epoch, loss, state, action, is_random)

                state = next_state

            results.append(step)
            if episode > 0 and episode % 10 == 0:
                self.save_models()

        _ = self.__server.receive_data()
        self.__connector.send_data(-2)  # Stop simulation

        self.plot(results)

    @staticmethod
    def report(step, episode, epoch, loss, state, action, is_random):
        print("Step {} Epoch {:03d}/{:03d} | Loss {:.2f} | State {} | Act {} | Random Act {}".
              format(step, episode, (epoch - 1), loss, state, action, is_random))

    @staticmethod
    def plot(results):
        plt.xlabel("Episode")
        plt.ylabel("Length of Episode")

        plt.plot(results)
        plt.show()

        plt.gcf().clear()

        results = np.array(results)

        mini_results = []
        for rhand in range(10, len(results), 10):
            lhand = len(mini_results)
            mini_results.append(np.average(results[lhand:rhand]))

        plt.xlabel("10 Episodes (each represents 10 episodes)")
        plt.ylabel("Average length of 10 Episodes")
        plt.plot(mini_results)
        plt.show()

        plt.gcf().clear()

