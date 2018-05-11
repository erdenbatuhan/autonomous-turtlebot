import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from random import random
from memory import Memory


class Agent:

    EPSILON = 0.05
    GAMMA = 0.99

    def __init__(self, connector, server):
        self.connector = connector
        self.server = server

        self.model = self.build_model()
        self.memory = Memory(max_memory=32768)

    @staticmethod
    def build_model():
        model = Sequential()

        model.add(Conv2D(32, (8, 8), padding="same", input_shape=(2, 80, 80), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(5))

        adam = Adam(lr=1e-4)
        model.compile(loss='mse', optimizer=adam)

        return model

    def load_model(self):
        path = "./data/model.h5"

        try:
            self.model.load_weights(filepath=path)
        except OSError:
            print("No pre-saved model found.")

    def save_model(self):
        path = "./data/model.h5"

        self.model.save_weights(filepath=path, overwrite=True)
        print("Model saved.")

    def get_random_action(self):
        epsilon_multiplier = 1. if random() < 0.5 else 2.  # Dice rolled

        if np.random.rand() <= (self.EPSILON * epsilon_multiplier):
            return np.random.randint(0, 5, size=1)[0]

        return None

    def get_next_action(self, state):
        random_action = self.get_random_action()
        actions = self.model.predict(state)[0]

        if random_action is not None:
            return random_action, True

        return np.argmax(actions), False

    def experience_replay(self, batch_size=64):
        len_memory = len(self.memory)

        inputs = np.zeros((min(len_memory, batch_size), 2, 80, 80))
        targets = np.zeros((inputs.shape[0], 5))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, done = self.memory.get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = self.model.predict(state)[0]

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.GAMMA * np.max(self.model.predict(next_state)[0])

        return self.model.train_on_batch(inputs, targets)

    def train(self, epoch, max_episode_length):
        self.load_model()

        reach_count = 0
        results = self.build_results()

        for episode in range(epoch):
            state = self.server.receive_data()
            state = state / 255.0

            terminal, crashed = False, False
            cumulative_reward, loss = 0., 0.

            step = 0
            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    print("Episode {}'s Report -> State {} | Crashed {} | Terminal {}".
                          format(episode, state, crashed, terminal))
                    self.connector.send_data(-1)  # Reset base

                    break

                action, is_random = self.get_next_action(state)
                self.connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.server.receive_data()
                next_state = next_state / 255.0

                cumulative_reward += reward

                if terminal:
                    reach_count += 1

                self.memory.remember_experience((state, action, reward, next_state, crashed))
                loss += self.experience_replay()

                self.report(step, episode, epoch, reach_count, state, action, is_random)
                state = next_state

            self.save_results(results, cumulative_reward, step, reach_count)

            # Save model each 20 rounds

            if episode % 20 == 1:
                self.save_model()

        _ = self.server.receive_data()
        self.connector.send_data(-2)  # Stop simulation

        return results

    @staticmethod
    def report(step, episode, epoch, reach_count, state, action, is_random):
        print("Step {} Epoch {:03d}/{:03d} | Reach count {} | State {}, {} | Act {} | Random Act {}".
              format(step, episode, (epoch - 1), reach_count, np.average(state[0][0]),  np.average(state[0][1]),
                     (action - 2), is_random))

    @staticmethod
    def build_results():
        return {
            "reach_counts": [],
            "steps_per_episode": [],
            "cumulative_reward_per_episode": []
        }

    @staticmethod
    def save_results(results, cumulative_reward, step, reach_count):
        results["reach_counts"].append(reach_count)
        results["steps_per_episode"].append(step - 1)
        results["cumulative_reward_per_episode"].append(cumulative_reward)

