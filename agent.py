import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


class Agent:

    __STATE_DIM = 1 + 8 * 8 + 1
    __NUM_ACTIONS = 3
    __BATCH_SIZE = 100
    __MAX_MEMORY = 1000
    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .95
    __EPSILON = .1

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__memory = Memory(max_memory=self.__MAX_MEMORY)
        self.__model = self.__build_model()
        self.__episodes = []

    @staticmethod
    def __report(step, episode, epoch, loss, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Loss {:.2f} | Reach count {} | " \
                  "Distance {:.2f} | Time Passed {:.2f} | Act {}"
        last_element = len(state) - 1
        print(message.format(step, episode, (epoch - 1), loss, reach_count, state[0], state[last_element], action))

    def __build_model(self):
        model = Sequential()

        model.add(Dense(200, input_shape=(self.__STATE_DIM, ), activation="relu"))
        model.add(Dense(200, activation="relu"))
        model.add(Dense(self.__NUM_ACTIONS, activation="linear"))
        model.compile(Adam(lr=self.__LEARNING_RATE), "mse")

        return model

    def __predict(self, state):
        return self.__model.predict(np.array([state]))[0]

    def __adapt(self):
        len_memory = len(self.__memory)

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), self.__STATE_DIM))
        targets = np.zeros((inputs.shape[0], self.__NUM_ACTIONS))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state = self.__memory.get_experience(ind, 0)
            terminal, crashed = self.__memory.get_experience(ind, 1)

            inputs[i] = state
            targets[i] = self.__predict(state)

            if terminal or crashed:
                targets[i, action] = reward
            else:
                Q = self.__predict(next_state)
                targets[i, action] = reward + self.__DISCOUNT_FACTOR * np.max(Q)

        return inputs, targets

    def __load_model(self):
        try:
            self.__model.load_weights("model.h5")
        except OSError:
            print("No pre-saved model found.")

    def __save_model(self):
            self.__model.save_weights("model.h5")

    def __get_best_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, self.__NUM_ACTIONS, size=1)[0]

        Q = self.__predict(state)
        print("Q: ", Q)

        return np.argmax(Q)

    def train(self, epoch, max_episode_length):
        self.__load_model()

        reach_count = 0
        self.__episodes = []

        for episode in range(epoch):
            state = self.__server.receive_data()
            step, terminal, crashed, loss = 0, False, False, 0.

            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    self.__connector.send_data(-1)  # Reset base
                    break

                action = self.__get_best_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.__server.receive_data()

                if terminal:
                    reach_count += 1

                self.__memory.remember_experience([[state, action, reward, next_state], [terminal, crashed]])
                inputs, targets = self.__adapt()
                loss += self.__model.train_on_batch(inputs, targets)

                self.__report(step, episode, epoch, loss, reach_count, state, action)
                state = next_state

            self.__episodes.append(step)
            if reach_count % 5 == 1:
                self.__save_model()

        self.__connector.send_data(-2)  # Stop simulation
        return self.__episodes

