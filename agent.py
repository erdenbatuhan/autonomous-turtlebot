import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


# TODO - Deep RL Algorithm
class Agent:

    __STATE_DIM = 1 + 8 * 8 + 1
    __NUM_ACTIONS = 3
    __BATCH_SIZE = 50
    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .99
    __EPSILON = .1

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__memory = Memory(500)
        self.__model1, self.__model2 = self.__build_model(), self.__build_model()
        self.__episodes = []

    @staticmethod
    def __report(step, episode, epoch, loss, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Pos {:.3f} | Act {}"
        print(message.format(step, episode, (epoch - 1), loss, reach_count, state[0, 0], (action - 1)))

    @staticmethod
    def __predict(model, state):
        return model.predict(np.array([state]))[0]

    def __build_model(self):
        model = Sequential()

        model.add(Dense(200, input_shape=(self.__STATE_DIM, ), activation="relu"))
        model.add(Dense(self.__NUM_ACTIONS, activation="linear"))
        model.compile(Adam(lr=self.__LEARNING_RATE), "mse")

        return model

    def __adapt_model(self):
        probability = random()

        len_memory = len(self.__memory)
        model = self.__model1 if probability > .5 else self.__model2

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), self.__STATE_DIM))
        targets = np.zeros((inputs.shape[0], self.__NUM_ACTIONS))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state = self.__memory.get_experience(ind, 0)
            terminal = self.__memory.get_experience(ind, 1)

            inputs[i] = state
            targets[i] = self.__predict(model, state)

            Q1 = self.__predict(self.__model1, next_state)
            Q2 = self.__predict(self.__model2, next_state)

            if terminal:
                targets[i] = reward
            elif probability > .5:
                targets[i] = reward * self.__DISCOUNT_FACTOR * Q2[np.argmax(Q1)]
            else:
                targets[i] = reward * self.__DISCOUNT_FACTOR * Q1[np.argmax(Q2)]

        return model, inputs, targets

    def __get_best_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, self.__NUM_ACTIONS, size=1)[0]

        Q1, Q2 = self.__predict(self.__model1, state), self.__predict(self.__model2, state)
        return np.argmax(np.add(Q1, Q2))

    def train(self, epoch, max_episode_length):
        reach_count = 0
        self.__episodes = []

        for episode in range(epoch):
            state = self.__server.receive_data()
            loss, terminal, step = 0., False, 0

            while not terminal:
                step += 1
                if step > max_episode_length:
                    break

                action = self.__get_best_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal = self.__server.receive_data()[0]

                if terminal:
                    reach_count += 1

                self.__memory.remember_experience([[state, action, reward, next_state], terminal])
                model, inputs, targets = self.__adapt_model()
                loss += model.train_on_batch(inputs, targets)

                if step % 100 == 1 or terminal:
                    self.__report(step, episode, epoch, loss, reach_count, state, action)

                state = next_state

            self.__episodes.append(step)

        self.__connector.send_data(-1)
        return self.__episodes

