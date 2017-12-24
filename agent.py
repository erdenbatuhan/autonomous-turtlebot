import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


# TODO - Deep RL Algorithm
class Agent:

    __MODELS_DIR = "./models/"
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
        message = "Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | " \
                  "Distance {:.3f} | Time Passed {:.3f} | Act {}"
        last_element = len(state) - 1
        print(message.format(step, episode, (epoch - 1), loss, reach_count,
                             state[0], state[last_element], (action - 1)))

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
            terminal, crashed = self.__memory.get_experience(ind, 1)

            inputs[i] = state
            targets[i] = self.__predict(model, state)

            Q1 = self.__predict(self.__model1, next_state)
            Q2 = self.__predict(self.__model2, next_state)

            if terminal or crashed:
                targets[i] = reward
            elif probability > .5:
                targets[i] = reward * self.__DISCOUNT_FACTOR * Q2[np.argmax(Q1)]
            else:
                targets[i] = reward * self.__DISCOUNT_FACTOR * Q1[np.argmax(Q2)]

        return model, inputs, targets

    def __load_models(self):
        try:
            self.__model1.load_weights(self.__MODELS_DIR + "model1.h5")
            self.__model2.load_weights(self.__MODELS_DIR + "model2.h5")
        except OSError:
            pass

    def __save_models(self):
        self.__model1.save_weights(self.__MODELS_DIR + "model1.h5")
        self.__model2.save_weights(self.__MODELS_DIR + "model2.h5")

    def __get_best_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, self.__NUM_ACTIONS, size=1)[0]

        Q1, Q2 = self.__predict(self.__model1, state), self.__predict(self.__model2, state)
        return np.argmax(np.add(Q1, Q2))

    def train(self, epoch, max_episode_length):
        self.__load_models()

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
                model, inputs, targets = self.__adapt_model()
                loss += model.train_on_batch(inputs, targets)

                self.__report(step, episode, epoch, loss, reach_count, state, action)
                state = next_state

            self.__episodes.append(step)
            if episode % 10 == 0:
                self.__save_models()

        self.__connector.send_data(-2)  # Stop simulation
        return self.__episodes

