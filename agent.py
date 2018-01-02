import util
import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


class Agent:

    __NUM_ACTIONS = 3
    __BATCH_SIZE = 250
    __MAX_MEMORY = 2500
    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .95
    __EPSILON = .1

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__memory = Memory(max_memory=self.__MAX_MEMORY)
        self.__distances_per_episode = []

        self.__distance_model = self.__build_model()
        self.__depth_model = self.__build_model()

    @staticmethod
    def __report(step, episode, epoch, loss_distance, loss_depth, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Loss Distance {:.2f} | Loss Depth {:.2f} | " \
                  "Reach count {} | Distance {:.2f} | Act {}"
        print(message.format(step, episode, (epoch - 1), loss_distance, loss_depth, reach_count, state[0], action))

    def __build_model(self):
        model = Sequential()

        model.add(Dense(100, input_shape=(3, ), activation="linear"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(self.__NUM_ACTIONS, activation="linear"))
        model.compile(Adam(lr=self.__LEARNING_RATE), "mse")

        return model

    def __load_models(self):
        try:
            self.__distance_model.load_weights("distance_model.h5")
            self.__depth_model.load_weights("depth_model.h5")
        except OSError:
            print("No pre-saved model found.")

    def __save_models(self):
        self.__distance_model.save_weights("distance_model.h5", overwrite=True)
        self.__depth_model.save_weights("depth_model.h5", overwrite=True)

    def __get_model(self, model_name):
        return self.__distance_model if model_name == "distance" else self.__depth_model

    def __predict(self, model_name, state):
        model = self.__get_model(model_name)
        return model.predict(np.array([state]))[0]

    def __get_best_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, self.__NUM_ACTIONS, size=1)[0]

        # Only use depth model when there is a risk of collision
        Q_distance = self.__predict("distance", state)
        Q_depth = [0, 0, 0]

        if state[1] != 0:
            Q_depth = self.__predict("depth", state)

        Q = np.add(Q_distance, Q_depth)
        return np.argmax(Q)

    def __adapt(self, model_name):
        len_memory = len(self.__memory)

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), 3))
        targets = np.zeros((inputs.shape[0], self.__NUM_ACTIONS))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state = self.__memory.get_experience(ind, 0)
            terminal, crashed = self.__memory.get_experience(ind, 1)
            inputs[i] = state
            targets[i] = self.__predict(model_name, state)

            if terminal or crashed:
                targets[i, action] = reward
            else:
                Q = self.__predict(model_name, next_state)
                targets[i, action] = reward + self.__DISCOUNT_FACTOR * np.max(Q)

        return inputs, targets

    def __learn(self, model_name):
        model = self.__get_model(model_name)
        inputs, targets = self.__adapt(model_name)

        return model.train_on_batch(inputs, targets)

    def train(self, epoch, max_episode_length):
        self.__load_models()

        reach_count = 0
        self.__distances_per_episode = []

        for episode in range(epoch):
            state = self.__server.receive_data()
            step, terminal, crashed, loss_distance, loss_depth = 0, False, False, 0., 0.

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

                loss_distance += self.__learn("distance")
                loss_depth += self.__learn("depth")

                if step % 20 == 1 or terminal:
                    self.__report(step, episode, epoch, loss_distance, loss_depth, reach_count, state, action)

                state = next_state

            distance = state[0]
            self.__distances_per_episode.append(distance)

            if reach_count % 2 == 1:
                self.__save_models()

        self.__connector.send_data(-2)  # Stop simulation
        return self.__distances_per_episode

