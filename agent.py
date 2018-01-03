import util
import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


class Agent:

    __BATCH_SIZE = 250
    __MAX_MEMORY = 25000
    __LEARNING_RATE = .1
    __DISCOUNT_FACTOR = .99
    __EPSILON = .1

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__memory = Memory(max_memory=self.__MAX_MEMORY)
        self.__distances_per_episode = []

        self.__greedy_model = self.__build_model()
        self.__safe_model = self.__build_model()

    @staticmethod
    def __report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Loss Greedy {:.2f} | Loss Safe {:.2f} | " \
                  "Reach count {} | State {} | Act {}"
        print(message.format(step, episode, (epoch - 1), loss_greedy, loss_safe,
                             reach_count, state, action))

    def __build_model(self):
        model = Sequential()

        model.add(Dense(200, input_shape=(1, ), activation="relu"))
        model.add(Dense(200, input_shape=(1, ), activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(Adam(lr=self.__LEARNING_RATE), "mse")

        return model

    def __load_models(self):
        try:
            self.__greedy_model.load_weights("greedy_model.h5")
            self.__safe_model.load_weights("safe_model.h5")
        except OSError:
            print("No pre-saved model found.")

    def __save_models(self):
        self.__greedy_model.save_weights("greedy_model.h5", overwrite=True)
        self.__safe_model.save_weights("safe_model.h5", overwrite=True)

    def __get_model(self, model_name):
        return self.__greedy_model if model_name == "greedy" else self.__safe_model

    def __predict(self, model_name, state):
        model = self.__get_model(model_name)

        model_state = state[model_name]
        model_actions = model.predict(np.array([model_state]))[0]

        return np.array(model_actions)

    def __get_next_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, 3, size=1)[0]

        greedy_actions = self.__predict("greedy", state)
        safe_actions = self.__predict("safe", state)

        actions = np.add(greedy_actions, safe_actions)
        return np.argmax(actions)

    def __adapt(self, model_name):
        len_memory = len(self.__memory)

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), 1))
        targets = np.zeros((inputs.shape[0], 3))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state = self.__memory.get_experience(ind, 0)
            terminal, crashed = self.__memory.get_experience(ind, 1)

            actions = self.__predict(model_name, state)
            next_actions = self.__predict(model_name, next_state)

            inputs[i] = state[model_name]
            targets[i] = actions

            if terminal or crashed:
                targets[i, action] = reward[model_name]
            else:
                targets[i, action] = reward[model_name] + self.__DISCOUNT_FACTOR * np.max(next_actions)

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
            step, terminal, crashed, loss_greedy, loss_safe = 0, False, False, 0., 0.

            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    print("Episode {}'s Report -> State {} | Crashed {} | Terminal {}".format(episode, state, crashed, terminal))
                    self.__connector.send_data(-1)  # Reset base

                    break

                action = self.__get_next_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.__server.receive_data()

                if terminal:
                    reach_count += 1

                self.__memory.remember_experience([[state, action, reward, next_state], [terminal, crashed]])

                loss_greedy += self.__learn("greedy")
                loss_safe += self.__learn("safe")

                self.__report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action)
                state = next_state

            distance = state["greedy"]
            self.__distances_per_episode.append(distance)

            if reach_count % 2 == 1:
                self.__save_models()

        self.__connector.send_data(-2)  # Stop simulation
        return self.__distances_per_episode

