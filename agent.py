import numpy as np
import time
import pickle as pkl

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from random import random
from memory import Memory


class Agent:

    EPSILON = 1
    EXPLORATION_RATE = 0.001
    EPSILON_LOWEST = 10
    GAMMA = 0.99

    def __init__(self, connector, server):
        self.connector = connector
        self.server = server

        self.model = self.build_model()
        self.memory = Memory(max_memory=5000)

    @staticmethod
    def build_model():
        model = Sequential()

        model.add(Conv2D(32, (8, 8), padding="same", input_shape=(1, 80, 80), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))

        adam = Adam(lr=1e-4)
        model.compile(loss='mse', optimizer=adam)

        return model

    def load_model(self):
        path = "./data/model.h5"

        try:
            self.model.load_weights(filepath=path)
            self.EXPLORATION_RATE = 1  # No exploration!
        except OSError:
            print("No pre-saved model found.")

    def save_model(self):
        path = "./data/model.h5"

        self.model.save_weights(filepath=path, overwrite=True)
        print("Model saved.")

    def get_random_action(self):
        if np.random.rand() <= self.EPSILON:
            return np.random.randint(0, 2, size=1)[0]

        return None

    def get_next_action(self, state):
        random_action = self.get_random_action()
        actions = self.model.predict(state)[0]

        print(actions)

        if random_action is not None:
            return random_action, True

        return np.argmax(actions), False

    def experience_replay(self, batch_size=32):
        len_memory = len(self.memory)

        inputs = np.zeros((min(len_memory, batch_size), 1, 80, 80))
        targets = np.zeros((inputs.shape[0], 2))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, done = self.memory.get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = self.model.predict(state)[0]

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.GAMMA * np.max(self.model.predict(next_state)[0])

        self.EPSILON = (self.EPSILON - self.EXPLORATION_RATE) \
            if self.EPSILON > self.EPSILON_LOWEST else self.EPSILON_LOWEST
        return self.model.train_on_batch(inputs, targets)

    def train(self):
        steps = []
        self.load_model()

        state = self.server.receive_data()
        lifetime, step, loss, crashed = 0., 0, 0., False

        while True:
            lifetime += 1
            step += 1

            action, is_random = self.get_next_action(state)
            self.connector.send_data(int(action))

            next_state, reward, _, crashed = self.server.receive_data()

            self.memory.remember_experience((state, action, reward, next_state, crashed))
            loss += self.experience_replay()

            self.report(step, action, is_random, crashed)
            state = next_state

            if crashed:
                self.connector.send_data(2)
                time.sleep(1)
                state, _, _, crashed = self.server.receive_data()

                steps.append(step)
                step = 0.

            if step > 0 and lifetime % 250 == 0:
                self.save_model()

                output = open("./data/steps.pkl", "wb")
                pkl.dump(steps, output)
                output.close()

    def report(self, step, action, is_random, crashed):
        print("Epsilon {} | Step {} | Act {} | Random Act {} | Crashed {}".
              format(self.EPSILON, step, (action - 3), is_random, crashed))

