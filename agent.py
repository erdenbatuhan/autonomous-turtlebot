import numpy as np
import time
import pickle as pkl

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

from random import random
from memory import Memory


class Agent:

    EPSILON = 1
    EXPLORATION_RATE = 0.001
    EPSILON_LOWEST = 0.2
    GAMMA = 0.99

    def __init__(self, connector, server):
        self.connector = connector
        self.server = server

        self.greedy_model = self.build_greedy_model()
        self.safe_model = self.build_safe_model()

        self.safe_model2 = self.build_safe_model()

        self.memory = Memory(max_memory=5000)

    @staticmethod
    def build_greedy_model():
        model = Sequential()

        model.add(Dense(24, input_shape=(1,)))
        model.add(LeakyReLU(alpha=0.01))

        for i in range(2 - 1):
            model.add(Dense(24))
            model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(4, activation="linear"))
        model.compile(optimizer=Adam(lr=0.01), loss="mse")

        return model

    @staticmethod
    def build_safe_model():
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
        greedy_path = "./data/greedy_model.h5"
        safe_path = "./data/safe_model.h5"

        try:
            self.greedy_model.load_weights(filepath=greedy_path)
            self.safe_model.load_weights(filepath=safe_path)

            self.EXPLORATION_RATE = 1  # No exploration!
        except OSError:
            print("No pre-saved model found.")

    def save_model(self):
        greedy_path = "./data/greedy_model.h5"
        safe_path = "./data/model.h5"

        self.greedy_model.save_weights(filepath=greedy_path, overwrite=True)
        self.safe_model.save_weights(filepath=safe_path, overwrite=True)
        print("Model saved.")

    def get_random_action(self):
        if np.random.rand() <= self.EPSILON:
            return np.random.randint(0, 2, size=1)[0]

        return None

    def get_next_action(self, observation):
        if observation[3]:
            actions = self.safe_model.predict(observation[1])[0]
            return np.argmax(actions) - 2
        else:
            actions = self.greedy_model.predict(observation[0])[0]
            return np.argmax(actions)
        '''
        if np.min(half_states) < 1200:
            print("OBSTACLE!   " + str(half_states))
        else:
            print(half_states)

        return np.argmax(half_states)
        '''

    def experience_replay(self, batch_size=32):
        model_id = 1 if random() < 0.5 else 2  # Dice rolled

        main_model = self.safe_model if model_id == 1 else self.safe_model2
        side_model = self.safe_model2 if model_id == 1 else self.safe_model

        len_memory = len(self.memory)

        inputs = np.zeros((min(len_memory, batch_size), 1, 80, 80))
        targets = np.zeros((inputs.shape[0], 2))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, done = self.memory.get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = main_model.predict(state)[0]

            if done:
                targets[i, action] = reward
            else:
                # Double Q-Learning Algorithm
                Q1 = main_model.predict(next_state)[0]
                Q2 = side_model.predict(next_state)[0]

                targets[i, action] = reward + self.GAMMA * Q2[np.argmax(Q1)]

        self.EPSILON = (self.EPSILON - self.EXPLORATION_RATE) \
            if self.EPSILON > self.EPSILON_LOWEST else self.EPSILON_LOWEST
        return main_model.train_on_batch(inputs, targets)

    def train(self):
        steps = []
        self.load_model()

        observation = self.server.receive_data()
        lifetime, step, loss, crashed = 0., 0, 0., False

        while True:
            lifetime += 1
            step += 1

            action = self.get_next_action(observation)

            print(observation)
            self.connector.send_data(int(action))

            next_observation, _, _, crashed = self.server.receive_data()

            #self.memory.remember_experience((state, action, reward, next_state, crashed))
            #loss += self.experience_replay()

            self.report(step, action, False, crashed)

            observation_prev = observation
            observation = next_observation

            print(observation)

            while observation[2]:
                if observation_prev[0] >= 0:
                    self.connector.send_data(0)
                else:
                    self.connector.send_data(3)

                    observation, _, _, _ = self.server.receive_data()

            if crashed:
                self.connector.send_data(4)
                time.sleep(5)
                observation, _, _, crashed = self.server.receive_data()

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

