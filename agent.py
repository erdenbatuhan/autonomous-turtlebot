import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


class Agent:

    __MODELS_DIRECTORY = "./models"
    __BATCH_SIZE = 128
    __MAX_MEMORY = 32768
    __HIDDEN_SIZE = 100
    __REGULARIZATION_FACTOR = .01
    __LEARNING_RATE = .01
    __EXPLORATION_RATE = .999  # Closer to 1 means more exploration.

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__epsilon = 1.
        self.__epsilon_min = .1

        self.__models = {
            "greedy": ((self.__build_model(input_size=1, num_layers=2), self.__build_model(input_size=1, num_layers=2)),
                       1, Memory(model_name="greedy", max_memory=self.__MAX_MEMORY), .99),
            "safe": ((self.__build_model(input_size=3, num_layers=3), self.__build_model(input_size=3, num_layers=3)),
                     3, Memory(model_name="safe", max_memory=self.__MAX_MEMORY), .9),
        }
        self.__is_collision_risk_detected = lambda step, state: step > 10 and np.min(state["safe"][0]) < .5

    def __report(self, step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Epsilon {:.4f} | " \
                  "Loss Greedy {:.2f} | Loss Safe {:.2f} | Reach count {} | State {} | Act {}"
        print(message.format(step, episode, (epoch - 1), self.__epsilon,
                             loss_greedy, loss_safe, reach_count, state, action))

    def __build_model(self, input_size, num_layers):
        model = Sequential()

        # Initial Layer
        model.add(Dense(self.__HIDDEN_SIZE, input_shape=(input_size, )))
        model.add(LeakyReLU(alpha=0.01))

        # Additional layers
        for i in range(num_layers - 1):
            model.add(Dense(self.__HIDDEN_SIZE))
            model.add(LeakyReLU(alpha=0.01))

        # Dropout
        model.add(Dropout(.3))

        # Output Layer
        model.add(Dense(3, activation="linear"))

        # Compiler
        optimizer = Adam(lr=self.__LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")

        return model

    def __load_model(self, model_name):
        path = self.__MODELS_DIRECTORY + "/" + model_name + "_model.h5"

        try:
            self.__models[model_name][0][0].load_weights(filepath=path)
            self.__epsilon = self.__epsilon_min  # No exploration!
        except OSError:
            print("No pre-saved model found for " + model_name + " model.")

    def __load_models(self):
        self.__load_model("greedy")
        self.__load_model("safe")

    def __save_model(self, model_name):
        path = self.__MODELS_DIRECTORY + "/" + model_name + "_model.h5"

        self.__models[model_name][0][0].save_weights(filepath=path, overwrite=True)
        print("Model of " + model_name + " model saved.")

    def __save_models(self):
        self.__save_model("greedy")
        self.__save_model("safe")

    def __get_random_action(self):
        epsilon_multiplier = 1 if random() < .75 else 1.5  # Dice rolled
        if np.random.rand() <= (self.__epsilon * epsilon_multiplier):
            print("Random action is being chosen with epsilon={}".format(self.__epsilon * epsilon_multiplier))
            return np.random.randint(0, 3, size=1)[0]

        return -1  # No random

    def __get_next_action(self, step, state):
        next_action = self.__get_random_action()

        if next_action != -1:
            return next_action

        # Double Q-Learning Algorithm
        greedy_actions = np.add(self.__models["greedy"][0][0].predict(state["greedy"])[0],
                                self.__models["greedy"][0][1].predict(state["greedy"])[0])
        safe_actions = np.add(self.__models["safe"][0][0].predict(state["safe"])[0],
                              self.__models["safe"][0][1].predict(state["safe"])[0])

        if np.argmax(greedy_actions) == np.argmax(safe_actions) or self.__is_collision_risk_detected(step, state):
            return np.argmax(safe_actions)

        actions = np.add(greedy_actions, safe_actions)
        best_action = np.argmax(actions)

        if best_action != 1 and actions[1] == np.amax(actions):
            return 1  # Go Forward

        return best_action

    def __adapt(self, model):
        model_id = 0 if random() < .5 else 1  # Dice rolled
        len_memory = len(model[2])

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), model[1]))
        targets = np.zeros((inputs.shape[0], 3))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, done = model[2].get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = model[0][model_id].predict(state)[0]

            if done:
                targets[i, action] = reward
            else:
                # Double Q-Learning Algorithm
                Q1 = model[0][model_id].predict(next_state)[0]
                Q2 = model[0][1 - model_id].predict(next_state)[0]

                targets[i, action] = reward + model[3] * Q2[np.argmax(Q1)]

        if self.__epsilon > self.__epsilon_min:
            self.__epsilon *= self.__EXPLORATION_RATE

        return model[0][model_id].train_on_batch(inputs, targets)

    def train(self, epoch, max_episode_length):
        self.__load_models()

        reach_count = 0
        results = {
            "distance_per_episode": [],
            "cumulative_reward_per_episode": [],
            "steps_per_episode": [],
            "reach_counts": []
        }

        for episode in range(epoch):
            state = self.__server.receive_data()
            step, loss_greedy, loss_safe, cumulative_reward, terminal, crashed = 0, 0., 0., 0., False, False

            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    print("Episode {}'s Report -> State {} | Crashed {} | Terminal {}".
                          format(episode, state, crashed, terminal))
                    self.__connector.send_data(-1)  # Reset base

                    break

                action = self.__get_next_action(step, state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.__server.receive_data()
                cumulative_reward += reward["greedy"] + reward["safe"]

                if terminal:
                    reach_count += 1

                self.__models["greedy"][2].remember_experience((
                    state["greedy"], action, reward["greedy"], next_state["greedy"], terminal))
                self.__models["safe"][2].remember_experience((
                    state["safe"], action, reward["safe"], next_state["safe"], crashed))

                loss_greedy += self.__adapt(self.__models["greedy"])
                loss_safe += self.__adapt(self.__models["safe"])

                self.__report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action)
                state = next_state

            distance = state["greedy"][0][0]

            results["distance_per_episode"].append(distance)
            results["cumulative_reward_per_episode"].append(cumulative_reward)
            results["steps_per_episode"].append(step - 1)
            results["reach_counts"].append(reach_count)

            if reach_count > 0:
                self.__save_models()
                self.__models["greedy"][2].save_memory()
                self.__models["safe"][2].save_memory()

        _ = self.__server.receive_data()
        self.__connector.send_data(-2)  # Stop simulation

        return results

