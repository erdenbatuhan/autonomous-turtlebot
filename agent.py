import util
import numpy as np
from random import random
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


class Agent:

    __BATCH_SIZE = 64
    __MAX_MEMORY = 2048
    __HIDDEN_SIZE = 100
    __REGULARIZATION_FACTOR = .01
    __LEARNING_RATE = .01
    __EXPLORATION_RATE = .999  # Closer to 1 means more exploration.

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__epsilon = 1.
        self.__epsilon_min = .2

        self.__models = {
            "greedy": ((self.__build_model(input_size=1, num_layers=2), self.__build_model(input_size=1, num_layers=2)),
                       1, Memory(max_memory=self.__MAX_MEMORY), .99),
            "safe": ((self.__build_model(input_size=3, num_layers=3), self.__build_model(input_size=3, num_layers=3)),
                     3,  Memory(max_memory=self.__MAX_MEMORY), .9),
        }
        self.__safe_model_usable = lambda step, state: step > 10 and np.min(state["safe"][0]) < 1.

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

    def __load_models(self):
        try:
            self.__models["greedy"][0][0].load_weights("greedy.h5")
            self.__models["safe"][0][0].load_weights("safe.h5")

            self.__epsilon = self.__epsilon_min  # No exploration!
        except OSError:
            print("No pre-saved models found.")

    def __save_models(self):
        self.__models["greedy"][0][0].save_weights("greedy.h5", overwrite=True)
        self.__models["safe"][0][0].save_weights("safe.h5", overwrite=True)

    def __get_next_action(self, step, state):
        epsilon_multiplier = 1 if random() < .75 else 1.5  # Dice rolled
        if np.random.rand() <= (self.__epsilon * epsilon_multiplier):
            print("Random action is being chosen with epsilon={}".format(self.__epsilon * epsilon_multiplier))
            return np.random.randint(0, 3, size=1)[0]

        # Double Q-Learning Algorithm
        greedy_actions = np.average([self.__models["greedy"][0][i].predict(state["greedy"])[0] for i in range(2)],
                                    axis=0)
        safe_actions = np.zeros(3, dtype=float)

        if self.__safe_model_usable(step, state):
            safe_actions = np.average([self.__models["safe"][0][i].predict(state["safe"])[0] for i in range(2)],
                                      axis=0)

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
                if self.__safe_model_usable(step, state):
                    loss_safe += self.__adapt(self.__models["safe"])

                self.__report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action)
                state = next_state

            distance = state["greedy"][0]

            results["distance_per_episode"].append(distance)
            results["cumulative_reward_per_episode"].append(cumulative_reward)
            results["steps_per_episode"].append(step - 1)
            results["reach_counts"].append(reach_count)

            if reach_count > 0 and reach_count % 5 == 0:
                self.__save_models()
                # self.__memory.save_memory()

        _ = self.__server.receive_data()
        self.__connector.send_data(-2)  # Stop simulation

        return results

