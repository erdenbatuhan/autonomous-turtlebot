import util
import numpy as np
from memory import Memory

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


class Agent:

    __BATCH_SIZE = 50
    __MAX_MEMORY = 1000
    __HIDDEN_SIZE = 96
    __INPUT_SIZE = 5
    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .99
    __EPSILON = .1

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.__memory = Memory(max_memory=self.__MAX_MEMORY)
        self.__model = self.__build_model()

    @staticmethod
    def __report(step, episode, epoch, loss, reach_count, state, action):
        message = "Step {} Epoch {:03d}/{:03d} | Loss {:.2f} | Reach count {} | State {} | Act {}"
        print(message.format(step, episode, (epoch - 1), loss, reach_count, state, action))

    def __build_model(self):
        model = Sequential()

        model.add(Dense(self.__HIDDEN_SIZE, input_shape=(self.__INPUT_SIZE, ), activation="relu"))
        model.add(Dense(self.__HIDDEN_SIZE, input_shape=(self.__INPUT_SIZE, ), activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer=Adam(lr=self.__LEARNING_RATE), loss="mse")

        return model

    def __load_model(self):
        try:
            self.__model.load_weights("model.h5")
        except OSError:
            print("No pre-saved model found.")

    def __save_model(self):
        self.__model.save_weights("model.h5", overwrite=True)

    def __predict(self, state):
        return np.array(self.__model.predict(np.array([state]))[0])

    def __get_next_action(self, state):
        if np.random.rand() <= self.__EPSILON:
            return np.random.randint(0, 3, size=1)[0]

        actions = self.__predict(state)
        return np.argmax(actions)

    def __adapt(self):
        len_memory = len(self.__memory)

        inputs = np.zeros((min(len_memory, self.__BATCH_SIZE), self.__INPUT_SIZE))
        targets = np.zeros((inputs.shape[0], 3))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state = self.__memory.get_experience(ind, 0)
            terminal, crashed = self.__memory.get_experience(ind, 1)

            actions = self.__predict(state)
            next_actions = self.__predict(next_state)

            inputs[i] = state
            targets[i] = actions

            if terminal or crashed:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.__DISCOUNT_FACTOR * np.max(next_actions)

        return inputs, targets

    def train(self, epoch, max_episode_length):
        self.__load_model()

        reach_count = 0
        results = {
            "distance_per_episode": [],
            "cumulative_reward_per_episode": [],
            "steps_per_episode": [],
            "reach_counts": []
        }

        for episode in range(epoch):
            state = self.__server.receive_data()
            step, loss, cumulative_reward, terminal, crashed = 0, 0., 0., False, False

            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    print("Episode {}'s Report -> State {} | Crashed {} | Terminal {}".
                          format(episode, state, crashed, terminal))
                    self.__connector.send_data(-1)  # Reset base

                    break

                action = self.__get_next_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.__server.receive_data()
                cumulative_reward += reward

                if terminal:
                    reach_count += 1

                self.__memory.remember_experience([[state, action, reward, next_state], [terminal, crashed]])

                inputs, targets = self.__adapt()
                loss += self.__model.train_on_batch(inputs, targets)

                self.__report(step, episode, epoch, loss, reach_count, state, action)
                state = next_state

            distance = util.c(state[0], state[1])

            results["distance_per_episode"].append(distance)
            results["cumulative_reward_per_episode"].append(cumulative_reward)
            results["steps_per_episode"].append(step - 1)
            results["reach_counts"].append(reach_count)

            if reach_count % 2 == 1:
                self.__save_model()
                self.__memory.save_memory()

        _ = self.__server.receive_data()
        self.__connector.send_data(-2)  # Stop simulation

        return results

