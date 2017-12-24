import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import host
from random import randint

# TODO - Deep RL Algorithm
class Agent:

    __LEARNING_RATE = .01
    __DISCOUNT_FACTOR = .99
    __EPSILON = .1

    def __init__(self):
        self.Q = {}  # Each Q Value is 2D, 8x8 Matrix.
        self.model = None
        self.server = host.HostServer()
        self.server.listen()
        self.connector = host.HostConnector()

    def __get_best_action(self, state):
        """
        mini_Q = np.zeros(3, dtype=np.float32)

        mini_Q[0] = np.average(self.Q[state][:, 0:2])  # LEFT
        mini_Q[1] = np.average(self.Q[state][:, 2:6])  # FORWARD
        mini_Q[2] = np.average(self.Q[state][:, 6:8])  # RIGHT

        return np.argmax(mini_Q)
        """
        Q = self.model.predict(np.asarray(state))
        print(Q)
        action = np.argmax(Q)
        print(action)
        action = randint(0, 2)

        return action

    def __learn(self):
        pass

    def adapt(self):
        pass

    def initialize_model(self, hidden_size, input_size, num_actions, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(input_size, ), activation="relu"))

        if hidden_size <= 100:
            model.add(Dense(hidden_size, activation="sigmoid"))

        model.add(Dense(num_actions, activation="linear"))
        model.compile(Adam(lr=learning_rate), "mse")

        self.model = model

    def train(self, epoch, max_episode_length):
        reach_count = 0
        self.episodes = []
        state = self.server.receive_data()

        for episode in range(epoch):
            loss = 0.
            terminal = False
            step = 0

            while not terminal:
                step += 1
                if step > max_episode_length:
                    break

                action = self.__get_best_action(state)
                self.connector.send_data(action)
                data = self.server.receive_data()  # Data from simulation [State, reward, terminal]
                next_state, reward, terminal = data[0], data[1], data[2]
                # distance, depth, time_passed = state[0], state[1], state[2]
                if terminal:
                    reach_count += 1

                # self.remember([[state, action, reward, next_state], terminal])  # store experience
                model, inputs, targets = self.adapt()  # adapt model
                #loss += model.train_on_batch(inputs, targets)

                #if step % 100 == 1 or terminal:
                #    print("Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Pos {:.3f} | Act {}".
                #          format(step, episode, (epoch - 1), loss, reach_count, state[0, 0], (action - 1)))

                state = next_state

            self.episodes.append(step)

agent = Agent()
agent.initialize_model(100, 2, 3, 0.1)
agent.train(1000, 1000)