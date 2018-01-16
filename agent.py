import numpy as np
from random import random
from model import Model


class Agent:

    EPSILON = 0.05

    def __init__(self, connector, server):
        self.__connector = connector
        self.__server = server

        self.greedy_model = Model(name="greedy", input_size=1, output_size=5, hidden_size=100, num_layers=2,
                                  max_memory=32768, learning_rate=0.01, discount_factor=0.99)
        self.safe_model = Model(name="safe", input_size=3, output_size=5, hidden_size=100, num_layers=4,
                                max_memory=32768, learning_rate=0.01, discount_factor=0.9)

        self.get_first_true = lambda arr: [index for index, item in enumerate(arr) if item][0]
        self.is_collision_risk_detected = lambda state: np.min(state["safe"][0]) < 0.4

    def load_models(self):
        # Greedy
        self.greedy_model.load_model()
        self.greedy_model.memory.load_memory()
        # Safe
        self.safe_model.load_model()
        self.safe_model.memory.load_memory()

    def save_models(self):
        # Greedy
        self.greedy_model.save_model()
        self.greedy_model.memory.save_memory()
        # Safe
        self.safe_model.save_model()
        self.safe_model.memory.save_memory()

    def get_random_action(self):
        epsilon_multiplier = 1. if random() < 0.5 else 2.  # Dice rolled

        if np.random.rand() <= (self.EPSILON * epsilon_multiplier):
            return np.random.randint(0, 5, size=1)[0]

        return None

    @staticmethod
    def get_votes(actions, vote_count):
        actions_cp = actions.copy()
        votes = []

        for _ in range(vote_count):
            vote = np.argmax(actions_cp)
            votes.append(vote)

            actions_cp[vote] = -np.inf

        return votes

    def get_next_action(self, state):
        random_action = self.get_random_action()

        # Actions of each model with Double Qs
        greedy_actions = np.add(self.greedy_model.models[0].predict(state["greedy"])[0],
                                self.greedy_model.models[1].predict(state["greedy"])[0])
        safe_actions = np.add(self.safe_model.models[0].predict(state["safe"])[0],
                              self.safe_model.models[1].predict(state["safe"])[0])

        shared_actions = np.add(greedy_actions, safe_actions)

        # Give full control to safe model in case of collision
        if self.is_collision_risk_detected(state):
            print("Collision risk detected!")
            return np.argmax(safe_actions), False
        elif random_action is not None:
            return random_action, True
        
        # Action votes of each model
        greedy_votes = self.get_votes(greedy_actions, 3)
        safe_votes = self.get_votes(safe_actions, 3)

        try:
            first_common_vote = self.get_first_true(np.equal(greedy_votes, safe_votes))
            return first_common_vote, False
        except IndexError:  # When both of the models couldn't reach to a consensus
            return np.argmax(shared_actions), False

    @staticmethod
    def experience_replay(model, batch_size=128):
        model_id = 0 if random() < 0.5 else 1  # Dice rolled
        len_memory = len(model.memory)

        inputs = np.zeros((min(len_memory, batch_size), model.input_size))
        targets = np.zeros((inputs.shape[0], 5))

        for i, ind in enumerate(np.random.randint(0, len_memory, inputs.shape[0])):
            state, action, reward, next_state, done = model.memory.get_experience(ind)

            inputs[i:i+1] = state
            targets[i] = model.models[model_id].predict(state)[0]

            if done:
                targets[i, action] = reward
            else:
                # Double Q-Learning Algorithm
                Q1 = model.models[model_id].predict(next_state)[0]
                Q2 = model.models[1 - model_id].predict(next_state)[0]

                targets[i, action] = reward + model.discount_factor * Q2[np.argmax(Q1)]

        return model.models[model_id].train_on_batch(inputs, targets)

    def train(self, epoch, max_episode_length):
        self.load_models()

        reach_count = 0
        results = self.build_results()

        for episode in range(epoch):
            state = self.__server.receive_data()

            terminal, crashed = False, False
            cumulative_reward_greedy, cumulative_reward_safe = 0., 0.
            loss_greedy, loss_safe = 0., 0.

            step = 0
            while True:
                step += 1
                if step > max_episode_length or crashed or terminal:
                    print("Episode {}'s Report -> State {} | Crashed {} | Terminal {}".
                          format(episode, state, crashed, terminal))
                    self.__connector.send_data(-1)  # Reset base

                    break

                action, is_random = self.get_next_action(state)
                self.__connector.send_data(int(action))

                next_state, reward, terminal, crashed = self.__server.receive_data()

                cumulative_reward_greedy += reward["greedy"]
                cumulative_reward_safe += reward["safe"]

                if terminal:
                    reach_count += 1

                self.greedy_model.memory.remember_experience((state["greedy"], action, reward["greedy"],
                                                              next_state["greedy"], terminal))
                self.safe_model.memory.remember_experience((state["safe"], action, reward["safe"],
                                                            next_state["safe"], crashed))

                loss_greedy += self.experience_replay(model=self.greedy_model)
                loss_safe += self.experience_replay(model=self.safe_model)

                self.report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action, is_random)
                state = next_state

            self.save_results(results, state, cumulative_reward_greedy, cumulative_reward_safe, step, reach_count)

            if reach_count > 0:
                self.save_models()

        _ = self.__server.receive_data()
        self.__connector.send_data(-2)  # Stop simulation

        return results

    @staticmethod
    def report(step, episode, epoch, loss_greedy, loss_safe, reach_count, state, action, is_random):
        print("Step {} Epoch {:03d}/{:03d} | Loss Greedy {:.2f} | Loss Safe {:.2f} | Reach count {} | State {} "
              "| Act {} | Random Act {}".format(step, episode, (epoch - 1), loss_greedy, loss_safe, reach_count, state,
                                                (action - 2), is_random))

    @staticmethod
    def build_results():
        return {
            "reach_counts": [],
            "steps_per_episode": [],
            "distance_per_episode": [],
            "cumulative_reward_per_episode": {
                "Both": [],
                "Greedy": [],
                "Safe": []
            }
        }

    @staticmethod
    def save_results(results, state, cumulative_reward_greedy, cumulative_reward_safe, step, reach_count):
        results["reach_counts"].append(reach_count)
        results["steps_per_episode"].append(step - 1)
        results["distance_per_episode"].append(state["greedy"][0][0])

        results["cumulative_reward_per_episode"]["Both"].append(cumulative_reward_greedy + cumulative_reward_safe)
        results["cumulative_reward_per_episode"]["Greedy"].append(cumulative_reward_greedy)
        results["cumulative_reward_per_episode"]["Safe"].append(cumulative_reward_safe)

