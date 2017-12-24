from matplotlib import pyplot as plt
from agent import Agent
import host


EPOCH = 1000
MAX_EPISODE_LENGTH = 1000


def plot_learning_curve(episodes):
    plt.plot(episodes)
    plt.xlabel("Episode")
    plt.ylabel("Length of episode")
    plt.show()


def main():
    connector = host.HostConnector()
    server = host.HostServer()
    server.listen()

    agent = Agent(connector=connector, server=server)
    episodes = agent.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)

    plot_learning_curve(episodes)


if __name__ == '__main__':
    main()

