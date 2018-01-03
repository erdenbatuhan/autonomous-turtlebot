from matplotlib import pyplot as plt
from agent import Agent
import host


EPOCH = 500
MAX_EPISODE_LENGTH = 500


def plot_learning_curve(distances_per_episode):
    plt.plot(distances_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Distance to destination")

    properties = str(EPOCH) + "_" + str(MAX_EPISODE_LENGTH)
    plt.savefig("distances_per_episode_" + properties + ".png")

    plt.show()


def main():
    connector = host.HostConnector()
    server = host.HostServer()

    server.listen()
    print("Listening vm.")

    agent = Agent(connector=connector, server=server)
    distances_per_episode = agent.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)

    if distances_per_episode is not None:
        plot_learning_curve(distances_per_episode)


if __name__ == '__main__':
    main()

