import host
from matplotlib import pyplot as plt
from agent import Agent


EPOCH = 5000
MAX_EPISODE_LENGTH = 500


def plot_results(results):
    result_types = {
        "distance_per_episode": "Distance to Destination",
        "cumulative_reward_per_episode": "Cumulative Reward",
        "steps_per_episode": "Number of Steps",
        "reach_counts": "Reach Count"
    }

    for result_type in results.keys():
        result = results[result_type]

        plt.xlabel("Episode")
        plt.ylabel(result_types[result_type])

        plt.plot(result)
        plt.savefig("./results/" + result_type + "_" + str(EPOCH) + "_" + str(MAX_EPISODE_LENGTH) + ".png")

        plt.gcf().clear()


def main():
    connector = host.HostConnector()
    server = host.HostServer()

    server.listen()
    print("Listening vm.")

    agent = Agent(connector=connector, server=server)
    results = agent.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)

    plot_results(results=results)


if __name__ == '__main__':
    main()

