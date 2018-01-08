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

    for i in results.keys():
        result = results[i]

        plt.xlabel("Episode")
        plt.ylabel(result_types[i])

        try:
            legends = []

            for j in result.keys():
                line, = plt.plot(result[j], label=j)
                legends.append(line)

            plt.legend(legends, list(result.keys()))
        except AttributeError:
            plt.plot(result)

        plt.savefig("./results/" + i + "_" + str(EPOCH) + "_" + str(MAX_EPISODE_LENGTH) + ".png")
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

