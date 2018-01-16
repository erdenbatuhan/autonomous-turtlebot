import host
from agent import Agent


EPOCH = 250
MAX_EPISODE_LENGTH = 125


def main():
    connector = host.HostConnector()
    server = host.HostServer()

    server.listen()
    print("Listening vm.")

    agent = Agent(connector=connector, server=server)
    agent.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)


if __name__ == '__main__':
    main()

