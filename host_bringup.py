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
    agent.train()


if __name__ == '__main__':
    main()

