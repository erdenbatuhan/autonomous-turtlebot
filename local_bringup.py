from local_agent import Agent


def main():
    print("Hello Turtle!")

    env = Environment()
    env.reset_base()

    agent = Agent(env)
    agent.train()


if __name__ == '__main__':
    main()

