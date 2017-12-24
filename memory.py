class Memory:

    def __init__(self, max_memory):
        self.__memory = []
        self.__max_memory = max_memory

    def remember(self, experience):
        self.__memory.append(experience)
        if len(self.__memory) > self.__max_memory:
            del self.__memory[0]

