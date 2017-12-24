class Memory:

    def __init__(self, max_memory):
        self.__memory = []
        self.__max_memory = max_memory

    def __len__(self):
        return len(self.__memory)

    def get_experience(self, i, j):
        return self.__memory[i][j]

    def remember_experience(self, experience):
        self.__memory.append(experience)
        if len(self.__memory) > self.__max_memory:
            del self.__memory[0]

