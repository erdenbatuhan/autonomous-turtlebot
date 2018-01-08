import pickle


class Memory:

    __MEMORIES_DIRECTORY = "./memories"

    def __init__(self, model_name, max_memory):
        self.__model_name = model_name
        self.__max_memory = max_memory

        self.__memory = []
        self.__len_memory = 0

        self.__load_memory()

    def __len__(self):
        return self.__len_memory

    def get(self):
        return self.__memory

    def __load_memory(self):
        try:
            with open(self.__MEMORIES_DIRECTORY + "/" + self.__model_name + "_memory.pkl", "rb") as memory_reader:
                self.__memory = pickle.load(memory_reader)
        except OSError:
            print("No pre-saved memory found for " + self.__model_name + " model.")

        self.__len_memory = len(self.__memory)

        if self.__len_memory > self.__max_memory:
            diff = self.__len_memory - self.__max_memory

            print("Maximum memory size expected ({}), got ({}). Removing first {} elements..".
                  format(self.__max_memory, self.__len_memory, diff))

            self.__memory = self.__memory[diff:]
            self.__len_memory -= diff

    def save_memory(self):
        with open(self.__MEMORIES_DIRECTORY + "/" + self.__model_name + "_memory.pkl", "wb") as memory_writer:
            pickle.dump(self.__memory, memory_writer)
            print("({}) Memory of " + self.__model_name + " model saved.".format(self.__len_memory))

    def get_experience(self, i):
        return self.__memory[i]

    def remember_experience(self, experience):
        self.__memory.append(experience)
        self.__len_memory += 1

        if self.__len_memory > self.__max_memory:
            del self.__memory[0]
            self.__len_memory -= 1

