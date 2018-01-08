import pickle


class Memory:

    MEMORIES_DIRECTORY = "./memories"

    def __init__(self, model_name, max_memory):
        self.model_name = model_name
        self.max_memory = max_memory

        self.memory = []
        self.len_memory = 0

    def __len__(self):
        return self.len_memory

    def get(self):
        return self.memory

    def load_memory(self):
        try:
            with open(self.MEMORIES_DIRECTORY + "/" + self.model_name + "_memory.pkl", "rb") as memory_reader:
                self.memory = pickle.load(memory_reader)
        except OSError:
            print("No pre-saved memory found for " + self.model_name + " model.")

        self.len_memory = len(self.memory)

        if self.len_memory > self.max_memory:
            diff = self.len_memory - self.max_memory

            print("Maximum memory size expected ({}), got ({}). Removing first {} elements..".
                  format(self.max_memory, self.len_memory, diff))

            self.memory = self.memory[diff:]
            self.len_memory -= diff

    def save_memory(self):
        with open(self.MEMORIES_DIRECTORY + "/" + self.model_name + "_memory.pkl", "wb") as memory_writer:
            pickle.dump(self.memory, memory_writer)
            print("({}) Memory of " + self.model_name + " model saved.".format(self.len_memory))

    def get_experience(self, i):
        return self.memory[i]

    def remember_experience(self, experience):
        self.memory.append(experience)
        self.len_memory += 1

        if self.len_memory > self.max_memory:
            del self.memory[0]
            self.len_memory -= 1

