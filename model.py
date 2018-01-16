from memory import Memory
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


class Model:

    MODELS_DIRECTORY = "./models"

    def __init__(self, name, input_size, output_size, hidden_size, num_layers, max_memory, learning_rate, discount_factor):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.discount_factor = discount_factor

        self.memory = Memory(model_name=name, max_memory=max_memory)
        self.models = [self.build_model(hidden_size, num_layers, learning_rate),
                       self.build_model(hidden_size, num_layers, learning_rate)]

    def build_model(self,hidden_size, num_layers, learning_rate):
        model = Sequential()

        model.add(Dense(hidden_size, input_shape=(self.input_size, )))
        model.add(LeakyReLU(alpha=0.01))

        for i in range(num_layers - 1):
            model.add(Dense(hidden_size))
            model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(self.output_size, activation="linear"))
        model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

        return model

    def get_path(self):
        try:
            path = self.MODELS_DIRECTORY + "/" + self.name + "_model.h5"
        except TypeError:
            path = self.MODELS_DIRECTORY + "/model.h5"

        return path

    def get_error_msg(self):
        try:
            message = "for " + self.name + " model."
        except TypeError:
            message = "for model."

        return message

    def load_model(self):
        try:
            for model in self.models:
                model.load_weights(filepath=self.get_path())
        except OSError:
            print("No pre-saved model found " + self.get_error_msg())

    def save_model(self):
        self.models[0].save_weights(filepath=self.get_path(), overwrite=True)
        print("Model " + self.get_error_msg())

