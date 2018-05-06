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
        self.discount_factor = discount_factor

        self.memory = Memory(model_name=name, max_memory=max_memory)
        self.models = [self.build_model(input_size, output_size, hidden_size, num_layers, learning_rate),
                       self.build_model(input_size, output_size, hidden_size, num_layers, learning_rate)]

    @staticmethod
    def build_model(input_size, output_size, hidden_size, num_layers, learning_rate):
        model = Sequential()

        model.add(Dense(hidden_size, input_shape=(input_size, )))
        model.add(LeakyReLU(alpha=0.01))

        for i in range(num_layers - 1):
            model.add(Dense(hidden_size))
            model.add(LeakyReLU(alpha=0.01))

        model.add(Dropout(.3))
        model.add(Dense(output_size, activation="linear"))

        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")

        return model

    def load_model(self):
        path = self.MODELS_DIRECTORY + "/" + self.name + "_model.h5"

        try:
            for model in self.models:
                model.load_weights(filepath=path)
        except OSError:
            print("No pre-saved model found for " + self.name + " model.")

    def save_model(self):
        path = self.MODELS_DIRECTORY + "/" + self.name + "_model.h5"

        self.models[0].save_weights(filepath=path, overwrite=True)
        print("Model of " + self.name + " model saved.")

