# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform

import numpy as np

SIZE_MIDDLE_FIRST_LAYER = 5
NUMBER_OF_OUTPUTS = 2
NUMBER_OF_INPUTS = 4

def activation(x):
    return x


def activation_array(array):
    res = []
    for item in array:
        res.append(activation(item))
    return res


def derivative_sigmoid(x):
    return np.exp(-x) / ((1+np.exp(-x))^2)


class NeuralNetwork:
    # Fields
    input = np.zeros(NUMBER_OF_INPUTS)
    weights_input = np.zeros((SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
    biases_input = np.zeros(SIZE_MIDDLE_FIRST_LAYER)

    neurons_first_layer = np.zeros(SIZE_MIDDLE_FIRST_LAYER)
    weights_first_layer = np.zeros((NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
    biases_first_layer = np.zeros(NUMBER_OF_OUTPUTS)
    output = np.zeros(NUMBER_OF_OUTPUTS)

    # Constructor
    def __init__(self):
        self.weights_input[0][0] = 1
        self.weights_first_layer[0][0] = 1

        # self.weights_input = np.random.uniform(low=-1, high=1, size=(SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
        # self.weights_first_layer = np.random.uniform(low=-1, high=1, size=(NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
        # self.biases_input = np.random.uniform(low=-5, high=5, size=SIZE_MIDDLE_FIRST_LAYER)
        # self.biases_first_layer = np.random.uniform(low=-5, high=5, size=NUMBER_OF_OUTPUTS)

    def activate(self, inputs):
        self.input = inputs
        self.neurons_first_layer = activation_array(np.add(np.dot(self.weights_input, self.input), self.biases_input))
        self.output = activation_array(np.add(np.dot(self.weights_first_layer, self.neurons_first_layer), self.biases_first_layer))
        return self.output

    def loss(self, ):
        pass


def main():
    ml = NeuralNetwork()



if __name__ == '__main__':
    main()

