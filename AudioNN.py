import numpy as np

def sigmoid(x):
    # clip the input values between -500 and 500 to avoid overflow errors
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def activate(inputs, weights):
    # perform net inputs
    h = 0
    for x,w in zip(inputs, weights):
        h += x*w

    # perform activation
    return sigmoid(h)

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # activations = []
        # for i in range(len(layers)):
        #     a = np.zeros(layers[i])
        #     activations.append(a)
        # self.activations = activations
        #
        # derivatives = []
        # for i in range(len(layers) - 1):
        #     d = np.zeros((layers[i], layers[i+1]))
        #     derivatives.append(d)
        # self.derivatives = derivatives


    def forward_propagate(self, inputs):

        activations = inputs
        #self.activations[0]=inputs

        for w in self.weights:
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)
            # activations = self.activation_func(net_inputs)
            # self.activations[i+1] = activations
        return activations

    def _sigmoid(self, x):
        return np.divide(1, 1 + np.exp(-x))


if __name__=="__main__":

    mlp = MLP()

    inputs = np.random.rand(mlp.num_inputs)

    output = mlp.forward_propagate(inputs)

    print(f'The network inputs is: {inputs}')
    print(f'The network outputs is: {output}')