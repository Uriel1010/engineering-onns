import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import The_Free_Spoken_Digit_dataset as ds
from tqdm import tqdm


def sigmoid(x):
    # clip the input values between -500 and 500 to avoid overflow errors
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))



def relu(x):
    return x if x > 0 else 0


def activate(inputs, weights, af=sigmoid):
    # num = 0
    # for i in range(len(inputs)):
    #     num += inputs[i] * weights[i]
    num = np.dot(weights, inputs)
    return af(num)


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

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0]=inputs

        for i,w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self.activation_func(net_inputs)
            self.activations[i+1] = activations
        return activations

    def back_propagate(self,error,verbose=False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshape = delta.reshape(delta.shape[0], -1).T
            current_activation = self.activations[i]
            current_activation_reshape = current_activation.reshape(current_activation.shape[0],-1)
            self.derivatives[i] = np.dot(current_activation_reshape, delta_reshape)
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print(f"Derivatives for W{self.derivatives[i]}")
        return error

    def gradient_decent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print(f"original W{i}\t: {weights}")
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            #print(f"Updated W{i}\t: {weights}")

    # def train(self,inputs,targets,epochs,learning_rate):
    #     for i in range(epochs):
    #         sum_error = 0
    #         for input,target in zip(inputs,targets):
    #             results = self.forward_propagate(input)
    #             error = target - results
    #             self.back_propagate(error)
    #             self.gradient_decent(learning_rate)
    #             sum_error += self._mse(target, results)
    #         #print(f"Error: {sum_error/len(inputs)} at epoch {i}")
    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            with tqdm(total=len(inputs)) as pbar:
                for j in range(len(inputs)):
                    input = inputs[j]
                    target = targets[j]
                    results = self.forward_propagate(input)
                    error = target - results
                    self.back_propagate(error)
                    self.gradient_decent(learning_rate)
                    sum_error += self._mse(target, results)
                    pbar.update(1)
            print(f"Error: {sum_error / len(inputs)} at epoch {i}")



    def activation_func(self, x):
        return sigmoid(x)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x )

    def _mse(self, target, results):
        return np.average((target - results)**2)


if __name__ == "__main__":
    # load TIDIGITS dataset, extract features, and preprocess data
    x_data, y_data = ds.load_data("free-spoken-digit-dataset/recordings")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    # create an instance of the MLP class
    num_inputs = len(x_train[0])  # number of features extracted from audio data
    num_hidden = [100]#, 200]  # number of nodes in each hidden layer
    num_outputs = len(set(y_train))  # number of classes (digits)
    mlp = MLP(num_inputs, num_hidden, num_outputs)

    # train the network
    mlp.train(x_train, y_train, epochs=50, learning_rate=0.1)

    # test the network on the test set
    predictions = []
    for input, target in zip(x_test, y_test):
        prediction = mlp.forward_propagate(input)
        predictions.append(np.argmax(prediction))

    # evaluate performance
    acc = np.mean(np.array(predictions) == y_test)
    print(f"Test Accuracy: {acc:.4f}")
