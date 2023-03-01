import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x if x > 0 else 0


def activate(inputs, weights, af=sigmoid):
    # num = 0
    # for i in range(len(inputs)):
    #     num += inputs[i] * weights[i]
    num = np.dot(weights, inputs)
    return af(num)


if __name__ == "__main__":
    inputs = [1, 2, 3]
    weights = [0.1, 0.5, 0.2]
    result = activate(inputs, weights, relu)
    print(result)
