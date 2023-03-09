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

if __name__=="__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print(output)