import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, w):
    return sigmoid(np.matmul(X, w))


def loss(X, Y, w):
    predictions = predict(X, w)
    first_term = Y * np.log(predictions)
    second_term = (1 - Y) * np.log(1 - predictions)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    return np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

