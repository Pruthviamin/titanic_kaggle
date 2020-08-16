{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(z):\n",
    "    return 1 / (1 + np.exp(-z))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(X, w):\n",
    "    return sigmoid(np.matmul(X, w))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss(X, Y, w):\n",
    "    predictions = predict(X, w)\n",
    "    first_term = Y * np.log(predictions)\n",
    "    second_term = (1 - Y) * np.log(1 - predictions)\n",
    "    return -np.average(first_term + second_term)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient(X, Y, w):\n",
    "    return np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(X, Y, iterations, lr):\n",
    "    w = np.zeros((X.shape[1], 1))\n",
    "    for i in range(iterations):\n",
    "        print(\"Iteration %4d => Loss: %.20f\" % (i, loss(X, Y, w)))\n",
    "        w -= gradient(X, Y, w) * lr\n",
    "    return w"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
