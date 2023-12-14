'''
Implementation for Bot1
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

from colorama import init, Back, Style
from Image import Image
import numpy as np
import random
init(autoreset=True)

D = 10
alpha = 0.05


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, w):
    return sigmoid(np.dot(x, w))


def loss(x, y, w):
    return -y * np.log(predict(x, w)) - (1 - y) * np.log(1 - predict(x, w))


def stocastic_gradient(X, dataset, alpha):
    n = D ** 2
    w = np.zeros(n)
    newerror, olderror = 0, 0

    threshold = 0.0001
    while abs(newerror - olderror) >= threshold:
        img = random.choice(X)
        y = dataset[img]
        x = img.pixels
        w = predict(x, w) - alpha * (predict(x, w) - y) * x
        olderror = newerror
        newerror = loss(x, y, w)
    return w, newerror



def create_dataset():
    dataset = {}
    for i in range(500):
        image = Image()
        image.create_image()
        dataset[image] = image.is_dangerous
    return dataset


if __name__ == '__main__':
    dataset = create_dataset()
    X, y = dataset.keys(), dataset.values()
    split = int(0.75 * len(X))
    training_data = list(X)[:split]
    testing_data = list(X)[split:]

    w = stocastic_gradient(training_data, dataset, 0.005)[0]

    count = 0
    for image in testing_data:
        ans = dataset[image]
        pred = predict(image.pixels, w)

        if ans == np.round(pred):
            count += 1
    print(count / len(testing_data))