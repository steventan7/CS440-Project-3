'''
Implementation for Bot1
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

from Image import Image
import numpy as np
import random

D = 20
ALPHA = 0.005
THRESHOLD = 0.001


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, w):
    return sigmoid(np.dot(x, w))


def loss(x, y, w):
    return -y * np.log(predict(x, w)) - (1 - y) * np.log(1 - predict(x, w))


def stocastic_gradient(X, dataset):
    n = D ** 2 * 5
    w = np.linspace(-0.025, 0.025, n)
    new_error, old_error = 0, THRESHOLD

    while abs(new_error - old_error) >= THRESHOLD:
        img = random.choice(X)
        y = dataset[img]
        x = img.pixels
        w = w - ALPHA * (predict(x, w) - y) * x
        old_error = new_error
        new_error = loss(x, y, w)
        print(abs(new_error - old_error))
    return w, new_error


def create_dataset():
    dataset = {}
    for i in range(5000):
        image = Image()
        image.create_image()
        dataset[image] = image.is_dangerous
    return dataset


if __name__ == '__main__':
    success_rate = 0
    N = 100
    for i in range(N):
        dataset = create_dataset()
        X, y = dataset.keys(), dataset.values()
        split = int(0.80 * len(X))
        training_data = list(X)[:split]
        testing_data = list(X)[split:]

        w = stocastic_gradient(training_data, dataset)[0]
        count = 0
        for image in testing_data:
            ans = dataset[image]
            pred = predict(image.pixels, w)

            if ans == np.round(pred):
                count += 1
        success_rate += count / len(testing_data)
    print(success_rate / N)


