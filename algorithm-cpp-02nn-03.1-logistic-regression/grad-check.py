import math

def example1():

    k_area = 0.1
    k_dist = 0.5
    b = 0.2

    area = 2
    dist = 3
    label = 0

    def f(x):
        predict = x * area + k_dist * dist + b
        logistic = 1 / (1 + math.exp(-predict))
        loss = -(math.log(logistic) * label + math.log(1 - logistic) * (1 - label))

        area_derivative = (logistic - label) * area
        return area_derivative, loss

    area_derivative, f0 = f(k_area)

    dx = 1e-6
    _, f1 = f(k_area + dx)
    derivative = (f1 - f0) / dx

    print(area_derivative)
    print(derivative)


def example2():

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    x = 0.8
    x_derivative = sigmoid(x) * (1 - sigmoid(x))

    dx = 1e-6
    f0 = sigmoid(x)
    f1 = sigmoid(x + dx)
    derivative = (f1 - f0) / dx
    print(x_derivative)
    print(derivative)


example1()
example2()