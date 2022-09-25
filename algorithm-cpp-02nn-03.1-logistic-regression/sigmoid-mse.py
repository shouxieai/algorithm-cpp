import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

label = 0
x     = np.linspace(-20, +20, 100)
y0    = (sigmoid(x) - label) ** 2
dx0   = 2 * (sigmoid(x) - label) * sigmoid(x) * (1 - sigmoid(x))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y0, "b-", label="f")
plt.plot(x, dx0, "r-", label="f prime")
plt.title("Sigmoid + MSE")
plt.legend()

##################################################################################################

label = 0
x     = np.linspace(-10, +10, 100)
y0    = -(np.log(sigmoid(x)) * label + np.log(1 - sigmoid(x)) * (1 - label))
dx0   = sigmoid(x) - label

plt.subplot(1, 2, 2)
plt.plot(x, y0, "b-", label="f")
plt.plot(x, dx0, "r-", label="f prime")
plt.title("Sigmoid + CrossEntropy")
plt.legend()

plt.savefig("sigmoid-mse.jpg")