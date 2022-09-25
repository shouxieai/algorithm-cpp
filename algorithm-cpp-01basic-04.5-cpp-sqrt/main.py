import numpy as np
import matplotlib.pyplot as plt


def load(file):
    lines = open(file, "r").readlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    return np.array(list(map(float, lines)))

def plot_lines(xs, ys):

    for x, y in zip(xs, ys):
        plt.plot(x, y, "r*", linewidth=3)
        plt.plot([x, x], [0, y], "r-", linewidth=3)


sqrt_gradient_descent = load("workspace/sqrt_gradient_descent")
sqrt_newton_method1   = load("workspace/sqrt_newton_method1")
sqrt_newton_method2   = load("workspace/sqrt_newton_method2")
sqrt_newton_method3   = load("workspace/sqrt_newton_method3")
xmin = min([sqrt_gradient_descent.min(), sqrt_newton_method1.min(), sqrt_newton_method2.min(), sqrt_newton_method3.min()])
xmax = max([sqrt_gradient_descent.max(), sqrt_newton_method1.max(), sqrt_newton_method2.max(), sqrt_newton_method3.max()])

plt.figure(figsize=(30, 10))

C = 9
x = np.linspace(xmin, xmax, 1000)

y = (x * x - C) ** 2
plt.subplot(1, 4, 1)
plt.title(f"GD y = (x * x - C) ** 2\n{len(sqrt_gradient_descent)} iters")
plt.plot(x, y, "b-", linewidth=1)
y = (sqrt_gradient_descent * sqrt_gradient_descent - C) ** 2
plot_lines(sqrt_gradient_descent, y)

y = 2 * (x * x - C) * 2 * x
plt.subplot(1, 4, 2)
plt.title(f"NT1 y = 2 * (x * x - C) * 2 * x\n{len(sqrt_newton_method1)} iters")
plt.plot(x, y, "b-")
y = 2 * (sqrt_newton_method1 * sqrt_newton_method1 - C) * 2 * sqrt_newton_method1
plot_lines(sqrt_newton_method1, y)

y = (x * x - C) ** 2
plt.subplot(1, 4, 3)
plt.title(f"NT2 y = (x * x - C) ** 2\n{len(sqrt_newton_method2)} iters")
plt.plot(x, y, "b-")
y = (sqrt_newton_method2 * sqrt_newton_method2 - C) ** 2
plot_lines(sqrt_newton_method2, y)

y = x * x - C
plt.subplot(1, 4, 4)
plt.title(f"NT3 y = x * x - C\n{len(sqrt_newton_method3)} iters")
plt.plot(x, y, "b-")
y = sqrt_newton_method3 * sqrt_newton_method3 - C
plot_lines(sqrt_newton_method3, y)

plt.savefig("figure.jpg")