import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv").to_numpy()
x, y = data.T
plt.plot(x, y, "b-")

xm, ym = np.mean(data, axis=0)
xs, ys  = np.std(data, axis=0)
x = (x - xm) / xs
y = (y - ym) / ys

############################# 训练流程 ####################################

k_identity = 0.1
k_sin      = 0.1
k_cos      = 0.1
b          = 0
lr         = 0.01

for i in range(1000):
    predict = x * k_identity + np.sin(x) * k_sin + np.cos(x) * k_cos + b
    loss = 0.5 * ((predict - y) ** 2).sum()

    if i % 100 == 0:
        print(f"Iter: {i}, Loss: {loss:.3f}")

    dk_identity = ((predict - y) * x).sum()
    dk_sin      = ((predict - y) * np.sin(x)).sum()
    dk_cos      = ((predict - y) * np.cos(x)).sum()
    db          = (predict - y).sum()
    k_identity = k_identity - dk_identity * lr
    k_sin      = k_sin      - dk_sin * lr
    k_cos      = k_cos      - dk_cos * lr
    b          = b - db * lr

restore_x = x * xs + xm
restore_y = predict * ys + ym

print(f"模型参数: k_identity = {k_identity:.6f}, k_sin = {k_sin:.6f}, k_cos = {k_cos:.6f}, b = {b:.6f}")
print(f"数据集: xm = {xm:.6f}, ym = {ym:.6f}, xs = {xs:.6f}, b = {ys:.6f}")

year = 2023
x = (year - xm) / xs
predict = x * k_identity + np.sin(x) * k_sin + np.cos(x) * k_cos + b
predict_price = predict * ys + ym
print(f"预计 {year} 年，上海的房价将会是：{predict_price:.3f}元")

plt.plot(restore_x, restore_y, "r-")
plt.savefig("figure.jpg")