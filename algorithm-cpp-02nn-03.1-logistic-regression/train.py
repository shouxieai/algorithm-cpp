import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv").to_numpy()
area_mean, distance_mean, _ = np.mean(data, axis=0)
area_std, distance_std, _   = np.std(data, axis=0)

area, distance, label = data.T
area     = (area     - area_mean)     / area_std
distance = (distance - distance_mean) / distance_std

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

############################# 训练流程 ####################################
k_area     = 0.1
k_distance = 0.1
b  = 0
lr = 0.1

for i in range(1000):
    predict  = area * k_area + distance * k_distance + b
    logistic = sigmoid(predict)
    loss     = -(label * np.log(logistic) + (1 - label) * np.log(1 - logistic)).sum()

    if i % 100 == 0:
        print(f"Iter: {i}, Loss: {loss:.3f}")

    dk_area     = ((logistic - label) * area).sum()
    dk_distance = ((logistic - label) * distance).sum()
    db          = (logistic - label).sum()
    k_area      = k_area - dk_area * lr
    k_distance  = k_distance - dk_distance * lr
    b           = b - db * lr

print(f"k_area = {k_area}\nk_distance = {k_distance}\nb = {b}")
print(f"area_mean = {area_mean:.6f}\narea_std = {area_std:.6f}\ndistance_mean = {distance_mean:.6f}\ndistance_std = {distance_std:.6f}")

