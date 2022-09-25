import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv").to_numpy()
area_mean, distance_mean, _ = np.mean(data, axis=0)
area_std, distance_std, _   = np.std(data, axis=0)

area, distance, label = data.T
area     = (area     - area_mean)     / area_std
distance = (distance - distance_mean) / distance_std

############################# 训练流程 ####################################
data_matrix  = np.stack([np.ones((len(data),)), area, distance], axis=-1)
label_matrix = label.T.reshape(-1, 1)
label_matrix[label_matrix == 0] = -1
theta        = np.array([0, 0.1, 0.1]).reshape(3, 1) #np.random.normal(0, 1, size=(3, 1))

lr = 1
batch_size = len(data)

for i in range(10):
    r        = data_matrix @ theta - label_matrix
    loss     = (r**2).sum() / batch_size
    print(f"Iter: {i}, Loss: {loss:.3f}")

    J        = data_matrix
    grad     = np.linalg.inv(J.T @ J) @ J.T @ r
    theta    = theta - lr * grad

k_area     = theta[1, 0]
k_distance = theta[2, 0]
b          = theta[0, 0]

print(f"k_area = {k_area}\nk_distance = {k_distance}\nb = {b}")
print(f"area_mean = {area_mean:.6f}\narea_std = {area_std:.6f}\ndistance_mean = {distance_mean:.6f}\ndistance_std = {distance_std:.6f}")