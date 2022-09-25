import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv")
happiness   = data.query("happiness == 1")
unhappiness = data.query("happiness == 0")

k_area = -3.7876112879878283
k_distance = -4.1771458278023506
b = 0.09789983280685588
area_mean = 225.294118
area_std = 133.112851
distance_mean = 2252.941176
distance_std = 1816.952085

# 分类线是
# k_area * x + k_distance * y + b = 0
# y = -(k_area * x + b) / k_distance
x      = np.array([0, 500])
norm_x = (x - area_mean) / area_std
norm_y = -(k_area * norm_x + b) / k_distance
y      = norm_y * distance_std + distance_mean

plt.plot(happiness.area,   happiness.distance, "b*", label="happiness")
plt.plot(unhappiness.area, unhappiness.distance, "r*", label="unhappiness")
plt.plot(x, y, "g-")
plt.legend()

plt.xlim([0, 500])
plt.ylim([0, 6000])
plt.xlabel("House Area")
plt.ylabel("Distance to subway")
plt.title("Are you living in happiness?")
plt.savefig("figure.jpg")