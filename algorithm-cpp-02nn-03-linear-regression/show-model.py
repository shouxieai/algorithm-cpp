import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

k1, k2, k3, b, xm, ym, xstd, ystd = 0.068349, 1.215169, -0.574370, 0.327223, 2011.000000, 23469.564453, 6.633250, 19034.007812

data = pd.read_csv("workspace/shanghai.csv")
data_array = data.to_numpy()
x, y = data_array.T

x = (x - xm) / xstd
p = x * k1 + np.sin(x) * k2 + np.cos(x) * k3 + b
pred = p * ystd + ym

plt.plot(data.year, data.price, "b-", label="data")
plt.plot(data.year, pred, "r-", label="predict")
plt.xlim([2000, 2022])
plt.ylim([0, 60000])
plt.legend()
plt.savefig("figure.jpg")