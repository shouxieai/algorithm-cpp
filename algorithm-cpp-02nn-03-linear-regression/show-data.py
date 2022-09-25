import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv")
data_array = data.to_numpy()

plt.plot(data.year, data.price, "b-")
plt.xlim([2000, 2022])
plt.ylim([0, 60000])
plt.savefig("figure.jpg")