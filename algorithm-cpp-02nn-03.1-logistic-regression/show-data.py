import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("workspace/shanghai.csv")
happiness   = data.query("happiness == 1")
unhappiness = data.query("happiness == 0")

plt.plot(happiness.area,   happiness.distance, "b*", label="happiness")
plt.plot(unhappiness.area, unhappiness.distance, "r*", label="unhappiness")
plt.legend()
plt.xlim([0, 500])
plt.ylim([0, 6000])
plt.xlabel("House Area")
plt.ylabel("Distance to subway")
plt.title("Are you living in happiness?")
plt.savefig("figure.jpg")