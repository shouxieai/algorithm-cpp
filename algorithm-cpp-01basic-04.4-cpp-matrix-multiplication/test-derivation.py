import torch
import torch.nn as nn


X = nn.parameter.Parameter(torch.tensor([
    [1, 2],
    [2, 1],
    [0, 2]
], dtype=torch.float32))

theta = nn.parameter.Parameter(torch.tensor([
    [5, 1, 0],
    [2, 3, 1]
], dtype=torch.float32))

loss = (X @ theta).sum()
loss.backward()

print(f"Loss = {loss.item()}")
print(f"dloss / dX = \n{X.grad}")
print(f"dloss / dtheta = \n{theta.grad}")

print("================手动计算矩阵的导数===================")
G = torch.ones_like(X @ theta)
X_grad = G @ theta.data.T
theta_grad = X.data.T @ G
print(f"dloss / dX = \n{X_grad}")
print(f"dloss / dtheta = \n{theta_grad}")