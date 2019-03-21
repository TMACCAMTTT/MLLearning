#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=('population', 'profit'))
data.plot(kind='scatter', x='population', y='profit', figsize=(12, 8))
plt.show()

data.insert(0, 'ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]


def computeCost(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner)/(2 * len(X))


X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0])
cost = computeCost(X, y, theta)
print(cost)

def gradientDescent(X, y, theta, alpha, iters):
    parameters = int(theta.shape[1])
    temp = np.zeros(theta.shape)
    cost = np.zeros(iters)
    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - (alpha/len(X) * sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

alpha = 0.01
iters = 1000
theta, cost = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(data.population.min(), data.population.max(), 100)
f = theta[0, 0] + theta[0, 1] * x

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(x, f, 'r', label='prediction')
# ax.scatter(data.population, data.profit, label='training data')
# ax.legend(loc=2)
# ax.set_xlabel('population')
# ax.set_ylabel('profit')
# ax.set_title('prediction vs training data')
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('iters')
# ax.set_ylabel('cost')
# ax.set_title('Error vs Training Epoch')
# plt.show()

#多变量线性回归
path2 = 'ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()
data2.insert(0, 'ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix([0, 0, 0])

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(computeCost(X2, y2, g2))

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('iters')
ax.set_ylabel('cost2')
ax.set_title('Error vs Training Epoch')
plt.show()
