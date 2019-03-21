#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()

data.insert(0, 'ones', 1)#在data的第0列之前插入一列，列名为'ones',数值为1,然后用向量化的方案计算代价和梯度
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]#iloc:左闭右开，选取0,1，...cols-2这些行赋给X
y = data.iloc[:, cols-1:cols]


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


X = np.matrix(X.values)#必须先将X转换为矩阵，否则类型不匹配
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))
p = computeCost(X, y, theta)
print('Cost:', p)


# def gradientDescent(X, y, theta, alpha, iters):
#     temp = np.matrix(np.zeros(theta.shape))#生成与theta同型的零矩阵
#     parameters = int(theta.shape[1])#括号中为theta的列数,即内层循环的次数
#     cost = np.zeros(iters)#cost数组，长度为1000，初始化为全0，一直更新
#
#     for i in range(iters):
#         error = (X * theta.T) - y
#
#         for j in range(parameters):
#             term = np.multiply(error, X[:, j])#对应元素相乘
#             temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))#theta_j = theta_j - (alpha/m * error * X_j)
#
#         theta = temp
#         cost[i] = computeCost(X, y, theta)
#
#     return theta, cost


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - (alpha / len(X)) * sum(term)

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)


x = np.linspace(data.Population.min(), data.Population.max(), 100)#以最小值为起点，最大值为终点，等距离取100个点，返回得到的数组
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)#图的位置
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()





