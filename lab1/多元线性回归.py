import numpy as np
import matplotlib.pyplot as plt

# 生成数据点


N = 30
x = np.reshape(np.linspace(0, 0.9, N), (N, 1))
y = np.cos(10 * x ** 2) + 0.1 * np.sin(100 * x)

plt.scatter(x, y, color='blue', label='Data points')
plt.title('Scatter')
plt.legend()
plt.show()


def gaussian(x, mean, scale):
    gaussian_vec=np.array([np.exp(-(x - mu) ** 2 / (2 * scale ** 2)) for mu in mean])
    return np.array(np.append(np.ones(x.shape[0]),gaussian_vec).reshape(1 + len(mean), len(x)).T)


points = np.reshape(np.linspace(0, 0.9, 300), (300, 1))#取300个均衡点，转化成一个300*1的向量
fit_x = gaussian(points, [0.1, 0.3, 0.9], 0.2)

fit_y = np.cos(10*points**2) + 0.1 * np.sin(100*points)
w = np.linalg.inv(fit_x.T @ fit_x) @ fit_x.T @ fit_y  # w=((X.T * X)-1*X.T*y).T这里的*在代码里是矩阵乘法用@
plt.plot(points, fit_x @ w)

plt.scatter(x, y, color='blue', label='Data points')
plt.title('Fitting curve')
plt.legend()
plt.show()