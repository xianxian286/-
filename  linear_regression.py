import pandas as pd
import numpy as np

# 从文件中读取数据（需要提供文件路径）
file_path = 'advertising_data.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 提取解释变量（TV, radio, newspaper）和目标变量（sales）
X = data[['TV', 'radio', 'newspaper']].values
Y = data['sales'].values

# 构造矩阵 A：在解释变量后添加一列 1 表示截距项
A = np.hstack([X, np.ones((X.shape[0], 1))])

# 计算最小二乘法解：x* = (A^T A)^(-1) A^T b
AT_A = np.dot(A.T, A)  # A^T A
AT_Y = np.dot(A.T, Y)  # A^T Y
x_star = np.linalg.solve(AT_A, AT_Y)  # 解方程 (A^T A)x = A^T Y

# 提取回归系数和截距
coefficients = x_star[:-1]  # 回归系数 (TV, radio, newspaper)
intercept = x_star[-1]       # 截距

print("回归系数:", coefficients)
print("截距:", intercept)

# 使用模型预测销售额
y_pred = np.dot(A, x_star)

# 计算误差平方和 (SSE) 和均方误差 (MSE)
sse = np.sum((Y - y_pred) ** 2)
mse = sse / len(Y)

# 计算决定系数 (R^2)
ss_total = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - sse / ss_total

print("误差平方和 (SSE):", sse)
print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r_squared)
