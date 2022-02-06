# 1.1.1 벡터와 행렬
import numpy as np

x = np.array([1, 2, 3])
x.__class__
x.shape
x.ndim

W = np.array([[1, 2, 3], [4, 5, 6]])
W.shape
W.ndim

# 1.1.2 행렬의 원소별 연산
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
W + X
W * X

# 1.1.3 브로드캐스트
A = np.array([[1, 2], [3, 4]])
A * 10

A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
A * b

# 1.1.4 벡터의 내적과 행렬의 곱
# 벡터의 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)

# 행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.matmul(A, B)