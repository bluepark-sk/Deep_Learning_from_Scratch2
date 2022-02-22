# 8.1.3 Decoder 개션 1
import numpy as np

T, H = 5, 4
hs = np.random.randn(T, H)
a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])

ar = a.reshape(5, 1).repeat(4, axis=1)
# print(ar.shape)

t = hs * ar
# print(t.shape)

c = np.sum(t, axis=0)
# print(c.shape)

# 미니배치 처리용 가중합
N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
a = np.random.randn(N, T)
ar = a.reshape(N, T, 1).repeat(H, axis=2) # ar shape : (N, T, H)

t = hs * ar # t shape : (N, T, H)
# print(t.shape)

c = np.sum(t, axis=1) # c shape : (N, H)
# print(c.shape)

# Weight Sum 계층
class WeightSum:
    def __init__(self):
        self.parmas, self.grads = [], []
        self.cache = None
    
    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)

        return c
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dhs = dt * ar
        dar = dt * dhs
        da = np.sum(dar, axis=2)

        return dhs, da