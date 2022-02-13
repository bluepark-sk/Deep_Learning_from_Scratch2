# 4.1.2 Embedding 계층 구현
import numpy as np

W = np.arange(21).reshape(7, 3)
print(W)
print(W[2])

idx = np.array([1, 0, 3, 0])
print(W[idx])

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        # dW[self.idx] = dout
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        # np.add.at(dW, self.idx, dout)
        return None