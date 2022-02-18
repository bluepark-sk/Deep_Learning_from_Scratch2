# 6.3 LSTM 구현
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:4*H]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        
        return h_next, c_next
    
    def backward(self, dh, dc):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        dc = (dc + dh * o * (1 - (np.tanh(c_next)**2)))
        dc_prev = dc * f

        df = dc * c_prev * f * (1-f)
        dg = dc * i * (1 - g**2)
        di = dc * g * i * (1 - i)
        do = dh * np.tanh(c_next) * o * (1 - o)
        
        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = np.sum(dA, axis=0)

        dh_prev = np.dot(dA, Wh.T)
        dx = np.dot(dA, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev