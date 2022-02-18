import numpy as np
import sys
sys.path.append('..')
from common.layers import Embedding, Affine, SoftmaxWithLoss

# TimeEmbedding 계층 (내가 직접 짠거)
class TimeEmbedding:
    def __init__(self, Wws):
        self.params = [Wws]
        self.grads = [np.zeros_like(Wws)]
        self.layers = None
        self.Wws = Wws
    
    def forward(self, ws):
        N, T = ws.shape
        V, D = self.Wws.shape

        embs = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.Wws)
            embs[:, t, :] = layer.forward(ws[:, t])
            self.layers.append(layer)

        return embs
    
    def backward(self, dembs):
        N, T, D = dembs.shape
        
        grad = 0

        for t in reversed(range(T)):
            layer = self.layers[t]
            layer.backward(dembs[:, t, :])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad

        return None

# TimeAffine 계층 (내가 직접 짠거)
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.layers = None
        self.hs = None
    
    def forward(self, hs):
        W, b = self.params
        N, T, H = hs.shape
        H, D = W.shape
        self.hs = hs

        ys = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Affine(W, b)
            ys[:, t, :] = layer.forward(hs[:, t, :])
            self.layers.append(layer)

        return ys
    
    def backward(self, dys):
        N, T, D = dys.shape
        N, T, H = self.hs.shape

        dxs = np.empty((N, T, H), dtype='float')
        grads = [0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx = layer.backward(dys[:, t, :])
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        return dxs

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x
    
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelㅇㅔ 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx

class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask