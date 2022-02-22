# 8.1.5 Decoder 개선 3
from turtle import forward
import numpy as np
from better_decoder_1 import WeightSum
from better_decoder_2 import AttentionWeight

class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None
    
    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout=1):
        dhs1, da = self.weight_sum_layer.backward(dout)
        dhs2, dh = self.attention_weight_layer.backward(da)
        dhs = dhs1 + dhs2
        return dhs, dh

# Time Attention 계층
class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None
    
    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape

        self.layers = []
        self.attention_weights = []
        out = np.empty_like(hs_dec)

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        
        return out
    
    def backward(self, dout=1):
        N, T, H = dout.shape

        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in reversed(range(T)):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        
        return dhs_enc, dhs_dec