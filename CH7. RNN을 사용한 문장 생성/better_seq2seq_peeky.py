# 7.4.2 엿보기(Peeky)
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H+D, 4*H) / np.sqrt(H+D)).astype('f') # peeky
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H+H, V) / np.sqrt(H+H)).astype('f') # peeky
        affine_b = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        
        self.cache = None
    
    def forward(self, xs, h):
        self.lstm.set_state(h)

        N, T = xs.shape
        N, H = h.shape

        xs = self.embed.forward(xs) # xs shape : (N, T, D)

        hs = np.repeat(h, T, axis=0).reshape(N, T, H) # hs shape : (N, T, H)
        out = np.concatenate((hs, xs), axis=2) # out shape : (N, T, H+D)
        out = self.lstm.forward(out) # out shape : (N, T, H)

        out = np.concatenate((hs, out), axis=2) # out shape : (N, T, H+H)
        score = self.affine.forward(out) # score shape : (N, T, V)

        self.cache = H

        return score
    
    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore) # dout shape : (N, T, H+H)
        dh_affine, dout = np.split(dout, [H], axis=2) # dh_affine shape : (N, T, H) / dout shape : (N, T, H)
        dh_affine = np.sum(dh_affine, axis=1) # dh_affine shape : (N, H)
        
        dout = self.lstm.backward(dout) # dout shape : (N, T, H+D)
        dh_lstm, dout = np.split(dout, [H], axis=2) # dh_lstm shape : (N, T, H) / dout shape : (N, T, D)
        dh_lstm = np.sum(dh_lstm, axis=1) # dh_lstm shape : (N, H)

        dh = self.lstm.dh # dh shape : (N, H)

        dh = dh_affine + dh_lstm + dh # dh shape : (N, H)

        return dh
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1] # h shape : (N, H) = (1, H)
        peeky_h = h.reshape(1, 1, H) # peeky_h shape : (1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1)) # x shape : (N, T) = (1, 1)
            out = self.embed.forward(x) # out shape : (N, T, D) = (1, 1, D)

            out = np.concatenate((peeky_h, out), axis=2) # out shape : (N, T, H+D) = (1, 1, H+D)
            out = self.lstm.forward(out) # out shape : (N, T, H) = (1, 1, H)

            out = np.concatenate((peeky_h, out), axis=2) # out shape : (N, T, H+H) = (1, 1, H+H)
            score = self.affine.forward(out) # out shape : (N, T, V) = (1, 1, V)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
        
        return sampled

from seq2seq import Encoder, Seq2seq
from common.time_layers import TimeSoftmaxWithLoss

class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)
        
        V, D, H = vocab_size, wordvec_size, hidden_size

        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads