# 8.2.1 Encoder 구현
import sys
sys.path.append('..')
from common.time_layers import *
from better_decoder_3 import TimeAttention

class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs
    
    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

# 8.2.2 Decoder 구현
class AttentionDecoder(Decoder):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)

        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.cache = H
        
    def forward(self, xs, enc_hs):
        self.lstm.set_state(enc_hs[:, -1, :])

        embed = self.embed.forward(xs)
        dec_hs = self.lstm.forward(embed)
        c = self.attention.forward(enc_hs, dec_hs)
        
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score
    
    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout1, dout2 = np.split(dout, [H], axis=2)

        dhs_enc, dhs_dec = self.attention.backward(dout1)

        dhs_dec += dout2
        dhs_dec = self.lstm.backward(dhs_dec)
        dhs_dec = self.embed.backward(dhs_dec)

        dhs_enc[:, -1, :] += self.lstm.dh

        return dhs_enc
    
    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(enc_hs[:, -1, :])

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape(1, 1)

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)

            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        
        return sampled

# 8.2.3 seq2seq 구현
class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)

        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads