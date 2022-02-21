# 7.1.2 문장 생성 구현
import sys
sys.path.append('..')
import numpy as np
from common.layers import softmax
from common.time_layers import Rnnlm, BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1) # input shape : (N, T) = (1, 1)
            score = self.predict(x) # score shape : (N, T, vocab_size) = (1, 1, vocab_size)
            p = softmax(score.flatten()) # score.flatten().shape : (N * T * vocab_size,) = (vocab_size,) 1차원으로 변경
            
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        
        return word_ids

from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen() # 가중치 초깃값 이용
model.load_params('Rnnlm.pkl') # 학습한 가중치 이용

# 시작(start) 문자와 건너뜀(skip) 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]

# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)