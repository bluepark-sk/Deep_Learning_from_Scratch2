# 2.4.3 SVD에 의한 차원 감소
import numpy as np
import matplotlib.pyplot as plt
from word_statistics import preprocess
from word_similarity import create_co_matrix
from pmi import ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0]) # 동시발생 행렬
print(W[0]) # PPMI 행렬
print(U[0]) # SVD

print(U[0, :2]) # 2차원 벡터로 차원 감소

# 그래프
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()