# 4.3.3 CBOW 모델 평가
import sys
sys.path.append('..')
from common.util import most_similar
import pickle

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 유추 문제 (비유 문제) = word2vec의 단어 분산 표현을 사용하면 유추 문제를 벡터의 덧셈과 뺄셈으로 풀 수 있다.
# man : woman = king : ?
# vec(woman) - vec(man) = vec(?) - vec(king)
from common.util import analogy

analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs) # 현재형 - 과거형
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs) # 단수형 - 복수형
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs) # 비교급