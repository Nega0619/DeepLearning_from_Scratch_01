import numpy as np

# 오차 제곱합 SSE sum of squares for error
# 데이터가 1차원이 아닐 수 있으므로 np.sum을 취한다. 책에서는 y_k, t_k로 표기되어 있다.
def sum_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 교차 엔트로피 오차 Cross Entrophy Error
'''
아래와 같이 하면 발생하는 문제
1. 잠와.
'''
def preivious_cross_entrophy(y, t):
    return -np.sum(t*np.log(y))