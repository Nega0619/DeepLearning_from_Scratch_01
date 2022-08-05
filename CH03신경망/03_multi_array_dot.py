'''
a = x1*w1 + x2*w2 + b를 구현 한 것

각 요소들의 배열 크기 값에 유의하면서 작성.
'''

import numpy as np

x = np.array([1.0, 0.5])
w1 = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
b1 = np.array([0.1, 0.2, 0.3])

# # 다차원 배열을 이용한 신경망 연산 구현
# if __name__ == '__main__':
#     a1 = np.dot(x, w1) + b1
#     print(a1)

# # 위 신경망에 활성화 함수까지 있다면 다음과 같습니다.
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# if __name__ == '__main__':
#     a1 = np.dot(x, w1) + b1
#     z1 = sigmoid(a1)
#     print(a1, z1, sep= '\n')