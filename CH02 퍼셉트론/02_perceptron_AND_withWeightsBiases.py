# 가중치와 편향이 적용한 AND Gate 퍼셉트론 적용

import numpy as np

def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x, w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    print(and_gate(0, 0))
    print(and_gate(1, 0))
    print(and_gate(0, 1))
    print(and_gate(1, 1))