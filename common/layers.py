import numpy as np

class Relu:
    def __init__(self):
        self.mask = None            # True / False로 구성된 넘파이 배열
                                    # 순전파의 입력인 x의 원소 값이 0 이하인 인덱스는 True, 그 외(0보다 큰 원소)는  False로 유지합니다.
                                    # 역전파때는 순전파 때 만들어둔 mask를 써서 mask가 True인 곳에는 상류에서 전파된 dout을 0으로 설정합니다.

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/ (1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out             # sigmoid를 backpropagation을 이용해서 미분하고, 수식을 정리하면 sigmoid의 backprob값이 y(1-y)로 정리된다는 것을 알 수 있다.

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)                         # 전치행렬을 구하는건 .T 인가보다.
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx