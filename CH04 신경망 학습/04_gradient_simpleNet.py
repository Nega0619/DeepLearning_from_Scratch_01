import sys, os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entrophy_error
from common.graident import numerical_gradient
import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        this_loss = cross_entrophy_error(y, t)
        return this_loss
