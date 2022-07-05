import os
import sys
print(sys.path)
sys.path.append(os.getcwd())
print(sys.path)

from DeepLearning_from_Scratch_01.common.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

