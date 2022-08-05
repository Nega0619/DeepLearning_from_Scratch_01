import os
from pyexpat.errors import XML_ERROR_ABORTED
import sys
sys.path.append(os.getcwd())

'''
패키지를 읽어오는 방법이 이거밖엔 없나..
sys.path는 어떻게 사용되는걸까.
거기 리스트에 있는 주소에 from 아랫절을 다 넣어보고 있으면 ok 없으면 error를 내뱉는 구조인가?
'''

from DeepLearning_from_Scratch_01.common.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# mini batch 구하기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # np.random : 0 이상, train_size미만의 수에서 batch_size개의 수를 고르는 것
print('batch_mask:', batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# mini batch의 cross entrophy 구하기
'''
y : softmax의 배열값
t : one_hot_label된 배열값 인 경우에 사용하는 것
'''
def cross_entrophy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(-t * np.log(y + 1e-7)) / batch_size

# 예측 값이 라벨링된 경우 사용되는 mini batch cross_entrophy
'''
얜 어떤방식으로 동작하는지모르겠다는.. 줄 알았는데 갑자기 혼자 해결함 ^^ 자가치료쩐다
------------------------------------------------------------------------------
y : softmax의 배열값
t : 정답 라벨 값 ex) 3 이나 7, 0 등의 숫자 값
'''
def cross_entrophy_error_with_labeled(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshaep(1, y.size)

    batch_size = y.shape[0]
    '''
    batch_size개수의 softmax 된 y값이 존재.
    return y[batch_size번째의, t번째 예측값]을 가져와라. 라는 뜻!
    '''
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e+7)) / batch_size


