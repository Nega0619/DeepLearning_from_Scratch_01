import os
import sys
import numpy as np
import pickle
from mnist import load_mnist


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) 

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__ == '__main__':
    print(__file__)
    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print(os.getcwd())

    # batch 처리 전
    print('\nbatch 전')
    x, t = get_data()
    network = init_network()

    accuracy_count = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_count += 1

    print('Accuracy: ', str(float(accuracy_count)/len(x)))

    # batch 처리 후, axis = 1을 해준 것에 유의
    print('\nbatch 후')
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_count = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis = 1)
        accuracy_count += np.sum(p == t[i:i+batch_size])

    print('Accuracy: ', str(float(accuracy_count)/len(x)))
