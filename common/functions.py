import numpy as np

def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x-max_x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

# def cross_entrophy_error(y, t):
#     return -np.sum(t * np.log(y))

def cross_entrophy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(-t * np.log(y + 1e-7)) / batch_size    