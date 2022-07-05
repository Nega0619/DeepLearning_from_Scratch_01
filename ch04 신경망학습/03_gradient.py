import numpy as np

def function_2(x):
    return x[0]**2+ x[1]**2

# 편미분이 적용된 기울기 구하기
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range((x.size)):
        tmp_val = x[idx]

        #f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

        return grad

'''
step_num이란, 경사법에 따른 반복 횟수라고 책에 명시(131pg)

이게 epochs? 인가?
'''
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for n in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x 