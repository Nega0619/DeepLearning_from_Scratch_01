# 수치 미분의 나쁜 예
'''
1e-50 : 매우 매우 작은 값.
파이썬의 경우, 소수점 아래 0이 8자리?정도를 넘어가면 그냥 0.0으로 표기합니다. 
반올림 오차 rounding error 문제 때문입니다.
'''
def vanishing_numerical_differentiation(f,x):
    h = 1e-50
    return (f(x+h) - f(x)) / h

# 중앙 차분이 적용된 미분값
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)- f(x-h))/ 2*h

