# And Gate 기능을 가진 퍼셉트론 구현

def and_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp > theta:
        return 1
    else:
        return 0

if __name__ == '__main__':
    print(and_gate(0,0))
    print(and_gate(1,0))
    print(and_gate(0,1))
    print(and_gate(1,1))