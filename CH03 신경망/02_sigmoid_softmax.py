import numpy as np

'''
x input이 1개 일 경우 : 하나의 값에 대한 sigmoid 값 계산
x input이 여러개 일 경우 : 여러개 값에 대한 '각각'의 sigmoid값 계산
---------------------------------------------------------------
(1_np.exp(-x))에서 괄호 빼면 overflow 발생 함
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))


# 간단한 softmax
'''
아래 식은 overflow를 고려하지 않은 계산법
'''
def simple_softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

# overflow 고려한 softmax
'''
input이 1개일 경우..는 존재하지 않겠지 분류해야하는데.
input이 여러개 일 경우, np.exp(x)를 통해 모든 요소가 exp값을 갖게되고,
sum_exp_x는 np.sum(exp_x)로 인해 스칼라 값이 된다.
그렇기때문에 y = exp_x / sum_exp_x값은 각 요소별 softmax값이 된다. 
?? softmax는 클래스값이 나와야하는거아닌가? 그냥 요소 전체가 나와도괜찮나?
--> 후에 mnist에서 softmax사용하는곳을 보니 np.argmax()를 이용해서 최대 값을 뽑더라~!
---------------------------------------------------------
현재는 max -> exp 순서로 되어있는데,
exp -> max의 순서로 되어야 하지않나 라는 생각을 했다.

위의 경우에서는 exp를 구할 때, argument값으로 x - max_x로 되어있고,
아래의 경우에서는 sum_exp_x를 구할 때 argument값으로 exp-x - max_exp_x를 해주어야한다고 생각했다.

두 경우 상관은없을것 같은게, softmax는 분류하기 위한 방법이고 그 값 자체가 중요하진 않다.
exp함수의 경우 단조증가함수이기 때문에 각 요소의 값 대소가 변할리가 없으므로 전자나 후자의 방법 모두 가능할 것이라 생각이 든다.
-------------------------------------------------------------------
아니네^^ 
일단 softmax값의 합이 1이 어야하는데 이거부터맞지않음
깔깔 나중에다시볼래 ^_^/
'''
def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x-max_x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return  y

def my_softmax(x):
    exp_x = np.exp(x)
    max_exp_x = np.max(exp_x)
    sum_exp_x = np.sum(exp_x - max_exp_x)
    y = exp_x / sum_exp_x
    return y


if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))
    y = my_softmax(a)
    print(y)
    print(np.sum(y))