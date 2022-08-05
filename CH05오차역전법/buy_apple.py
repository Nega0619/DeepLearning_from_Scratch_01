import sys, os
sys.path.append(os.getcwd())
from DeepLearning_from_Scratch_01.CH05오차역전법.layer_naive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print('순전파 price', price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print('역전파:',dapple, dapple_num, dtax)