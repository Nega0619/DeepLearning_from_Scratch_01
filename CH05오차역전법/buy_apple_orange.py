from layer_naive import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_oragne_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer(apple, apple_num)
orange_price = mul_oragne_layer(orange, orange_num)
all_price = add_apple_orange_layer(apple_price, orange_price)
price = mul_tax_layer(tax, all_price)

print('최종 가격:', price)

# 역전파
dout = 1
dall_price, dtax = mul_tax_layer.backward(dout)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_oragne_layer.backward(dorange_price)

print('역전파 :', dapple, dapple_num, dorange, dorange_num, dapple_price, dorange_price, dtax, dout)
