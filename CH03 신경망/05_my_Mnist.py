# DeepLearning_from_scratch_01 깃허브에 있는 mnist.py 가져온 것
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# normalize : 입력이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화 할 것인지 여부 결정
# flatten, normalize 말고도 one-hot-label도 존재

# 각 데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)