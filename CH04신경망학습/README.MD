- 차례

4.1 데이터에서 학습한다.

4.2 손실함수

4.3 수치 미분

4.4 기울기

4.5 학습 알고리즘 구현하기

# 4.1 데이터에서 학습한다.

## ML과 DL

- 두 가지 전부..

어떠한 데이터 묶음을 이용해서 데이터에 대한 인사이트를 얻으려고 하는 것은 동일하다.

- ML Machine Learning의 경우..

데이터를 살펴보거나? 정제하거나, 알고리즘을 적용하여 데이터에 대한 어떤 특징을 뽑아낸다. 그리고 이 특징에 대해서 기계학습을 이용하여 기준을 세우게 된다. 그리고 새로운 데이터가 들어오면 그 기준에 부함하는지 아닌지를 판단하게 된다.

예를 들어, 영상 데이터의 경우, 사람이 HOG나 SIFT와 같은 영상처리 기법을 사용하여 벡터를 추출하고, 이 벡터들에 대한 특징을 기계학습인 SVM, KNN을 이용하여 기준을 세우는 것이다.

즉, 사람의 개입이 필 ! 수 !

- DL Deep Learning의 경우..

사람의 개입이 필요없다. -> End to End machine learning이라고도 함. (데이터 입력에서 목표한 결과를 사람의 개입없이 얻는다는 뜻)

임의의 데이터에서 Weight라는 가중치를 이용하여 '학습'을 반복하면서 자율적으로 데이터에대한 탐색과 기준을 세우는 것.

## test 데이터와 train 데이터로 나누어야 하는 이유

범용 능력을 확인하기 위함

mnist와 같이 생각하면 좋겠다. 글씨체 판독은 '임의의 사람'이 쓴 '임의의 글자'를 구별해 내야한다는 것.

cross validation랑 같이 생각하면 좋을 듯 ^^ 이건 답 외웠으니 스킵한다.


# 4.2 손실 함수 = 비용함수 = cost func = loss func

손실 함수란, 학습이 잘되고 있는지 판단하는 지표. 굳이 콕 집어서 얘기하자면, 얼마나 신경망 성능이 '나쁘냐'를 의미

## 손실함수 종류

### MSE mean squared error

내가 가장 처음 배운거! 

> 1/n * sigma(정답 - 예측)^2

- MSE, MAE의 장단점 비교한 여기 글 좋네요 : 

### RMSE root? mean square error

### CROSS ENTROPY ERROR

> -sigma(정답*log예측값)

> 자연로그 그래프 넣긔

###### 질문

- 여기에 나온 loss func의 예시들은 전부 softmax의 결과값에 대한 loss들이다. 만약 예측이 클래스로 나오면, 혹은 회귀로 나오면? 회귀로나오면 ........................... (나중에 생각해보도록 하자 ^^)



## 미니배치학습

미니배치란, loss 값을 구할 때 모든 값을 고려해서 계산하기에는 시간과 비용이 많이 든다. 이를 위하여, 모든 데이터 중 일부만을 랜덤으로 선택하여 loss값을 계산하는 방법을 미니배치라고 한다.

즉, 미니배치는 loss의 근사값을 구하는 것이다.

손실함수도 이 미니배치값으로 계산하게 된다.

### 손실함수를 사용하는 이유

1. 미분을 이용하여 가중치를 업데이트 하게 되기 때문

미분을 사용하는 이유는, 경사하강법을 적용할 수 있기 때문이다. 기울기를 알면, 해당 가중치를 어느 방향으로, 어느 정도 이동시켜야 할 지를 알 수 있다.

2. 정확도가 아니라 loss값으로 손실함수를 사용하는 이유

정확도는 0의 값이 많이 나오는데 이 지표로 가중치를 업데이트하게 되면, 미분값이 자주 0이 되어 제대로 업데이트가 되지 않는다.

또한, 연속된 값이 아니라, 이산적인 값이기 때문에 가중치를 미세하게 변경하였을 때 loss에 어느정도의 영향을 미치는지 추적하기 어렵다.

위 이유는, 계단함수를 activation function으로 사용하지 않는 이유와도 일맥상통합니다.

# 4.3 수치미분

- 수치 미분을 컴퓨터로 구현할 경우 두 가지 문제점이 발생합니다.

1. ( f(x+h) - f(x) ) / h 미분에서 h값을 너무 작게 설정할 시, rounding error가 발생한다.

> np.float32(1e-50) 출력값 : 0.0

1e-50은 매우 작은 숫자 임에도 불구하고, 0.0으로 되어 올바르게 표현할 수 없습니다. 너무 작은 값을 이용하면 컴퓨터가 계산을 하지 못하는 것.

해결책 : 10^(-4) 정도의 값을 사용한다.

cf. 1e-50 은 0.00000 ... 1 : 소수점 아래 0이 50개

2. 차분에 의해 발생하는 수치 미분과 해석적 미분의 차이

- 실제 미분과 차분의 차이

실제 미분 : 접선에서의 기울기

차분 : x + h 와 x사이의 함수 f의 차분 즉, 근사값

이로인한 오차가 발생한다 (아래 사진 참고)

> 사진넣긔

해결책 : f(x+h) - f(x-h)일 때의 f차분계산 (중심 차분, 중앙 차분)

cf. 차분 : 임의 두 점에서 함수 값들의 차이
cf. 전방 차분 : f(x+h)와 x의 차분
cf. 아주 작은 차분으로 미분하는 것: 수치미분, 수식을 전개해 미분하는것은 해석적 analytic이라는 말을 이용하여 해석적으로 미분하다 등으로 표현합니다. 해석적 미분은 오차를 포함하지 않는 진정한 미분값을 구해줍니다. (해석적 미분 : 수학시간에 배운 미분, 수치 미분 : 근사치 계산)

- 편미분을 구할 때는..

> f = x[0]**2 + x[1]**2

라는 식이 있을 때, 편미분을 구하려면, x[0]에 대해 구할 때는 x[1]의 값을, x[1]에 대해 구할 때는 x[0]에 대한 값을 대입해서, 하나의 변수에 대한 식을 구한 다음, 미분을 해줍니다.


# 4.4 기울기

기울기란, 모든 변수의 편미분을 벡터로 정리한 것 입니다.

> 이미지 넣기

기울기를 가시화 하면 다음과 같으며, 기울기 벡터는 현재 위치에서 함수의 가장 낮은 값(최솟값)을 가리킵니다.

정확히는, 기울기 벡터는 각 지점에서 기울기가 낮아지는 방향을 가리킵니다. (local_minimum을 의미, global minimum일 수도 있음)

최솟값과 거리가 멀 수록 벡터의 크기가 커집니다.

- 기울기가 어디에 쓰일까? 경사하강법

경사하강법은 기울기를 이용하여 가중치의 업데이트 값을 조절하는 방식입니다.

> w = w - learning_rate * 기울기(dl/dx)

- 경사하강법의 업데이트 값의 형상 = W의 형상과 동일해야 함

