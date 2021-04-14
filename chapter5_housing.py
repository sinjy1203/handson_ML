## 데이터 로드
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

##
x, y = housing['data'], housing['target']

##
import numpy as np

size = len(x)

idx = np.random.permutation(size)
train_idx = idx[: int(np.round(size * 0.8))]
test_idx = idx[int(np.round(size * 0.8)):]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

## 훈련
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svr_reg = Pipeline([
    ("scaler", StandardScaler()),
    ('svr', SVR())
])

svr_reg.fit(x_train, y_train)

## 테스트
from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_test, svr_reg.predict(x_test))))
