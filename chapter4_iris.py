## 데이터셋로드
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

## 데이터셋 준비
x = iris['data']
y = iris['target']

x_bias = np.c_[np.ones([len(x), 1]), x]

size = len(y)

## target onehot
y_onehot = np.zeros((len(y), len(np.unique(y))))
y_onehot[np.arange(len(y)), y] = 1

## train test split
rnd_idx = np.arange(size)
np.random.shuffle(rnd_idx)

split_idx = np.int(np.around(size * 0.8))

x_train, x_test = x[rnd_idx[: split_idx]], x[rnd_idx[split_idx:]]
y_train, y_test = y_onehot[rnd_idx[: split_idx]], y_onehot[rnd_idx[split_idx:]]

## softmax
class Softmax_regression:
    def __init__(self, x, y, lr=0.01):
        self.theta = np.random.randn(x.shape[1], y.shape[1])
        self.lr = lr

    def predict_proba(self, x):
        s = x.dot(self.theta)
        s_exp = np.exp(s)
        p = s_exp / np.sum(s_exp, axis=-1)[:, np.newaxis]
        return p

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=-1)

    def loss(self, x, y):
        size = len(y)
        p = self.predict_proba(x)
        return - np.sum(y * np.log(p)) / size

    def fit(self, x, y):
        size = len(y)
        pred_proba = self.predict_proba(x)
        grad = x.T.dot(pred_proba - y) / size
        self.theta -= self.lr * grad

## training
num_epoch = 10000
lr = 0.1
sof_reg = Softmax_regression(x_train, y_train, lr=lr)

loss_minimum = np.inf
count = 0
for epoch in range(num_epoch):
    sof_reg.fit(x_train, y_train)
    loss = sof_reg.loss(x_train, y_train)
    val_loss = sof_reg.loss(x_test, y_test)
    if epoch % 50 == 0:
        print("EPOCH : %04d | TRAIN : %.3f | VAL : %.3f" % (epoch, loss, val_loss))

    if val_loss > loss_minimum:
        count += 1
        print("%d over fitting !!!!!!!!!!!!!!" % count)
        print("LOSS : %.3f | MINIMUM LOSS : %.3f" % (val_loss, loss_minimum))
        if count >= 10:
            break
    else:
        loss_minimum = val_loss
        count = 0

##

