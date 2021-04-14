## 데이터 로드
from sklearn import datasets

data = datasets.load_iris()

##
x, y = data['data'][:, (2,3)], data['target']
x_s_v, y_s_v = x[y <= 1], y[y <= 1]

## 데이터 시각화
import matplotlib.pyplot as plt

x_s = x_s_v[y_s_v == 0]
x_v = x_s_v[y_s_v == 1]
plt.plot(x_s[:, 0], x_s[:, 1], "b.", label='setosa')
plt.plot(x_v[:, 0], x_v[:, 1], "r.", label='versicolor')
plt.xlabel('length')
plt.ylabel('width')
plt.legend(loc='best')

## LinearSVC, SVC, SGDClassifier training
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
x_s_v_ = std_scaler.fit_transform(x_s_v)

linear_svc = LinearSVC()
linear_svc.fit(x_s_v_, y_s_v)

svc = SVC(kernel='linear')
svc.fit(x_s_v_, y_s_v)

sgd_clf = SGDClassifier()
sgd_clf.fit(x_s_v_, y_s_v)

## 분류기 시각화
import matplotlib.pyplot as plt

a1 = - linear_svc.coef_[0, 0] / linear_svc.coef_[0, 1]
b1 = - linear_svc.intercept_[0] / linear_svc.coef_[0, 1]
line1 = std_scaler.inverse_transform([[-10, -10 * a1 + b1], [10, 10 * a1 + b1]])

a2 = - svc.coef_[0, 0] / svc.coef_[0, 1]
b2 = - svc.intercept_[0] / svc.coef_[0, 1]
line2 = std_scaler.inverse_transform([[-10, -10 * a2 + b2], [10, 10 * a2 + b2]])

a3 = - sgd_clf.coef_[0, 0] / sgd_clf.coef_[0, 1]
b3 = - sgd_clf.intercept_[0] / sgd_clf.coef_[0, 1]
line3 = std_scaler.inverse_transform([[-10, -10 * a3 + b3], [10, 10 * a3 + b3]])

x_s = x_s_v[y_s_v == 0]
x_v = x_s_v[y_s_v == 1]

plt.plot(x_s[:, 0], x_s[:, 1], "b.")
plt.plot(x_v[:, 0], x_v[:, 1], "r.")
plt.plot(line1[:, 0], line1[:, 1], "y-", label='linear_svc')
plt.plot(line2[:, 0], line2[:, 1], "g--", label='svc')
plt.plot(line3[:, 0], line3[:, 1], "c:", label='sgd_classifier')
plt.xlabel('length')
plt.ylabel('width')
plt.axis([0, 5.5, 0, 2])
plt.legend(loc="best")
plt.show()
