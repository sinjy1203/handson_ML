## mnist 데이터 불러오기
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist["data"], mnist["target"]

## train, test 나누기
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

## 정규화
from sklearn.preprocessing import StandardScaler
std_transform = StandardScaler()
x_train_transformed = std_transform.fit_transform(x_train)

## grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

param_grid = [{'weights': ['uniform'], 'n_neighbors': list(range(1, 100))},
              {'weights': ['distance'], 'n_neighbors': list(range(1, 100))}]

knn_clf = KNeighborsClassifier()
random_grid_search = RandomizedSearchCV(knn_clf, param_grid, 10,
                                        scoring='accuracy', cv=3, verbose=2,
                                        return_train_score=True)
random_grid_search.fit(x_train_transformed, y_train)

## training
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_transformed, y_train)

## 테스트
from sklearn.metrics import accuracy_score

x_test_transformed = std_transform.transform(x_test)
y_pred = knn_clf.predict(x_test_transformed)
acc1 = accuracy_score(y_test, y_pred) # 0.944

## data augmentation
import numpy as np

def img_shift(img, y):
    img = img.reshape(-1, 28, 28)
    aug1 = np.zeros(img.shape)
    aug2 = np.zeros(img.shape)
    aug3 = np.zeros(img.shape)
    aug4 = np.zeros(img.shape)

    aug1[:, :-1, :] = img[:, 1:, :] # 위방향
    aug2[:, 1:, :] = img[:, :-1, :] # 아래방향
    aug3[:, :, :-1] = img[:, :, 1:] # 왼쪽방향
    aug4[:, :, 1:] = img[:, :, :-1] # 오른쪽방향

    aug = np.concatenate((img, aug1, aug2, aug3, aug4), axis=0)

    return aug.reshape(-1, 784), np.concatenate((y, y, y, y, y), axis=0)

##
x_train_aug, y_train_aug = img_shift(x_train, y_train)

## 정규화
std_transform = StandardScaler()
x_train_aug_transformed = std_transform.fit_transform(x_train_aug)

## training
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_aug_transformed, y_train_aug)

## test
from sklearn.metrics import accuracy_score

x_test_transformed = std_transform.transform(x_test)
y_pred = knn_clf.predict(x_test_transformed)
acc2 = accuracy_score(y_test, y_pred) # 0.96

##

