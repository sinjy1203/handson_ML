# 데이터 로드
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

##
x = mnist['data']
y = mnist['target']

##
x_train, y_train = x[:60000], y[:60000]
x_test, y_test = x[60000:], y[60000:]

## svc training
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])
param = [
    {'svc__kernel': ['linear'], 'svc__C': [1e-2, 1e-1, 1, 10]},
    {'svc__kernel': ['poly'], 'svc__C': [1e-2, 1e-1, 1, 10], 'svc__degree': [2, 3, 5, 10], 'svc__coef0': [1e-1, 0, 1, 10]},
    {'svc__kernel': ['rbf'], 'svc__C': [1e-2, 1e-1, 1, 10], 'svc__gamma': [1e-2, 1e-1, 1, 10]}
]

search_cv = RandomizedSearchCV(svc, param_distributions=param, n_iter=5, scoring='accuracy', cv=3, return_train_score=True, verbose=2)
search_cv.fit(x_train, y_train)

## test
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, search_cv.predict(x_test)))

##

