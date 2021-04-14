## datasets load
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=10000, noise=0.4)

## split train test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2)

## gird search
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param = {'max_leaf_nodes': list(range(1, 50))}
search_cv = GridSearchCV(DecisionTreeClassifier(),
                         param_grid=param, scoring='accuracy',
                         n_jobs=None,
                         cv=3, verbose=2, return_train_score=True)
search_cv.fit(x, y)

## test
from sklearn.metrics import accuracy_score

test_acc = accuracy_score(y_test, search_cv.predict(x_test))

## random forest
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=1000, test_size=len(x_train) - 100)

## forest train
from sklearn.base import clone
forest = [clone(search_cv.best_estimator_) for _ in range(1000)]
acc_lst = []

for clf, idx in zip(forest, rs.split(x_train)):
    x_split = x_train[idx[0]]
    y_split = y_train[idx[0]]

    clf.fit(x_split, y_split)

    acc_lst.append(accuracy_score(y_test, clf.predict(x_test)))

## test
import numpy as np

print(np.mean(acc_lst))

##
from scipy.stats import mode

a = np.array([[1,2,3,4],
              [3,4,1,3],
              [3,2,3,4]])
print(mode(a))

##
pred_lst = [clf.predict(x_test) for clf in forest]
pred_forest = mode(pred_lst)

##
print(accuracy_score(y_test, pred_forest.mode.reshape(-1, 1)))

##

