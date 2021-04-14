## data load
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

##
x = mnist['data']
y = mnist['target']

## train test split
x_train, x_val, x_test = x[:50000], x[50000:60000], x[60000:]
y_train, y_val, y_test = y[:50000], y[50000:60000], y[60000:]

## classifier training
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

rf_clf = RandomForestClassifier(n_jobs=-1)
svc = SVC()
et_clf = ExtraTreesClassifier(n_jobs=-1)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('svc', svc), ('et', et_clf)],
    voting='hard'
)

##
from sklearn.metrics import accuracy_score

for clf in (rf_clf, svc, et_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    acc_val = accuracy_score(y_val, y_pred)
    print(clf.__class__.__name__, acc_val)

rf_clf = RandomForestClassifier(n_jobs=-1)
svc = SVC(probability=True)
et_clf = ExtraTreesClassifier(n_jobs=-1)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('svc', svc), ('et', et_clf)],
    voting='soft'
)

for clf in (rf_clf, svc, et_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    acc_val = accuracy_score(y_val, y_pred)
    print(clf.__class__.__name__, acc_val)

##
import time
start = time.time()
rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(x_train, y_train)
print(time.time() - start)
