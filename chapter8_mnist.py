##
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

##
from sklearn.model_selection import train_test_split

x = mnist['data']
y = mnist['target']

##
x_train, x_test, y_train, y_test = \
    x[:60000], x[60000:], y[:60000], y[60000:]

##
from sklearn.ensemble import RandomForestClassifier
import time

rnd_clf = RandomForestClassifier()

start = time.time()
rnd_clf.fit(x_train, y_train)
print(time.time() - start) # 60.60

##
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, rnd_clf.predict(x_test))
print(acc) # 0.97

##
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
reduced_x_train = pca.fit_transform(x_train)

rnd_clf = RandomForestClassifier()

start = time.time()
rnd_clf.fit(reduced_x_train, y_train)
print(time.time() - start) # 97.45

##
reduced_x_test = pca.transform(x_test)
acc = accuracy_score(y_test, rnd_clf.predict(reduced_x_test))
print(acc) # 0.94

##
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)

start = time.time()
log_clf.fit(x_train, y_train)
print(time.time() - start) # 17

##
acc = accuracy_score(y_test, log_clf.predict(x_test))
print(acc) # 0.92

##
log_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)

start = time.time()
log_clf.fit(reduced_x_train, y_train)
print(time.time() - start) # 5.5

##
acc = accuracy_score(y_test, log_clf.predict(reduced_x_test))
print(acc) # 0.92

##
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
x_train_reduced = tsne.fit_transform(x_test)

##
import matplotlib.pyplot as plt

plt.plot(x_train_reduced[:, 0], x_train_reduced[:, 1], '.', s=1)
plt.show()

##
import matplotlib.pyplot as plt
for i in range(10):
    x_ = x_train_reduced[y_test == i]
    name = str(i)
    plt.scatter(x_[:, 0], x_[:, 1], s=1, label=name)
plt.legend(loc='best')
plt.show()

##
plt.figure(figsize=(10, 7))
plt.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], c=y_test, cmap='jet', s=2)
plt.colorbar()
plt.show()

##
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_test_reduced = pca.fit_transform(x_test)

##
plt.figure(figsize=(10, 7))
plt.scatter(x_test_reduced[:, 0], x_test_reduced[:, 1], c=y_test, cmap='jet', s=3)
plt.colorbar()
plt.show()

##
from 
