import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


step_size = 0.2
color = plt.get_cmap('RdBu')
clf_name = "KNN"
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)


data_sets = [make_classification(n_features=2, n_classes=3, n_informative=2, n_redundant=0, random_state=1,
                                 n_clusters_per_class=1),
             make_moons(noise=0.3, random_state=1),
             make_circles(noise=0.2, factor=0.5, random_state=1),
             make_blobs(n_features=2, centers=3, cluster_std=1.0, random_state=1)]


figure = plt.figure(figsize=(8, 8))

plt_idx = 0

for ds_idx, ds in enumerate(data_sets):
    X, y = ds

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (training)")

    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (testing)")

    # TODO: train classifier
    clf.fit(X_train, y_train)


    score = clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(clf_name + " (training)")


    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.6)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(clf_name + " (testing)")
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()
