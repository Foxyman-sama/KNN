from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNearestNeighbors:
    def __init__(self, n_neighbors=5, regression=False):
        self.n_neighbors = n_neighbors
        self.regression = regression

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    # Вычисляет расстояние с помощью теоремы Пифагора (самая простая и общепринятая метрика)
    def _euclidean_distances(self, x_test_i):
        # np.sum с axis=1 считает сумму по рядам и потом вычисляет корень с суммы
        return np.sqrt(np.sum((self.X_train - x_test_i) ** 2, axis=1))

    def _make_prediction(self, x_test_i):
        # Вычисляем расстояние до всех соседей
        distances = self._euclidean_distances(x_test_i)
        k_nearest_indexes = np.argsort(distances)[: self.n_neighbors]
        targets = self.y_train[k_nearest_indexes]
        return np.mean(targets) if self.regression else np.bincount(targets).argmax()

    def predict(self, X_test):
        return np.array([self._make_prediction(x) for x in X_test])


def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    plt.show()


X1, y1 = load_iris(return_X_y=True, as_frame=True)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1.values, y1.values, random_state=0
)
print(X1, y1, sep="\n")

X2, y2 = load_diabetes(return_X_y=True, as_frame=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2.values, y2.values, random_state=0
)
print(X2, y2, sep="\n")

knn_clf = KNearestNeighbors()
knn_clf.fit(X1_train, y1_train)
knn_clf_pred_res = knn_clf.predict(X1_test)
knn_clf_accuracy = accuracy_score(y1_test, knn_clf_pred_res)

print(f"KNN classifie accuracy: {knn_clf_accuracy:}")
print(knn_clf_pred_res)

knn_reg = KNearestNeighbors(regression=True)
knn_reg.fit(X2_train, y2_train)
knn_reg_pred_res = knn_reg.predict(X2_test)
knn_reg_r2 = r2_score(y2_test, knn_reg_pred_res)

print(f"KNN regressor R2 score: {knn_reg_r2}")
print(knn_reg_pred_res)

sk_knn_clf = KNeighborsClassifier()
sk_knn_clf.fit(X1_train, y1_train)
sk_knn_clf_pred_res = sk_knn_clf.predict(X1_test)
sk_knn_clf_accuracy = accuracy_score(y1_test, sk_knn_clf_pred_res)

print(f"sk KNN classifier accuracy: {sk_knn_clf_accuracy:}")
print(sk_knn_clf_pred_res)

feature_indexes = [2, 3]
title1 = "KNeighborsClassifier surface"
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_knn_clf, feature_indexes, title1)
