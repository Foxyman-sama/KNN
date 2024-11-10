from operator import delitem
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_svmlight_file
import csv
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch


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


# https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset?resource=download
# Загрузка данных из CSV 700
df = pd.read_csv("user_behavior_dataset.csv")

# Просмотр первых 5 строк данных до преобразования
print(df.head())

# Преобразуем категориальные столбцы в числовые значения
# Для этого будем использовать LabelEncoder для колонок, которые содержат текстовые данные

# Преобразуем 'Device Model' в числовые значения
encoder_device = LabelEncoder()
df["Device Model"] = encoder_device.fit_transform(df["Device Model"])

# Преобразуем 'Operating System' в числовые значения
encoder_os = LabelEncoder()
df["Operating System"] = encoder_os.fit_transform(df["Operating System"])

# Преобразуем 'Gender' в числовые значения
encoder_gender = LabelEncoder()
df["Gender"] = encoder_gender.fit_transform(df["Gender"])

# Просмотр первых 5 строк данных после преобразования
print(df.head())

# Теперь данные можно разделить на признаки и целевую переменную
X = df.drop(
    columns=["User ID", "User Behavior Class"]
)  # Убираем 'User ID' и 'User Behavior Class' (это целевая переменная)

# Вариант 1: выбор двух признаков вручную
# X = df[["Age", "Screen On Time (hours/day)"]]
y = df["User Behavior Class"]

# Вариант 2: используем PCA для уменьшения признаков
# PCA — это линейный метод, который помогает найти наиболее важные оси для данных (главные компоненты), которые объясняют наибольшую вариацию в данных.
# Эти оси являются комбинацией исходных признаков.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# Масштабируем данные (особенно важно, если признаки имеют разные масштабы, как в данном случае: часы и мегабайты)
# Масштабирование признаков важно, чтобы привести все признаки к одному масштабу, улучшить сходимость и производительность модели.
# Целевая переменная не масштабируется, чтобы сохранить её смысл (классы или реальный диапазон значений) и избежать искажения результата модели
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучим модель (например, Logistic Regression)
sk_knn_clf = KNeighborsClassifier(n_neighbors=3)
sk_knn_clf.fit(X_train_scaled, y_train)

# Визуализация решающей поверхности с помощью PCA
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train_scaled, y_train.values, clf=sk_knn_clf, legend=2)
plt.title("Решающая поверхность")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
