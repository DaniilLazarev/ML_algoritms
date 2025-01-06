import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score




class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be between 0 and 1")

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        eps = 1e-15
        N, D = X.shape
        X = np.hstack([np.ones((N, 1)), X])
        self.weights = np.ones(D + 1)

        for i in range(1, self.n_iter + 1):
            Y1 = 1 / (1 + np.exp(-np.dot(X, self.weights)))
            Log_Loss = (1 / N) * np.sum((y * np.log(Y1 + eps)) + (1 - y) * np.log(1 - Y1 + eps))
            gradient = (1 / N) * (Y1 - y) @ X
            self.weights -= self.learning_rate * gradient
            y_pred = self.predict(X[:, 1:])
            score = self.calculate_metric(y, X[:, 1:])  # Передаем X для ROC AUC
            if verbose and i % verbose == 0:
                print(f"{i} | loss: {Log_Loss} | {self.metric}: {score:.2f}")
                
            if self.best_score is None or score > self.best_score:
                self.best_score = score

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Добавляем столбец единиц
        N = X.shape[0]
        X = np.hstack([np.ones((N, 1)), X])
        # Рассчитываем логиты (вероятности)
        logits = 1 / (1 + np.exp(-np.dot(X, self.weights)))
        return logits

    def get_best_score(self):
        return self.best_score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Получаем вероятности
        probabilities = self.predict_proba(X)
        # Превращаем вероятности в классы
        return (probabilities > 0.5).astype(int)

    def accuracy(self, y, y_pred):
        correct_predictions = np.sum(y == y_pred)
        total_predictions = len(y)
        return correct_predictions / total_predictions

    def precision(self, y, y_pred):
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(self, y, y_pred):
        tp = np.sum((y == 1) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1(self, y, y_pred):
        prec = self.precision(y, y_pred)
        rec = self.recall(y, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def roc_auc(self, y, y_pred_proba):
        return roc_auc_score(y, y_pred_proba)

    def calculate_metric(self, y, X):
        y_pred = self.predict(X)
        if self.metric == 'accuracy':
            return self.accuracy(y, y_pred)
        elif self.metric == 'precision':
            return self.precision(y, y_pred)
        elif self.metric == 'recall':
            return self.recall(y, y_pred)
        elif self.metric == 'f1':
            return self.f1(y, y_pred)
        elif self.metric == 'roc_auc':
            y_pred_proba = self.predict_proba(X)
            return self.roc_auc(y, y_pred_proba)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
