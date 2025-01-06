import numpy as np
import pandas as pd
import random

class MyLineReg:
    def __init__(self, weights=None, n_iter=100, learning_rate=0.1, metric=None,
                 reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        if l1_coef < 0:
            raise ValueError('l1_coef не может быть отрицательным')
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        # Фиксируем сид для воспроизводимости
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Сбрасываем индексы, чтобы избежать проблем с перемешиванием
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Добавляем столбец единиц для учёта bias (свободного члена)
        X = pd.concat([pd.DataFrame({0: [1] * len(X)}), X], axis=1)
        
        # Инициализируем веса
        self.weights = np.ones(X.shape[1])

        for iteration in range(1, self.n_iter + 1):
            # Определяем размер мини-пакета
            if self.sgd_sample is None:
                indices = np.arange(len(X))
            elif isinstance(self.sgd_sample, int):
                # Если размер выборки больше количества данных, уменьшаем его до длины данных
                sample_size = min(self.sgd_sample, len(X))
                indices = random.sample(range(len(X)), sample_size)
            elif isinstance(self.sgd_sample, float) and 0 < self.sgd_sample <= 1:
                sample_size = max(1, int(len(X) * self.sgd_sample))  # Должен быть хотя бы 1 пример
                indices = random.sample(range(len(X)), sample_size)
            else:
                raise ValueError('sgd_sample должен быть либо целым числом, либо дробным числом от 0.0 до 1.0')

            # Формируем подвыборку
            X_batch = X.iloc[indices]
            y_batch = y.iloc[indices]

            # Вычисляем предсказания и ошибки
            Y = X_batch.dot(self.weights)
            errors = y_batch - Y

            # Вычисляем градиент
            gradient = -2 * X_batch.T.dot(errors) / len(X_batch)

            # Добавляем регуляризацию, если она задана
            if self.reg == 'l1':
                gradient += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                gradient += 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
            
            # Определяем текущую скорость обучения
            lr = self.learning_rate if not callable(self.learning_rate) else self.learning_rate(iteration)

            # Обновляем веса
            self.weights -= lr * gradient
            
            if verbose:
                if self.metric:
                    metric_value = self._calculate_metric(y, X.dot(self.weights))
                    if verbose and iteration % verbose == 0:
                        print(f"{iteration} | loss: {self._calculate_metric(y, X.dot(self.weights)):.2f} | {self.metric}: {metric_value:.2f}")
                print(f"Iteration {iteration}, learning rate: {lr}")

        # Расчет финальных предсказаний на всем датасете
        final_prediction = X.dot(self.weights)
        self.best_score = self._calculate_metric(y, final_prediction)
        if verbose:
            print("Training complete.")
    
    def get_coef(self):
        return self.weights[1:]  # Возвращаем веса, исключая bias

    def predict(self, X: pd.DataFrame):
        X = pd.concat([pd.DataFrame({0: [1] * len(X)}), X], axis=1)
        return X.dot(self.weights)
    
    def _calculate_metric(self, y, Y):
        if self.metric == 'mae':
            return np.mean(np.abs(y - Y))
        elif self.metric == 'mse':
            return np.mean((y - Y) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y - Y) ** 2))
        elif self.metric == 'mape':
            return np.mean(np.abs((y - Y) / y)) * 100
        elif self.metric == 'r2':
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - Y) ** 2)
            return 1 - (ss_residual / ss_total)
    
    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def get_best_score(self):
        return self.best_score
