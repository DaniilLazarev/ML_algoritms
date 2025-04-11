import numpy as np
import pandas as pd
from scipy.spatial import distance

class MyKMeans():
    def __init__(self, n_clusters: int = 3,
                 max_iter: int = 10, 
                 n_init: int = 3,
                 random_state: int = 42):
        
        self.n_clusters = n_clusters 
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        
    def fit(self, X: pd.DataFrame):
        try:
            n, m = X.shape
            #print(f'----- Размер выборки: кол-во строк {n}, кол-во столбцов {m} -----')
        except ValueError:
            raise ValueError('Обучающая выборка должна быть двумерным массивом')
    
    # Преобразуем X в массив NumPy
        X = X.to_numpy()
        np.random.seed(self.random_state)
    # Лучшая кластеризация
        self.inertia_ = float('inf')
        self.cluster_centers_ = None
        best_labels = None

        for _ in range(self.n_init):
            

        # Инициализация центроидов
            centroid_objects = np.random.uniform(
                X.min(axis=0), X.max(axis=0), size=(self.n_clusters, X.shape[1])
            )
            prev_centroid_objects = np.zeros_like(centroid_objects)

            step = 0
            while step < self.max_iter:
            # Вычисляем расстояния от точек до центроидов
                cluster_distance = np.hstack([
                    np.linalg.norm(X - centroid, axis=1).reshape(-1, 1)
                    for centroid in centroid_objects
                ])
                cluster_distance = np.hstack([
                    np.linalg.norm(X - centroid, axis=1).reshape(-1, 1)
                    for centroid in centroid_objects
                ])
                cluster_distance = np.hstack([
                np.sqrt(
                    np.power(X - centroid, 2).sum(axis=1)
                    ).reshape(-1,1)
                for centroid in centroid_objects
        ])
                
                
                
                
                cluster_labels = cluster_distance.argmin(axis=1)

            # Сохраняем текущие центроиды
                prev_centroid_objects = centroid_objects.copy()

            # Обновляем центроиды
                centroid_objects = np.array([
                    X[cluster_labels == i].mean(axis=0) if (cluster_labels == i).sum() > 0
                    else np.random.uniform(X.min(axis=0), X.max(axis=0))
                    for i in range(self.n_clusters)
                ])

            # Условие остановки
                centroid_shift = np.linalg.norm(centroid_objects - prev_centroid_objects, axis=1)
                if np.all(centroid_shift < 1e-4):
                    break

            step += 1

        # Вычисляем WCSS для текущей инициализации
            wcss = 0
            for k in range(self.n_clusters):
                points_in_cluster = X[cluster_labels == k]
                if len(points_in_cluster) > 0:
                    distances = np.linalg.norm(points_in_cluster - centroid_objects[k], axis=1)**2
                    wcss += np.sum(distances)

        # Сохраняем лучшую кластеризацию
            if wcss < self.inertia_:
                self.inertia_= wcss
                self.cluster_centers = centroid_objects
                best_labels = cluster_labels

   