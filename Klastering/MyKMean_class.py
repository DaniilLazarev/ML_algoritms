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
            print(f'----- Размер выборки кол-во сторк {n} и колл-во столбцов {m} -----')
        except ValueError:
            print('Обуч выборка должна быть двумерным массивом')
           
        for i in range(self.n_init):
            np.random.seed(seed=self.random_state)
    
            centroid_objects = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(self.n_clusters, X.shape[1]))
        
            centroid_objects_prev = [np.zeros(m) for i in range(self.n_clusters)]
        
            weight_evolution = [distance.euclidean(centroid_objects_prev[i], centroid_objects[i]) for i in range(self.n_clusters)]
        
            step = 0

            wcss = 0
            for k in range(self.n_clusters):
                 points_in_cluster = X[cluster_labels == k]  # Точки в кластере
                 centroid = centroid_objects[k]  # Центроид текущего кластера
                 distances = np.linalg.norm(points_in_cluster - centroid, axis=1)**2
                 wcss += np.sum(distances)
                 
            if wcss < best_wcss:
                best_wcss = wcss
                best_centroids = centroid_objects
                best_labels = cluster_labels

            while sum(weight_evolution[i] >0.001 for i in range(self.n_clusters))!=0 and step < self.max_iter:
                centroid_objects_prev = centroid_objects.copy()
            
                cluster_distance = np.hstack([
                np.sqrt(
                    np.power(X - centroid, 2).sum(axis=1)
                ).reshape(-1,1)
                for centroid in centroid_objects])
            
                cluster_lables = cluster_distance.argmin(axis=1)
            
                centroid_objects = [X[cluster_lables==i].mean(axis=0) for i in range(self.n_clusters)]
            
                weight_evolution = [distance.euclidean(centroid_objects_prev[j], centroid_objects[j]) for j in range(self.n_clusters)]
            
            
                step =  step + 1
        
        
        # Пройдете заданное количество шагов (параметр max_iter).
        # Повторяет пункты 2-3 несколько раз (параметр n_init)
        
    def __str__(self):
        return f"MyKMeans class: n_clusters={self.n_clusters}, 
    max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"