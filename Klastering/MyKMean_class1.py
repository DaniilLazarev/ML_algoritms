import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class MyKMeans():
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = []
        self.inertia_ = 0
    
    def __str__(self):  
        return f"MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"
    def __closest_centroid(self, points, centroids):
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def __distance_centroid(self, points, centroids):
        distances = np.sqrt(((points - centroids[np.newaxis, :])**2).sum(axis=1))
        return distances

    def fit(self, X):
        np.random.seed(self.random_state)
        X = X.to_numpy()
        _, m = X.shape
        centroids = np.empty((self.n_clusters, m))
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)
        wcss_arr = np.empty((self.n_init))
        print(f' --- иницилизация wcss_arr{wcss_arr}  --- ')
        centroids_arr = np.empty((self.n_init, self.n_clusters, m))
        print(f' --- иницилизация centroids_arr{centroids_arr}  --- ')
        for n_i in range(self.n_init):
            # инициализация кластеров
            for item in range(self.n_clusters):
                centroids[item] = np.random.uniform(min, max)
            # ищем ближайший кластера и двигаем их
            for _ in range(self.max_iter):
                closest_indx = self.__closest_centroid(X, centroids)
                for k in range(self.n_clusters):
                    cluster = X[closest_indx == k]
                    centroids[k] = np.mean(cluster, axis=0) if np.any(cluster) else centroids[k]
            
            # ищем внутрикластерные расстояния
            closest_indx = self.__closest_centroid(X, centroids)
            wcss = 0
            for k in range(self.n_clusters):
                cluster = X[closest_indx == k]
                dis_centr = np.sum((self.__distance_centroid(cluster, centroids[k])**2), axis = 0)
                wcss += dis_centr
            wcss_arr[n_i] = wcss
            centroids_arr[n_i] = centroids
        
        # берем минимальное wcss
        min_index = np.argmin(wcss_arr)
        self.inertia_ = wcss_arr[min_index]
        self.cluster_centers_ = centroids_arr[min_index]
    #
    def predict(self, X):
        return self.__closest_centroid(X.to_numpy(), self.cluster_centers_)
    
    
file_path = 'C:\ML\Ml алгоритмы\ML_algoritms\Klastering\clustering.pkl'
with open(file_path, 'rb') as f:
    data_clustering = pickle.load(f)
    
X2 = pd.DataFrame(data_clustering['X'])