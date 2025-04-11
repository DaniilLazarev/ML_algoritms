from scipy.spatial import distance
import numpy as np

def k_means(X: np.array, 
            k: int = 2, 
            eps: float = 0.001,
            num_iteration: 
            int = 50, 
            verbose: bool = True) -> np.array:
    ''' 
    :param X: обучающ выборка
    :param k: кол-во кластеров 
    '''
    
    try:
        n, m = X.shape
        print(f'----- Размер выборки кол-во сторк {n} и колл-во столбцов {m} -----')
    except ValueError:
        print('Обуч выборка должна быть двумерным массивом')
        
    # инициализируем центроиды случайным элементом выборки 
    # центроидов столько, сколько и кластеров
    centroid_objects = [
        X[obj_id, :]
        for obj_id in np.random.randint(0, n, size = k)
    ]
    print(f'-----Иницилизация центроидов {centroid_objects} -----')
    print(f'----- начальная точка  -----')
    centroid_objects_prev = [np.zeros(m) for i in range(k)]
    # расстоние между предыдущем и текущем положением каждого центроида (оптимизиурем именно это)
    weight_evolution = [distance.euclidean(centroid_objects_prev[i], centroid_objects[i]) for i in range(k)]
    # step = 0 - условие остановки 
    step = 0
    while sum(weight_evolution[i] >eps for i in range(k))!=0 and step < num_iteration:
        #зачем копировать? типо начальное условие
        centroid_objects_prev = centroid_objects.copy()
        cluster_distance = np.hstack([
            np.sqrt(
                np.power(X - centroid, 2).sum(axis=1)
            ).reshape(-1,1)
            for centroid in centroid_objects
        ])
        print(f'----- {cluster_distance}  -----')
        #Используется для горизонтальной конкатенации массивов. Это о
        # бъединяет одномерные массивы расстояний для
        # каждого центроида в одну матрицу, где каждая 
        # колонка соответствует расстояниям до определенного центроида.
        #axis = 1 по строкаим
        # находим минимальное расстояние в каждой строчке - это будет кластер объекта
        cluster_lables = cluster_distance.argmin(axis=1)
        #print(f'----- {cluster_lables}  -----')
         # усредняем координаты объектов каждого кластера - это новое положение центроида
         # вот здесь 
        centroid_objects = [
            X[cluster_lables==i].mean(axis=0) for i in range(k)
        ]
         # вычисляем расстояние между центроидами на соседних итерациях
        weight_evolution = [
            distance.euclidean(centroid_objects_prev[j], centroid_objects[j])
            for j in range(k)
        ]
        if verbose:
            print("step %s, cluster shift: %s" % (step, weight_evolution))
        # обновлённые кластера
        print(centroid_objects)
        print(np.vstack(centroid_objects))
        step += 1
    return np.vstack(centroid_objects), cluster_lables



import matplotlib.pyplot as plt
import pickle

file_path = 'C:\ML\Ml алгоритмы\ML_algoritms\Klastering\clustering.pkl'
with open(file_path, 'rb') as f:
    data_clustering = pickle.load(f)
    
X = np.array(data_clustering['X'])
X1 = np.random.rand(100,1)
X = np.hstack((X, X1))

print('num points: %d' % X.shape[0])

Y = np.array(data_clustering['Y'])

centroids, labels = k_means(X, k=2)

plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, marker='o', alpha=0.8, label='data')
plt.plot(centroids[:, 0], centroids[:, 1], marker='+', mew=10, ms=20)
plt.show()