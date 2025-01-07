from scipy.spatial import distance
import numpy as np

def k_means(X: np.array, 
            k: int = 2, 
            eps: float = 0.001,
            num_iteration: 
            int = 10, 
            verbose: bool = True) -> np.array:
    ''' 
    :param X: обучающ выборка
    :param k: кол-во кластеров 
    '''
    
    try:
        n, m = X.shape
    except ValueError:
        print('Обуч выборка должна быть двумерным массивом')
        
    # инициализируем центроиды случайным элементом выборки 
    # центроидов столько, сколько и кластеров
    centroid_objects = [
        X[obj_id, :]
        for obj_id in np.random.randit(0, n, size = k)
    ]