import pandas as pd
import numpy as np


class MyTreeClf():
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs 
    
    #def __str__(self):
        #return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'
    @staticmethod
    def get_best_split(X: pd.DataFrame, y: pd.Series):
        col_name = ''
        best_ig = -np.inf  # Начинаем с отрицательной бесконечности
        split_value = 0
        N = X.shape[0]
        eps = 1e-11  # Для избежания log(0)
        
        for col in X.columns:
            values = np.sort(X[col].unique())  # Уникальные отсортированные значения
            
            # Энтропия до разбиения
            p1 = np.sum(y == 0) / N
            p2 = np.sum(y == 1) / N
            s0 = - p1 * np.log2(p1 + eps) - p2 * np.log2(p2 + eps)
            
            #print(f"Feature: {col}, Initial Entropy (S0): {s0:.4f}")
            
            for i in range(1, len(values)):
                threshold = (values[i] + values[i - 1]) / 2  # Разделитель между двумя соседними уникальными значениями
                left_mask = X[col] <= threshold
                right_mask = X[col] > threshold
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Проверка на пустые подвыборки
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Энтропия для левой подвыборки
                p11 = np.sum(y_left == 0) / len(y_left)
                p12 = np.sum(y_left == 1) / len(y_left)
                s_left = - p11 * np.log2(p11 + eps) - p12 * np.log2(p12 + eps)
                
                # Энтропия для правой подвыборки
                p21 = np.sum(y_right == 0) / len(y_right)
                p22 = np.sum(y_right == 1) / len(y_right)
                s_right = - p21 * np.log2(p21 + eps) - p22 * np.log2(p22 + eps)
                
                # Средневзвешенная энтропия после разбиения
                s1 = (len(y_left) / N) * s_left
                s2 = (len(y_right) / N) * s_right
                
                #print(f"Threshold: {threshold:.4f}, Left Entropy: {s_left:.4f}, Right Entropy: {s_right:.4f}")
                #print(f"Weighted Entropy after split: {(s1 + s2):.4f}")
                
                # Прирост информации
                ig = s0 - (s1 + s2)
                
                #print(f"Information Gain (IG): {ig:.4f}")
                
                # Обновляем лучший сплит, если прирост информации больше
                if ig > best_ig:
                    best_ig = ig
                    col_name = col
                    split_value = threshold
                    
        return col_name, split_value, best_ig