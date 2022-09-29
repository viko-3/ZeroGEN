import numpy as np
import math

ssw_matrix_path='/home/s2136015/Code/DeepTarget_final/data/prot_ssw.npy'
ssw_matrix=np.load(ssw_matrix_path)
def calculate_distance(i,j):
    distance_ij = 1-(ssw_matrix[i][j] / math.sqrt(ssw_matrix[i][i] * ssw_matrix[j][j]))
    return distance_ij

if __name__ == '__main__':
    distance_matrix = np.zeros_like(ssw_matrix)
    num = len(distance_matrix)
    for i in range(num):
        for j in range(num):
            distance_matrix[i][j]=calculate_distance(i,j)
    np.save("prot_distance.npy", distance_matrix)