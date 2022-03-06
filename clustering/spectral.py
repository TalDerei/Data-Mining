import enum
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as graph
from random import randint

class spectral:
    # default constructor
    def __init__(self, K):
        self.K = K

    def spectral_clustering(self, data):
        # size of NxN matrix
        matrix_size = data.shape[0]

        # compute similarity matrix
        self.compute_similarity_matrix(data, matrix_size)

        # compute adjacency matrix to construct KNN graph
        self.compute_adjacency_matrix(matrix_size)

    def compute_similarity_matrix(self, data, matrix_size):
        self.similarity_matrix = []
        for idx in enumerate(data):
            self.similarity_matrix.append([np.sqrt(np.sum((idx[1] - value)**2)) for value in data])
        # print(self.similarity_matrix)

    def compute_adjacency_matrix(self, matrix_size):
        self.adacancy_matrix = [[] for _ in range(matrix_size)]
        for i in range(matrix_size):
            for j in range(matrix_size):
                if (self.similarity_matrix[i][j] < 0.5):
                    self.adacancy_matrix[i].append(0)
                else:
                    self.adacancy_matrix[i].append(1)
        # print(self.adacancy_matrix)

def main(): 
    obj = spectral(2)
    
    # read input file
    x,y = np.loadtxt("square.txt", delimiter=',', unpack=True)

    # convert data to 2D array
    data = np.column_stack((x, y))     
    
    # run spectral clustering algorithm    
    obj.spectral_clustering(data)

# define special variable to execute main function
if __name__== "__main__": 
    main()