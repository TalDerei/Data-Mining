import numpy as np
import pandas as pd
import seaborn as sns
import kmeans_clustering as km
import matplotlib.pyplot as graph
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import math
from scipy.sparse import csgraph
from sklearn import preprocessing

class spectral:
    # default constructor
    def __init__(self, K):
        self.K = K

    def spectral_clustering(self, user_input):
        self.dimentionality_reduction(user_input)

        # convert data to 2D array
        self.data = self.data.to_numpy()

        # size of NxN matrix
        matrix_size = self.data.shape[0]

        # construct similarity matrix
        self.compute_similarity_matrix(self.data)

        # construct adjacency matrix to construct KNN graph
        self.compute_adjacency_matrix(matrix_size, user_input)

        # construct degree matrix
        self.compute_degree_matrix()

        # construct laplacian matrix
        self.compute_laplacian_matrix()

        # determine eigenvalues and eigenvectors
        self.eigen()

        # graphing using k-means algorithm
        self.plot(self.data, user_input)
    
    # dimentionality reduction
    def dimentionality_reduction(self, user_input):
        # read input file and clean datasets
        if (user_input == 'square.txt' or user_input == 'elliptical.txt'):
            self.K = 2
            self.data = pd.read_csv(user_input, header = None, delimiter = " ")
        if (user_input == 'cho.txt'): 
            self.K = 5
            self.data = pd.read_csv(user_input, header = None, delimiter = " ")
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
        if (user_input == 'iyer.txt'):
            self.K = 10
            self.data = pd.read_csv(user_input, header = None, delimiter = " ")
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])

    # compute similarity matrix using KNN similarity
    def compute_similarity_matrix(self, data):
        self.similarity_matrix = []
        for vector in enumerate(data):
            self.similarity_matrix.append([np.sqrt(np.sum((vector[1] - point)**2)) for point in data])
        
        print("Similarity Matrix is: ")
        # print(self.similarity_matrix)

    # compute adjacency by defining some previosuly threshold, k
    def compute_adjacency_matrix(self, matrix_size, user_input):
        self.adjacency_matrix = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if (self.similarity_matrix[i][j] == 0):
                    self.adjacency_matrix[i][j] = 0
                else:
                    if (user_input == 'square.txt'):
                        if ((self.similarity_matrix[i][j]) < 1):
                            self.adjacency_matrix[i][j] = 0
                        else:
                            self.adjacency_matrix[i][j] = 1
                    elif (user_input == 'elliptical.txt'):
                        if ((self.similarity_matrix[i][j]) < 0.15):
                            self.adjacency_matrix[i][j] = 0
                        else:
                            self.adjacency_matrix[i][j] = 1
                    elif (user_input == 'cho.txt'):
                        if ((self.similarity_matrix[i][j]) < 2):
                            self.adjacency_matrix[i][j] = 0
                        else:
                            self.adjacency_matrix[i][j] = 1
                    elif (user_input == 'iyer.txt'):
                        if ((self.similarity_matrix[i][j]) < 2):
                            self.adjacency_matrix[i][j] = 0
                        else:
                            self.adjacency_matrix[i][j] = 1

        print("\n Adjacency Matrix is: ")
        print(self.adjacency_matrix)

    # compute degree matrix by summing up values along diagnole
    def compute_degree_matrix(self):
        self.degree_matrix = np.diag(np.sum(np.array(self.adjacency_matrix), axis=1))
        
        print("\n Degree Matrix is: ")
        print(self.degree_matrix)
        
    # compute laplacian matrix 
    def compute_laplacian_matrix(self): 
        # unormalized laplacian
        self.unormalized_laplacian_matrix = self.degree_matrix - self.adjacency_matrix 
        print("\n Unormalized Laplacian Matrix: ")
        print(self.unormalized_laplacian_matrix)

        # normalized laplacian
        # self.normalized_laplacian_matrix = csgraph.laplacian(self.adjacency_matrix, normed=True)

    # compute eigenvalues and eigenvectors
    def eigen(self):
        self.eigen_value, self.eigen_vector = np.linalg.eig(self.unormalized_laplacian_matrix)
        
        print('\n Eigenvalues:')
        print(self.eigen_value)

        print('\n Eigenvectors:')
        print(self.eigen_vector)

    # map eigenvectors to data and plot
    def plot(self, data, user_input):
        if (user_input == "square.txt" or user_input == "elliptical.txt"):
            print("Clustering Results: ")
            self.eigen_vector[:,1][self.eigen_vector[:,1] < 0] = 0
            self.eigen_vector[:,1][self.eigen_vector[:,1] > 0] = 1
            graph.scatter(data[:, 0], data[:, 1],c = self.eigen_vector[:,1])
            graph.show()
            print(self.eigen_vector[:,1])

        if (user_input == "iyer.txt"):
            print(self.eigen_vector)
            np.savetxt("spectral_iyer.txt", self.eigen_vector)
            k = km.kmeans(5, 100)
            KMeans = k.run_kmeans("spectral_iyer.txt")
            
        if (user_input == "cho.txt"):
            print(self.eigen_vector)
            np.savetxt("spectral_cho.txt", self.eigen_vector)
            k = km.kmeans(5, 100)
            KMeans = k.run_kmeans("spectral_cho.txt")        

def main(): 
    # prompt user input
    user_input = input("Please enter file name: ")
    
    # run spectral clustering algorithm    
    obj = spectral(2)
    obj.spectral_clustering(user_input)

# define special variable to execute main function
if __name__== "__main__": 
    main()