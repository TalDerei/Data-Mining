import numpy as np
import pandas as pd
import seaborn as sns
import kmeans_clustering as km
import matplotlib.pyplot as graph
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.cluster import KMeans

class spectral:
    # default constructor
    def __init__(self, K):
        self.K = K

    def spectral_clustering(self, data, user_input):
        # convert data to 2D array
        self.data = data.to_numpy()

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
        self.plot(self.data)

    # compute similarity matrix using KNN similarity
    def compute_similarity_matrix(self, data):
        self.similarity_matrix = []
        for vector in enumerate(data):
            self.similarity_matrix.append([np.sqrt(np.sum((vector[1] - point)**2)) for point in data])
        
        print("Similarity Matrix is: ")
        print(self.similarity_matrix)

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

        print("\n Adjacency Matrix is: ")
        print(self.adjacency_matrix)

    # compute degree matrix by summing up values along diagnole
    def compute_degree_matrix(self):
        self.degree_matrix = np.diag(np.sum(np.array(self.adjacency_matrix), axis=1))
        
        print("\n Degree Matrix is: ")
        print(self.degree_matrix)
        
    # compute laplacian matrix 
    def compute_laplacian_matrix(self):
        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix   
        
        print("\n Laplacian Matrix is: ")
        print(self.laplacian_matrix)

    # compute eigenvalues and eigenvectors
    def eigen(self):
        self.eigen_value, self.eigen_vector = np.linalg.eig(self.laplacian_matrix)
        
        print('\n Eigenvalues:')
        print(self.eigen_value)

        print('\n Eigenvectors:')
        print(self.eigen_vector)

    # map eigenvectors to data and plot
    def plot(self, data):
        self.eigen_vector[:,1][self.eigen_vector[:,1] < 0] = 0
        self.eigen_vector[:,1][self.eigen_vector[:,1] > 0] = 1
        graph.scatter(data[:, 0], data[:, 1],c = self.eigen_vector[:,1])
        graph.show()

def main(): 
    # prompt user input
    user_input = input("Please enter file name: ")
    
    # read input file
    data = pd.read_csv(user_input, header = None, delimiter = " ") 
    
    # run spectral clustering algorithm    
    obj = spectral(2)
    obj.spectral_clustering(data, user_input)

# define special variable to execute main function
if __name__== "__main__": 
    main()