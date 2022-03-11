import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as graph

class kmeans:
    # default constructor
    def __init__(self, K, max_iterations):
        self.K = K                             # number of clusters                      
        self.max_iterations = max_iterations   # list of lists for cluster

    # run k-means algorithm
    def run_kmeans(self, user_input):
        # perform dimentionality reduction
        self.dimentionality_reduction(user_input)
        
        columns = len(self.data.columns)

        # convert input to 2D array
        self.data = self.data.to_numpy()

        # choose k random points as centroids
        self.create_centroids(self.data, columns)

        # NxN dimentional array for clusters
        self.clusters = np.array([]).reshape(0, columns) 

        # run k-means until convergence
        for _ in range(self.max_iterations):
            # create cluster
            self.clusters = self.create_cluster(self.centroids)

            # update centroids by calculating mean of all points associated with centroid
            for idx in enumerate(self.clusters):
                self.centroids[idx[0]] = np.mean(idx[1], axis=0)
        
        # plot datasets
        self.plot_data(self.centroids, self.clusters, columns, user_input)
    
    # dimentionality reduction
    def dimentionality_reduction(self, user_input):
        # read input file and clean datasets
        if (user_input == 'cho.txt'): 
            self.K = 5
            self.data = pd.read_csv(user_input, header = None, delimiter = "	")
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
        if (user_input == 'iyer.txt'):
            self.K = 5
            self.data = pd.read_csv(user_input, header = None, delimiter = ",")
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
        if (user_input == 'square.txt' or user_input == 'elliptical.txt'):
            self.K = 2
            self.data = pd.read_csv(user_input, header = None, delimiter = " ")
        if (user_input == 'spectral_square.txt' or user_input == 'spectral_elliptical.txt'):
            self.K = 2
            self.data = pd.read_csv(user_input, header = None, delimiter = " ")

    # choose k random centroid
    def create_centroids(self, data, columns):
        # NxN dimentional array for centroids
        self.centroids = np.array([]).reshape(0, columns) 

        for k in range(self.K):     
            rand_index = random.randint(0, data.shape[0])
            self.centroids = np.append(self.centroids, [self.data[rand_index]], axis = 0)
    
    # create clusters
    def create_cluster(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for i in enumerate(self.data):
            # assign points to clusters
            min_distance = [np.sqrt(np.sum((i[1] - centroid)**2)) for centroid in centroids]
            index = np.argmin(min_distance)
            clusters[index].append(i[1])
        return clusters

    # plot data points 
    def plot_data(self, centroids, clusters, columns, user_input):
        print(user_input)
        # helper arrays
        self.temp = np.array([]).reshape(0, columns)
        self.temp_new = [np.array([]).reshape(0, columns) for _ in range(self.K)]

        # create 2D array of clusters
        counter = 0
        while (counter != self.K):
            for idx in clusters[counter]:
                self.temp = np.append(self.temp, [idx], axis = 0)
            self.temp_new[counter] = np.append(self.temp_new[counter], self.temp, axis = 0)
            self.temp = np.array([]).reshape(0, columns)
            counter += 1

        if (user_input == 'square.txt' or user_input == 'elliptical.txt' or user_input == 'spectral.txt'): 
            x,y = np.split(centroids, 2, axis = 1)
            graph.scatter(x, y, color = 'red', marker = "x", alpha = 1)
            graph.scatter(self.temp_new[0][:,0],self.temp_new[0][:,1], c = 'blue')
            graph.scatter(self.temp_new[1][:,0],self.temp_new[1][:,1], c = 'red')
            graph.title('K-Means Dataset')
            graph.show()
        if (user_input == 'cho.txt' or user_input == 'iyer.txt'):
            graph.scatter(self.centroids[:,0],self.centroids[:,1], c = 'red', marker = "x", alpha = 1)
            graph.scatter(self.temp_new[0][:,0],self.temp_new[0][:,1], c = 'blue')
            graph.scatter(self.temp_new[1][:,0],self.temp_new[1][:,1], c = 'red')
            graph.scatter(self.temp_new[2][:,0],self.temp_new[2][:,1], c = 'yellow')
            graph.scatter(self.temp_new[3][:,0],self.temp_new[3][:,1], c = 'green')
            graph.scatter(self.temp_new[4][:,0],self.temp_new[4][:,1], c = 'orange')
            graph.title('K-Means Dataset')
            graph.show()
            
def main(): 
    # call constructor and create object
    obj = kmeans(2, 100)

    # prompt user input
    user_input = input("Please enter file: ")

    # run kmeans algorithm
    obj.run_kmeans(user_input)

# define special variable to execute main function
if __name__== "__main__": 
    main()