import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as graph

class kmeans:
    K = 0
    total_points = 0
    total_values = 0
    max_iterations = 0

    # default constructor
    def __init__(self, K, max_iterations):
        self.K = K                                   # number of clusters                      
        self.max_iterations = max_iterations         # list of lists for cluster
        self.centroids = np.array([]).reshape(0, 2)  # 2D array for centroids
        self.clusters = np.array([]).reshape(0, 2)   # 2D array for clusters

    # run k-means algorithm
    def kmeans(self, x, y):
        # convert input to 2D array
        self.data = np.column_stack((x, y))

        # choose k random points as centroids
        self.create_centroids(self.data)

        # run k-means until convergence
        for _ in range(self.max_iterations):
            # create cluster
            self.clusters = self.create_cluster(self.centroids)

            self.plot_data(self.centroids, self.clusters)

            # update centroids
            self.update_centroids(self.clusters)

            # convergence

    # choose k random centroid
    def create_centroids(self, data):
        for k in range(self.K):     
            rand_index = random.randint(0, data.shape[0])
            self.centroids = np.append(self.centroids, [self.data[rand_index]], axis = 0)
    
    # create clusters
    def create_cluster(self, centroids):
        clusters = [[] for _ in range(self.K)]
        # clusters = np.array([]).reshape(0, 2)
        for idx in enumerate(self.data):
            centroid_index = self.closest_centroid(idx, centroids)
            clusters[centroid_index].append(idx[1])
        return clusters
    
    # assign points to clusters
    def closest_centroid(self, idx, centroids):
        # using list comprehension, calculate minimum distance between centroid and points
        distances = [np.sqrt(np.sum((idx[1] - centroid)**2)) for centroid in centroids]
        closest_index = np.argmin(distances)
        return closest_index
   
    # calculate mean of all points associated with centroid
    def update_centroids(self, clusters):
        for idx in enumerate(clusters):
            self.centroids[idx[0]] = np.mean(idx[1], axis=0)

    # plot data points 
    def plot_data(self, centroids, clusters):
        temp = np.array([]).reshape(0, 2)
        temp1 = np.array([]).reshape(0, 2)
        # clusters = np.array([]).reshape(0, 2)
        counter = 0
        for idx in clusters[counter]:
            # print(idx)
            # print(idx)
            temp = np.append(temp, [idx], axis = 0)
        
        for idx in clusters[1]:
            # print(idx)
            # print(idx)
            temp1 = np.append(temp1, [idx], axis = 0)
        # print(np.size(temp))

        # print(self.data)
        print(temp)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(temp1)

        x,y=np.split(centroids, 2, axis=1)
        graph.scatter(x, y, color = 'red', marker = "x", alpha=1)
        graph.scatter(temp[:,0],temp[:,1])
        graph.scatter(temp1[:,0],temp1[:,1])
        graph.title('Original Dataset')
        graph.show()


def main(): 
    # call constructor
    obj = kmeans(2, 5)

    # read input file
    x, y = np.loadtxt("square.txt", delimiter=',', unpack=True)

    obj.kmeans(x, y)

# define special variable to execute main function
if __name__== "__main__": 
    main()
