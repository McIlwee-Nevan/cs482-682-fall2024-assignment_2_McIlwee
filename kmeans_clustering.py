import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        
        self.dataset_file = dataset_file
        self.data = None
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        
    def model_fit(self, n_clusters = 3):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        self.model = KMeans(n_clusters=n_clusters, random_state=0).fit(self.data)
        cluster_centers = self.model.cluster_centers_
        return cluster_centers
    
    def make_plots(self):
        '''
        Makes a plot of the K-Means clusters
        '''
        if self.model is None:
            print('Model has not been initialized yet. Initializing with default number of clusters: 3')
            self.model_fit()
        
        cluster_centers = self.model.cluster_centers_
        num_clusters = clusters_centers.shape[0]
        labels = np.arange(num_clusters)

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.title("K-Means: %d Clusters" % num_clusters)
        plt.contourf(xx, yy, Z, alpha=0.6)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.model.labels_, edgecolors='k', marker='o', s=50)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=labels, edgecolors='k', marker='s', s=100)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    parser.add_argument('-c', '--clusters', type=int, default=3, help='Number of clusters')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit(args.clusters)
    print(clusters_centers)
    classifier.make_plots()
    