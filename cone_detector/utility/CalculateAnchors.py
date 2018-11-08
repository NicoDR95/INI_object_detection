import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class CalculateAnchors(object):

    def __init__(self, parameters, dataset, annotations_dir,
                 n_clusters, n_init, max_iter):
        self.parameters = parameters
        self.dataset = dataset
        self.annotations_dir = annotations_dir
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

    def get_anchors(self):
        output_w = self.parameters.output_w
        output_h = self.parameters.output_h

        dataset_annotations = self.dataset.get_dataset_dict(self.annotations_dir)
        # print(dataset_annotations)

        input = []
        for image_ann in dataset_annotations:
            for obj in image_ann['object']:
                width = obj['xmax'] - obj['xmin']
                width = width / image_ann['width'] * output_w
                height = obj['ymax'] - obj['ymin']
                height = height / image_ann['height'] * output_h
                input.append([width, height])
        input = np.array(input)
        # print(input)
        # print(input[:, 0])

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, max_iter=500).fit(input)
        kmeans_pred = kmeans.predict(input)
        centers = kmeans.cluster_centers_
        print("The suggested anchors values with {} clusters are the following:\n".format(self.n_clusters))
        print(centers)
        print("\nValues are couples of widht and height.")
        print("\nPlease copy paste these values in the main to use them:")
        to_copy = []
        for x in centers:
            to_copy.append(round(x[0], 2))
            to_copy.append(round(x[1], 2))

        print(to_copy)
        plt.scatter(input[:, 0], input[:, 1], c=kmeans_pred, s=10, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
        plt.xlabel("BoundingBox Width")
        plt.ylabel("BoundingBox Height")
        plt.show()
