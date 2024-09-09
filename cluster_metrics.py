#author: L. Stival

import torch
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd

class ClusterMetrics:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.silhouette_score = silhouette_score

    def calculate_metrics(self):

        unique_labels = torch.unique(self.labels)
     
        cohesion = 0
        separation = 0

        for label in unique_labels:
            label_samples = self.samples[self.labels == label]
            centroid = torch.mean(label_samples, dim=0)
            distance = torch.norm(label_samples - centroid, dim=1)
            cohesion += torch.sum(distance)
            separation += torch.sum(torch.min(torch.norm(label_samples - centroid, dim=1, keepdim=True), dim=0)[0])

        cohesion /= len(self.samples)
        separation /= len(self.samples)

        davies_bouldin_index = davies_bouldin_score(self.samples, self.labels)
        silhouette_score = self.silhouette_score(self.samples, self.labels)

        metrics = ['Davies-Bouldin Index', 'Cohesion', 'Separation', 'Silhouette Score']
        values = [davies_bouldin_index, cohesion, separation, silhouette_score]

        return values

if __name__ == "__main__":
    # Usage
    samples = torch.rand(100, 256)
    labels = torch.randint(0, 10, (100,))

    metrics_calculator = ClusterMetrics(samples, labels)
    metrics_df = metrics_calculator.calculate_metrics()
    print(metrics_df)

