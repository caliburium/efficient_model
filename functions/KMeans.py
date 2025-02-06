import torch

"""
class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def initialize_centroids_kmeans_pp(self, data):

        # Initialize centroids using K-Means++ algorithm.

        n_samples = data.size(0)
        centroids = [data[torch.randint(0, n_samples, (1,)).item()]]

        for _ in range(1, self.n_clusters):
            distances = torch.min(torch.cdist(data, torch.stack(centroids)), dim=1)[0]
            probs = distances / distances.sum()
            new_centroid = data[torch.multinomial(probs, 1).item()]
            centroids.append(new_centroid)

        return torch.stack(centroids)

    def fit(self, data):
        data = data.to(self.device)
        n_samples, n_features = data.size()

        # Initialize centroids using K-Means++
        centroids = self.initialize_centroids_kmeans_pp(data)

        for i in range(self.max_iter):
            # Compute distances and assign labels
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)

            # Compute new centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = data[labels == k]
                if cluster_points.size(0) == 0:
                    # Handle empty cluster by reinitializing
                    new_centroids.append(data[torch.randint(0, n_samples, (1,)).item()])
                else:
                    new_centroids.append(cluster_points.mean(dim=0))

            new_centroids = torch.stack(new_centroids)

            # Check for convergence
            shift = torch.norm(new_centroids - centroids, dim=1).sum()
            centroids = new_centroids

            if shift <= self.tol:
                break

        self.centroids = centroids
        self.labels = labels

        return labels

    def predict(self, data):

        # Predict cluster assignments for new data points.

        data = data.to(self.device)
        distances = torch.cdist(data, self.centroids)
        return torch.argmin(distances, dim=1)
"""

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def fit(self, data):
        data = data.to(self.device)
        n_samples, n_features = data.size()

        random_indices = torch.randperm(n_samples)[:self.n_clusters]
        centroids = data[random_indices]

        for i in range(self.max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([data[labels == k].mean(dim=0) for k in range(self.n_clusters)])
            shift = torch.norm(new_centroids - centroids, dim=1).sum()
            centroids = new_centroids
            if shift <= self.tol:
                break

        self.centroids = centroids
        self.labels = labels

        return labels

    def predict(self, data):
        data = data.to(self.device)
        distances = torch.cdist(data, self.centroids)

        return torch.argmin(distances, dim=1)
