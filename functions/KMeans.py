import torch

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
