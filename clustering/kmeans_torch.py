import torch
import torch.nn as nn


class KMeansTorch(nn.Module):
    def __init__(self, num_clusters, max_iter=100, tol=1e-4, device='cuda'):
        super(KMeansTorch, self).__init__()
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device(device)  # device¸¦ ÀÎ½ºÅÏ½º º¯¼ö·Î ÀúÀå
        self.centroids = None

    def fit(self, X):
        assert X.ndim == 2, "Input data should be a 2D tensor"
        X = X.to(self.device)  # X¸¦ ÁöÁ¤µÈ device·Î ÀÌµ¿

        labels = torch.zeros(X.size(0), dtype=torch.long, device=self.device)

        indices = torch.randperm(X.size(0))[:self.num_clusters]
        self.centroids = X[indices]

        for i in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids = torch.stack([X[labels == j].mean(dim=0) for j in range(self.num_clusters)])

            if torch.norm(new_centroids - self.centroids, p=None) < self.tol:
                break

            self.centroids = new_centroids

        return labels

    def predict(self, X):
        X = X.to(self.device)  # X¸¦ ÁöÁ¤µÈ device·Î ÀÌµ¿
        distances = torch.cdist(X, self.centroids)
        labels = torch.argmin(distances, dim=1)
        return labels
