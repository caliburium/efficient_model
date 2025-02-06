import torch

def entropy(labels):
    """
    Calculate entropy of a label distribution.
    Args:
        labels (torch.Tensor): Tensor of labels (N,)
    Returns:
        float: Entropy value
    """
    n = labels.size(0)
    unique, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / n
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()


def mutual_information(cluster_labels, true_labels):
    """
    Calculate mutual information between two label sets.
    Args:
        cluster_labels (torch.Tensor): Cluster labels (N,)
        true_labels (torch.Tensor): True labels (N,)
    Returns:
        float: Mutual information
    """
    n = cluster_labels.size(0)
    unique_cluster, cluster_counts = torch.unique(cluster_labels, return_counts=True)
    unique_true, true_counts = torch.unique(true_labels, return_counts=True)

    mi = 0.0
    for cluster, cluster_count in zip(unique_cluster, cluster_counts):
        cluster_indices = (cluster_labels == cluster)
        for true, true_count in zip(unique_true, true_counts):
            true_indices = (true_labels == true)
            joint_count = torch.sum(cluster_indices & true_indices)
            if joint_count > 0:
                joint_prob = joint_count.float() / n
                cluster_prob = cluster_count.float() / n
                true_prob = true_count.float() / n
                mi += joint_prob * torch.log(joint_prob / (cluster_prob * true_prob) + 1e-10)
    return mi.item()


def normalized_mutual_info(cluster_labels, true_labels):
    """
    Calculate Normalized Mutual Information (NMI).
    Args:
        cluster_labels (torch.Tensor): Cluster labels (N,)
        true_labels (torch.Tensor): True labels (N,)
    Returns:
        float: Normalized Mutual Information
    """
    cluster_labels = cluster_labels.view(-1)
    true_labels = true_labels.view(-1)
    mi = mutual_information(cluster_labels, true_labels)
    h_cluster = entropy(cluster_labels)
    h_true = entropy(true_labels)
    nmi = 2 * mi / (h_cluster + h_true + 1e-10)
    return nmi
