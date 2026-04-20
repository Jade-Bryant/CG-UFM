import torch
import torch.nn.functional as F

class NadarayaWatsonAggregator:
    """
    Nadaraya-Watson Kernel Regression for Patch Boundary Aggregation.
    Used during sliding window inference to smoothly blend overlapping patches.
    """
    def __init__(self, bandwidth: float = 0.1, kernel_type: str = 'gaussian'):
        """
        Args:
            bandwidth: Hyperparameter sigma for the Gaussian kernel
            kernel_type: Type of kernel to use (e.g., 'gaussian')
        """
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        
    def _compute_weights(self, distances: torch.Tensor):
        """
        Computes kernel weights based on distances.
        
        Args:
            distances: Tensor of shape (N, M)
        Returns:
            weights: Tensor of shape (N, M)
        """
        if self.kernel_type == 'gaussian':
            # w(d) = exp(-d^2 / (2 * sigma^2))
            weights = torch.exp(- (distances ** 2) / (2 * self.bandwidth ** 2))
        else:
            raise NotImplementedError(f"Kernel {self.kernel_type} not supported.")
            
        return weights

    def aggregate(self, global_points: torch.Tensor, patch_points: torch.Tensor, patch_values: torch.Tensor):
        """
        Aggregates patch_values onto global_points using Nadaraya-Watson.
        
        Args:
            global_points: The global dense point cloud, shape (N_global, 3)
            patch_points: The points from the predicted patches, shape (N_patch, 3)
            patch_values: The values to be aggregated (e.g., normals, features, or survival probs), shape (N_patch, V)
            
        Returns:
            aggregated_values: The blended values on the global points, shape (N_global, V)
        """
        # Note: In practice, computing pairwise distances for the entire global cloud
        # is O(N_global * N_patch), which is memory intensive.
        # A robust implementation would use a neighbor search (e.g., Faiss, PyTorch3D KNN, Open3D KDTree)
        # Here we demonstrate the exact dense computation for scaffolding.
        
        N_g = global_points.shape[0]
        N_p = patch_points.shape[0]
        V = patch_values.shape[1]
        
        # Pairwise distance matrix (N_g, N_p)
        dist_matrix = torch.cdist(global_points.unsqueeze(0), patch_points.unsqueeze(0)).squeeze(0)
        
        # Compute weights (N_g, N_p)
        weights = self._compute_weights(dist_matrix)
        
        # Normalize weights along the patch dimension (N_p)
        # Add epsilon to prevent division by zero
        weight_sums = weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_weights = weights / weight_sums
        
        # Nadaraya-Watson estimator: y_hat = sum(w_i * y_i) / sum(w_i)
        # (N_g, N_p) @ (N_p, V) -> (N_g, V)
        aggregated_values = torch.matmul(normalized_weights, patch_values)
        
        return aggregated_values
