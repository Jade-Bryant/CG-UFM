import torch

def normalize_patch(points):
    """
    Centers a 3D point cloud patch and normalizes its scale.
    Input:
        points (N, 3) tensor
    Output:
        normalized_points (N, 3) tensor
        centroid (1, 3) tensor
        scale (float)
    """
    centroid = torch.mean(points, dim=0, keepdim=True)
    centered = points - centroid
    scale = torch.max(torch.sqrt(torch.sum(centered**2, dim=1)))
    normalized_points = centered / scale
    return normalized_points, centroid, scale
