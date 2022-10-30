import math
from typing import List


def compute_base_R(n_dim: int, n_samples: int, n_levels: int, layer_witdh: int, split_factor: float) -> float:
    """Compute the base size of the multi-level latent grid

    Args:
        n_dim (int): Number of dimensions (choices = [3, 4] for 2D, 3D time-varying datasets resepectively)
        n_samples (int): combined size of the multilevel grid
        n_levels (int): Number of grid levels
        layer_witdh (int): Feature dimension of the latent grid
        split_factor (float): split factor

    Returns:
        float: base size of the multi-level grid
    """
    s_sum = 0
    for level in range(n_levels):
        s_sum += math.pow(split_factor, n_dim * level)

    return round(math.pow(n_samples * (n_levels / layer_witdh) / s_sum, 1 / n_dim))


def compute_all_R(n_dim: int, n_samples: int, n_levels: int, layer_witdh: int, split_factor: float) -> List[float]:
    """Computes the individual sizes of the levels in the multi-level latent grid

    Args:
        n_dim (int): Number of dimensions (choices = [3, 4] for 2D, 3D time-varying datasets resepectively)
        n_samples (int): combined size of the multilevel grid
        n_levels (int): Number of grid levels
        layer_witdh (int): Feature dimension of the latent grid
        split_factor (float): split factor

    Returns:
        float: List of sizes of all the grid levels
    """
    base_r = compute_base_R(n_dim, n_samples, n_levels, layer_witdh, split_factor)
    f_size = []
    for ldx in range(n_levels):
        r = int(base_r * math.pow(split_factor, ldx))
        f_size.append(r)
    return f_size
