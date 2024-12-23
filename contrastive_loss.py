import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = torch.dot(z_i, z_j) / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch.
    """
    N = out_left.shape[0]  # total number of training examples
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        # Compute similarity between all augmented examples and z_k, z_k_N
        sim_k = torch.exp(torch.matmul(out, z_k) / tau)  # [2*N]
        sim_k_N = torch.exp(torch.matmul(out, z_k_N) / tau)  # [2*N]
        
        # Exclude self-similarity
        sim_k[k] = 0
        sim_k_N[k+N] = 0
        
        # Compute l(k, k+N) and l(k+N, k)
        l_k_k_N = -torch.log(sim_k[k+N] / torch.sum(sim_k))
        l_k_N_k = -torch.log(sim_k_N[k] / torch.sum(sim_k_N))
        
        # Add to total loss
        total_loss += l_k_k_N + l_k_N_k
    
    # Divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2 * N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    norm_left = torch.nn.functional.normalize(out_left, dim=1)
    norm_right = torch.nn.functional.normalize(out_right, dim=1)
    pos_pairs = torch.sum(norm_left * norm_right, dim=1, keepdim=True)  # Dot product along the feature dimension
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    norm_out = torch.nn.functional.normalize(out, dim=1)  # Normalize each vector
    sim_matrix = torch.matmul(norm_out, norm_out.T)  # Compute pairwise dot product
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    # Step 1: Compute the denominator values for all augmented samples.
    exponential = torch.exp(sim_matrix / tau)  # e^{sim / tau}, shape [2*N, 2*N]
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).bool()
    
    # Apply the binary mask
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Compute the denominator values
    denom = torch.sum(exponential, dim=1, keepdim=True)  # [2*N, 1]
    
    # Step 2: Compute similarity between positive pairs
    positive_pairs = sim_positive_pairs(out_left, out_right)  # [N, 1]
    
    # Expand positive pairs to match 2*N samples
    numerator = torch.exp(torch.cat([positive_pairs, positive_pairs], dim=0) / tau)  # [2*N, 1]
    
    # Step 4: Compute the total loss
    loss = -torch.sum(torch.log(numerator / denom)) / (2 * N)
    return loss


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
