import torch
from torch.nn.functional import binary_cross_entropy
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ELBO(pred, target, mu, sigma, free_bits):
    """
    Evidence Lower Bound
    Return KL Divergence and KL Regularization using free bits
    """
    device = pred.device
    # Reconstruction error
    # Pytorch cross_entropy combines LogSoftmax and NLLLoss
    likelihood = -binary_cross_entropy(pred, target, reduction='sum')
    # Regularization error
    sigma_prior = torch.tensor([1], dtype=torch.float, device=device)
    mu_prior = torch.tensor([0], dtype=torch.float, device=device)
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)
    elbo = torch.mean(likelihood) - torch.max(torch.mean(kl_div)-free_bits, torch.tensor([0], dtype=torch.float, device=device))
    
    return -elbo, kl_div.mean()
