import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Calculate Renyi divergence between two multivariate Gaussians
# mu is mean of 1st distribution, mean of 2nd distribution is 0
# var is variance of 1st distribution, gamma is variance of 2nd distribution
def renyi_divergence(mu, var, alpha, gamma=1):
    sigma_star = alpha * gamma + (1 - alpha) * var
    term1 = alpha / 2 * mu ** 2 / sigma_star

    term2_1 = var ** (1 - alpha) * gamma ** alpha
    term2_2 = torch.log(sigma_star / term2_1)
    term2 = -0.5 / (alpha - 1) * term2_2

    total = term1 + term2

    return torch.sum(total)

def renyi_crossentropy(mu, var, alpha, gamma=1):

    # var = logvar.exp()
    # var is sigma squared
    # alpha_var = log_alpha_var.exp()
    # print(f'alpha_var is {alpha_var}')
    # var = (alpha_var - 1) / (alpha - 1)
    # print(f'var is {var}')

    inside_log = (gamma + var * (alpha -1)) / gamma

    # print(f'var is {torch.sum(var)}')

    #
    # print(f'var sum is {torch.sum(var)}')
    # print(f'inside_log sum {torch.sum(inside_log)}')

    term1 = -torch.log(inside_log)
    # term1 = -log_alpha_var
    # print(f'term1 sum is {torch.sum(term1)}')

    # term2 = (1-alpha) * gamma ** 64 * torch.log(2*3.14159)**64
    term3 = -mu ** 2 / var
    term3 = torch.nan_to_num(term3)
    # print(f'term3 sum is {torch.sum(term3)}')

    term4 = mu **2 / var **2 * var * gamma / (gamma + var * (alpha-1))
    term4 = torch.nan_to_num(term4)
    # print(f'term4 sum is {torch.sum(term4)}')


    total = term1 + term3 + term4

    # print(f'return sum is {torch.sum(total)/(2-2*alpha) }')


    return torch.sum(total) / (2-2*alpha)

def gaussian_nll(yhat, y, var, reduction='mean'):
    """
    Compute the Gaussian negative log-likelihood for y under
    N(yhat, exp(log_var)).

    Args:
        yhat   : Tensor of shape (batch_size, ...) - predicted mean
        log_var  : Tensor of shape (batch_size, ...) - predicted log-variance
        y   : Tensor of the same shape as yhat
        reduction: 'sum' or 'mean' or 'none'

    Returns:
        Scalar (if reduction != 'none') or elementwise NLL (if 'none').
    """
    # var = exp(log_var)

    # The actual term: 0.5 * [ log(2π) + log_var + (y - yhat)^2 / var ]
    nll_elem = 0.5 * (
        math.log(2.0 * math.pi) + torch.log(var) + (y - yhat).pow(2) / var
    )

    if reduction == 'sum':
        return torch.sum(nll_elem)
    elif reduction == 'mean':
        return torch.mean(nll_elem)
    else:
        return nll_elem

# IB or CFB loss
def get_KLdivergence_loss(y, yhat, mu, var, beta, r_method='mean'):
    # mu_ref = 10
    # var_ref = 1

    var = torch.nn.functional.softplus(var) + 1e-6
    
    if r_method == 'mean':
    	divergence = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
    else:
    	divergence = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    # divergence = 0.5 * torch.mean((mu_ref - mu) ** 2 / var_ref + var / var_ref - 1 - torch.log(var / var_ref))
    # divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    yhat = torch.where(torch.isnan(yhat), torch.zeros_like(yhat), yhat)
    yhat = torch.where(torch.isinf(yhat), torch.zeros_like(yhat), yhat)

    cross_entropy = torch.nn.functional.mse_loss(yhat, y, reduction=r_method)
    # cross_entropy = cross_entropy / (torch.mean(y ** 2) + 1e-6)
    # cross_entropy = gaussian_nll(yhat, y, var, reduction='mean')
    # cross_entropy = -torch.nn.functional.gaussian_nll_loss(yhat, y, var, reduction='mean')

    # print("mu:", mu)
    # print("var:", var)
    # print("Divergence:", divergence)
    # print("Cross Entropy:", cross_entropy)

    return divergence + beta * cross_entropy


# RFIB loss
def get_RFIB_loss(y, yhat_fair, yhat, mu, logvar, alpha, beta1, beta2):
    if alpha == 0:
        divergence = 0
    elif alpha == 1:  # KL Divergence
        divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elif alpha < 1: # Renyi cross entropy
        divergence = renyi_crossentropy(mu, logvar, alpha)
    else: # Renyi divergence
        divergence = renyi_divergence(mu, logvar, alpha)

    IB_cross_entropy = torch.nn.functional.mse_loss(yhat, y, reduction='mean')
    CFB_cross_entropy = torch.nn.functional.mse_loss(yhat_fair, y, reduction='mean')

    # print(f'IB_cross_entropy is {IB_cross_entropy}')
    # print(f'beta1 * IB_cross_entropy is {beta1 * IB_cross_entropy}')
    # print(f'CFB_cross_entropy is {CFB_cross_entropy}')
    # print(f'beta2 * CFB_cross_entropy is {beta2 * CFB_cross_entropy}')

    loss = divergence + beta1 * IB_cross_entropy + beta2 * CFB_cross_entropy

    return loss
