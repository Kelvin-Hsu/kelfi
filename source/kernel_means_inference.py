"""
Kernel Means Inference Module.

Core methods for likelihood-free inference with kernel means.
"""
import numpy as np
import scipy.linalg as la
from kernels import gaussian_kernel_gramix, gaussian_density_gramix_multiple, gaussian_density_gramix, convert_anisotropic


def kernel_means_weights(y, x_sim, theta_sim, eps, beta, reg=None):
    """
    Compute the weights of the kernel means likelihood.
    
    Parameters
    ----------
    y : np.ndarray [size: (1, d)]
        Observed data or summary statistics
    x_sim : np.ndarray [size: (m, s, d)]
        Simulated data or summary statistics
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    eps : float or np.ndarray [size: () or (1,) for isotropic; (d,) for anistropic]
        The simulator noise level(s) for the epsilon-kernel or epsilon-likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    reg : float, optional
        The regularization parameter for the conditional kernel mean
        
    Returns
    -------
    np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    """
    # size: (m, 1)
    if x_sim.ndim == 3:
        data_epsilon_likelihood = gaussian_density_gramix_multiple(y, x_sim, eps).transpose()
    elif x_sim.ndim == 2:
        data_epsilon_likelihood = gaussian_density_gramix(y, x_sim, eps).transpose()
    else:
        raise ValueError('Simulated dataset is neither 2D or 3D.')
    
    # The number of simulations
    m = theta_sim.shape[0]
    
    # Set the regularization hyperparameter to some default value if not specified
    if reg is None:
        reg = 1e-3 * np.min(beta)
        
    # Compute the weights at O(m^3)
    theta_sim_gramix = gaussian_kernel_gramix(theta_sim, theta_sim, beta)
    lower = True
    theta_sim_gramix_cholesky = la.cholesky(theta_sim_gramix + m * reg * np.eye(m), lower=lower)
    weights = la.cho_solve((theta_sim_gramix_cholesky, lower), data_epsilon_likelihood)
    
    # size: (m, 1)
    return weights


def kernel_means_likelihood(theta_query, theta_sim, weights, beta):
    """
    Query the kernel means likelihood.
    
    Parameters
    ----------
    theta_query : np.ndarray [size: (n_query, p)]
        The parameters to query the likelihood at
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
        
    Returns
    -------
    np.ndarray [size: (n_query,)]
        The kernel means likelihood values at the query points
    """
    # size: (m, n_query)
    theta_evaluation_gramix = gaussian_kernel_gramix(theta_sim, theta_query, beta)
    
    # size: (n_query,)
    return np.dot(theta_evaluation_gramix.transpose(), weights).ravel()


def marginal_kernel_means_likelihood(theta_sim, weights, beta, prior_mean=None, prior_std=None):
    """
    Compute the marginal kernel means likelihood under a diagonal Gaussian prior.
    
    Parameters
    ----------
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    prior_mean : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The mean(s) of the diagonal Gaussian prior
    prior_std : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The standard deviation(s) of the diagonal Gaussian prior
        
    Returns
    -------
    float
        The marginal kernel means likelihood
    """
    # By defaut, the prior has zero mean
    if prior_mean is None:
        prior_mean = np.zeros((1, theta_sim.shape[-1]))
        
    # By default, the prior standard deviation is set to the same as the length scale of the parameter kernel
    if prior_std is None:
        prior_std = beta

    # Compute the final length scale and the ratio scalar coefficient of the resulting prior mean embedding 
    prior_embedding_length_scale = np.sqrt(beta ** 2 + prior_std ** 2)
    ratio = np.prod(convert_anisotropic(beta / prior_embedding_length_scale, theta_sim.shape[1]))
        
    # Compute the prior mean embedding [size: (m, 1)]
    prior_mean_embedding = ratio * gaussian_kernel_gramix(theta_sim, np.atleast_2d(prior_mean), prior_embedding_length_scale)
    
    # Compute the kernel means marginal likelihood
    return np.dot(prior_mean_embedding.ravel(), weights.ravel())


def approximate_marginal_kernel_means_likelihood(theta_samples, theta_sim, weights, beta):
    """
    Compute the approximate marginal kernel means likelihood using prior samples.
    
    Parameters
    ----------
    theta_samples : np.ndarray [size: (n_samples, p)]
        The parameters samples to marginalize over
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
        
    Returns
    -------
    float
        The approximate marginal kernel means likelihood
    """  
    return np.mean(kernel_means_likelihood(theta_samples, theta_sim, weights, beta))


def kernel_means_posterior(theta_query, theta_sim, weights, beta, prior_pdf, marginal_likelihood):
    """
    Query the kernel means posterior.
    
    Parameters
    ----------
    theta_query : np.ndarray [size: (n_query, p)]
        The parameters to query the likelihood at
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    prior_pdf : callable
        The prior probability density function
    marginal_likelihood: float
        The pre-computed marginal kernel means likelihood
        
    Returns
    -------
    np.ndarray [size: (n_query,)]
        The kernel means likelihood values at the query points
    """
    return kernel_means_likelihood(theta_query, theta_sim, weights, beta) * prior_pdf(theta_query).ravel() / marginal_likelihood
  

def kernel_means_posterior_embedding(theta_query, theta_sim, weights, beta, prior_mean=None, prior_std=None, marginal_likelihood=None):
    """
    Compute the kernel means posterior embedding.
    
    Parameters
    ----------
    theta_query : np.ndarray [size: (n_query, p)]
        The parameters to query the likelihood at
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    prior_mean : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The mean(s) of the diagonal Gaussian prior
    prior_std : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The standard deviation(s) of the diagonal Gaussian prior
    marginal_likelihood : float, optional
        The marginal likelihood value if it was precomputed
    """
    # By defaut, the prior has zero mean
    if prior_mean is None:
        prior_mean = np.zeros((1, theta_sim.shape[-1]))
        
    # By default, the prior standard deviation is set to the same as the length scale of the parameter kernel
    # Compute the final length scale of the resulting prior mean embedding
    if prior_std is None:
        prior_std = beta
        
    # Anisotropic constants
    sigma = convert_anisotropic(prior_std, theta_sim.shape[1])
    s = 1 / np.sqrt(2 / (beta ** 2) + 1 / (sigma ** 2))
    gamma = convert_anisotropic(beta / sigma, theta_sim.shape[1])
    denom = 2 + (gamma ** 2)
    
    # Broadcast: (m, d) -> (1, m, d)
    theta_sim_broad = theta_sim[np.newaxis, :, :]
    
    # Broadcast: (q, d) -> (q, 1, d)
    theta_query_broad = theta_query[:, np.newaxis, :]
    
    # Compute the quadratic term in the exponent with size: (q, m, d)
    first = theta_query_broad ** 2 + theta_sim_broad ** 2 + (gamma * prior_mean) **2
    second = theta_query_broad + theta_sim + gamma ** 2 * prior_mean
    c = first / denom - (second / denom) ** 2
    
    # Perform the integral and multiple each dimension to get size: (q, m)
    h = np.prod((s / sigma) * np.exp(- 0.5 * c / (s ** 2)), axis=-1)
    
    # Compute the marginal likelihood if it has not been computed already
    if marginal_likelihood is None:
        marginal_likelihood = kernel_means_marginal_likelihood(theta_sim, weights, beta, prior_mean=prior_mean, prior_std=prior_std)
    
    # size: (q,)
    return np.dot(h, weights).ravel() / marginal_likelihood


def approximate_kernel_means_posterior_embedding(theta_query, theta_sim, weights, beta, theta_samples, marginal_likelihood=None, beta_query=None):
    """
    Compute the approximate kernel means posterior embedding.
    
    Parameters
    ----------
    theta_query : np.ndarray [size: (n_query, p)]
        The parameters to query the likelihood at
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : np.ndarray [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    theta_samples : np.ndarray [size: (n_samples, p)]
        The parameters samples to marginalize over
    marginal_likelihood : float, optional
        The marginal likelihood value if it was precomputed
    """
    # Approximate the integral empirically with size: (q, m)
    if beta_query is None:
        beta_query = beta
    h = np.dot(gaussian_kernel_gramix(theta_query, theta_samples, beta_query), gaussian_kernel_gramix(theta_samples, theta_sim, beta)) / theta_samples.shape[0]
    
    # Compute the marginal likelihood if it has not been computed already
    if marginal_likelihood is None:
        marginal_likelihood = approximate_marginal_kernel_means_likelihood(theta_samples, theta_sim, weights, beta)
        
    # size: (q,)
    return np.dot(h, weights).ravel() / marginal_likelihood
    
    
def kernel_herding(mean_embedding_values_or_function, kernel_function, samples, n_super_samples, init_super_samples=None):
    """
    Obtain super samples via kernel herding.
    
    Parameters
    ----------
    mean_embedding_values_or_function : callable or np.ndarray [size: (n_samples,)]
        An mean embedding function of one argument or the mean embedding values queried at the samples
    kernel_function : callable
        A kernel function of two arguments on samples
    samples : np.ndarray (n_samples, n_dim)
        The original samples to samples from
    n_super_samples : int
        The number of super samples to obtain
    init_super_samples : np.ndarray [size: (n_init_super_samples, n_dim)], optional
        The super samples already obtained if one wants to continue sampling instead of restarting the herd
    """
    # Initialize the super samples
    if init_super_samples is None:
        n_init_super_samples = 0
        super_samples = np.zeros((n_super_samples, samples.shape[1]))
    else:
        n_init_super_samples = init_super_samples.shape[0]
        super_samples[:n_init_super_samples] = init_super_samples
    
    # Obtain the mean embedding quried at the samples
    if type(mean_embedding_values_or_function) is callable:
        mu = embedding_values_or_function(samples)
    else:
        mu = mean_embedding_values_or_function
        assert mu.shape[0] == samples.shape[0]
        
    # Perform standard kernel herding
    mu_hat_sum = 0
    for i in range(n_init_super_samples, n_super_samples):
        mu_hat = mu_hat_sum / (i + 1)
        objective = mu - mu_hat
        super_samples[i] = samples[np.argmax(objective)]
        mu_hat_sum += kernel_function(super_samples[[i]], samples).ravel()

    # size : (n_super_samples, n_dim)
    return super_samples