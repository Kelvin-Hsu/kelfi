"""
Kernel Means Learning Module.

Core methods for Bayesian learning of hyperparameters for likelihood-free inference with kernel means.
"""
import numpy as np
import tensorflow as tf
from tensorflow_kernels import gaussian_kernel_gramix, gaussian_density_gramix_multiple, gaussian_density_gramix, convert_anisotropic, atleast_2d


def kernel_means_weights(y, x_sim, theta_sim, eps, beta, reg=None):
    """
    Compute the weights of the kernel means likelihood.
    
    Parameters
    ----------
    y : tf.Tensor [size: (1, d)]
        Observed data or summary statistics
    x_sim : tf.Tensor [size: (m, s, d)]
        Simulated data or summary statistics
    theta_sim : tf.Tensor [size: (m, p)]
        Parameter values corresponding to the simulations
    eps : float or tf.Tensor [size: () or (1,) for isotropic; (d,) for anistropic]
        The simulator noise level(s) for the epsilon-kernel or epsilon-likelihood
    beta : float or tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    reg : float, optional
        The regularization parameter for the conditional kernel mean
        
    Returns
    -------
    tf.Tensor [size: (m, 1)]
        The weights of the kernel means likelihood
    """
    # size: (m, 1)
    if len(x_sim.get_shape().as_list()) == 3:
        data_epsilon_likelihood = tf.transpose(gaussian_density_gramix_multiple(y, x_sim, eps))
    elif len(x_sim.get_shape().as_list()) == 2:
        data_epsilon_likelihood = tf.transpose(gaussian_density_gramix(y, x_sim, eps))
    else:
        raise ValueError('Simulated dataset is neither 2D or 3D.')
        
    # The number of simulations
    m = theta_sim.get_shape().as_list()[0]
    
    # Set the regularization hyperparameter to some default value if not specified
    if reg is None:
        reg = 1e-3 * tf.reduce_min(beta)
        
    # Compute the weights at O(m^3)
    theta_sim_gramix = gaussian_kernel_gramix(theta_sim, theta_sim, beta)
    theta_sim_gramix_cholesky = tf.cholesky(theta_sim_gramix + m * reg * tf.eye(m))
    weights = tf.cholesky_solve(theta_sim_gramix_cholesky, data_epsilon_likelihood)
    
    # size: (m, 1)
    return weights


def kernel_means_likelihood(theta_query, theta_sim, weights, beta):
    """
    Query the kernel means likelihood.
    
    Parameters
    ----------
    theta_query : tf.Tensor [size: (n_query, p)]
        The parameters to query the likelihood at
    theta_sim : tf.Tensor [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : tf.Tensor [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
        
    Returns
    -------
    tf.Tensor [size: (n_query,)]
        The kernel means likelihood values at the query points
    """
    # size: (m, n_query)
    theta_evaluation_gramix = gaussian_kernel_gramix(theta_sim, theta_query, beta)
    
    # size: (n_query,)
    return tf.squeeze(tf.matmul(tf.transpose(theta_evaluation_gramix), weights))


def marginal_kernel_means_likelihood(theta_sim, weights, beta, prior_mean=None, prior_std=None):
    """
    Compute the marginal kernel means likelihood under a diagonal Gaussian prior.
    
    Parameters
    ----------
    theta_sim : tf.Tensor [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : tf.Tensor [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
    prior_mean : tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The mean(s) of the diagonal Gaussian prior
    prior_std : tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The standard deviation(s) of the diagonal Gaussian prior
        
    Returns
    -------
    float
        The marginal kernel means likelihood
    """
    # By defaut, the prior has zero mean
    if prior_mean is None:
        prior_mean = tf.zeros((1, theta_sim.get_shape().as_list()[-1]))
        
    # By default, the prior standard deviation is set to the same as the length scale of the parameter kernel
    if prior_std is None:
        prior_std = beta

    # Compute the final length scale and the ratio scalar coefficient of the resulting prior mean embedding 
    prior_embedding_length_scale = tf.sqrt(beta ** 2 + prior_std ** 2)
    ratio = tf.reduce_prod(convert_anisotropic(beta / prior_embedding_length_scale, theta_sim.get_shape().as_list()[1]))
        
    # Compute the prior mean embedding [size: (m, 1)]
    prior_mean_embedding = ratio * gaussian_kernel_gramix(theta_sim, atleast_2d(prior_mean), prior_embedding_length_scale)
    
    # Compute the kernel means marginal likelihood
    return tf.reduce_sum(tf.multiply(prior_mean_embedding, weights))


def approximate_marginal_kernel_means_likelihood(theta_samples, theta_sim, weights, beta):
    """
    Compute the approximate marginal kernel means likelihood using prior samples.
    
    Parameters
    ----------
    theta_samples : tf.Tensor [size: (n_samples, p)]
        The prior parameter samples to marginalize over
    theta_sim : tf.Tensor [size: (m, p)]
        Parameter values corresponding to the simulations
    weights : tf.Tensor [size: (m, 1)]
        The weights of the kernel means likelihood
    beta : float or tf.Tensor [size: () or (1,) for isotropic; (p,) for anistropic]
        The length scale(s) for the parameter kernel
        
    Returns
    -------
    float
        The approximate marginal kernel means likelihood   
    """
    return tf.reduce_mean(kernel_means_likelihood(theta_samples, theta_sim, weights, beta))


def kernel_means_hyperparameter_learning(y, x_sim, theta_sim, eps_tuple, beta_tuple, reg_tuple,
                                         eps_ratios=1., beta_ratios=1., offset=0.,
                                         prior_samples=None, prior_mean=None, prior_std=None,
                                         learning_rate=0.01, n_iter=1000, display_steps=10):
    """
    Bayesian hyperparameter learning for KELFI by maximizing the MKML.
    
    The API is written to take in numpy arrays. There is no need to pass in tensorflow tensors directly.
    If the prior is Gaussian, analytical forms exist. Specify its mean and standard deviation in 'prior_mean' and 'prior_std'.
    Otherwise, approximate forms exist by using prior samples. Provide the samples via 'prior_samples'.
    If neither is provided, the prior is assumed to be Gaussian with zero mean 
    and standard deviation equal to the length scale of the parameter kernel.
    For anisotropic cases, it could be convenient to learn the multiplier on fixed length scale ratios. This is 
    especially true for the parameter kernel, where the length scales can be set to a scaled multiple of the prior
    standard deviations. Provide these ratios in 'eps_ratios' and 'beta_ratios'.
    
    Parameters
    ----------
    y : np.ndarray [size: (1, d)]
        Observed data or summary statistics
    x_sim : np.ndarray [size: (m, s, d)]
        Simulated data or summary statistics
    theta_sim : np.ndarray [size: (m, p)]
        Parameter values corresponding to the simulations
    eps_tuple : tuple (eps_init, learn_flag)
        eps_init: float or np.ndarray [size: () or (1,) for isotropic; (d,) for anistropic]
            The initial simulator noise level(s) for the epsilon-kernel or epsilon-likelihood
        learn_flag: str
            Indicate learning for this hyperparameter with 'learn' and use 'fix' otherwise
    beta_tuple : tuple (beta_init, learn_flag)
        beta_init: float or np.ndarray [size: () or (1,) for isotropic; (d,) for anistropic]
            The initial length scale(s) for the parameter kernel
        learn_flag: str
            Indicate learning for this hyperparameter with 'learn' and use 'fix' otherwise
    reg_tuple : tuple (reg_init, learn_flag)
        reg_init: float
            The initial regularization parameter for the conditional kernel mean
        learn_flag: str
            Indicate learning for this hyperparameter with 'learn' and use 'fix' otherwise
    eps_ratios: float or np.ndarray [size: () or (1,) for isotropic; (d,) for anistropic]
        Fixed ratios for the simulator noise level(s) for the epsilon-kernel or epsilon-likelihood
    beta_ratios: float or np.ndarray [size: () or (1,) for isotropic; (d,) for anistropic]
        Fixed ratios for the length scale(s) for the parameter kernel
    offset : float
        A positive offset in case approximate marginal kernel means likelihood values are slightly negative.
    prior_samples : np.ndarray [size: (n_samples, p)]
        The parameters samples to marginalize over
    prior_mean : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The mean(s) of the diagonal Gaussian prior
    prior_std : np.ndarray [size: () or (1,) for isotropic; (p,) for anistropic]
        The standard deviation(s) of the diagonal Gaussian prior
    learning_rate : float
        The learning rate for the gradient update
    n_iter : int
        Number of iterations
    display_steps : int
        Number of iterations before displaying the current optimization status
    """
    # Short notation for converting into tensorflow constants or variables
    tfc = lambda array: tf.constant(array, dtype=tf.float32)
    tfv = lambda array: tf.Variable(array, dtype=tf.float32)
    
    # Convert hyperparameters to variable for learning or constants otherwise
    log_eps = tfv(np.log(eps_tuple[0])) if eps_tuple[1] == 'learn' else tfc(np.log(eps_tuple[0]))
    log_beta = tfv(np.log(beta_tuple[0])) if beta_tuple[1] == 'learn' else tfc(np.log(beta_tuple[0]))
    log_reg = tfv(np.log(reg_tuple[0])) if reg_tuple[1] == 'learn' else tfc(np.log(reg_tuple[0]))
    
    # Transform the hyperparameters into the actual hyperparameters if not already
    eps_ = tf.exp(log_eps) * eps_ratios
    beta_ = tf.exp(log_beta) * beta_ratios
    reg_ = tf.exp(log_reg)
    
    # Convert all data into constants
    y_, x_sim_, theta_sim_ = tfc(y), tfc(x_sim), tfc(theta_sim)
    
    # Compute the weights
    weights_ = kernel_means_weights(y_, x_sim_, theta_sim_, eps_, beta_, reg=reg_)
    
    # Compute the objective using either the analytical or empirical form
    if prior_samples is None:
        # By defaut, the prior has zero mean
        prior_mean_ = tf.zeros((1, theta_sim.get_shape().as_list()[-1])) if prior_mean is None else tfc(prior_mean)
        # By default, the prior standard deviation is set to the same as the length scale of the parameter kernel
        prior_std_ = beta if prior_std is None else tfc(prior_std)
        # Objective
        ml = marginal_kernel_means_likelihood(theta_sim_, weights_, beta_, prior_mean=prior_mean_, prior_std=prior_std_)
    else:
        # Use prior samples if the prior was not transformed into Gaussians beforehand
        theta_samples_ = tfc(prior_samples)
        # Objective
        ml = approximate_marginal_kernel_means_likelihood(theta_samples_, theta_sim_, weights_, beta_)
    objective = - tf.log(ml + offset)
    
    # Minimize the negative marginal likelihood
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    learn_step = opt.minimize(objective)       
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(n_iter):  
        if i % display_steps == 0:
            _, eps, beta, reg, marginal_likelihood = sess.run([learn_step, eps_, beta_, reg_, ml])
            print('i: ', i, ' | eps: ', eps, ' | beta: ', beta, ' | reg: ', reg, ' | ml: ', marginal_likelihood)
        else:
            sess.run(learn_step)
        
    # Return the learned hyperparameters
    return eps, beta, reg