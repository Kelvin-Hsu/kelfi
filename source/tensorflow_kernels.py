"""
Kernel Module.

Basic functions used for computing anisotropic Gaussian kernels.

Note: Matern and other kernels were removed from this version, since they were not used in the experiments in the end.
"""
import tensorflow as tf
import numpy as np


def sqdist_gramix(x_p, x_q, length_scale):
    """
    Compute a gram matrix of euclidean distances between two datasets under an isotropic or anisotropic length scale.
    
    Parameters
    ----------
    x_p : tf.Tensor [Size: (n_p, d)]
        A dataset
    x_q : tf.Tensor [Size: (n_q, d)]
        A dataset
    length_scale : float or tf.Tensor [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
        
    Returns
    -------
    tf.Tensor [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, d)
    z_p = x_p / length_scale
    # Size: (n_q, d)
    z_q = x_q / length_scale
    # Size: (n_p, n_q)
    d_pq = tf.matmul(z_p, tf.transpose(z_q))
    # Size: (n_p,)
    d_p = tf.reduce_sum(z_p ** 2, axis=1)
    # Size: (n_q,)
    d_q = tf.reduce_sum(z_q ** 2, axis=1)
    # Size: (n_p, n_q)
    return d_p[:, tf.newaxis] - 2 * d_pq + d_q


def gaussian_kernel_gramix(x_p, x_q, length_scale):
    """
    Compute a gram matrix of Gaussian kernel values between two datasets under an isotropic or anisotropic length scale.
    
    Parameters
    ----------
    x_p : tf.Tensor [Size: (n_p, d)]
        A dataset
    x_q : tf.Tensor [Size: (n_q, d)]
        A dataset
    length_scale : float or tf.Tensor [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
        
    Returns
    -------
    tf.Tensor [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, n_q)
    return tf.exp(-0.5 * sqdist_gramix(x_p, x_q, length_scale))


def atleast_1d(tensor):
    """
    Convert a tensor to at least 1 dimensional (equivalent to np.atleast_1d).
    
    Parameters
    ----------
    tensor : tf.Tensor
        A tensor
    
    Returns
    -------
    tf.Tensor
        A tensor of at least 1 dimension
    """
    return tf.reshape(tensor, [1]) if len(tensor.get_shape()) == 0 else tensor


def atleast_2d(tensor):
    """
    Convert a tensor to at least 2 dimensional (equivalent to np.atleast_2d).
    
    Parameters
    ----------
    tensor : tf.Tensor
        A tensor
    
    Returns
    -------
    tf.Tensor
        A tensor of at least 2 dimension
    """
    return tf.reshape(tensor, [1, 1]) if len(tensor.get_shape()) == 0 else tf.reshape(tensor, [1, tensor.get_shape().as_list()[-1]])


def convert_anisotropic(length_scale, d):
    """
    Convert isotropic length scale format to anisotropic length scale format.
    
    Parameters
    ----------
    length_scale : tf.Tensor [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
    d : int
        The dimensionality of the anisotropic kernel
    
    Returns
    -------
    tf.Tensor [Size: (d,)]
        The anisotropic length scale(s)
    """
    # Size: (1,) or (d,)
    length_scale_array = atleast_1d(length_scale)
    # Make sure it is not more than 1 dimensional
    assert len(length_scale_array.get_shape()) == 1
    # Make sure it is either isotropic or anisotropic
    assert length_scale_array.get_shape().as_list()[0] in [1, d]
    # Convert it to anistropic only if it is isotropic or keep it the same
    if length_scale_array.get_shape().as_list()[0] == 1:
        return tf.tile(length_scale_array, [d])
    else:
        return length_scale_array

    
def gaussian_density_gramix(x, mu, sigma):
    """
    Compute a gram matrix of Gaussian density values of a dataset for multiple means.
    Parameters
    ----------
    x : tf.Tensor [Size: (n, d)]
        A dataset
    mu : tf.Tensor [Size: (m, d)]
        A dataset
    sigma : float or tf.Tensor [Size: () or (1,) for isotropic; (d,) for anistropic]
        The standard deviations(s)
        
    Returns
    -------
    tf.Tensor [Size: (n, m)]
        The resulting gram matrix
    """
    # Compute the scaling factor of a diagonal multivariate Gaussian distribution
    d = mu.get_shape().as_list()[-1]
    const = (tf.sqrt(2 * np.pi) ** d) * tf.reduce_prod(convert_anisotropic(sigma, d))
    # Size: (n, m)
    return gaussian_kernel_gramix(x, mu, sigma) / const


def gaussian_density_gramix_multiple(x, mu, sigma):
    """
    Compute the average gram matrix of Gaussian density values of a dataset for multiple arrays of means.
    
    Parameters
    ----------
    x : tf.Tensor [Size: (r, n, d)]
        A dataset
    mu : tf.Tensor [Size: (m, s, d)]
        A dataset where the average is to be taken over the middle dimension (s)
    sigma : float or tf.Tensor [Size: () or (1,) for isotropic; (d,) for anistropic]
        The standard deviations(s)
        
    Returns
    -------
    tf.Tensor [Size: (n, m)]
        The resulting gram matrix
    """    
    # Obtain and ensure the shape matches
    r, n, d = tuple(x.get_shape().as_list())
    m, s, d = tuple(mu.get_shape().as_list())
    assert d == x.get_shape().as_list()[-1]
    assert d == mu.get_shape().as_list()[-1]
    # Size: (r * n, d)
    x_2d = tf.reshape(x, [r * n, d])
    # Size: (m * s, d)
    mu_2d = tf.reshape(mu, [m * s, d])
    # Size: (r * n, m * s)
    gramix_2d = gaussian_density_gramix(x_2d, mu_2d, sigma)
    # Size: (r, n, m, s)
    gramix_4d = tf.reshape(gramix_2d, [r, n, m, s])
    # Size: (n, m)
    # return tf.reduce_mean(tf.reduce_prod(gramix_4d, axis=1), axis=-1)
    return tf.reduce_prod(tf.reduce_mean(gramix_4d, axis=-1), axis=1)