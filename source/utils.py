"""
Utils Module.

Miscellaneous methods that are helpful.
"""
import numpy as np
import ghalton as gh

def halton_sequence(n_points=1, n_dims=1):
    """
    Generate Quasi Monte Carlo samples using the Halton sequence.
    
    Parameters
    ----------
    n_points : int
        The number of data points to generate
    n_dims : int
        The number of dimensions to generate the samples in

    Returns
    -------
    numpy.ndarray
        A dataset of size (n_points, n_dims)
    """
    sequencer = gh.Halton(n_dims)
    points = np.array(sequencer.get(n_points))
    return points