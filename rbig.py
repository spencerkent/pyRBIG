"""
This file defines the rbig function which iteratively gaussianizes data
"""

import numpy as np
from scipy.stats import ortho_group
from univariate_make_normal import univariate_make_normal

def rbig(data, num_iters, rotation_type, pdf_extension=0.1,
         pdf_resolution=1000, progress_report_interval=None):
  """
  Rotation-based iterative gaussianization

  Parameters
  ----------
  data : ndarray
      A 2D array giving an iid sample in each column. Number of rows is the
      number of components in each datapoint.
  num_iters : int
      The number of steps to run the sequence of marginal gaussianization and
      then rotation
  rotation_type : str
      One of {'PCA', 'random'}. The type of orthogonal linear transform to
      apply to gaussianized data at each iteration. Later will add ICA
      rotation based on FastICA
  pdf_extension : float
      The fraction by which to extend the support of the gaussianized marginal
      pdf compared to the empirical marginal pdf
  pdf_resolution : int
      The number of points at which to compute the gaussianized marginal pdfs.
      The functions that map from original data to gaussianized data at each
      iteration have to be stored so that we can invert them later - if working
      with high-dimensional data consider reducing this resolution to shorten
      computation time.
  progress_report_interval : int, optional
      If specified, report the RBIG iteration number every
      progress_report_interval iterations.
  """
  parameter_lookup = {'pdf_extension': pdf_extension,
                      'pdf_resolution': pdf_resolution,
                      'iterations': {}}
  #^ we'll use this to also store parameters of the gaussianizing transform
  # at each iteration

  num_components = data.shape[0]
  num_samples = data.shape[1]
  g_data = np.copy(data)  # gaussianized data

  for rbig_iter in range(num_iters):
    if progress_report_interval is not None:
      if rbig_iter % progress_report_interval == 0:
        print("Completed ", rbig_iter, "iterations of RBIG")
    # Marginal gaussianization
    for c_idx in range(num_components):
      if c_idx == 0:
        parameter_lookup['iterations'][rbig_iter] = {}

      g_data[c_idx, :], params = univariate_make_normal(g_data[c_idx, :],
                                                        pdf_extension,
                                                        pdf_resolution)
      parameter_lookup['iterations'][rbig_iter][c_idx] = params

    # Rotation
    if rotation_type == 'random':
      rand_ortho_matrix = ortho_group.rvs(num_components)
      g_data = np.dot(rand_ortho_matrix, g_data)
      parameter_lookup['iterations'][
          rbig_iter]['rotation_matrix'] = rand_ortho_matrix

    elif rotation_type == 'PCA':
      if num_components > num_samples or num_components > 10**6:
        # If the dimensionality of each datapoint is high, we probably
        # want to compute the SVD of the data directly to avoid forming a huge
        # covariance matrix
        U, _, _ = np.linalg.svd(g_data, full_matrices=True)
      else:
        # the SVD is more numerically stable then eig so we'll use it on the 
        # covariance matrix directly
        U, _, _ = np.linalg.svd(np.dot(g_data, g_data.T) / num_samples, 
                                full_matrices=True)

      g_data = np.dot(U.T, g_data)
      parameter_lookup['iterations'][
          rbig_iter]['rotation_matrix'] = U.T

    else:
      raise ValueError('Rotation type ' + rotation_type + ' not recognized')

  return g_data, parameter_lookup
