"""
This file defines functionality to invert an RBIG transform
"""

import numpy as np
from univariate_invert_normalization import univariate_invert_normalization

def invert_rbig(gaussian_data, transform_params, progress_report_interval=None):
  """
  Inverts an RBIG transform by using the saved transform params

  Parameters
  ----------
  gaussian_data : ndarray
      A 2D array giving an iid sample in each column. The components within
      each column have been gaussianized by RBIG.
  transform_params : dictionary
      Parameters of the forward transform. Should have toplevel keys of
      'pdf_extension': The fraction by which to extend the support of the
        gaussianized marginal pdf compared to the empirical marginal pdf
      'pdf_resolution': The number of points at which to compute the
        gaussianized marginal pdfs.
      'iterations': dictionary
        keys are [0, 1, 2, ..., num_iters] indexing the RBIG iteration
        values are dictionaries with keys [0, 1, 2, ..., num_components],
          base level dictionary has keys
          'empirical_pdf_support', 'empirical_pdf', 'uniform_cdf_support',
          and 'uniform_cdf':
  progress_report_interval : int, optional
      If specified, report the RBIG iteration number every 
      progress_report_interval iterations.

  Returns
  -------
  sampled_data : ndarray
      Data sampled under the inverse model
  """
  sampled_data = np.copy(gaussian_data)
  total_iters = 0
  for rbig_iter in range(len(transform_params['iterations'])-1, -1, -1):
    if progress_report_interval is not None:
      if rbig_iter % progress_report_interval == 0:
        print("Completed ", len(transform_params['iterations']) - rbig_iter,
              "iterations of Inverse-RBIG")
    # we have to go in reverse order
    rotation_matrix = transform_params['iterations'][rbig_iter]['rotation_matrix']
    sampled_data = np.dot(rotation_matrix.T, sampled_data)
    for component_idx in range(sampled_data.shape[0]):
      sampled_data[component_idx, :] = univariate_invert_normalization(
          sampled_data[component_idx, :],
          transform_params['iterations'][rbig_iter][component_idx])

  return sampled_data
