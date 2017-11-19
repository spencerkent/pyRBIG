"""
Applies a pre-learned RBIG transform to a dataset
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

def apply_rbig(data, transform_params):
  """
  Given a dataset and the parameters of a learned RBIG transorm apply to data

  Parameters
  ----------
  data : ndarray
      A 2D array giving an iid sample in each column. Number of rows is the
      number of components in each datapoint.
  transform_params
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
  Returns
  -------
  rbig_transformed : ndarray
      Same size as input, but gaussianized using the RBIG transform
  """
  rbig_transformed = np.copy(data)
  for rbig_iter in range(len(transform_params['iterations'])):
    if rbig_iter % 10 == 0:
      print("Completed ", rbig_iter, "iterations of RBIG on fresh data")
    for component_idx in range(data.shape[0]):
      # marginal uniformization
      rbig_transformed[component_idx, :] = interp1d(
          transform_params['iterations'][rbig_iter][
            component_idx]['uniform_cdf_support'],
          transform_params['iterations'][rbig_iter][
            component_idx]['uniform_cdf'])(
          rbig_transformed[component_idx, :])
      # marginal gaussianization
      rbig_transformed[component_idx] = \
          norm.ppf(rbig_transformed[component_idx, :])

    # rotation
    rbig_transformed = \
        np.dot(transform_params['iterations'][rbig_iter]['rotation_matrix'],
               rbig_transformed)

  return rbig_transformed





