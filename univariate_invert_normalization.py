"""
This file defines functionality for inverting the univariate gaussianization
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

def univariate_invert_normalization(uni_gaussian_data, trans_params):
  """
  Inverts the marginal normalization

  See the companion, univariate_make_normal.py, for more details
  """
  uni_uniform_data = norm.cdf(uni_gaussian_data)
  uni_data = univariate_invert_uniformization(uni_uniform_data, trans_params)
  return uni_data


def univariate_invert_uniformization(uni_uniform_data, trans_params):
  """
  Inverts the marginal uniformization transform specified by trans_params

  See the companion, univariate_make_normal.py, for more details
  """
  # simple, we just interpolate based on the saved CDF
  return interp1d(trans_params['uniform_cdf'],
                  trans_params['uniform_cdf_support'])(uni_uniform_data)
