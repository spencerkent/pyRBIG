"""
This file defines functionality for computing the jacobian of the rbig transform
"""
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

# TODO: do a performance analysis of this function, I don't think its
# particularly efficient

def rbig_jacobian(original_data, tranform_params):

  num_components = original_data.shape[0]
  num_samples = original_data.shape[1]  # in all likelihood a subset of the data
  jacobian = np.zeros((num_samples, num_components, num_components))

  data_rbig = np.copy(original_data)
  xx = np.zeros([num_components, num_samples])
  #^ some kind of mask
  xx[0, :] = np.ones(num_samples)
  for rbig_iter in range(len(transform_params['iterations'])):
    temp_gaussian = np.zeros((num_components, num_samples))
    gaussian_pdf = np.zeros((num_components, num_samples, rbig_iter))
    #^ not quite sure yet what this is keeping track of...
    for component_idx in range(num_components):
      interp_uniform = interp1d(transform_params['iterations'][rbig_iter][
                                    component_idx]['uniform_cdf_support'],
                                transform_params['iterations'][rbig_iter][
                                    component_idx]['uniform_cdf'])
      data_uniform = learned_interp(data_rbig[component_idx])
      temp_gaussian[component_idx, :] = norm.ppf(data_uniform)
      interp_gauss_pdf = interp1d(transform_params['iterations'][rbig_iter][
                                      component_idx]['empirical_pdf_support'],
                                  transform_params['iterations'][rbig_iter][
                                      component_idx]['empirical_pdf'])
      gaussian_pdf[component_idx, :, rbig_iter] = (
          interp_gauss_pdf(data_rbig[component_idx]) *
          (1 / norm.pdf(temp_gaussian[component_idx])))

    xx = np.dot(transform_params['iterations'][rbig_iter]['rotation_matrix'],
                gaussian_pdf[:, :, rbig_iter] * xx)

    data_rbig = np.dot(
        transform_params['iterations'][rbig_iter]['rotation_matrix'],
        temp_gaussian)

  jacobian[:, :, 0] = xx.T

  if num_components > 1:
    for x_parc in range(1, num_components):
      xx = np.zeros([num_components, num_samples])
      xx[x_parc, :] = np.ones(num_samples)
      for rbig_iter in range(len(tranform_params['iterations'])):
        xx = np.dot(
            transform_params['iterations'][rbig_iter]['rotation_matrix'],
            gaussian_pdf[:, :, rbig_iter] * xx)
      jacobian[:, :, x_parc] = xx.T

  return jacobian, data_rbig





