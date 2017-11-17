"""
This file defines a simple function for estimating data probability under rbig
"""
# WARNING: this has not been fully tested yet

def estimate_prob_with_rbig(original_data, transform_params,
                            num_samples_jacobian=0):
  """
  Computes the probability of original data under the generative RBIG model

  Parameters
  ----------
  original_data : ndarray
      Points at which the pdf is evaluated
  transform_params : dictionary
      paramters of the transformation. See invert_rbig.py for
      the structure this dictionary should have. This completely defines the
      generative model under RBIG
  num_samples_jacobian: int
      We might want to take several different estimates of the jacobian and
      average them together.

  Returns
  -------
  prob_data_input_domain : ndarray
      A 1D array giving the probability under the RBIG model for a each
      datapoint in the columns of original_data
  prob_data_gaussian_domain : ndarray
      A 1D array giving the probabilty of each datapoint in the columns of
      original_data in the gaussianized probability space.
  det_jacobian : float
      Determinant of the jacobian of the RBIG transorm
  jacobian : ndarray
      The jacobian between the input space and the gaussianized space. This
      is composed of the jacobian at each iteration with the rotation matrix
  """
  # not sure why we divide by 20...in the example script num_samples_jac is set
  # to 20 so maybe it's for a related reason...
  component_wise_std = np.std(original_data, axis=1) / 20

  num_components = original_data.shape[0]
  num_samples = original_data.shape[1]
  chunk_size = 2000
  #^ compute the jacobian for batches of samples that are this big
  full_chunks = np.floor(num_samples, chunk_size)
  leftover = np.mod(num_samples, chunk_size)

  prob_data_gaussian_domain = np.zeros([num_samples_jacobian, num_samples])
  prob_data_input_domain = np.zeros([num_samples_jacobian, num_samples])
  for jac_iter in range(num_samples_jacobian + 1):
    jacobians = np.zeros((num_samples, num_components, num_components))
    #^ a jacobian for each sample
    # TODO: figure out what's going on here
    if jac_iter < num_samples_jacobian:
      data_aux = original_data + component_wise_std[:, None]
    else:
      data_aux = original_data
    data_temp = np.zeros(data_aux.shape)

    # now compute the jacobian for each sample
    for base_iter in range(0, full_chunks*jacobian_chunk_size,
                           jacobian_chunk_size):
      (jacobians[base_iter:base_iter+jacobian_chunk_size, :, :],
       data_temp[:, base_iter:base_iter+jacobian_chunk_size]) = (
         rbig_jacobian(data_aux[:, base_iter:base_iter+jacobian_chunk_size],
                       transform_params))
    # cleanup the leftover
    if leftover > 0:
      (jacobians[full_chunks*jacobian_chunk_size:, :, :],
       data_temp[:, full_chunks*jacobian_chunk_size:]) = (
         rbig_jacobian(data_aux[:, full_chunks*jacobian_chunk_size:],
                       transform_params))

    det_jacobians = np.linalg.det(jacobians)
    #^ computes determinant for each sample's jacobian
    prob_data_gaussian_domain[jac_iter, :] = np.prod(
        (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(data_temp, 2))
        axis=0)
    #^ computes the total probability under the gaussian model for each datapoint
    prob_data_input_domain[jac_iter, :] = (
        prob_data_gaussian_domain[jac_iter] * np.abs(det_jacobians))
    prob_data_input_domain[jac_iter,
                           np.isnan(prob_data_input_domain[jac_iter])] = 0.0

  # now average over the different jacobian calculations we made
  prob_data_input_domain = np.mean(prob_data_input_domain, axis=0)
  prob_data_gaussian_domain = np.mean(prob_data_gaussian_domain, axis=0)
  det_jacobian = np.mean(det_jacobians, 0)

  return (prob_data_input_domain, prob_data_gaussian_domain,
          det_jacobian, jacobians)
