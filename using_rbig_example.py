"""
Example usage for RBIG with simple 2d data
"""
import numpy as np
from matplotlib import pyplot as plt
from rbig import rbig
from invert_rbig import invert_rbig

num_samples = 10000
x = np.abs(2*np.random.randn(1, num_samples))
y = np.sin(x) + 0.25*np.random.randn(1, num_samples)
data = np.vstack((x, y))

plt.figure(figsize=(10, 10))
plt.scatter(data[0], data[1], s=1)
plt.xlabel('X')
plt.xlabel('Y')
plt.title('Original Data')

# Obtain the RBIG transform for the data using *PCA* rotation and 50 iterations
rbig_transformed_data, trans_params = rbig(data, 50, 'PCA')
plt.figure()
plt.scatter(rbig_transformed_data[0], rbig_transformed_data[1], s=1)
plt.xlabel('X')
plt.xlabel('Y')
plt.title('Data after RBIG transform')

# Compare to random gaussian data
data_synth_gaussian = np.random.randn(data.shape[0], data.shape[1])
plt.figure()
plt.scatter(data_synth_gaussian[0], data_synth_gaussian[1], s=1)
plt.xlabel('X')
plt.xlabel('Y')
plt.title('Synthetically generated factorial gaussian data')

# Invert the random data to synthesize new data under the learned model
data_synth_inputspace = invert_rbig(data_synth_gaussian, trans_params)
plt.figure()
plt.scatter(data_synth_inputspace[0], data_synth_inputspace[1], s=1)
plt.xlabel('X')
plt.xlabel('Y')
plt.title('Synthetically generated data from the input distribution')

plt.show()
