import numpy as np
import seeds3d

# Load your 3D data
data = np.fromfile('../../array_3d.bin', dtype=np.float32)
data = data.reshape((48, 192, 192))

# Normalize the data if necessary
data = data / 255.0

# Create the SEEDS3D object
num_superpixels = 400
num_levels = 4
prior = 2
num_histogram_bins = 5
double_step = False
seeds = seeds3d.createSuperpixelSEEDS3D(192, 192, 48, 1, num_superpixels,
                                           num_levels, prior, num_histogram_bins, double_step)

# Iterate
seeds.iterate(data=data, num_iterations=4)

# Get the labels
labels = seeds.getLabels()