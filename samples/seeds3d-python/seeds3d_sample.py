import numpy as np
import nibabel as nib
import seeds3d


if __name__ == "__main__":
    # Load your 3D data
    image = nib.load('../../data/BraTS-GLI-00000-000-t1c.nii.gz')
    image_array = image.get_fdata() 
    affine = image.affine

    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    lowest_resolution_side = np.argmax(spacing, axis=-1)
    if lowest_resolution_side == 0:
        transpose_order = (0, 1, 2); de_transpose_order = (0, 1, 2)
    elif lowest_resolution_side == 1:
        transpose_order = (1, 0, 2); de_transpose_order = (1, 0, 2)
    elif lowest_resolution_side == 2:
        transpose_order = (2, 0, 1); de_transpose_order = (1, 2, 0)
    image_array = np.transpose(np.array(image_array).astype(np.float32), transpose_order)
    
    data = np.ascontiguousarray(image_array, dtype=np.float32)
    # Normalize the data if necessary
    data = data / 255.0

    # Create the SEEDS3D object
    num_superpixels = 432
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
    # print(labels.flags['C_CONTIGUOUS']) # True

    labels = np.transpose(labels, de_transpose_order)
    affine = np.array([
        [-1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, 4, 0],  
        [0, 0, 0, 1]
    ])
    img_nii = nib.Nifti1Image(labels, affine)
    nib.save(img_nii, '../../data/result-python.nii.gz')