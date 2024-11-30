import numpy as np
import nibabel as nib
import seeds3d


if __name__ == "__main__":
    # Load your 3D data
    img = nib.load('../../data/input.nii.gz')
    img_data = img.get_fdata()
    img_data = np.asarray(img_data).astype(np.float32).transpose((2, 0, 1))
    img_data = np.ascontiguousarray(img_data, dtype=np.float32)
    
    # Normalize the data if necessary
    img_data = img_data / 255.0

    # Create the SEEDS3D object
    num_superpixels = 1000
    num_levels = 4
    prior = 2
    histogram_bins = 15
    double_step = False
    seeds = seeds3d.createSuperpixelSEEDS3D(
        width=img_data.shape[2], height=img_data.shape[1], depth=img_data.shape[0], channels=1, 
        num_superpixels=num_superpixels,
        num_levels=num_levels, 
        prior=prior, 
        histogram_bins=histogram_bins, 
        double_step=double_step
    )

    # Iterate
    seeds.iterate(data=img_data, num_iterations=10)
    # Get the labels
    labels = seeds.getLabels()

    # Save labels
    labels = labels.transpose(1, 2, 0)
    affine = np.array([
        [-1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, 1, 0],  
        [0, 0, 0, 1]
    ])
    labels_nii = nib.Nifti1Image(labels, affine)
    nib.save(labels_nii, './result.nii.gz')