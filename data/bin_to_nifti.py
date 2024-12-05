import numpy as np
import nibabel as nib


if __name__ == '__main__':
    data = np.fromfile("./result.bin", dtype=np.int32)
    data = data.reshape(160, 160, 160)
    data = data.transpose(2, 1, 0)
    affine = np.array([
        [-1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, 1, 0],  
        [0, 0, 0, 1]
    ])
    img_nii = nib.Nifti1Image(data, affine)
    nib.save(img_nii, './result.nii.gz')