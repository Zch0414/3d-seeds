import numpy as np
import nibabel as nib


if __name__ == '__main__':
    data = np.fromfile("/Users/Zach/Zch/Research/seeds-3d/seeds-3d/seeds-3d/result.bin", dtype=np.int32)
    data = data.reshape(48, 192, 192)
    data = data.transpose(1, 2, 0)
    affine = np.array([
        [-1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, 4, 0],  
        [0, 0, 0, 1]
    ])
    img_nii = nib.Nifti1Image(data, affine)
    nib.save(img_nii, './result.nii.gz')