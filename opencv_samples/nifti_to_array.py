import numpy as np
import nibabel as nib


if __name__ == "__main__":
    image = nib.load('/Users/Zach/Zch/Research/seeds-3d/seeds-3d/samples/BraTS-GLI-00000-000-t1c.nii.gz')
    image_array = image.get_fdata() # [192, 192, 48]
    np.save('/Users/Zach/Zch/Research/seeds-3d/seeds-3d/samples/array_3d.npy', image_array)