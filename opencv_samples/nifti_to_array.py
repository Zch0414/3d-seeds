import numpy as np
import nibabel as nib


if __name__ == "__main__":
    image = nib.load('/Users/Zach/Zch/Research/seeds-3d/seeds-3d/opencv_samples/BraTS-GLI-00000-000-t1c.nii.gz')
    image_array = image.get_fdata() 
    image_array = np.transpose(np.array(image_array).astype(np.float32), (2, 0, 1))
    print(f'Array shape: {image_array.shape}; Max value: {image_array.max()}; Min value: {image_array.min()}; Type: {image_array.dtype}')
    image_array.tofile('/Users/Zach/Zch/Research/seeds-3d/seeds-3d/seeds-3d/array_3d.bin')