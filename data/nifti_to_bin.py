import numpy as np
import nibabel as nib


if __name__ == "__main__":
    data_path = './BraTS-GLI-00000-000-t1c.nii.gz'
    # data_path = '../UM/BRAIN_UM_00000972/save_pro/CORONAL-HEAD_WITHOUT-Protocol/nifti_pro.nii.gz'

    image = nib.load(data_path)
    image_array = image.get_fdata() 
    affine = image.affine

    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    lowest_resolution_side = np.argmax(spacing, axis=-1)
    if lowest_resolution_side == 0:
        transpose_order = (0, 1, 2)
    elif lowest_resolution_side == 1:
        transpose_order = (1, 0, 2)
    elif lowest_resolution_side == 2:
        transpose_order = (2, 0, 1)

    image_array = np.transpose(np.array(image_array).astype(np.float32), transpose_order)
    print(f'Array shape: {image_array.shape}; Max value: {image_array.max()}; Min value: {image_array.min()}; Type: {image_array.dtype}')
    image_array.tofile('./input.bin')