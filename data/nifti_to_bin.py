import numpy as np
import nibabel as nib

import torch
from monai.transforms import CropForeground, SpatialPad


if __name__ == "__main__":
    file_path = './BraTS-GLI-01666-000-t1c.nii.gz'

    img = nib.load(file_path)
    img_data = img.get_fdata()
    affine = img.affine
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    orientation = nib.orientations.aff2axcodes(affine)
    print(f"Raw data: shape({img_data.shape}); spacing({spacing}); orientation({orientation})")

    # clip 0.5% ~ 99.5% then project to 0 ~ 255
    lower = np.percentile(img_data, 0.5)
    upper = np.percentile(img_data, 99.5)
    img_data = np.clip(img_data, lower, upper)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255

    # assert the first dimension is channel
    img_data = img_data[None, ...]
    
    # foreground crop
    img_data = torch.FloatTensor(img_data)
    foreground_cropper = CropForeground(allow_smaller=True)
    img_data = foreground_cropper(img_data)

    # pad
    spatial_size = [160, 160, 160]
    padder = SpatialPad(spatial_size=spatial_size)
    img_data = padder(img_data)
    
    img_data = img_data.astype(np.float32).transpose((2, 0, 1)) # visualize the axial space in seeds3d-cpp.
    img_data.tofile('./input.bin')
    affine = np.array([
                [-1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, 1, 0],  
                [0, 0, 0, 1]
            ])
    img_nii = img_data.transpose(1, 2, 0)
    img_nii = nib.Nifti1Image(img_nii, affine)
    nib.save(img_nii, './input.nii.gz')