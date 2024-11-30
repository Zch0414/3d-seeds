import numpy as np
import nibabel as nib

import torch
from monai.transforms import CropForeground, SpatialPad, Resize


if __name__ == "__main__":
    file_path = './BraTS-GLI-01666-000-t1c.nii.gz'

    img = nib.load(file_path)
    img_data = img.get_fdata()
    print(f"Shape of raw data:{img_data.shape}")
    affine = img.affine
    orientation = nib.orientations.aff2axcodes(affine)

    # assert the orientation is ['L', 'P', 'S']
    orientation_transpose = []
    try:
        orientation_transpose.append(orientation.index('L'))
    except ValueError:
        orientation_transpose.append(orientation.index('R'))
    try:
        orientation_transpose.append(orientation.index('P'))
    except ValueError:
        orientation_transpose.append(orientation.index('A'))
    try:
        orientation_transpose.append(orientation.index('S'))
    except ValueError:
        orientation_transpose.append(orientation.index('I'))
    img_data = np.transpose(img_data, orientation_transpose)
    if 'R' in orientation:
        img_data = img_data[::-1, :, :]
    if 'A' in orientation:
        img_data = img_data[:, ::-1, :]
    if 'I' in orientation:
        img_data = img_data[:, :, ::-1]

    # clip 0.5% ~ 99.5% then project to 0 ~ 255
    lower = np.percentile(img_data, 0.5)
    upper = np.percentile(img_data, 99.5)
    img_data = np.clip(img_data, lower, upper)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255

    # prepocessing, assert the first dimension is channel
    img_data = img_data[None, ...]
    img_data = torch.FloatTensor(img_data)
    foreground_cropper = CropForeground(allow_smaller=True)
    img_data = foreground_cropper(img_data)
    img_data = torch.FloatTensor(img_data)
    longest_side = torch.tensor(img_data.size()[1:3]).max().item()
    spatial_size = [longest_side, longest_side, longest_side]
    padder = SpatialPad(spatial_size=spatial_size)
    img_data = padder(img_data)
    img_data = torch.FloatTensor(img_data)
    img_data = img_data.squeeze().numpy()
    img_data = img_data.astype(np.float32).transpose((2, 0, 1))
    print(f'Array shape: {img_data.shape}; Max value: {img_data.max()}; Min value: {img_data.min()}; Type: {img_data.dtype}')
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