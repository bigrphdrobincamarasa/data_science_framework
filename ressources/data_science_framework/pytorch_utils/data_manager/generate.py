import nibabel as nib
import numpy as np
from scipy.ndimage import rotate

if __name__ == '__main__':
    array = np.arange(23*37*13).reshape(23, 37, 13)/(23*37*13)

    for i in range(1, 10): 
        arr_ = i/10 * array

        # Save image
        image_ = nib.Nifti1Image(arr_, np.eye(4)) 
        nib.save(image_, 'input_{}'.format(i)) 

        # Save rotated image
        rotate_x = lambda x: rotate(x, reshape=False, angle=45, axes=(1, 2))
        rotate_y = lambda x: rotate(x, reshape=False, angle=90, axes=(2, 0))
        rotate_z = lambda x: rotate(x, reshape=False, angle=180, axes=(0, 1))
        rotate_xyz = lambda x: rotate_x(rotate_y(rotate_z(x)))

        image_ = nib.Nifti1Image(rotate_xyz(arr_), np.eye(4))
        nib.save(image_, 'rot_45_90_180_{}'.format(i))

        # Save flipped image
        flip_x_transform = lambda x: x[::-1, :, :]
        flip_y_transform = lambda x: x[:, ::-1, :]
        flip_z_transform = lambda x: x[:, :, ::-1]
        image_ = nib.Nifti1Image(flip_x_transform(arr_), np.eye(4))
        nib.save(image_, 'flip_x_{}'.format(i))

        image_ = nib.Nifti1Image(flip_y_transform(arr_), np.eye(4))
        nib.save(image_, 'flip_y_{}'.format(i)) 

        image_ = nib.Nifti1Image(flip_z_transform(arr_), np.eye(4))
        nib.save(image_, 'flip_z_{}'.format(i)) 

        # Save left or right gt
        tmp_ = np.zeros((23, 37, 13))
        tmp_[:10, :13, :6] = 1
        image_ = nib.Nifti1Image(tmp_, np.eye(4)) 
        nib.save(image_, 'right_gt_{}'.format(i)) 

        tmp_ = np.zeros((23, 37, 13))
        tmp_[-10:, -13:, -6:] = 1
        image_ = nib.Nifti1Image(tmp_, np.eye(4)) 
        nib.save(image_, 'left_gt_{}'.format(i)) 
        
