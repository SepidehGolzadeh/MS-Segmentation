"""
Data loading and preprocessing module for 2D MRI lesion segmentation.

This module:
- Loads 3D NIfTI MRI volumes and corresponding masks
- Extracts axial slices
- Resizes slices to a fixed spatial resolution
- Applies intensity normalization
- Prepares data for CNN training and evaluation
"""

import os
from typing import Tuple

import numpy as np
import nibabel as nib
import cv2  

class TrainingLoader:
    def __init__(self):
        self.images_path = []
        self.masks_path = []

        script_path = os.path.dirname(__file__)
        #training_path = "/kaggle/working/MSSEG1-Training/Training"
        training_path = os.path.join(script_path, "Training")

        for center in os.listdir(training_path):
            center_path = os.path.join(training_path, center)
            if not os.path.isdir(center_path):
                continue

            for patient in os.listdir(center_path):
                patient_path = os.path.join(center_path, patient)
                if not os.path.isdir(patient_path):
                    continue

                img = os.path.join(patient_path, "Preprocessed_Data", "FLAIR_preprocessed.nii.gz")
                mask = os.path.join(patient_path, "Masks", "Consensus.nii.gz")

                self.images_path.append(img)
                self.masks_path.append(mask)

    def load_training_data(self, normalize=True, img_size: Tuple[int, int] = (224, 224)):
        """
        Load 3D MRI volumes and corresponding segmentation masks, convert them into 
        2D axial slices, resize each slice to a fixed spatial resolution, and apply 
        intensity normalization.

        Processing steps:
            1. Load NIfTI image and mask volumes.
            2. Extract slices along the axial (z) dimension.
            3. Resize each slice to the specified img_size.
            4. Apply per-slice Gaussian normalization:
                   (x - mean) / (std + 1e-6)
            5. Clip normalized intensities to [-5, 5] and scale to [-1, 1].
            6. Add channel dimension for CNN compatibility.

        Args:
            normalize (bool): Whether to apply intensity normalization (currently always applied).
            img_size (Tuple[int, int]): Target (height, width) for resizing slices.

        Returns:
            images (np.ndarray): Array of shape (num_slices, H, W, 1) 
                                 containing normalized MRI slices (float32).
            masks (np.ndarray): Array of shape (num_slices, H, W, 1) 
                                containing corresponding segmentation masks.
        """
        images = []
        masks = []

        for img_path, mask_path in zip(self.images_path, self.masks_path):
            img_vol = nib.load(img_path).get_fdata().astype(np.float32)
            
            mask_vol = nib.load(mask_path).get_fdata()

            for i in range(img_vol.shape[2]):
                img_slice = img_vol[:, :, i]
                mask_slice = mask_vol[:, :, i]
                
                #if mask_slice.sum() == 0:
                #    continue

                # Resize
                img_slice = cv2.resize(img_slice, img_size)
                mask_slice = cv2.resize(mask_slice, img_size, interpolation=cv2.INTER_NEAREST)
                
                 # Gaussian normalization
                img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-6)

                # Optional: clip extreme values to [-5,5] and scale to [-1,1]
                img_slice = np.clip(img_slice, -5, 5) / 5


                images.append(img_slice.astype(np.float32))
                masks.append(mask_slice)
                
        # Convert to numpy arrays and add channel dimension
        images = np.array(images)[..., np.newaxis]
        masks  = np.array(masks)[..., np.newaxis]

        return images, masks


class TestingLoader:
    def __init__(self):
        self.images_path = []
        self.masks_path = []

        script_path = os.path.dirname(__file__)
        #testing_path = "/kaggle/input/datasets/atskyy1/msdatasets/MSSEG1-Testing/Testing"
        testing_path = os.path.join(script_path, "Testing")

        for center in os.listdir(testing_path):
            center_path = os.path.join(testing_path, center)
            if not os.path.isdir(center_path):
                continue

            for patient in os.listdir(center_path):
                patient_path = os.path.join(center_path, patient)
                if not os.path.isdir(patient_path):
                    continue
                
                img = os.path.join(patient_path, "Preprocessed_Data", "FLAIR_preprocessed.nii.gz")
                mask = os.path.join(patient_path, "Masks", "Consensus.nii.gz")
                
                self.images_path.append(img)
                self.masks_path.append(mask)
                

    def load_testing_data(self, normalize=True, img_size: Tuple[int, int] = (224, 224)):
        """
        Load 3D MRI volumes and corresponding segmentation masks, convert them into 
        2D axial slices, resize each slice to a fixed spatial resolution, and apply 
        intensity normalization.

        Processing steps:
            1. Load NIfTI image and mask volumes.
            2. Extract slices along the axial (z) dimension.
            3. Resize each slice to the specified img_size.
            4. Apply per-slice Gaussian normalization:
                   (x - mean) / (std + 1e-6)
            5. Clip normalized intensities to [-5, 5] and scale to [-1, 1].
            6. Add channel dimension for CNN compatibility.

        Args:
            normalize (bool): Whether to apply intensity normalization (currently always applied).
            img_size (Tuple[int, int]): Target (height, width) for resizing slices.

        Returns:
            images (np.ndarray): Array of shape (num_slices, H, W, 1) 
                                 containing normalized MRI slices (float32).
            masks (np.ndarray): Array of shape (num_slices, H, W, 1) 
                                containing corresponding segmentation masks.
        """
        images = []
        masks = []

        for img_path, mask_path in zip(self.images_path, self.masks_path):
            img_vol = nib.load(img_path).get_fdata().astype(np.float32)
            mask_vol = nib.load(mask_path).get_fdata()

            for i in range(img_vol.shape[2]):
                
                img_slice = img_vol[:, :, i]
                mask_slice = mask_vol[:, :, i]
                
                # Resize
                img_slice = cv2.resize(img_slice, img_size)
                mask_slice = cv2.resize(mask_slice, img_size, interpolation=cv2.INTER_NEAREST)
                
                 # Gaussian normalization
                img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-6)

                # Optional: clip extreme values to [-5,5] and scale to [-1,1]
                img_slice = np.clip(img_slice, -5, 5) / 5

                images.append(img_slice.astype(np.float32))
                masks.append(mask_slice)
            
        # Convert to numpy arrays and add channel dimension
        images = np.array(images)[..., np.newaxis]
        masks = np.array(masks)[..., np.newaxis]

        return images, masks