import nibabel as nib
import numpy as np
from nilearn.image import clean_img

def load_fmri(file_path):
    """
    Load fMRI data (.nii or .nii.gz).
    """
    try:
        img = nib.load(file_path)
        return img
    except Exception as e:
        print(f"Error loading fMRI: {e}")
        return None

def preprocess_fmri(img):
    """
    Basic fMRI preprocessing: detrending, standardizing.
    """
    if img is None:
        return None
    cleaned_img = clean_img(img, detrend=True, standardize=True)
    return cleaned_img
