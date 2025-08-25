# EEG-fMRI Fusion

This project aims to build a pipeline for **multimodal brain data fusion** using EEG (electroencephalography) and fMRI (functional magnetic resonance imaging).  
It includes preprocessing modules, fusion models, and visualization tools.

---

## Project Structure

EEG_fMRI_Fusion/
│── data/ # Raw datasets (EEG .edf/.set, fMRI .nii/.nii.gz)
│── preprocessing/
│ ├── eeg_preprocess.py
│ ├── fmri_preprocess.py
│── fusion/
│ ├── fusion_model.py # Models combining EEG + fMRI
│── utils/
│ ├── visualization.py
│ ├── helpers.py
│── notebooks/
│ ├── 01_eeg_preprocessing.ipynb
│ ├── 02_fmri_preprocessing.ipynb
│ ├── 03_fusion_training.ipynb
│── main.py # Main pipeline script
│── requirements.txt
│── README.md

---

## Installation

1. Clone this repository:
   git clone https://github.com/HP1514197/EEG_fMRI_Fusion.git
   cd EEG_fMRI_Fusion

2. Install dependencies:
   pip install -r requirements.txt

---

## Usage

1. Preprocess EEG data:
   python preprocessing/eeg_preprocess.py

2. Preprocess fMRI data:
   python preprocessing/fmri_preprocess.py

3. Train fusion model:
   python main.py

---   

## Features

1. EEG preprocessing (filtering, epoching, artifact removal).
2. fMRI preprocessing (smoothing, normalization, masking).
3. EEG + fMRI fusion model.
4. Visualization utilities.

---

## Requirements
See requirements.txt
