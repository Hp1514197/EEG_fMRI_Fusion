<p align="center">
 <img height=200px src="./eeg-fmri.jpg" alt="EEG-fMRI Fusion">
</p>

<h1 align="center">EEG-fMRI Fusion for Cognitive Dysfunction Analysis</h1>

<div align="center">

[![Python version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<h4>This project integrates EEG and fMRI data to analyze cognitive dysfunction using advanced machine learning and deep learning techniques. The fusion framework leverages the high temporal resolution of EEG with the high spatial resolution of fMRI to provide deeper insights into brain activity, aiding early diagnosis and treatment research in neurological disorders.</h4>

</div>

-----------------------------------------
### Inspiration

* Neurological disorders such as Alzheimer’s, Schizophrenia, and ADHD are often linked to abnormalities in brain connectivity and cognition.  
* EEG provides fine-grained temporal signals of brain activity, while fMRI captures spatial information. Individually, both modalities have limitations, but together they provide a more complete picture of brain function.  
* The motivation behind this project is to design an **EEG-fMRI fusion framework** that helps clinicians and researchers understand neural patterns, detect early signs of dysfunction, and support precision medicine.  

------------------------------------------
### Implementation Details

This project can be broken down into 3 main modules:

1. `EEG Preprocessing Module`  
   - Cleans raw EEG signals (artifact removal, filtering, segmentation).  
   - Extracts features such as spectral power, ERP components, and connectivity measures.  

2. `fMRI Preprocessing and Feature Extraction Module`  
   - Performs motion correction, normalization, and ROI extraction.  
   - Extracts spatial connectivity patterns and hemodynamic responses.  

3. `Fusion & Analysis Module`  
   - Applies multimodal fusion using deep learning (e.g., CNN + RNN, transformer-based models).  
   - Performs classification/regression for cognitive dysfunction detection.  
   - Visualizes multimodal brain activation patterns.  

Read more about methodology, preprocessing pipelines, and fusion architecture [here](./EEG_fMRI_Fusion_Implementation.pdf).

------------------------------------------
### Demo

* `EEG Signal Preprocessing`

<p align="center">
 <img height=400px src="./eeg-signal.png" alt="EEG Signal">
</p>

<br> 

* `EEG-fMRI Fusion Analysis`

<p align="center">
    <img src="./fusion-demo.gif" alt="EEG-fMRI Fusion">
</p>

------------------------------------------
### Prerequisites

1. [Python 3.10](https://www.python.org/downloads/release/python-3100/)  
2. [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [MNE](https://mne.tools/stable/index.html) (for EEG processing)  
3. [NiBabel](https://nipy.org/nibabel/), [Nilearn](https://nilearn.github.io/stable/index.html) (for fMRI data handling)  
4. [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/) (for fusion models)  

------------------------------------------
### Installation

* Step I: Clone the Repository
```sh
      $ git clone https://github.com/Hp1514197/EEG-fMRI-Fusion-Cognitive-Dysfunction
