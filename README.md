# Enhanced Locally low-rank Imaging for Tissue contrast Enhancement (ELITE)

This repository contains MATLAB and Python scripts to replicate our reconstruction framework for dynamic MRI radial k-space data, specifically designed for Dynamic Contrast-Enhanced (DCE) MRI. The reconstruction framework incorporates acceleration techniques based on low-rank (LLR) subspace modeling and machine learning. 

## Instructions

The main MATLAB code `TWIX_2_GRASP_seg_main.m` can be executed with the two options below by toggling the `flags.highRes` parameter inside `loadingGraspParams.m`.

### ELITE Reconstruction

Temporal basis estimation is performed using principal component analysis (PCA) for each tissue segment, with a default temporal resolution of 4.2 seconds (8 spokes), resulting in high spatial resolution image series containing 36 time frames. The matlab code is set by default to run this reconstruction. An example segementation mask file can be found under 'data' forder. To run the code, the corresponding raw k-space breast dataset ('fastMRI_breast_141_2.h5') can be freely downloaded from fastMRI: https://fastmri.med.nyu.edu/

### ELITE Reconstruction Aided by Residual Neural Network (ResNet)

A 2D ResNet was designed to effectively overcome the spatial undersampling penalty (i.e., streak artifacts) while preserving high temporal fidelity. ResNet was trained using low-resolution DCE simulated data generated by a Digital Reference Object (DRO) platform: https://doi.org/10.1002/mrm.30152. The low-resolution output images by ResNet is then integrated into the above ELITE reconstruction code for basis estimation, resulting in high spatial and temporal resolution image series with a 1-second (2 spokes) temporal resolution and 144 time frames.

## Introduction to ResNet  

All relevant ResNet Python code is available inside `ResNet_ELITE` folder, which includes:

- **Training script:** `Train_valid_DCE_ResNet.py`
- **Testing script:** `Test_DCE_ResNet.py`

## Train the ResNet Model from scratch

Unfortunately the data files (`Train_DCE_data.mat` and `Valid_DCE_data.mat`) and weightes ('`DCE_params_RESNET_epoch=150`') are too large to be hosted here but can be provided upon reasonable request.

To run:
```bash
python Train_valid_DCE_ResNet.py
```
## Test ResNet Model

To run the trained model, run:

```bash
python Test_DCE_ResNet.py
```

- The script uses a real undersampled DCE image series reconstructed with 2 spokes as an input (`GRASP_real_lowRes_BC23_slice69.mat`).
- The script will output a DCE image series with reduced streak artifacts while maintaining contrast dynamics ('GRASP_pred_lowRes_BC23_slice69.mat`).

## Run ELITE framework with ResNet output 

ResNet output image series ('GRASP_pred_lowRes_BC23_slice69.mat') can be found under 'data' folder and can loaded by 'TWIX_2_GRASP_seg_main.m' by setting flags.highRes parameter to True (inside `loadingGraspParams.m`). The ResNet image series output will go through a quick basis estimation, which will then be used for the reconstruction of a high spatial and temporal resolution final image series with a 1-second (2 spokes) temporal resolution and 144 time frames.

## Citation

If you use the ELITE DCE Breast data or code in your research, please cite our preprint:https://doi.org/10.1148/ryai.240345
