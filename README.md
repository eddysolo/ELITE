# Enhanced Locally low-rank Imaging for Tissue contrast Enhancement (ELITE)

This repository contains MATLAB and Python scripts to replicate our reconstruction framework for dynamic MRI radial k-space data, specifically designed for Dynamic Contrast-Enhanced (DCE) MRI. The reconstruction framework incorporates acceleration techniques based on low-rank (LLR) subspace modeling and machine learning. 

### ELITE Reconstruction

The main MATLAB code `TWIX_2_GRASP_seg_main.m` perfoms a temporal basis estimation using principal component analysis (PCA) for each tissue segment. By default, this code will reconstruct an image series with a temporal resolution of 4.2 seconds (8 spokes) containing 36 time frames. To run this code, please make sure `flags.highRes` (inside `loadingGraspParams.m`) is set to False. An example of a segementation mask file can be found inside `data` folder, and a corresponding raw k-space breast dataset ('fastMRI_breast_141_2.h5') can be freely downloaded from fastMRI: https://fastmri.med.nyu.edu/

### ELITE Reconstruction Aided by Residual Neural Network (ResNet)

A 2D ResNet code in Python was designed to effectively overcome the spatial undersampling penalty (i.e., streak artifacts) while preserving high temporal fidelity. ResNet was trained using low-resolution DCE simulated data generated by a Digital Reference Object (DRO) (https://doi.org/10.1002/mrm.30152). The reduced artifact output images by ResNet can then be utlized by the above ELITE reconstruction Matlab code for basis estimation, resulting in high spatial and temporal resolution image series with a 1-second (2 spokes) temporal resolution and 144 time frames.

ResNet Python code is available inside `ResNet_ELITE` folder, which includes:

- **Training script:** `Train_valid_DCE_ResNet.py`
- **Testing script:** `Test_DCE_ResNet.py`

#### Train the ResNet Model from scratch

Unfortunately the data files (`Train_DCE_data.mat` and `Valid_DCE_data.mat`) and weightes (`DCE_params_RESNET_epoch=150`) are too large to be hosted here but can be provided upon reasonable request.

To run:
```bash
python Train_valid_DCE_ResNet.py
```
#### Test ResNet Model

To run the trained model, run:

```bash
python Test_DCE_ResNet.py
```

- The script uses a real undersampled DCE image series reconstructed with 2 spokes as an input (`GRASP_real_lowRes_BC23_slice69.mat`).
- The script will output a DCE image series with reduced streak artifacts while maintaining contrast dynamics (`GRASP_pred_lowRes_BC23_slice69.mat`).

#### Run ELITE Reconstruction with ResNet output 

ResNet output image series (`GRASP_pred_lowRes_BC23_slice69.mat`) can be found inside `data` folder and can loaded by 'TWIX_2_GRASP_seg_main.m' by setting flags.highRes parameter to True (inside `loadingGraspParams.m`). The ResNet image series output will go through a quick basis estimation, which will then be used for the reconstruction of a high spatial and temporal resolution final image series with a 1-second (2 spokes) temporal resolution and 144 time frames. 

## Citation

If you use the ELITE DCE Breast data or code in your research, please cite our preprint: https://doi.org/10.1148/ryai.240345
