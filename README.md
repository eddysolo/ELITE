# Enhanced Locally low-rank Imaging for Tissue contrast Enhancement (ELITE)
This repository contains MATLAB scripts to replicate our reconstruction framework for dynamic MRI radial k-space data, specifically designed for Dynamic Contrast-Enhanced (DCE) MRI. The reconstruction framework incorporates acceleration techniques based on low-rank (LLR) subspace modeling and machine learning. An example of raw k-space breast DCE dataset is also included.

## Instructions
The main code 'TWIX_2_GRASP_seg_main.m' can be executed with the two options below by toggling flags.highRes parameter inside 'loadingGraspParams.m'

- **ELITE reconstruction**: Temporal basis estimation is performed using principal component analysis (PCA) for each tissue segment, with a temporal resolution of 4.2 seconds (8 spokes), resulting in high spatial resolution image series containing 36 time frames.
  
- **ELITE reconstruction aided by Residual Neural Network (ResNet)**: A 2D ResNet was designed to effectively overcome the spatial undersampling penalty (i.e., streak artifacts) while preserving high temporal fidelity. ResNet was trained using low-resolution simulated data generated by a Digital Reference Object (DRO) platform. The low-resolution output image series by ResNet were integrated into our reconstruction framework for basis estimation, resulting in high spatial and temporal resolution image series with a 1-second (2 spokes) temporal resolution and 144 time frames. 

## Manuscript
This framework is described in our latest manuscript titled 'Dynamic breast MRI with Flexible Temporal Resolution Aided by Deep Learning'

## Authors
<sup>1</sup>Eddy Solomon, <sup>1</sup>Jonghyun Bae, <sup>2</sup>Linda Moy, <sup>2</sup>Laura Heacock, <sup>2</sup>Li Feng, <sup>1</sup>Sungheon Gene Kim<br />
<sup>1</sup>Department of Radiology, Weill Cornell Medical College, New York, NY, United States<br /> 
<sup>2</sup>Center for Advanced Imaging Innovation and Research (CAI2R), Department of Radiology, New York University, New York, NY, United States









