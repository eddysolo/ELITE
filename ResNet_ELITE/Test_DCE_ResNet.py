import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import time
from scipy.io import loadmat, savemat
from scipy import io
import matplotlib.pyplot as plt
import h5py
import conv_Net_v2
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

caseNum =  ['BC23']
sliceNum = ['69']

for i in range(1): #range(24):

    fName = 'GRASP_real_lowRes_{}_slice{}.mat'.format(caseNum[i], sliceNum[i]) #input: real undersampled lowRes DCE GRASP

    f = loadmat(fName)
    img = torch.from_numpy(np.expand_dims(np.array(f['img']).transpose(2,0,1),axis=0)) # using torch.from_numpy for treating single dataset
    model = conv_Net_v2.DCEConvNet()

    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--gpu_id', type=str, default='1')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']

    device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.

    model.to(device)
    state_dict = torch.load('DCE_params_RESNET_epoch=150',map_location=device) #

    from collections import OrderedDict
    new_state_dict = OrderedDict() # read the params
    for k, v in state_dict['model_state_dict'].items():
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict) # load the params

    pred_X = model(img.to(device))
    pred_X = pred_X.cpu().detach().numpy().squeeze().transpose(1,2,0)


    io.savemat(
        'GRASP_pred_lowRes_{}_slice{}.mat'.format(caseNum[i], sliceNum[i]), {'result':pred_X.squeeze()}) #output: prediticted undersampled lowRes DCE GRASP
