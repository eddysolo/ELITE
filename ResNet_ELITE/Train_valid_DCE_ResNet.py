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
import mat73
import argparse
import os




class DCE_Train_ImgDataset(Dataset):
    def __init__(self):
        f = mat73.loadmat('Train_DCE_data.mat')
        self.X = torch.from_numpy(np.array(f['train_grasp']))
        self.y = torch.from_numpy(np.array(f['target']))

        self.len = self.X.shape[-1]
        print(f"Train_DCE_data size: {self.len}")

    def __getitem__(self, index):
        return self.X[..., index], self.y[..., index]

    def __len__(self):
        return self.len

class DCE_Valid_ImgDataset(Dataset):
    def __init__(self):
        f = mat73.loadmat('Valid_DCE_data.mat') 
        self.X = torch.from_numpy(np.array(f['train_grasp']))
        self.y = torch.from_numpy(np.array(f['target']))

        self.len = self.X.shape[-1]
        print(f"Valid_DCE_data size: {self.len}")

    def __getitem__(self, index):
        return self.X[..., index], self.y[..., index]

    def __len__(self):
        return self.len

def main():

    start = time.time()

    patch_wise_cost = 0

    # device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1

    Train_data = DCE_Train_ImgDataset()
    train_loader = DataLoader(dataset=Train_data, batch_size=batch_size, shuffle=True, num_workers=2) # using DataLoader for treating multiple datasets
    Valid_data = DCE_Valid_ImgDataset()
    valid_loader = DataLoader(dataset=Valid_data, batch_size=batch_size, shuffle=True, num_workers=2)
  
    Train_loss = []
    Valid_loss = []
    
    model = conv_Net_v2.DCEConvNet()
    
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--gpu_id', type=str, default='1')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']

    model.to(device)

    current_lr = 5e-5 #learning rate

    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr) #1e-4
    criterion = torch.nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995) # 0.995 minimizing the learning rate gamma=0.995

    max_epoch = 151
    for epoch in range(max_epoch):
        tLoss = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            inputs, truePar = data # inputs and truePar(target)
            #print(inputs.shape)
            inputs = inputs.permute(0, 3, 1, 2) # permute input dimensions
            truePar = truePar.permute(0, 3, 1,2) # permute target dimensions


            # print(inputs.shape)
            #batch_size = inputs.shape[0]
            inputs, truePar = Variable(inputs), Variable(truePar) #pytorch variable
            inputs = inputs.to(device)
            truePar = truePar.to(device)

            y_pred = model(inputs.float())
            #print(inputs.size())
            loss = criterion(y_pred.float().squeeze(), truePar.float().squeeze()) # comparing the prediction to the target

            print(epoch, i, loss.item())
            loss.backward()
            optimizer.step()
            tLoss.append(loss.item())

        Train_loss.append(sum(tLoss) / len(tLoss))

        if epoch%5 == 0:
            Last3 = min(Train_loss[-5:]) # every epoch you calc the loss
            if (Train_loss[-1] >= Last3) & (epoch > 200) & (current_lr > 1e-10):
                exp_lr_scheduler.step() # minimize the step
                print('Decreasing LR!')
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    print("Current learning rate is: {}".format(current_lr))

        vLoss = []
        for i, data in enumerate(valid_loader, 0):

            inputs, truePar = data
            #print(inputs.shape)
            inputs = inputs.permute(0, 3, 1, 2) # permute input dimensions
            truePar = truePar.permute(0, 3, 1, 2) # permute target dimensions
            inputs, truePar = Variable(inputs), Variable(truePar)
            inputs = inputs.to(device)
            truePar = truePar.to(device)
            #print(inputs.size())

            y_pred = model(inputs.float())
            loss = criterion(y_pred.float().squeeze(), truePar.float().squeeze())

            vLoss.append(loss.item())

        Valid_loss.append(sum(vLoss) / len(vLoss))
        Last3 = min(Valid_loss[-5:])

        if (Valid_loss[-1] == Last3) & (epoch > 20) & (current_lr > 1e-10):
            exp_lr_scheduler.step()
            print('Decreasing LR!')
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print("Current learning rate is: {}".format(current_lr))

        if epoch == 150:
            torch.save({'model_state_dict': model.state_dict()},
                       'DCE_params_RESNET_epoch={}'.format(epoch))
    #io.savemat(
    #    'DCE_loss_RESNET_epoch{}.mat'.format(max_epoch), {'TrainLoss': Train_loss, 'ValidLoss': Valid_loss})

if __name__ == '__main__':
    main()





