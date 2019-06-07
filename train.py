# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
import argparse
import os
import sys
import pickle
import time
from tqdm import tqdm
from torch import optim
from model import CSVAE
from torch.autograd import Variable

from config import train_config

# def train2(train_loader, model, criterion, optimizer, epoch):
#     '''
#     Standard training step for one epoch
#     Args: 
#         train_loader: Dataloder class of training loader
#         model: nn.module class of network graph
#         criterion: nn.loss class
#         optimizer: torch.optim.optimizer class
#         epoch: int of epoch numbr
#     '''
#     train_loss = []
#     s = time.time()
#     for i, (input_1, input_2, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
#         input_ = torch.cat((input_1, input_2), dim=1).cuda()
#         optimizer.zero_grad()
#         output_ = net.forward(input_)
#         loss = criterion(output_.view(-1, 123), target.cuda())
#         train_loss.append(loss.item())
#         loss.backward()
#         optimizer.step()
#     print('Train Epoch: ' + str(epoch) + ', time: ' + str(time.time() - s) + ', Train Loss: ' + str(np.average(train_loss)))
#     torch.cuda.empty_cache()



def train(train_loader, model, criterion, optimizer1, optimizer2, epoch):
    '''
    '''
    s = time.time()
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
        if model.useCUDA:
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x)
            y = Variable(y)
    
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        z, w, mu_z, mu_w_0, mu_w_1, sigma_z, sigma_w_0, sigma_w_1, out, pred_y = model.forward(x, y)
        loss_m1, loss_dict = model.loss(out, x, y, mu_z, mu_w_0, mu_w_1, sigma_z, sigma_w_0, sigma_w_1)
        loss_m2 = criterion(y, pred_y)
        loss = loss_m1 - loss_m2
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    #TODO: print training loss
    print()
    torch.cuda.empty_cache()

csvae = CSVAE()
if csvae.useCUDA:
    csvae.cuda()

criterion = torch.nn.BCELoss()
optimizer1 = optim.RMSprop(csvae.parameters(), lr=train_config['lr'])
optimizer2 = optim.RMSprop([csvae.enc1.parameters(), csvae.encMuZ.parameters(), csvae.encSigmaZ.parameters(),
                            csvae.encY.parameters(), csvae.predY.parameters()], lr=train_config['lr'])

for i in tqdm(range(start_epoch, epochs_ + 1), desc='Total'):
    print('epoch %d' % (i))
    train(train_loader, csvae, criterion, optimizer1, optimizer2, i)