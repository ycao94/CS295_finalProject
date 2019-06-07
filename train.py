# -*- coding: utf-8 -*-
import torch
from model import CSVAE
from config import train_config

def train2(train_loader, model, criterion, optimizer, epoch):
    '''
    Standard training step for one epoch
    Args: 
        train_loader: Dataloder class of training loader
        model: nn.module class of network graph
        criterion: nn.loss class
        optimizer: torch.optim.optimizer class
        epoch: int of epoch numbr
    '''
    train_loss = []
    s = time.time()
    for i, (input_1, input_2, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
        input_ = torch.cat((input_1, input_2), dim=1).cuda()
        optimizer.zero_grad()
        output_ = net.forward(input_)
        loss = criterion(output_.view(-1, 123), target.cuda())
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Train Epoch: ' + str(epoch) + ', time: ' + str(time.time() - s) + ', Train Loss: ' + str(np.average(train_loss)))
    torch.cuda.empty_cache()

def train(train_loader, model, criterion, optimizer, epoch):
    '''
    '''
    rec_loss = []
    
