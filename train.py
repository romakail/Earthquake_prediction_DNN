from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn

def train_RNN_full (RNN_cell,
                   device,
                   dataloader_train,
                   n_cycles=1,
                   learning_rate=0.0003,
                   earthquake_weight=1.,
                   lr_decay=1.):
    
    loss_massive = []
    
    RNN_cell.to(device)
    
    weights = torch.tensor([1., earthquake_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weights)
    
    i = 0
    for cycle in range(n_cycles):
        
        optimizer = torch.optim.Adam(RNN_cell.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        
        hid_state = RNN_cell.init_state(batch_size=1, device=device)
        for data in dataloader_train:
            
            inputs = data[0].to(device)
            labels = data[1].to(device)
            
            hid_state, outputs = RNN_cell.forward(inputs, hid_state)
            
            loss = criterion(outputs, labels.squeeze(0).long())
            loss_massive.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (type(hid_state) == tuple):
                for elem in hid_state:
                    elem.detach_()
            else:
                hid_state.detach_()
            
            if (i)%100==0:
                clear_output(True)
                print ("Done :", i, "/", dataloader_train.__len__() * n_cycles)
                plt.plot(loss_massive,label='loss')
                plt.legend()
                plt.show()
            i += 1
        learning_rate /= lr_decay

def train_RNN_part(RNN_cell,
                   device,
                   dataset_train,
                   n_cycles=1,
                   queue_lenght=1,
                   learning_rate=0.0003,
                   earthquake_weight=1.,
                   lr_decay=1.):
    
    loss_massive = []
    i = 0
    
    RNN_cell.to(device)
    
    weights = torch.tensor([1., earthquake_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weights)
    
    i = 0
    for cycle in range(n_cycles):
        
        optimizer = torch.optim.Adam(RNN_cell.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        
        hid_state = RNN_cell.init_state(batch_size=1, device=device)
        start = random.randint(0, dataset_train.__len__() - queue_lenght)
        
        for t in range(start, start + queue_lenght):
            
            data = dataset_train[t]
            inputs = data[0].unsqueeze(0).to(device)
            labels = data[1].unsqueeze(0).to(device)
            
            hid_state, outputs = RNN_cell.forward(inputs, hid_state)
            
            loss = criterion(outputs, labels.squeeze(1).long())
            loss_massive.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            i += 1
            
        if (i)%queue_lenght==0:
            clear_output(True)
            print ("Done :", cycle, "/", n_cycles)
            plt.plot(loss_massive,label='loss')
            plt.legend()
            plt.show()
        
        learning_rate /= lr_decay