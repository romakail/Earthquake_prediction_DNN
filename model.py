import torch
import torch.nn as nn
from torch.autograd import Variable

EMB_SIZE = 16
HID_SIZE = 32

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.CONV  = nn.Conv2d    (in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=False)
        
        self.BNORM =nn.BatchNorm2d(out_channels,
                                   eps=1e-05,
                                   momentum=0.1,
                                   affine=False)
 
        self.MAXPOOL = nn.MaxPool2d(3,
                                    stride=1,
                                    padding=1,
                                    dilation=1)
        
    def forward(self, x):
   
        x = self.CONV   (x)
        x = self.BNORM  (x)
        x = self.MAXPOOL(x)
        
        return x
    
class LSTMCell (nn.Module):
    
    def __init__ (self,
                  frequency_map,
                  embedding_size=16,
                  hidden_state_size=32,
                  n_cells_hor=200,
                  n_cells_ver=250,
                  device = torch.device('cpu')):
        super(self.__class__,self).__init__()
        
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        
        self.freq_map = Variable(torch.cat([1 - frequency_map.to(device),
                                            frequency_map.to(device)],
                                           dim=0).unsqueeze(0),
                                 requires_grad=True)
        
        self.emb_size = embedding_size
        self.hid_size = hidden_state_size
        
        self.embedding = nn.Sequential(ConvBlock(1,
                                                 self.emb_size,
                                                 3),
                                       nn.ReLU(),
                                       ConvBlock(self.emb_size,
                                                 self.emb_size,
                                                 3))
        self.hidden_to_result = nn.Sequential (ConvBlock (hidden_state_size, 
                                                          2, 
                                                          kernel_size=3),
                                               nn.Softmax (dim=1),
                                               )
        
        self.f_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        self.i_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        self.c_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Tanh())
        self.o_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        
    def forward (self, x, prev_state):
        (prev_c, prev_h) = prev_state
        x_emb = self.embedding(x)
        
        x_and_h = torch.cat([prev_h, x_emb], dim=1)
        
        f_i = self.f_t(x_and_h)
        i_i = self.i_t(x_and_h)
        c_i = self.c_t(x_and_h)
        o_i = self.o_t(x_and_h)
        
        next_c = prev_c * f_i + i_i * c_i
        next_h = torch.tanh(next_c) * o_i
        
        assert prev_h.shape == next_h.shape
        assert prev_c.shape == next_c.shape
        
        correction = self.hidden_to_result(next_h)[:, 0, :, :]
        prediction = torch.cat([self.freq_map for i in range(correction.shape[0])], dim=0)
        
        prediction[:, 0, :, :] -= correction 
        prediction[:, 1, :, :] += correction
        
        return (next_c, next_h), prediction
        
    def init_state (self, batch_size, device=torch.device("cpu")):
        return (Variable(torch.zeros(batch_size,
                                     self.hid_size,
                                     self.n_cells_hor,
                                     self.n_cells_ver,
                                     device=device)),
                Variable(torch.zeros(batch_size,
                                     self.hid_size,
                                     self.n_cells_hor,
                                     self.n_cells_ver,
                                     device=device)))
               
    