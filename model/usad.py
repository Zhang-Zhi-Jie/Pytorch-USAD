# -*- encoding: utf-8 -*-
'''
Filename         :usad.py
Description      :
Time             :2020/11/29 13:58:31
Author           :ZhiJie Zhang
Version          :1.0
'''


import torch
import torch.nn as nn 


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size//2, input_size // 4),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(input_size//4, hidden_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.linear3(self.linear2(self.linear1(x)))

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_size, input_size // 4),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size//4, input_size // 2),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(input_size//2, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.linear3(self.linear2(self.linear1(x)))

class USAD(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(USAD, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder1 = Decoder(input_size, hidden_size)
        self.decoder2 = Decoder(input_size, hidden_size)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat1 = self.decoder1(z)
        x_hat2 = self.decoder2(z)
        x_hat12 = self.decoder2(self.encoder(x_hat1))
        return z, x_hat1, x_hat2, x_hat12