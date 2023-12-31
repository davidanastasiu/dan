#!/usr/bin/env python
# coding: utf-8

import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils.utils2 import *
from sklearn.metrics import mean_absolute_percentage_error
import logging
logging.basicConfig(filename = "Decoder_LSTM.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderLSTM(nn.Module):
    def __init__(self, opt):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.r_shift = opt.r_shift
        self.h_style_hr = 1
        
        self.lstm00 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True)
        self.lstm01 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True)
        self.lstm02 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True)
        self.lstm03 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True)

        self.L_out00 = nn.Linear(self.hidden_dim*2, 1)
        self.L_out03 = nn.Linear(self.hidden_dim*2, 1)

        self.cnn00 = nn.Conv1d(self.hidden_dim*2, self.hidden_dim, 7, stride=1, padding=3)
        self.cnn01 = nn.Conv1d(self.hidden_dim, 1, 3, stride=1, padding=1)

    def forward(self, x1, x2, x3, encoder_h, encoder_c):  # x1: time sin & cos;   x2: hinter vectors

        h0 = encoder_h[0] 
        c0 = encoder_c[0]
        h2 = encoder_h[1]
        c2 = encoder_c[1]
        h3 = encoder_h[2]
        c3 = encoder_c[2]
        sig = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        x = x1     
          
        # far points   
        o0, (hn, cn) = self.lstm00(x1, (h0,c0))
        out00 = self.L_out00(o0)
        out0 = torch.squeeze(out00, dim=2)            
        
        # near points        
        o3, (hn, cn) = self.lstm03(x1, (h3,c3))
        out03 = self.L_out03(o3)
        out3 = torch.squeeze(out03, dim=2)        

        # If there is a weak hinter, the model try to generate a trainable hinter from last r_shift length of rain data
        if (self.r_shift > 0):
            """
            As a weak hinter, use trained hidden state to help predict.   
            """ 
            o1, (hn, cn) = self.lstm01(x1, (h2,c2))
        else:
            h0 = torch.zeros(self.layer_dim*2, x2.size(0), self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim*2, x2.size(0), self.hidden_dim).to(device) 
            o1, (hn, cn) = self.lstm02(x2, (h0,c0)) 

        """
        Indicator refine layer
        """             
        out = o1.permute(0, 2, 1)  
        cnn_out = F.relu(self.cnn00(out))
        cnn_out = self.cnn01(cnn_out)
        Ind = cnn_out.permute(0, 2, 1)
     
        """
        Generate 0-1 label matrix a for far points   
        Use sig as a gate control
        """        
        aa = torch.round(torch.squeeze(sig(Ind*4), dim=2))  
        a = torch.squeeze(sig(Ind*4), dim=2)
        a = torch.mul(a, aa)
        b = torch.ones(a.size()).to(device)   

        """
        Generate 0-1 label matrix b for near points   
        """                   
        b = b - aa              
        out4 = a.mul(out0.detach()) + b.mul(out3.detach())    


        
        return out0, Ind, out3, out4

