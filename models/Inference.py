# Inference......................
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
from models.DANet import *
from models.EncoderLSTM import *
from models.DecoderLSTM import *
from models.ResidueLSTM import *
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import logging
logging.basicConfig(filename = "Inference.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DAN_I:

    def __init__(self, opt):

        self.logger = logging.getLogger()
        self.logger.info("I am logging...")
        self.opt = opt
        self.sensor_id = opt.stream_sensor
        
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_watersheds = opt.watershed
        self.is_prob_feature = 1
        self.TrainEnd = opt.model
        self.opt_hinter_dim = opt.watershed
        self.r_shift = opt.r_shift
        ENCODER_BLOCK = 'encoder'
        DECODER_BLOCK = 'decoder'
        RESIDUE_BLOCK = 'residue'
        stacks = opt.stack_types.split(',')
        stack_list = []
        for i in range(len(stacks)):
            stack_list.append(eval(stacks[i]))        
        self.stack_types = tuple(stack_list) 
                              
        if opt.event_focus_level > 0:
            self.is_over_sampling = 1
        else:
            self.is_over_sampling = 0

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer

        self.net = DANet(self.opt, stack_types=self.stack_types)
        self.optimizer = torch.optim.Adam(self.net.parameters(),self.opt.learning_rate)  

        self.criterion = nn.MSELoss(reduction='sum')
        self.criterion1 = nn.HuberLoss(reduction='sum')
        self.criterion_KL = nn.KLDivLoss(reduction="sum")

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        
        norm = np.loadtxt(self.expr_dir + '/' + str(self.sensor_id) + '_Norm.txt',dtype=float,delimiter=None)
        print("norm is: ", norm)

        self.mean = norm[0]
        self.std = norm[1]
        self.R_mean = norm[2]
        self.R_std = norm[3]
        

    def std_denorm_dataset(self, predict_y0):
        pre_y = []
        a2 = log_std_denorm_dataset(self.mean, self.std, predict_y0, pre_y)

        return a2

    def inference_test(self, x_test, y_input1, y_input2):
                                
        y_predict = []
        d_out = torch.tensor([]).to(device)
        self.net.eval()

        with torch.no_grad():
            
            x_test = torch.from_numpy(np.array(x_test, np.float32)).to(device)
            y_input1 = torch.from_numpy(np.array(y_input1, np.float32)).to(device)
            y_input2 = torch.from_numpy(np.array(y_input2, np.float32)).to(device)

            h0 = torch.zeros(self.layer_dim*2, x_test.size(0),self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim*2, x_test.size(0),self.hidden_dim).to(device)
            out0, out1, out2, Ind, out3 = self.net(y_input1, y_input2, x_test, h0, c0)
            y_predict = [out3[0][i].item() for i in range(len(out3[0]))]
            y_predict = np.array(y_predict).reshape(1,-1) 
            
        return y_predict


    def model_load(self):
        
        if(self.is_over_sampling == 1):
            OS = "_OS" + str(self.opt.event_focus_level)
        else:
            OS = "_OS-null"

        if(self.is_watersheds == 1) :
            if(self.is_prob_feature == 0) :
                watersheds = "shed"
            else:
                watersheds = "Shed-ProbFeature"
        elif(self.is_prob_feature == 0) :
            watersheds = "solo"
        else:
            watersheds = "ProbFeature"
        
        basic_path = self.test_dir + '/' + str(self.sensor_id) + OS
        basic_model_path = self.expr_dir + '/' + str(self.sensor_id) + OS 
        net_name = basic_model_path + '_' + watersheds + str(self.TrainEnd) + "_net.pt"

        model = DANet(self.opt, stack_types=self.stack_types)
        model.load_state_dict(torch.load(net_name), strict=False)
        
        self.net = model        
        self.gm3 = torch.load(self.expr_dir + '/' + str(self.sensor_id) + 'GMM.pt')
    
    def predict(self, test_point, stream_data, rain_data=None):
        
        time_str = test_point
        self.net.eval()
        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        test_month = []
        test_day = []
        test_hour = []
        new_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        for i in range(self.predict_days):
            new_time_temp = new_time + timedelta(minutes=15)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
   
            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)  
            
            new_time = new_time_temp
            
        y2 = cos_date(test_month, test_day, test_hour) 
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(test_month, test_day, test_hour) 
        y3 = [[ff] for ff in y3]        
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])
       
        if(self.is_prob_feature==1):
            x_test = np.array(log_std_normalization_1(stream_data, self.mean, self.std), np.float32).reshape(self.train_days,-1)
        else:
            x_test = np.array(log_std_normalization_1(stream_data, self.mean, self.std), np.float32).reshape(self.train_days,-1)
        
        if self.opt_hinter_dim > 0:
            if rain_data is None:
                raise ValueError("Rain data is required.")
            y4 = np.array(log_std_normalization_1(rain_data, self.R_mean, self.R_std)).reshape(self.predict_days, -1)
        else:
            y4 = x_test[-1*self.predict_days:]
            weights3 = self.gm3.weights_
            data_prob3 = self.gm3.predict_proba(y4)
            prob_in_distribution3 = data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2]

            prob_like_outlier3 = 1 - prob_in_distribution3
            rain_data = prob_like_outlier3.reshape((len(y4), 1))
            y4 = np.array(log_std_normalization_1(rain_data, self.R_mean, self.R_std)).reshape(self.predict_days, -1)

        x_test = [x_test]
        y_input2 = [y4]
        y_predict = self.inference_test(x_test, y_input1, y_input2)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
        test_predict = np.array(self.std_denorm_dataset(y_predict))
        diff_predict = []
        test_predict = (test_predict + abs(test_predict))/2 
        
        return test_predict
    

