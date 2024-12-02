#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random  # unused?
import pandas as pd  # unused?
import math
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader  # unused?
from matplotlib import pyplot as plt


# 1-order difference preprocessing log + diff + std
def diff_order_1(data):
    a = data
    b = a[:-1]
    a = a[1:] - b
    c = np.array([0] + a.tolist())
    d = np.nanmin(c)
    c = c - d

    return c


def log_std_normalization(sensor_data_val):
    a = np.log(np.array(sensor_data_val) + 1)
    c = a
    mean = np.nanmean(c)
    print("mean is: ", mean)
    std = np.nanstd(c)
    print("std is ", std)
    c = (c - mean) / std

    return c, mean, std


def log_std_normalization_1(sensor_data_val, mean, std):
    a = np.log(np.array(sensor_data_val) + 1)
    c = a
    c = (c - mean) / std

    return c


def log_std_denorm_dataset(mean, std, predict_y0, y_pre):
    a2 = predict_y0
    a2 = [ii * std + mean for ii in a2]
    a3 = a2
    a3 = [((np.e) ** ii) - 1 for ii in a3]

    return a3


def gen_Nan_tag(sensor_data):
    data = np.array(sensor_data["datetime"].fillna(np.nan))
    b = np.isnan(data)
    c = 1 - b  # 1 means none, else o
    tag = 1 - c  # 0 means none, else 1

    return tag


def gen_month_tag(sensor_data):
    sensor_month = sensor_data["datetime"].str[5:7]
    a = sensor_month.str[:]
    a = a.astype(int)
    tag = np.array(a.fillna(np.nan))
    tag = -1 * tag  # month number with negtive label, all less or equal to 0

    return tag


# generate time feature as month+day+hour, transfer str to int, then we have a sequence of meaningful int
def gen_time_feature(sensor_data):

    sensor_month = sensor_data["datetime"].str[5:7]
    sensor_day = sensor_data["datetime"].str[8:10]
    sensor_hour = sensor_data["datetime"].str[11:13]
    month = sensor_month.astype(np.int8)
    month = np.array(month.fillna(np.nan))
    day = sensor_day.astype(np.int8)
    day = np.array(day.fillna(np.nan))
    hour = sensor_hour.astype(np.int8)
    hour = np.array(hour.fillna(np.nan))

    return month, day, hour


def cos_date(month, day, hour):
    t = []
    for i in range(len(month)):
        a = math.cos(((month[i] - 1) * 30.5 + day[i]) * 2 * (math.pi) / 365)
        t.append(a)

    return t


def sin_date(month, day, hour):
    t = []
    for i in range(len(month)):
        a = math.sin(((month[i] - 1) * 30.5 + day[i]) * 2 * (math.pi) / 365)
        t.append(a)

    return t


# prepare train and val set
class RnnDataset(TensorDataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "type4":
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def plot(gt, pre):
    plt.figure(figsize=(15, 3))
    plt.ylim(-1, 900)
    plt.xlabel("time ( hours )")
    plt.ylabel("streamflow")
    plt.plot(np.array(gt), "black", label="Ground Truth")
    plt.plot(np.array(pre), "blue", label="Predicted")
