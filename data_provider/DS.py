import numpy as np
import random
import pandas as pd
from ..utils.utils2 import (
    log_std_normalization,
    gen_month_tag,
    gen_time_feature,
    cos_date,
    sin_date,
    RnnDataset,
    log_std_normalization_1,
)
import os
from sklearn.mixture import GaussianMixture
from scipy import stats
import torch
from torch.utils.data import DataLoader

class DS:

    def __init__(self, opt, trainX, R_X):

        self.opt = opt
        self.trainX = trainX
        self.R_X = R_X
        self.mean = 0
        self.std = 0
        self.R_mean = 0
        self.R_std = 0
        self.tag = []
        self.sensor_data = []
        self.data = []
        self.data_time = []
        self.sensor_data_norm = []
        self.sensor_data_norm1 = []

        self.R_sensor_data = []
        self.R_data = []
        self.R_data_time = []
        self.R_sensor_data_norm = []
        self.R_sensor_data_norm1 = []

        self.val_points = []
        self.test_points = []
        self.test_start_time = self.opt.test_start
        self.test_end_time = self.opt.test_end
        self.opt_hinter_dim = self.opt.watershed
        self.gm3 = GaussianMixture(
            n_components=3,
        )

        self.norm_percen = 0
        self.kruskal = opt.oversampling
        self.event_focus_level = opt.event_focus_level

        self.train_days = self.opt.input_len
        self.predict_days = self.opt.output_len
        self.val_near_days = self.opt.output_len
        self.lens = self.train_days + self.predict_days + 1
        self.batch_size = opt.batchsize

        self.is_prob_feature = 1
        self.val_data_loader = []
        self.train_data_loader = []
        self.month = []
        self.day = []
        self.hour = []

        self.h_value = []
        self.sampled_h_value = []
        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.sensor_id = opt.stream_sensor

        self.read_dataset()

        # save mean and std of dataset
        norm = []
        norm.append(self.get_mean())
        norm.append(self.get_std())
        norm.append(self.get_R_mean())
        norm.append(self.get_R_std())
        np.savetxt(self.expr_dir + "/" + "Norm.txt", norm)
        norm = np.loadtxt(self.expr_dir + "/" + "Norm.txt", dtype=float, delimiter=None)
        print("norm is: ", norm)
        if self.opt.mode == "train":
            self.val_dataloader()
            self.train_dataloader()
        else:
            self.refresh_dataset(trainX, R_X)

    def get_trainX(self):

        return self.trainX

    def get_data(self):

        return self.data

    def get_sensor_data(self):

        return self.sensor_data

    def get_sensor_data_norm(self):

        return self.sensor_data_norm

    def get_sensor_data_norm1(self):

        return self.sensor_data_norm1

    def get_R_data(self):

        return self.R_data

    def get_R_sensor_data(self):

        return self.R_sensor_data

    def get_R_sensor_data_norm(self):

        return self.R_sensor_data_norm

    def get_R_sensor_data_norm1(self):

        return self.R_sensor_data_norm1

    def get_val_data_loader(self):

        return self.val_data_loader

    def get_train_data_loader(self):

        return self.train_data_loader

    def get_val_points(self):

        return self.val_points

    def get_test_points(self):

        return self.test_points

    def get_mean(self):

        return self.mean

    def get_std(self):

        return self.std

    def get_R_mean(self):

        return self.R_mean

    def get_R_std(self):

        return self.R_std

    def get_month(self):

        return self.month

    def get_day(self):

        return self.day

    def get_hour(self):

        return self.hour

    def get_tag(self):

        return self.tag

    # Fetch dataset from data file, do preprocessing
    def read_dataset(self):

        # read sensor data to vector
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        print("for sensor ", self.opt.stream_sensor, "start_num is: ", start_num)
        # foot label of train_end
        train_end = (
            self.trainX[self.trainX["datetime"] == self.opt.train_point].index.values[0]
            - start_num
        )
        print("train set length is : ", train_end)

        # the whole dataset
        self.sensor_data = self.trainX[
            start_num: train_end + start_num
        ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        self.sensor_data_norm, self.mean, self.std = log_std_normalization(self.data)
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]

        if self.is_prob_feature == 1:
            clean_data = []
            for ii in range(len(self.data)):
                if (self.data[ii] is not None) and (np.isnan(self.data[ii]) != 1):
                    clean_data.append(self.data[ii])
            sensor_data_prob = np.array(clean_data).reshape(-1, 1)
            self.gm3.fit(sensor_data_prob)
            torch.save(self.gm3, self.expr_dir + "/" + "GMM.pt")
            print("gm3.means are: ", self.gm3.means_)
            print("gm3.covariances are: ", self.gm3.covariances_)
            print("gm3.weights are: ", self.gm3.weights_)
            weights3 = self.gm3.weights_
            data_prob3 = self.gm3.predict_proba(sensor_data_prob)
            prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))

            recover_data = []
            temp = [
                0.5
            ]  # use to fill nan value, which will not be selected into the train and validation data set.
            jj = 0
            for ii in range(len(self.data)):
                if (self.data[ii] is not None) and (np.isnan(self.data[ii]) != 1):
                    recover_data.append(prob_like_outlier3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(temp)
            prob_like_outlier3 = np.array(recover_data, np.float32).reshape(
                len(self.data), 1
            )

            print("Finish prob indicator generating.")

        if self.opt_hinter_dim >= 1:  # defined by the "watershed" parameter
            # read Rain data to vector
            R_start_num = self.R_X[
                self.R_X["datetime"] == self.opt.start_point
            ].index.values[0]
            print("for sensor ", self.opt.rain_sensor, "start_num is: ", R_start_num)
            R_train_end = (
                self.R_X[self.R_X["datetime"] == self.opt.train_point].index.values[0]
                - R_start_num
            )
            print("R_X set length is : ", R_train_end)
            self.R_sensor_data = self.R_X[
                R_start_num: R_train_end + R_start_num
            ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan))
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = [[ff] for ff in self.R_sensor_data_norm]
        else:  # use GMM indicator
            self.R_data = prob_like_outlier3
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = prob_like_outlier3.squeeze()
            self.R_sensor_data_norm = self.R_sensor_data_norm1

        self.tag = gen_month_tag(self.sensor_data)

        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]

    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence, between Sep and May), tag it as 3
    # For those points near this point (near is defined as a parameter), tag them as 4, which should also not be included in the training

    def val_dataloader(self):

        print("Begin to generate val_dataloader!")
        DATA = []  # unused?
        Label = []  # unused?

        near_len = self.predict_days

        # randomly choose val data
        random.seed(self.opt.val_seed)

        ii = 0
        while ii < self.opt.val_size:

            i = random.randint(self.predict_days, len(self.data) - self.lens - 1)
            # Hydro year is from September to May, for all experiments and all methods.
            if (
                (not np.isnan(self.data[i: i + self.lens]).any())
                and (not np.isnan(self.R_data[i: i + self.lens]).any())
                and (
                    self.tag[i + self.train_days] <= -9
                    or -6 < self.tag[i + self.train_days] < 0
                    or 2 <= self.tag[i + self.train_days] <= 3
                )
            ):

                self.tag[i + self.train_days] = 2  # tag 2 means in validation set

                for k in range(near_len):
                    self.tag[i + self.train_days - k] = (
                        3  # tag 3 means near points of validation set
                    )
                    self.tag[i + self.train_days + k] = 3

                point = self.data_time[i + self.train_days]
                self.val_points.append([point])
                ii = ii + 1

        self.opt.name = "%s" % (self.opt.model)
        val_dir = os.path.join(self.opt.outf, self.opt.name, "val")
        file_name = os.path.join(val_dir, "validation_timestamps_24avg.tsv")

        pd_temp = pd.DataFrame(data=self.val_points, columns=["Hold Out Start"])
        pd_temp.to_csv(file_name, sep="\t")

    # train_dataloader can only be run after val_dataloader to avoid sampled validation points
    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence between Sep and May, and tag is not 3 and 4),
    # select it as a train point, tag it as 5

    def train_dataloader(self):

        print("Begin to generate train_dataloader!")
        DATA = []
        Label = []

        # randomly choose train data
        random.seed(self.opt.train_seed)

        ii = 0
        while ii < self.opt.train_volume:

            i = random.randint(
                self.predict_days, len(self.data) - 31 * self.predict_days - 1
            )

            if (
                (not np.isnan(self.data[i: i + self.lens]).any())
                and (not np.isnan(self.R_data[i: i + self.lens]).any())
                and (
                    self.tag[i + self.train_days] <= -9
                    or -6 < self.tag[i + self.train_days] < 0
                )
            ):

                data0 = np.array(
                    self.sensor_data_norm1[i: (i + self.train_days)]
                ).reshape(self.train_days, -1)
                label00 = np.array(
                    self.sensor_data_norm[
                        (i + self.train_days): (
                            i + self.train_days + self.predict_days
                        )
                    ]
                )
                label01 = np.array(
                    self.data[
                        (i + self.train_days): (
                            i + self.train_days + self.predict_days
                        )
                    ]
                )
                label0 = [[ff] for ff in label00]

                b = i + self.train_days
                e = i + self.train_days + self.predict_days

                label2 = cos_date(
                    self.month[b:e], self.day[b:e], self.hour[b:e]
                )  # represent cos(int(data)) here
                label2 = [[ff] for ff in label2]

                label3 = sin_date(
                    self.month[b:e], self.day[b:e], self.hour[b:e]
                )  # represent sin(int(data)) here
                label3 = [[ff] for ff in label3]

                label4 = np.array(
                    self.R_sensor_data_norm[
                        (i + self.train_days - self.predict_days): (
                            i + self.train_days
                        )
                    ]
                )
                label4 = [[ff] for ff in label4]

                label5 = np.array(self.R_sensor_data_norm[b:e])
                label5 = [[ff] for ff in label5]

                label = np.concatenate((label0, label2), 1)
                label = np.concatenate((label, label3), 1)
                label = np.concatenate((label, label4), 1)
                label = np.concatenate((label, label5), 1)

                if (
                    (label01[:72] == label01[72:144]).all()
                    and (label01[144:216] == label01[216:]).all()
                    and (label01[72:144] == label01[144:216]).all()
                ):
                    a = 0
                    b = 0
                else:
                    a, b = stats.kruskal(
                        label01[:72], label01[72:144], label01[144:216], label01[216:]
                    )
                self.h_value.append(a)
                if a > self.kruskal:
                    DATA.append(data0)
                    Label.append(label)
                    self.tag[i + self.train_days] = 4  # tag 4 means in train set
                    ii = ii + 1
                    self.sampled_h_value.append(a)
                else:
                    kk = random.randint(0, 99)
                    if kk <= self.event_focus_level:
                        DATA.append(data0)
                        Label.append(label)
                        self.tag[i + self.train_days] = 4  # tag 4 means in train set
                        ii = ii + 1
                        self.sampled_h_value.append(a)

        dataset1 = RnnDataset(DATA, Label)
        self.train_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def refresh_dataset(self, trainX, R_X):
        self.trainX = trainX
        self.R_X = R_X
        # read sensor data to vector
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        # the whole dataset
        k = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
        self.sensor_data = self.trainX[start_num:k]
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        self.sensor_data_norm = log_std_normalization_1(
            self.data, self.mean, self.std
        )  # use old mean & std
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]

        if self.is_prob_feature == 1:

            clean_data = []
            for ii in range(len(self.data)):
                if (self.data[ii] is not None) and (np.isnan(self.data[ii]) != 1):
                    clean_data.append(self.data[ii])
            sensor_data_prob = np.array(clean_data).reshape(-1, 1)

            weights3 = self.gm3.weights_
            data_prob3 = self.gm3.predict_proba(sensor_data_prob)
            prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape(len(sensor_data_prob), 1)

            recover_data = []
            temp0 = [0.5]
            jj = 0
            for ii in range(len(self.data)):
                if (self.data[ii] is not None) and (np.isnan(self.data[ii]) != 1):
                    recover_data.append(prob_like_outlier3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(temp0)
            prob_like_outlier3 = np.array(recover_data).reshape(len(self.data), 1)
            print("Finish prob indicator updating.")

        if self.opt_hinter_dim >= 1:
            # read Rain data to vector
            R_start_num = self.R_X[
                self.R_X["datetime"] == self.opt.start_point
            ].index.values[0]
            R_k = self.R_X[self.R_X["datetime"] == self.test_end_time].index.values[0]
            self.R_sensor_data = self.R_X[
                R_start_num:R_k
            ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan))
            self.R_sensor_data_norm = log_std_normalization_1(
                self.R_data, self.R_mean, self.R_std
            )  # use old mean & std
            self.R_sensor_data_norm1 = [[ff] for ff in self.R_sensor_data_norm]
        else:
            self.R_data = prob_like_outlier3
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = prob_like_outlier3.squeeze()
            self.R_sensor_data_norm = self.R_sensor_data_norm1

        self.tag = gen_month_tag(self.sensor_data)  # update

        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)  # update

        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]

    def gen_test_data(self):

        self.test_points = []
        self.refresh_dataset(self.trainX, self.R_X)
        print("Begin to generate test_points!")

        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]

        begin_num = (
            self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0]
            - start_num
        )
        end_num = (
            self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
            - start_num
        )

        for i in range(int((end_num - begin_num - self.predict_days) / 16)):
            point = self.data_time[begin_num + i * 16]
            if not np.isnan(
                np.array(
                    self.data[
                        begin_num
                        + i * 16
                        - self.train_days: begin_num
                        + i * 16
                        + self.predict_days
                    ]
                )
            ).any():
                self.test_points.append([point])
