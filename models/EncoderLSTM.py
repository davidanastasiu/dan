#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import random
import logging

logging.basicConfig(filename="Encoder_LSTM.log", filemode="w", level=logging.DEBUG)
random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim

        super(EncoderLSTM, self).__init__()

        """
        1D CNN block. It performs representation learning on different length of local context,
        it does Conv1d and ReLU.
        """
        self.cnn0 = nn.Conv1d(1, self.cnn_dim, 11, stride=11, padding=0)
        self.cnn1 = nn.Conv1d(1, self.cnn_dim, 7, stride=7, padding=0)
        self.cnn2 = nn.Conv1d(1, self.cnn_dim, 3, stride=3, padding=0)

        """
        LSTM block. It does position embedding to hidden state according to different length of local context,
        it does stacked LSTM only.
        """

        # far points
        self.lstm0 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm1 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim,
            bidirectional=True,
            batch_first=True,
        )
        # distribution_indicator
        self.lstm3 = nn.LSTM(
            1, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True
        )
        # near points
        self.lstm4 = nn.LSTM(
            1, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True
        )

    def forward(self, x1, x2, x3, h0, h3, h, c):
        c0 = c

        x = x3.permute(0, 2, 1)
        cnn_out0 = torch.tanh(self.cnn0(x))
        cnn_out0 = cnn_out0.permute(0, 2, 1)  # short subsequence

        cnn_out1 = torch.tanh(self.cnn1(x))
        cnn_out1 = cnn_out1.permute(0, 2, 1)  # middle subsequence

        cnn_out2 = torch.tanh(self.cnn2(x))
        cnn_out2 = cnn_out2.permute(0, 2, 1)  # long subsequence

        out, (hn3, cn3) = self.lstm3(x2, (h, c0))  # Ind

        out, (hn0, cn0) = self.lstm0(cnn_out0, (h, c0))  # Far
        out, (hn1, cn1) = self.lstm1(cnn_out1, (h, c0))
        out, (hn2, cn2) = self.lstm2(cnn_out2, (h, c0))
        hn0 = hn0 + hn1 + hn2
        cn1 = cn0 + cn1 + cn2

        x = x.permute(0, 2, 1)

        out, (hn4, cn4) = self.lstm4(x, (h, c0))  # Near

        hn = []
        cn = []
        hn.append(hn0)  # Far
        hn.append(hn3)  # Ind
        hn.append(hn4)  # Near
        cn.append(cn0)
        cn.append(cn3)
        cn.append(cn4)

        return hn, cn
