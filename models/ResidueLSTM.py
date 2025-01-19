#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import random
import logging

logging.basicConfig(filename="Residue_LSTM.log", filemode="w", level=logging.DEBUG)
random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidueLSTM(nn.Module):
    def __init__(self, opt):
        super(ResidueLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer

        self.lstm00 = nn.LSTM(
            2, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True
        )
        self.lstm02 = nn.LSTM(
            1, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True
        )
        self.lstm03 = nn.LSTM(
            2, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True
        )

        self.L_out00 = nn.Linear(self.hidden_dim * 2, 1)
        self.L_out02 = nn.Linear(self.hidden_dim * 2, 1)
        self.L_out03 = nn.Linear(self.hidden_dim * 2, 1)

        self.L_out04 = nn.Linear(2, self.hidden_dim)
        self.L_out05 = nn.Linear(self.hidden_dim, 1)

        """
        Distribution Gate Indicator.
        """
        self.cnn00 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 7, stride=1, padding=3)
        self.cnn01 = nn.Conv1d(self.hidden_dim, 1, 3, stride=1, padding=1)

    def forward(
        self, x1, out0, Ind, out3, encoder_h, encoder_c
    ):  # x1: time sin & cos;   x2: hinter vectors

        # Initialize hidden and cell state with zeros
        h0 = encoder_h[0]
        c0 = encoder_c[0]
        h3 = encoder_h[2]
        c3 = encoder_c[2]

        # far points
        o0, (hn, cn) = self.lstm00(x1, (h0, c0))
        out00 = self.L_out00(o0)
        out0 = out0 + torch.squeeze(out00, dim=2)

        # near points
        o3, (hn, cn) = self.lstm03(x1, (h3, c3))
        out03 = self.L_out03(o3)
        out3 = out3 + torch.squeeze(out03, dim=2)

        out2, (hn, cn) = self.lstm02(Ind, (h0, c0))
        out02 = self.L_out02(out2)

        out03 = torch.unsqueeze(out3, dim=2)
        out02 = torch.cat([out02, out03], dim=2)
        out4 = self.L_out04(out02)
        out4 = self.L_out05(out4)
        out4 = torch.squeeze(out4, dim=2)

        return out0, out3, out4
