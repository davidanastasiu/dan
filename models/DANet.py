#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import random
from .EncoderLSTM import EncoderLSTM
from .DecoderLSTM import DecoderLSTM
from .ResidueLSTM import ResidueLSTM
import logging

logging.basicConfig(filename="DANet.log", filemode="w", level=logging.DEBUG)
random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DANet(nn.Module):
    ENCODER_BLOCK = "encoder"
    DECODER_BLOCK = "decoder"
    RESIDUE_BLOCK = "residue"

    def __init__(
        self,
        opt,
        device=torch.device("cuda"),
        stack_types=(ENCODER_BLOCK, DECODER_BLOCK, RESIDUE_BLOCK),
    ):
        super(DANet, self).__init__()
        self.opt = opt
        self.stack_types = stack_types
        self.stacks = []
        self.parameters = []
        self.device = device
        print("| DANet")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f"| --  Stack {stack_type.title()} (#{stack_id})")
        block_init = DANet.set_block(stack_type)
        block = block_init(self.opt)
        self.parameters.extend(block.parameters())

        return block

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def set_block(block_type):
        if block_type == DANet.ENCODER_BLOCK:
            return EncoderLSTM
        elif block_type == DANet.DECODER_BLOCK:
            return DecoderLSTM
        else:
            return ResidueLSTM

    def forward(self, x1, x2, x3, h0, c0):
        h = h0
        c = c0
        Ind = x2
        out0 = torch.squeeze(torch.zeros(x2.size()).to(device), dim=2)
        out1 = torch.squeeze(torch.zeros(x2.size()).to(device), dim=2)
        out2 = torch.squeeze(torch.zeros(x2.size()).to(device), dim=2)

        for stack_id in range(len(self.stacks)):
            if stack_id == 0:
                h, c = self.stacks[stack_id](x1, x2, x3, h0, h0, h0, c0)
            else:
                if (
                    stack_id < len(self.stacks) - 1
                    and self.stack_types[stack_id] == "encoder"
                ):
                    h, c = self.stacks[stack_id](x1, Ind, x3, h[0], h[2], h0, c0)

                elif self.stack_types[stack_id] == "decoder":
                    """
                    As a decoder block, use original input and trained Ind.
                    Account for the accuracy of Ind.

                    """
                    o0, Ind, o1, o2 = self.stacks[stack_id](x1, Ind, x3, h, c)
                    out0 = out0 + o0
                    out1 = out1 + o1
                    out2 = out2 + o2
                    out4 = out2

                else:
                    """
                    As a residue block, use accumulated output as input and trained Ind.
                    Account for the accuracy of all.

                    """
                    o0, o1, o4 = self.stacks[stack_id](x1, out0, Ind, out1, h, c)
                    out0 = o0
                    out1 = o1
                    out4 = out4 + o4

        Ind = torch.squeeze(Ind, dim=2)

        return out0, out1, out2, Ind, out4
