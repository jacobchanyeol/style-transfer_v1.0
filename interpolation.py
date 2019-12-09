# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:05:15 2019

@author: jacobchoi
"""

import argparse
import os
import sys
import time
import re
import neural_style

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16