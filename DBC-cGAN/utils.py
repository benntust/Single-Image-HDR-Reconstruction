import os
import imageio
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from torch.nn import init
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter 

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_Pixelwise = torch.nn.L1Loss()
writer = SummaryWriter()  

def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)

def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)

def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)

def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

def get_lr_scheduler(optimizer):
    """Learning Rate Scheduler"""
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler

def set_requires_grad(networks, requires_grad=False):
    """Prevent a Network from Updating"""
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad

def denormalize(array):
    mean = 0.5 #np.array([[[0.5]], [[0.5]], [[0.5]]])
    var = 0.5 #np.array([[[0.5]], [[0.5]], [[0.5]]])
    array = array*var + mean
    # The normalize code -> t.sub_(m).div_(s)
    return array

def thres(gray, threshold, smaller):
    threshold = np.ones(np.shape(gray))*threshold
    if(smaller):
        res = (gray<=threshold).astype(np.float32)
    else:
        res = (gray>=threshold).astype(np.float32)
    pix = np.sum(res)
    return pix