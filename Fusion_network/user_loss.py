import torch
import torch.nn as nn
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


class mu_loss(object):
    def __init__(self, gamma=2.2, percentile=99):
        self.gamma = gamma
        self.percentile = percentile

    def __call__(self, pred, label):
        '''pred = pred ** self.gamma   # I think those are linear already.
        label = label ** self.gamma'''
        norm_perc = np.max(label.data.cpu().numpy().astype(np.float32))
        mu_pred = norm_mu_tonemap(label, norm_perc)
        mu_label = norm_mu_tonemap(pred, norm_perc)
        return nn.L1Loss()(mu_pred, mu_label)

class mu2_loss(object):
    def __init__(self, gamma=2.2, percentile=99):
        self.gamma = gamma
        self.percentile = percentile

    def __call__(self, pred, label):
        '''pred = pred ** self.gamma   # I think those are linear already.
        label = label ** self.gamma'''
        norm_perc = np.percentile(label.data.cpu().numpy().astype(np.float32), self.percentile)
        mu_pred = norm_mu_tonemap(label, norm_perc)
        mu_label = norm_mu_tonemap(pred, norm_perc)
        return nn.MSELoss()(mu_pred, mu_label)

class mu_tonemap_loss(object):
    def __init__(self):
        pass

    def __call__(self, pred, label):
        label = mu_tonemap(label)
        pred = mu_tonemap(pred)
        return nn.MSELoss()(label, pred)

class edge_loss(nn.Module):
    def __init__(self, channels=3):
        super(edge_loss, self).__init__()
        kernel = torch.tensor([[-1., -1., -1.],
                            [-1., 8., -1.],
                            [-1., -1., -1.]])
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.laplace_kernel = kernel.to(device)
        self.percentile = 99

    def forward(self, pre, gt):
        norm_perc = np.percentile(gt.data.cpu().numpy().astype(np.float32), self.percentile)
        gt = norm_mu_tonemap(gt, norm_perc)
        pre = norm_mu_tonemap(pre, norm_perc)
        img = torch.nn.functional.pad(pre, (2, 2, 2, 2), mode='reflect')
        lap_img = torch.nn.functional.conv2d(img, self.laplace_kernel, groups=img.shape[1])
        lap_img = torch.abs(torch.tanh(lap_img))
        gt = torch.nn.functional.pad(gt, (2, 2, 2, 2), mode='reflect')
        lap_gt = torch.nn.functional.conv2d(gt, self.laplace_kernel, groups=img.shape[1])
        lap_gt = torch.abs(torch.tanh(lap_gt))
        return nn.L1Loss()(lap_img, lap_gt)

