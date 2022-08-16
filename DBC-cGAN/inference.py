import os
from cv2 import imwrite
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter 

from config import *
from dataloader import *
from discriminator import Discriminator
from attU2Net import init_weights, AttU_Net
from utils import make_dirs, set_requires_grad, get_lr_scheduler, denormalize, thres
from psnr_ssim import calculate_psnr

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loss #
class edge_loss(nn.Module):
    def __init__(self, channels=3):
        super(edge_loss, self).__init__()
        kernel = torch.tensor([[-1., -1., -1.],
                            [-1., 8., -1.],
                            [-1., -1., -1.]])
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.laplace_kernel = kernel.to(device)

    def forward(self, pre, gt):
        img = torch.nn.functional.pad(pre, (2, 2, 2, 2), mode='reflect')
        lap_img = torch.nn.functional.conv2d(img, self.laplace_kernel, groups=img.shape[1])
        lap_img = torch.abs(torch.tanh(lap_img))
        gt = torch.nn.functional.pad(gt, (2, 2, 2, 2), mode='reflect')
        lap_gt = torch.nn.functional.conv2d(gt, self.laplace_kernel, groups=img.shape[1])
        lap_gt = torch.abs(torch.tanh(lap_gt))
        return criterion_Pixelwise(lap_img, lap_gt)
criterion_Adversarial = nn.BCELoss()
criterion_Pixelwise = nn.L1Loss()
edge_loss = edge_loss()

def test(data_loader, G_A2B, G_B2A, epoch):
    A2B_losses=[]
    B2A_losses=[]
    psnr_A2B_losses=[]
    psnr_B2A_losses=[]
    bright_count = 0
    dark_count = 0
    
    G_A2B.eval()
    G_B2A.eval()
    with torch.no_grad():
        for i, (bright, dark) in enumerate(data_loader):
            if(i!=534):
                continue
            bright_skip = False
            dark_skip = False
            # over-/under-exposed images
            bright_numpy = np.squeeze(bright,0)
            bright_numpy = bright_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            bright_numpy = denormalize(bright_numpy)
            gray = cv2.cvtColor(bright_numpy, cv2.COLOR_RGB2GRAY)
            gray = gray.flatten()*255
            over_exposed_pix = thres(gray, 249, False)
            under_exposed_pix = thres(gray, 6, True)
            loss_const = (over_exposed_pix>len(gray)*0.25) or (under_exposed_pix>len(gray)*0.25)
            if loss_const:
                bright_count += 1
                bright_skip = True
            
            # over-/under-exposed images
            dark_numpy = np.squeeze(dark,0)
            dark_numpy = dark_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            dark_numpy = denormalize(dark_numpy)
            gray = cv2.cvtColor(dark_numpy, cv2.COLOR_RGB2GRAY)
            gray = gray.flatten()*255
            over_exposed_pix = thres(gray, 249, False)
            under_exposed_pix = thres(gray, 6, True)
            loss_const = (over_exposed_pix>len(gray)*0.25) or (under_exposed_pix>len(gray)*0.25)
            if loss_const:
                dark_count += 1
                dark_skip = True
            
            padding = 32
            dark = torch.nn.functional.pad(dark, (padding, padding, padding, padding), 'reflect')
            bright = torch.nn.functional.pad(bright, (padding, padding, padding, padding), 'reflect')
            dark = dark.to(device)
            bright = bright.to(device)

            if(not bright_skip):
                fake_dark = G_A2B(bright)
                #A2B_loss = criterion_Pixelwise(fake_dark, dark) #+ config.edge_lambda * edge_loss(fake_dark, dark)
                #A2B_losses.append(A2B_loss.item())
                fake_dark_numpy = np.squeeze(fake_dark,0)
                fake_dark_numpy = fake_dark_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_dark_numpy = fake_dark_numpy[padding:-padding, padding:-padding, :]
                fake_dark_numpy = denormalize(fake_dark_numpy)
                psnr_A2B_loss = calculate_psnr(fake_dark_numpy,dark_numpy)
                psnr_A2B_losses.append(psnr_A2B_loss)
                fake_dark_numpy=np.clip(fake_dark_numpy*255,0,255)
                fake_dark_numpy=fake_dark_numpy[:,:,::-1]
                dark_numpy=np.clip(dark_numpy*255,0,255)
                dark_numpy=dark_numpy[:,:,::-1]
                cv2.imwrite(os.path.join(config.inference_path, 'test/fake_dark_Unet_SC_%d_%.2f.jpg' %(i,psnr_A2B_loss)),fake_dark_numpy)
                #cv2.imwrite(os.path.join(config.inference_path, 'unet/gt_dark_unet_%d.jpg' %(i)),dark_numpy)
            
            if(not dark_skip):
                fake_bright = G_B2A(dark)
                #B2A_loss = criterion_Pixelwise(fake_bright, bright) #+ config.edge_lambda * edge_loss(fake_bright, bright)
                #B2A_losses.append(B2A_loss.item())
                fake_bright_numpy = np.squeeze(fake_bright,0)
                fake_bright_numpy = fake_bright_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_bright_numpy = fake_bright_numpy[padding:-padding, padding:-padding, :]
                fake_bright_numpy = denormalize(fake_bright_numpy)
                psnr_B2A_loss = calculate_psnr(fake_bright_numpy,bright_numpy)
                psnr_B2A_losses.append(psnr_B2A_loss)
                fake_bright_numpy=np.clip(fake_bright_numpy*255,0,255)
                fake_bright_numpy=fake_bright_numpy[:,:,::-1]
                bright_numpy=np.clip(bright_numpy*255,0,255)
                bright_numpy=bright_numpy[:,:,::-1]
                cv2.imwrite(os.path.join(config.inference_path, 'test/fake_bright_Unet_SC_%d_%.2f.jpg' %(i,psnr_B2A_loss)),fake_bright_numpy)
                #cv2.imwrite(os.path.join(config.inference_path, 'unet_SC/gt_bright_unet_SC_%d.jpg' %(i)),dark_numpy)
        
        print("Test | Epochs [{}/{}] | G_A2B Loss {:.4f} PSNR {}"
                        .format(epoch+1, config.num_epochs, np.mean(A2B_losses), np.mean(psnr_A2B_losses)))
        print("Test | Epochs [{}/{}] | G_B2A Loss {:.4f} PSNR {}"
                        .format(epoch+1, config.num_epochs, np.mean(B2A_losses), np.mean(psnr_B2A_losses)))
        print("Test | Epochs [{}/{}] bright domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, bright_count))
        print("Test | Epochs [{}/{}] dark domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, dark_count))

    del bright, dark, fake_bright, fake_dark

    return np.mean(psnr_A2B_losses), np.mean(psnr_B2A_losses)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    valiset = get_vali_dataset()
    vali_loader = DataLoader(valiset, batch_size = 1, shuffle = False)
    # Prepare Networks #
    G_A2B = AttU_Net().to(device)
    G_B2A = AttU_Net().to(device)
    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_A2B32.29_aug.pkl')))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_B2A31.23_aug.pkl')))
    test(vali_loader, G_A2B, G_B2A, 0)