import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from itertools import chain
#from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter 

from config import *
from dataloader_joint import *
from sam_5level import Sam_Net
from attU2Net import init_weights, AttU_Net
from user_loss import edge_loss, mu_loss, mu2_loss
from utils import ev_alignment, set_requires_grad, get_lr_scheduler, thres, denormalize
from psnr_ssim import psnr_tanh_norm_mu_tonemap, mu_tonemap, normalized_psnr

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l1 = nn.L1Loss()
l2 = nn.MSELoss()
mu_loss = mu_loss()
mu2_loss = mu2_loss()


writer = SummaryWriter() 

def test(data_loader, Darker, Brighter, epoch):
    test_loss=[]
    psnrs=[]
    norm_psnrs=[]
    ssims=[]
    norm_ssims=[]
    over_count = 0
    under_count = 0
    total_batch = len(data_loader)
    Darker.eval()
    Brighter.eval()
    #F.eval()
    with torch.no_grad():
        for i, ((ldr, hdr, t)) in enumerate(data_loader):
            if(i!=517):
                continue
            # over-/under-exposed images
            ldr_numpy = torch.squeeze(ldr,0)
            ldr_numpy = ldr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            ldr_numpy = denormalize(ldr_numpy)
            gray = cv2.cvtColor(ldr_numpy, cv2.COLOR_RGB2GRAY)
            gray = gray.flatten()*255
            #print(np.max(gray))
            #print(np.min(gray))
            over_exposed_pix = thres(gray, 249, False)
            #print(over_exposed_pix)
            under_exposed_pix = thres(gray, 6, True)
            if(over_exposed_pix>=len(gray)*0.25):
                over_count += 1
                continue
            if(under_exposed_pix>=len(gray)*0.25):
                under_count += 1
                continue

            lower_exposed_pix = thres(gray, 40, True)
            higher_exposed_pix = thres(gray, 235, False)
            lower = lower_exposed_pix>len(gray)*0.1
            higher = higher_exposed_pix>len(gray)*0.1
            if((lower and higher) or (not lower and not higher)):
                contribution = 1
            elif(not lower and higher):
                contribution = 2
            elif(lower and not higher):
                contribution = 0

            # Data Preparation #
            padding = 32
            ldr = torch.nn.functional.pad(ldr, (padding, padding, padding, padding), 'reflect')
            #print(np.shape(ldr))
            ldr = ldr.to(device)
            hdr = hdr.to(device)

            ###################
            # Generate images #
            ###################
            print(contribution)
            if(contribution == 0): #under exposure case
                t_list=np.array([t,4*t,16*t],dtype=np.float32)
                dark = ldr
                # produce normal and over
                normal = Brighter(dark)
                bright = Brighter(normal)
            elif(contribution == 1): #normal exposure case
                t_list=np.array([t/4,t,4*t],dtype=np.float32)
                normal = ldr
                # produce over and under
                dark = Darker(normal)
                bright = Brighter(normal)
            elif(contribution == 2): #over exposure case
                t_list=np.array([t/16,t/4,t],dtype=np.float32)
                bright = ldr
                # produce normal and under
                normal = Darker(bright)
                dark = Darker(normal)

            
            #################
            # adjust images #
            #################

            # denormalize 
            dark = torch.add(torch.mul(dark,0.5),0.5)
            normal = torch.add(torch.mul(normal,0.5),0.5)
            bright = torch.add(torch.mul(bright,0.5),0.5)
            
            ##################
            # branch Debevec #
            ##################
            dark_numpy = torch.squeeze(dark,0)
            dark_numpy = dark_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            dark_numpy = dark_numpy[padding:-padding, padding:-padding, :]
            dark_numpy = np.clip((dark_numpy*255).astype(np.uint8),0,255)
            normal_numpy = torch.squeeze(normal,0)
            normal_numpy = normal_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            normal_numpy = normal_numpy[padding:-padding, padding:-padding, :]
            normal_numpy = np.clip((normal_numpy*255).astype(np.uint8),0,255)
            bright_numpy = torch.squeeze(bright,0)
            bright_numpy = bright_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            bright_numpy = bright_numpy[padding:-padding, padding:-padding, :]
            bright_numpy = np.clip((bright_numpy*255).astype(np.uint8),0,255)

            image_list=[dark_numpy,normal_numpy,bright_numpy]
            calibrateDebevec = cv2.createCalibrateDebevec(samples=120,random=True)  
            ###采样点数120个，采样方式为随机，一般而言，采用点数越多，采样方式越随机，最后的CRF曲线会越加平滑
            print(i)
            responseDebevec = calibrateDebevec.process(image_list, t_list)  #获得CRF
            merge_Debevec = cv2.createMergeDebevec()
            hdrDebevec = merge_Debevec.process(image_list, t_list, responseDebevec)

            ############
            # evaluate #
            ############

            hdr_numpy = torch.squeeze(hdr,0)
            hdr_numpy = hdr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            psnr, ssim = psnr_tanh_norm_mu_tonemap(hdrDebevec,hdr_numpy)
            norm_psnr, norm_ssim = normalized_psnr(hdrDebevec,hdr_numpy)
            hdrDebevec=hdrDebevec[:,:,::-1]
            cv2.imwrite("./debevec/%d.hdr" %(i),hdrDebevec)
            
            psnrs.append(psnr)
            ssims.append(ssim)
            norm_psnrs.append(norm_psnr)
            norm_ssims.append(norm_ssim)
            

        writer.add_scalar('test/PSNR', np.mean(psnrs), epoch)
        writer.add_scalar('test/loss', np.mean(test_loss), epoch)

        print("Test | Epochs [{}/{}] | mu2 Loss {:.4f} PSNR {} SSIM {} Norm_PSNR {} Norm_SSIM {}"
                        .format(epoch+1, config.num_epochs, np.mean(test_loss), np.mean(psnrs), np.mean(ssims), np.mean(norm_psnrs), np.mean(norm_ssims)))
        print("Test | Epochs [{}/{}] has {} over exposure images of {} images".format(epoch+1, config.num_epochs, over_count, total_batch))
        print("Test | Epochs [{}/{}] has {} under exposure images of {} images".format(epoch+1, config.num_epochs, under_count, total_batch))
        
    return np.mean(psnrs)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    valiset = get_vali_dataset()
    vali_loader = DataLoader(valiset, batch_size = 1, shuffle = False)
    # Prepare Networks #
    F = Sam_Net().to(device)
    Darker = AttU_Net().to(device)
    Brighter = AttU_Net().to(device)
    # Initial weight
    Darker.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_A2B_CGAN_SC.pkl')))
    Brighter.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_B2A_CGAN_SC.pkl')))
    test(vali_loader, Darker, Brighter, epoch=0)