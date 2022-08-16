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

def test(data_loader, Darker, Brighter, F, epoch):
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
    F.eval()
    with torch.no_grad():
        for i, ((ldr, hdr)) in enumerate(data_loader):
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
            ldr = ldr.to(device)
            hdr = hdr.to(device)

            ###################
            # Generate images #
            ###################
            if(contribution == 0): #under exposure case
                dark = ldr
                # produce normal and over
                normal = Brighter(dark)
                bright = Brighter(normal)
            elif(contribution == 1): #normal exposure case
                normal = ldr
                # produce over and under
                dark = Darker(normal)
                bright = Brighter(normal)
            elif(contribution == 2): #over exposure case
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
            if(i%303==0):
                writer.add_image("image/test/dark", dark, global_step=i//300, dataformats='NCHW')
                writer.add_image("image/test/normal", normal, global_step=i//300, dataformats='NCHW')
                writer.add_image("image/test/bright", bright, global_step=i//300, dataformats='NCHW')
            
            ############
            # evaluate #
            ############

            pred_hdr = F(dark, normal, bright)
            loss = mu2_loss(pred_hdr,hdr)
            '''print(loss)
            print(pred_hdr)
            print(hdr)'''
            pred_hdr_numpy = torch.squeeze(pred_hdr,0)
            pred_hdr_numpy = pred_hdr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            hdr_numpy = torch.squeeze(hdr,0)
            hdr_numpy = hdr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            psnr, ssim = psnr_tanh_norm_mu_tonemap(pred_hdr_numpy,hdr_numpy)
            norm_psnr, norm_ssim = normalized_psnr(pred_hdr_numpy,hdr_numpy)
            #pred_hdr_numpy=pred_hdr_numpy[:,:,::-1]
            #cv2.imwrite("./HDR-Real_test/%05d/ben.hdr" %(i),pred_hdr_numpy)
            
            psnrs.append(psnr)
            ssims.append(ssim)
            norm_psnrs.append(norm_psnr)
            norm_ssims.append(norm_ssim)
            test_loss.append(loss.item())

            if(i%303==0):
                pred_hdr_numpy = np.clip(mu_tonemap(pred_hdr_numpy), 0, 1)
                hdr_numpy = np.clip(mu_tonemap(hdr_numpy), 0, 1)
                writer.add_image("image/test/zpred", pred_hdr_numpy, global_step=i//300, dataformats='HWC')
                writer.add_image("image/test/zgt", hdr_numpy, global_step=i//300, dataformats='HWC')
            

        writer.add_scalar('test/PSNR', np.mean(psnrs), epoch)
        writer.add_scalar('test/loss', np.mean(test_loss), epoch)

        print("Test | Epochs [{}/{}] | mu2 Loss {:.4f} PSNR {} SSIM {} Norm_PSNR {} Norm_SSIM {}"
                        .format(epoch+1, config.num_epochs, np.mean(test_loss), np.mean(psnrs), np.mean(ssims), np.mean(norm_psnrs), np.mean(norm_ssims)))
        print("Test | Epochs [{}/{}] has {} over exposure images of {} images".format(epoch+1, config.num_epochs, over_count, total_batch))
        print("Test | Epochs [{}/{}] has {} under exposure images of {} images".format(epoch+1, config.num_epochs, under_count, total_batch))
        
    return np.mean(psnrs)

def train():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Prepare Data Loader #
    trainset = get_train_dataset()
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = True)
    valiset = get_vali_dataset()
    vali_loader = DataLoader(valiset, batch_size = 1, shuffle = False)
    total_batch = len(train_loader)
    print(total_batch)

    # Prepare Networks #
    F = Sam_Net().to(device)
    Darker = AttU_Net().to(device)
    Brighter = AttU_Net().to(device)
    # Initial weight
    Darker.load_state_dict(torch.load(os.path.join(config.weights_path, 'Darkerbest_single.pkl')))
    Brighter.load_state_dict(torch.load(os.path.join(config.weights_path, 'Brighterbest_single.pkl')))
    #F.load_state_dict(torch.load(os.path.join(config.weights_path, 'SamNetbest_single.pkl')))
    init_weights(F,init_type='kaiming')   
    
    # Optimizers #
    F_optim = torch.optim.Adam(F.parameters(), lr=config.lr, betas=(0.5, 0.999))
    Darker_optim = torch.optim.Adam(Darker.parameters(), lr=config.lr, betas=(0.5, 0.999))
    Brighter_optim = torch.optim.Adam(Brighter.parameters(), lr=config.lr, betas=(0.5, 0.999))

    F_optim_scheduler = get_lr_scheduler(F_optim)
    Darker_optim_scheduler = get_lr_scheduler(Darker_optim)
    Brighter_optim_optim_scheduler = get_lr_scheduler(Brighter_optim) 
    
    # Training #
    print("Training SamNet started with total epoch of {}.".format(config.num_epochs))
    best_psnr = 0

    for epoch in range(config.num_epochs):
        under_count = 0
        over_count = 0
        losses, PSNRs= [], []
        bright_losses=[]
        dark_losses=[]
        if(epoch<=11):
            F_optim_scheduler.step()
            continue
            
        for i, (ldr, ldr_b1, ldr_b2, ldr_d1, ldr_d2, hdr) in enumerate(train_loader):
            Darker.eval()
            Brighter.eval()
            F.train()
            # over-/under-exposed images
            ldr_numpy = torch.squeeze(ldr,0)
            ldr_numpy = ldr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            ldr_numpy = denormalize(ldr_numpy)
            gray = cv2.cvtColor(ldr_numpy, cv2.COLOR_RGB2GRAY)
            gray = gray.flatten()*255
            over_exposed_pix = thres(gray, 249, False)
            under_exposed_pix = thres(gray, 6, True)
            if(over_exposed_pix>len(gray)*0.25):
                over_count += 1
                continue
            if(under_exposed_pix>len(gray)*0.25):
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
            ldr = ldr.to(device)
            ldr_b1 = ldr_b1.to(device)
            ldr_b2 = ldr_b2.to(device)
            ldr_d1 = ldr_d1.to(device)
            ldr_d2 = ldr_d2.to(device)
            hdr = hdr.to(device)

            # Initialize Optimizers #
            Darker_optim.zero_grad()
            Brighter_optim.zero_grad()

            #####################
            # train first stage #
            #####################

            if(contribution == 0): #under exposure case
                dark = ldr
                # produce normal and over
                normal = Brighter(dark)
                G_loss_b2 = l1(ldr_b2, normal)
                bright = Brighter(normal)
                G_loss_b1 = l1(ldr_b1, bright)
                G_loss = (G_loss_b2 + G_loss_b1)/2
                G_loss.backward()
                Brighter_optim.step()
                bright_losses.append(G_loss.item())

            elif(contribution == 1): #normal exposure case
                normal = ldr
                # produce over and under
                dark = Darker(normal)
                G_loss_d1 = l1(ldr_d1, dark)
                G_loss_d1.backward()
                Darker_optim.step()
                dark_losses.append(G_loss_d1.item())

                bright = Brighter(normal)
                G_loss_b2 = l1(ldr_b2, bright)
                G_loss_b2.backward()
                Brighter_optim.step()
                bright_losses.append(G_loss_b2.item())

            elif(contribution == 2): #over exposure case
                bright = ldr
                # produce normal and under
                normal = Darker(bright)
                G_loss_d1 = l1(ldr_d1, normal)
                dark = Darker(normal)
                G_loss_d2 = l1(ldr_d2, dark)
                G_loss = (G_loss_d1 + G_loss_d2)/2
                G_loss.backward()
                Darker_optim.step()
                dark_losses.append(G_loss.item())
            
            #################
            # adjust images #
            #################

            # denormalize 
            dark = torch.add(torch.mul(dark,0.5),0.5)
            normal = torch.add(torch.mul(normal,0.5),0.5)
            bright = torch.add(torch.mul(bright,0.5),0.5)
            if(i%500==0):
                writer.add_image("image/train/dark", dark, global_step=i/500, dataformats='NCHW')
                writer.add_image("image/train/normal", normal, global_step=i/500, dataformats='NCHW')
                writer.add_image("image/train/bright", bright, global_step=i/500, dataformats='NCHW')

            ################
            # train SamNet #
            ################

            if(contribution == 0): #under exposure case
                dark = ldr
                # produce normal and over
                normal = Brighter(dark)
                bright = Brighter(normal)

            elif(contribution == 1): #normal exposure case
                normal = ldr
                # produce over and under
                dark = Darker(normal)
                bright = Brighter(normal)

            elif(contribution == 2): #over exposure case
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
            
            # Initialize Optimizers #
            Darker_optim.zero_grad()
            Brighter_optim.zero_grad()
            F_optim.zero_grad()

            pred_hdr = F(dark, normal, bright)
            loss = mu2_loss(pred_hdr, hdr)
            loss.backward()
            Darker_optim.step()
            Brighter_optim.step()
            F_optim.step()
            losses.append(loss.item())

            pred_hdr_numpy = torch.squeeze(pred_hdr,0)
            pred_hdr_numpy = pred_hdr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            hdr_numpy = torch.squeeze(hdr,0)
            hdr_numpy = hdr_numpy.detach().cpu().numpy().transpose((1, 2, 0))
            train_psnr = psnr_tanh_norm_mu_tonemap(pred_hdr_numpy,hdr_numpy)
            PSNRs.append(train_psnr)

            if(i%500==0):
                pred_hdr_numpy = np.clip(mu_tonemap(pred_hdr_numpy), 0, 1)
                hdr_numpy = np.clip(mu_tonemap(hdr_numpy), 0, 1)
                writer.add_image("image/train/zpred", pred_hdr_numpy, global_step=i/500, dataformats='HWC')
                writer.add_image("image/train/zgt", hdr_numpy, global_step=i/500, dataformats='HWC')
    
            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("CycleGAN | Epochs [{}/{}] | Iterations [{}/{}] | G Dark Loss {:.4f} | G Bright Loss {:.4f}"
                    .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(dark_losses), np.average(bright_losses)))
                print("SamNet | Epochs [{}/{}] | Iterations [{}/{}] | G Loss {:.4f} | PSNR {:.4f}"
                    .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(losses), np.average(PSNRs)))
        
        writer.add_scalar('train/Dark_loss', np.average(dark_losses), epoch)
        writer.add_scalar('train/Bright_loss', np.average(bright_losses), epoch)
        writer.add_scalar('train/Fusion_loss', np.average(losses), epoch)
        print("Epochs [{}/{}] has {} over exposure images".format(epoch+1, config.num_epochs, over_count))
        print("Epochs [{}/{}] has {} under exposure images".format(epoch+1, config.num_epochs, under_count))
        # Adjust Learning Rate
        Darker_optim_scheduler.step()
        Brighter_optim_optim_scheduler.step()
        F_optim_scheduler.step()

        psnr = test(vali_loader, Darker, Brighter, F, epoch)

        # Save Model Weights
        if psnr>best_psnr:
            best_psnr = psnr
            torch.save(Darker.state_dict(), os.path.join(config.weights_path, 'Darkerbest_single.pkl'))
            torch.save(Brighter.state_dict(), os.path.join(config.weights_path, 'Brighterbest_single.pkl'))
            torch.save(F.state_dict(), os.path.join(config.weights_path, 'SamNetbest_single.pkl'))

    # Make a GIF file #
    #make_gifs_train('Pix2Pix', config.samples_path)

    # Plot Losses #
    #plot_losses(D_losses, G_losses, config.plots_path)

    print("Training finished.")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if(1):
        valiset = get_vali_dataset()
        vali_loader = DataLoader(valiset, batch_size = 1, shuffle = False)
        # Prepare Networks #
        #F = Sam_Net().to(device)
        Darker = AttU_Net().to(device)
        Brighter = AttU_Net().to(device)
        # Initial weight
        Darker.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_A2B_CGAN_SC.pkl')))
        Brighter.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_B2A_CGAN_SC.pkl')))
        #F.load_state_dict(torch.load(os.path.join(config.weights_path, 'SamNetbest_joint_23.90.pkl')))
        test(vali_loader, Darker, Brighter, epoch=0)
    else:
        train()