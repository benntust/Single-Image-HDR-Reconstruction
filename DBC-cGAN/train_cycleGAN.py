import os
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

writer = SummaryWriter() 

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

            dark = dark.to(device)
            bright = bright.to(device)

            if(not bright_skip):
                fake_dark = G_A2B(bright)
                A2B_loss = criterion_Pixelwise(fake_dark, dark) #+ config.edge_lambda * edge_loss(fake_dark, dark)
                A2B_losses.append(A2B_loss.item())
                fake_dark_numpy = np.squeeze(fake_dark,0)
                fake_dark_numpy = fake_dark_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_dark_numpy = denormalize(fake_dark_numpy)
                psnr_A2B_loss = calculate_psnr(fake_dark_numpy,dark_numpy)
                psnr_A2B_losses.append(psnr_A2B_loss)
                if(i==750):
                    writer.add_image("image/dark", dark_numpy, global_step=epoch, dataformats='HWC')
                    writer.add_image("image/dark_fake", fake_dark_numpy, global_step=epoch, dataformats='HWC')
            
            if(not dark_skip):
                fake_bright = G_B2A(dark)
                B2A_loss = criterion_Pixelwise(fake_bright, bright) #+ config.edge_lambda * edge_loss(fake_bright, bright)
                B2A_losses.append(B2A_loss.item())
                fake_bright_numpy = np.squeeze(fake_bright,0)
                fake_bright_numpy = fake_bright_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_bright_numpy = denormalize(fake_bright_numpy)
                psnr_B2A_loss = calculate_psnr(fake_bright_numpy,bright_numpy)
                psnr_B2A_losses.append(psnr_B2A_loss)
                if(i==750):
                    writer.add_image("image/bright", bright_numpy, global_step=epoch, dataformats='HWC')
                    writer.add_image("image/bright_fake", fake_bright_numpy, global_step=epoch, dataformats='HWC')
                
        writer.add_scalar('test/PSNR_A2B', np.mean(psnr_A2B_losses), epoch)
        writer.add_scalar('test/G_A2B_loss', np.mean(A2B_losses), epoch)
        writer.add_scalar('test/PSNR_B2A', np.mean(psnr_B2A_losses), epoch)
        writer.add_scalar('test/G_B2A_loss', np.mean(B2A_losses), epoch)
        print("Test | Epochs [{}/{}] | G_A2B Loss {:.4f} PSNR {}"
                        .format(epoch+1, config.num_epochs, np.mean(A2B_losses), np.mean(psnr_A2B_losses)))
        print("Test | Epochs [{}/{}] | G_B2A Loss {:.4f} PSNR {}"
                        .format(epoch+1, config.num_epochs, np.mean(B2A_losses), np.mean(psnr_B2A_losses)))
        print("Test | Epochs [{}/{}] bright domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, bright_count))
        print("Test | Epochs [{}/{}] dark domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, dark_count))

    del bright, dark, fake_bright, fake_dark

    return np.mean(psnr_A2B_losses), np.mean(psnr_B2A_losses)

def train():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Samples, Weights and Results Path #
    paths = [config.samples_path, config.weights_path, config.plots_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    trainset = get_train_dataset()
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = True)
    valiset = get_vali_dataset()
    vali_loader = DataLoader(valiset, batch_size = 1, shuffle = False)
    total_batch = len(train_loader)
    print(total_batch)

    # Prepare Networks #
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    G_A2B = AttU_Net().to(device)
    G_B2A = AttU_Net().to(device)
    #print(summary(G,(3,512,512)))

    # Initial weight
    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_A2B_CGAN.pkl')))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'G_B2A_CGAN.pkl')))
    #init_weights(G_A2B,init_type='kaiming')
    #init_weights(G_B2A,init_type='kaiming')
    init_weights(D_A,init_type='kaiming')
    init_weights(D_B,init_type='kaiming')

    # Optimizers #
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(0.5, 0.999))
    GA2B_optim = torch.optim.Adam(G_A2B.parameters(), lr=config.lr, betas=(0.5, 0.999))
    GB2A_optim = torch.optim.Adam(G_B2A.parameters(), lr=config.lr, betas=(0.5, 0.999))

    D_A_optim_scheduler = get_lr_scheduler(D_A_optim)
    D_B_optim_scheduler = get_lr_scheduler(D_B_optim)
    GA2B_optim_scheduler = get_lr_scheduler(GA2B_optim) 
    GB2A_optim_scheduler = get_lr_scheduler(GB2A_optim)  

    # Training #
    print("Training CycleGAN started with total epoch of {}.".format(config.num_epochs))
    best_psnr_A2B = 0
    best_psnr_B2A = 0

    for epoch in range(config.num_epochs):
        bright_count = 0
        dark_count = 0
        D_A_losses, D_B_losses, GA2B_losses, GB2A_losses = [], [], [], []
        GA2B_D_losses, GB2A_D_losses = [], []
        
        for i, (bright, dark) in enumerate(train_loader):
            G_A2B.train()
            G_B2A.train()
            D_A.train()
            D_B.train()
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
            
            # Data Preparation #
            bright = bright.to(device)
            dark = dark.to(device)

            # Initialize Optimizers #
            D_A_optim.zero_grad()
            D_B_optim.zero_grad()
            #G_optim.zero_grad()
            GA2B_optim.zero_grad()
            GB2A_optim.zero_grad()

            ###################
            # Train Generator #
            ###################

            # Prevent Discriminator Update during Generator Update #
            set_requires_grad([D_A, D_B], requires_grad=False)
            set_requires_grad([G_B2A, G_A2B], requires_grad=True)
            
            if(not bright_skip):
                fake_dark = G_A2B(bright)
                prob_fake_dark = D_B(fake_dark, bright)
                # Adversarial Loss #
                real_labels = torch.ones(prob_fake_dark.size()).to(device) # G wants to produce real images
                G_mse_loss_A2B = criterion_Adversarial(prob_fake_dark, real_labels)
                # Reconstruction loss #
                G_loss_reconstruct_A2B = criterion_Pixelwise(fake_dark, dark) #+ config.edge_lambda * edge_loss(fake_dark, dark)
                '''# Cycle loss
                cycle_bright = G_B2A(fake_dark)
                G_loss_reconstruct_ABA = criterion_Pixelwise(cycle_bright, bright) + config.edge_lambda * edge_loss(cycle_bright, bright)'''
                # Total
                G_loss_A2B = G_mse_loss_A2B + config.l1_lambda * G_loss_reconstruct_A2B #+ G_loss_reconstruct_ABA)
                G_loss_A2B.backward()
                GA2B_optim.step()

                # Add items to Lists #
                #GA2B_D_losses.append(G_mse_loss_A2B.item())
                GA2B_losses.append(G_loss_reconstruct_A2B.item())

                fake_dark_numpy = np.squeeze(fake_dark,0)
                fake_dark_numpy = fake_dark_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_dark_numpy = denormalize(fake_dark_numpy)
                if(i%1000==0):
                    writer.add_image("image/train_dark", dark_numpy, global_step=i/1000, dataformats='HWC')
                    writer.add_image("image/train_dark_fake", fake_dark_numpy, global_step=i/1000, dataformats='HWC')
            
            if(not dark_skip):
                fake_bright = G_B2A(dark)
                prob_fake_bright = D_A(fake_bright, dark)
                # Adversarial Loss #
                real_labels = torch.ones(prob_fake_bright.size()).to(device)
                G_mse_loss_B2A = criterion_Adversarial(prob_fake_bright, real_labels)
                # Reconstruction loss #
                G_loss_reconstruct_B2A = criterion_Pixelwise(fake_bright, bright) #+ config.edge_lambda * edge_loss(fake_bright, bright)
                '''# Cycle loss
                cycle_dark = G_A2B(fake_bright)
                G_loss_reconstruct_BAB = criterion_Pixelwise(cycle_dark, dark) + config.edge_lambda * edge_loss(cycle_dark, dark)'''
                # Total
                G_loss_B2A = G_mse_loss_B2A + config.l1_lambda * G_loss_reconstruct_B2A #+ G_loss_reconstruct_BAB)
                G_loss_B2A.backward()
                GB2A_optim.step()

                # Add items to Lists #
                GB2A_losses.append(G_loss_reconstruct_B2A.item())
                #GB2A_D_losses.append(G_mse_loss_B2A.item())

                fake_bright_numpy = np.squeeze(fake_bright,0)
                fake_bright_numpy = fake_bright_numpy.detach().cpu().numpy().transpose((1, 2, 0))
                fake_bright_numpy = denormalize(fake_bright_numpy)
                if(i%1000==0):
                    writer.add_image("image/train_bright", bright_numpy, global_step=i/1000, dataformats='HWC')
                    writer.add_image("image/train_bright_fake", fake_bright_numpy, global_step=i/1000, dataformats='HWC')
            
            #######################
            # Train Discriminator #
            #######################

            # Prevent Discriminator Update during Generator Update #
            set_requires_grad([D_A, D_B], requires_grad=True)
            set_requires_grad([G_B2A, G_A2B], requires_grad=False)

            ## train D_A ##
            # Adversarial Loss #
            if(not dark_skip):
                # real
                prob_real = D_A(bright, dark)
                real_labels = torch.ones(prob_real.size()).to(device)
                D_real_loss = criterion_Adversarial(prob_real, real_labels)
                # fake
                fake_bright = G_B2A(dark)
                prob_fake = D_A(fake_bright.detach(), dark)
                fake_labels = torch.zeros(prob_fake.size()).to(device) # D wants to give images producing by G zeros
                D_fake_loss = criterion_Adversarial(prob_fake, fake_labels)

                # Calculate Total Discriminator Loss #
                D_A_loss = torch.mean(D_real_loss + D_fake_loss)

                # Back Propagation and Update #
                D_A_loss.backward()
                D_A_optim.step()

                D_A_losses.append(D_A_loss.item())

            ## train D_B ##
            # Adversarial Loss #
            if(not bright_skip):
                # real
                prob_real = D_B(dark, bright)
                real_labels = torch.ones(prob_real.size()).to(device)
                D_real_loss = criterion_Adversarial(prob_real, real_labels)
                # fake
                fake_dark = G_A2B(bright)
                prob_fake = D_B(fake_dark.detach(), bright)
                fake_labels = torch.zeros(prob_fake.size()).to(device) # D wants to give images producing by G zeros
                D_fake_loss = criterion_Adversarial(prob_fake, fake_labels)

                # Calculate Total Discriminator Loss #
                D_B_loss = torch.mean(D_real_loss + D_fake_loss)

                # Back Propagation and Update #
                D_B_loss.backward()
                D_B_optim.step()

                D_B_losses.append(D_B_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("CycleGAN | Epochs [{}/{}] | Iterations [{}/{}] | D_A Loss {:.4f} | D_B Loss {:.4f} | GA2B Loss {:.4f} | GB2A Loss {:.4f}| GA2B_D Loss {:.4f} | GB2A_D Loss {:.4f}"
                    .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_A_losses), np.average(D_B_losses), np.average(GA2B_losses), np.average(GB2A_losses), np.average(GA2B_D_losses), np.average(GB2A_D_losses)))
        
        writer.add_scalar('train/D/D_A_loss', np.average(D_A_losses), epoch)
        writer.add_scalar('train/D/D_B_loss', np.average(D_B_losses), epoch)
        writer.add_scalar('train/GA2B_loss', np.average(GA2B_losses), epoch)
        writer.add_scalar('train/GB2A_loss', np.average(GB2A_losses), epoch)

        print("Epochs [{}/{}] bright domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, bright_count))
        print("Epochs [{}/{}] dark domain has {} over-/under exposure images".format(epoch+1, config.num_epochs, dark_count))
        
        if(epoch % config.print_val_every_epoch==0):
            psnr_A2B, psnr_B2A = test(vali_loader, G_A2B, G_B2A, epoch)
            # Save Model Weights #
            if psnr_A2B > best_psnr_A2B:
                best_psnr_A2B = psnr_A2B
                torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'G_A2B_MAX_test.pkl'))
            if psnr_B2A > best_psnr_B2A:
                best_psnr_B2A = psnr_B2A
                torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'G_B2A_MAX_test.pkl'))

        # Adjust Learning Rate #
        D_A_optim_scheduler.step()
        D_B_optim_scheduler.step()
        GA2B_optim_scheduler.step()
        GB2A_optim_scheduler.step()
        

    print("Training finished.")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()