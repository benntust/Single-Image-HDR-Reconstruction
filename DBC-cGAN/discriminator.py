import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.ndf = 32
        self.out_channels = 1

        '''model = [
            nn.Conv2d(self.in_channels*2, self.ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        n_blocks = 3

        for i in range(n_blocks):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ndf * mult * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [
            nn.Conv2d(self.ndf * mult * 2, self.out_channels, kernel_size=4, stride=1, padding=1, bias=True), 
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)'''

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.ndf, 3, 1, 'same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # 256 x 256

            nn.Conv2d(self.ndf, self.ndf*2, 3, 1, 'same', bias=False),
            #nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # 128 x 128

            nn.Conv2d(self.ndf*2, self.ndf*4, 3, 1, 'same', bias=False),
            #nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # 64 x 64

            nn.Conv2d(self.ndf*4, self.ndf*8, 3, 1, 'same', bias=False),
            #nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # 32 x 32

            #nn.Conv2d(self.ndf*8, self.ndf*8, 4, 1, 1, bias=False),
            #nn.BatchNorm2d(self.ndf*8),
            #nn.LeakyReLU(0.2, inplace=True),
            # 31 x 31

            nn.Conv2d(self.ndf*8, self.out_channels, 3, 1, 'same', bias=False),
            # 30 x 30 (PatchGAN)
            nn.Sigmoid()
        )

    def forward(self, x, label):
        out = torch.cat((x, label), dim=1) # concat input and G result
        out = self.main(out)
        return out