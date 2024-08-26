import os

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import torchvision.utils as vutils

import wandb

import time

 

# 初始化 wandb

wandb.init(project="dcgan-cifar100&10",name="gan-style2")

 

# 数据预处理与加载

def get_dataloader(image_size=64, batch_size=128):

    transform = transforms.Compose([

        transforms.Resize(image_size),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

    dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

 

# 模型定义

class DCGAN(nn.Module):

    def __init__(self, nz, ngf, ndf, nc):

        super(DCGAN, self).__init__()

        self.generator = self.build_generator(nz, ngf, nc)

        self.discriminator = self.build_discriminator(ndf, nc)

 

    def build_generator(self, nz, ngf, nc):

        return nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(ngf * 8),

            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 2),

            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf),

            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),

            nn.Tanh()

        )

 

    def build_discriminator(self, ndf, nc):

        return nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 2),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 4),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()

        )

 

    def initialize_weights(self):

        for m in self.modules():

            classname = m.__class__.__name__

            if classname.find('Conv') != -1:

                nn.init.normal_(m.weight.data, 0.0, 0.02)

            elif classname.find('BatchNorm') != -1:

                nn.init.normal_(m.weight.data, 1.0, 0.02)

                nn.init.constant_(m.bias.data, 0)

 

# 模型训练

def train_dcgan(model, dataloader, num_epochs=100, lr=0.0002, beta1=0.5, device="cuda:5"):

    criterion = nn.BCELoss().to(device)

    optimizerD = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerG = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))

 

    real_label = 1.0

    fake_label = 0.0

 

    os.makedirs('results', exist_ok=True)

    fixed_noise = torch.randn(64, model.generator[0].in_channels, 1, 1, device=device)

   

    model.to(device)

    model.initialize_weights()

 

    start_time = time.time()

 

    for epoch in range(num_epochs):

        for i, data in enumerate(dataloader, 0):

            # 更新判别器

            model.discriminator.zero_grad()

            real_data = data[0].to(device)

            batch_size = real_data.size(0)

            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            output = model.discriminator(real_data).view(-1)

            lossD_real = criterion(output, label)

            lossD_real.backward()

            D_x = output.mean().item()

 

            noise = torch.randn(batch_size, model.generator[0].in_channels, 1, 1, device=device)

            fake_data = model.generator(noise)

            label.fill_(fake_label)

            output = model.discriminator(fake_data.detach()).view(-1)

            lossD_fake = criterion(output, label)

            lossD_fake.backward()

            D_G_z1 = output.mean().item()

            optimizerD.step()

 

            # 更新生成器

            model.generator.zero_grad()

            label.fill_(real_label)

            output = model.discriminator(fake_data).view(-1)

            lossG = criterion(output, label)

            lossG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

 

            if i % 50 == 0:

                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD_real.item() + lossD_fake.item()} Loss_G: {lossG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

 

        wandb.log({

            "epoch": epoch + 1,

            "Loss_D": lossD_real.item() + lossD_fake.item(),

            "Loss_G": lossG.item(),

            "D(x)": D_x,

            "D(G(z1))": D_G_z1,

            "D(G(z2))": D_G_z2

        })

 

        vutils.save_image(real_data, f'results/real_samples_epoch_{epoch}.png', normalize=True)

        fake = model.generator(fixed_noise)

        vutils.save_image(fake.detach(), f'results/fake_samples_epoch_{epoch}.png', normalize=True)

 

    print(f'Training finished. Total Time: {time.time() - start_time:.2f} seconds')

    wandb.log({"training_time": time.time() - start_time})

    torch.save(model.generator.state_dict(), './dcgan_generator.pth')

    torch.save(model.discriminator.state_dict(), './dcgan_discriminator.pth')

    wandb.finish()

 

# 主程序入口

if __name__ == "__main__":

    dataloader = get_dataloader()

    dcgan_model = DCGAN(nz=100, ngf=64, ndf=64, nc=3)

    train_dcgan(dcgan_model, dataloader)