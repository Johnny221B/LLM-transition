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

wandb.init(project="dcgan-cifar100&10",name="gan-style1")

 

# 配置参数

config = {

    "image_size": 64,

    "batch_size": 128,

    "num_epochs": 100,

    "learning_rate": 0.0002,

    "beta1": 0.5,

    "nz": 100,  # Size of z latent vector (i.e. size of generator input)

    "ngf": 64,  # Size of feature maps in generator

    "ndf": 64,  # Size of feature maps in discriminator

    "device": torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),

    "real_label": 1.0,

    "fake_label": 0.0,

    "results_dir": "results"

}

 

# 数据预处理和加载

def get_dataloader(config):

    transform = transforms.Compose([

        transforms.Resize(config["image_size"]),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

 

    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    return dataloader

 

# 定义生成器模型

class Generator(nn.Module):

    def __init__(self, config):

        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(config["nz"], config["ngf"] * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(config["ngf"] * 8),

            nn.ReLU(True),

            nn.ConvTranspose2d(config["ngf"] * 8, config["ngf"] * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ngf"] * 4),

            nn.ReLU(True),

            nn.ConvTranspose2d(config["ngf"] * 4, config["ngf"] * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ngf"] * 2),

            nn.ReLU(True),

            nn.ConvTranspose2d(config["ngf"] * 2, config["ngf"], 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ngf"]),

            nn.ReLU(True),

            nn.ConvTranspose2d(config["ngf"], 3, 4, 2, 1, bias=False),

            nn.Tanh()

        )

 

    def forward(self, x):

        return self.main(x)

 

# 定义判别器模型

class Discriminator(nn.Module):

    def __init__(self, config):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(3, config["ndf"], 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(config["ndf"], config["ndf"] * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ndf"] * 2),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(config["ndf"] * 2, config["ndf"] * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ndf"] * 4),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(config["ndf"] * 4, config["ndf"] * 8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(config["ndf"] * 8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(config["ndf"] * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()

        )

 

    def forward(self, x):

        return self.main(x).view(-1)

 

# 初始化模型和优化器

def initialize_models(config):

    netG = Generator(config).to(config["device"])

    netD = Discriminator(config).to(config["device"])

 

    netG.apply(weights_init)

    netD.apply(weights_init)

 

    optimizerD = optim.Adam(netD.parameters(), lr=config["learning_rate"], betas=(config["beta1"], 0.999))

    optimizerG = optim.Adam(netG.parameters(), lr=config["learning_rate"], betas=(config["beta1"], 0.999))

 

    return netG, netD, optimizerD, optimizerG

 

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)

 

# 训练过程

def train_dcgan(dataloader, netG, netD, optimizerD, optimizerG, config):

    criterion = nn.BCELoss().to(config["device"])

    real_label = config["real_label"]

    fake_label = config["fake_label"]

 

    os.makedirs(config["results_dir"], exist_ok=True)

 

    start_time = time.time()

   

    for epoch in range(config["num_epochs"]):

        for i, data in enumerate(dataloader, 0):

            # 更新判别器

            netD.zero_grad()

            real_cpu = data[0].to(config["device"])

            b_size = real_cpu.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=config["device"])

            output = netD(real_cpu)

            errD_real = criterion(output, label)

            errD_real.backward()

            D_x = output.mean().item()

 

            noise = torch.randn(b_size, config["nz"], 1, 1, device=config["device"])

            fake = netG(noise)

            label.fill_(fake_label)

            output = netD(fake.detach())

            errD_fake = criterion(output, label)

            errD_fake.backward()

            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

 

            # 更新生成器

            netG.zero_grad()

            label.fill_(real_label)

            output = netD(fake)

            errG = criterion(output, label)

            errG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

 

            if i % 50 == 0:

                print(f'[{epoch + 1}/{config["num_epochs"]}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

 

        # 保存和记录

        wandb.log({

            "epoch": epoch + 1,

            "Loss_D": errD.item(),

            "Loss_G": errG.item(),

            "D(x)": D_x,

            "D(G(z1))": D_G_z1,

            "D(G(z2))": D_G_z2

        })

 

        fake = netG(torch.randn(config["batch_size"], config["nz"], 1, 1, device=config["device"]))

        vutils.save_image(real_cpu, f'{config["results_dir"]}/real_samples_epoch_{epoch + 1}.png', normalize=True)

        vutils.save_image(fake.detach(), f'{config["results_dir"]}/fake_samples_epoch_{epoch + 1}.png', normalize=True)

 

    end_time = time.time()

    training_time = end_time - start_time

    print(f'Training Time: {training_time:.2f} seconds')

 

    wandb.log({"training_time": training_time})

    torch.save(netG.state_dict(), os.path.join(config["results_dir"], 'dcgan_generator.pth'))

    torch.save(netD.state_dict(), os.path.join(config["results_dir"], 'dcgan_discriminator.pth'))

    wandb.finish()

 

# 主程序

def main():

    dataloader = get_dataloader(config)

    netG, netD, optimizerD, optimizerG = initialize_models(config)

    train_dcgan(dataloader, netG, netD, optimizerD, optimizerG, config)

 

if __name__ == "__main__":

    main()