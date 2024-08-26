import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
import time

wandb.init(project="dcgan-cifar100&10")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

netG = Generator()
netD = Discriminator()

netG.apply(lambda m: m.__class__.__name__ == 'ConvTranspose2d' and nn.init.normal_(m.weight.data, 0.0, 0.02) or (m.__class__.__name__ == 'BatchNorm2d' and nn.init.normal_(m.weight.data, 1.0, 0.02) and nn.init.constant_(m.bias.data, 0)))
netD.apply(lambda m: m.__class__.__name__ == 'Conv2d' and nn.init.normal_(m.weight.data, 0.0, 0.02) or (m.__class__.__name__ == 'BatchNorm2d' and nn.init.normal_(m.weight.data, 1.0, 0.02) and nn.init.constant_(m.bias.data, 0)))

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

os.makedirs('results', exist_ok=True)

num_epochs = 100
real_label = 1.
fake_label = 0.

start_time = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        real_label_tensor = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        fake_label_tensor = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        netD.zero_grad()
        output = netD(real_cpu).view(-1)
        errD_real = nn.BCELoss()(output, real_label_tensor)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach()).view(-1)
        errD_fake = nn.BCELoss()(output, fake_label_tensor)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        optimizerD.step()

        netG.zero_grad()
        output = netD(fake).view(-1)
        errG = nn.BCELoss()(output, real_label_tensor)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            wandb.log({
                "epoch": epoch,
                "Loss_D": errD_real.item() + errD_fake.item(),
                "Loss_G": errG.item(),
                "D(x)": D_x,
                "D(G(z1))": D_G_z1,
                "D(G(z2))": D_G_z2
            })
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD_real.item() + errD_fake.item()} Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

end_time = time.time()
training_time = end_time - start_time
print('Training finished. Total time: {:.2f} seconds'.format(training_time))
wandb.finish()

torch.save(netG.state_dict(), './generator.pth')
torch.save(netD.state_dict(), './discriminator.pth')