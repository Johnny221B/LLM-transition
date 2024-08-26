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

data_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.layer_stack = nn.Sequential(
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

    def forward(self, z):
        return self.layer_stack(z)

class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.layer_stack = nn.Sequential(
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

    def forward(self, x):
        return self.layer_stack(x).view(-1)

generator = Gen()
discriminator = Disc()

generator.apply(lambda m: m.__class__.__name__ == 'ConvTranspose2d' and nn.init.normal_(m.weight.data, 0.0, 0.02))
discriminator.apply(lambda m: m.__class__.__name__ == 'Conv2d' and nn.init.normal_(m.weight.data, 0.0, 0.02))

optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

os.makedirs('results', exist_ok=True)

num_epochs = 100
label_real = 1.
label_fake = 0.

start_time = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(data_loader, 0):
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        real_labels = torch.full((batch_size,), label_real, dtype=torch.float, device=device)

        discriminator.zero_grad()
        real_output = discriminator(real_images)
        err_real = nn.BCELoss()(real_output, real_labels)
        err_real.backward()
        real_score = real_output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size,), label_fake, dtype=torch.float, device=device)
        fake_output = discriminator(fake_images.detach())
        err_fake = nn.BCELoss()(fake_output, fake_labels)
        err_fake.backward()
        fake_score = fake_output.mean().item()
        optimizer_d.step()

        generator.zero_grad()
        output_g = discriminator(fake_images)
        err_g = nn.BCELoss()(output_g, real_labels)
        err_g.backward()
        optimizer_g.step()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(data_loader)}] Loss_D: {err_real.item() + err_fake.item()} Loss_G: {err_g.item()} D(x): {real_score} D(G(z)): {fake_score}')

    if epoch % 5 == 0:
        vutils.save_image(real_images, f'results/real_samples_epoch_{epoch}.png', normalize=True)
        vutils.save_image(fake_images.detach(), f'results/fake_samples_epoch_{epoch}.png', normalize=True)

end_time = time.time()
print(f'Training finished. Total time: {end_time - start_time:.2f} seconds')
wandb.finish()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
