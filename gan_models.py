"""Generador y discriminador de una GAN condicional (cDCGAN) para imágenes 64x64 RGB."""
import torch
import torch.nn as nn

NZ = 100        # tamaño del ruido z (espacio latente)
IMG_SIZE = 64


class Generator(nn.Module):
    """G(z, y): ruido + clase -> imagen falsa en [-1, 1]. Nunca ve datos reales."""

    def __init__(self, n_classes, nz=NZ, ngf=64, emb=50):
        super().__init__()
        self.nz, self.n_classes = nz, n_classes
        self.embed = nn.Embedding(n_classes, emb)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + emb, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),   # 8x8
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),   # 16x16
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),       # 32x32
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),             # 64x64
            nn.Tanh(),
        )

    def forward(self, z, y):
        h = torch.cat([z, self.embed(y)], dim=1)
        return self.net(h.unsqueeze(-1).unsqueeze(-1))


class Discriminator(nn.Module):
    """D(x, y): imagen + clase -> logit de 'es real'. sigmoid(logit) = D(x) del paper."""

    def __init__(self, n_classes, ndf=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, IMG_SIZE * IMG_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),                      # 32x32
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),                # 16x16
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),            # 8x8
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),            # 4x4
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),                  # 1x1
        )

    def forward(self, x, y):
        mapa = self.embed(y).view(-1, 1, IMG_SIZE, IMG_SIZE)  # la clase entra como un canal extra
        return self.net(torch.cat([x, mapa], dim=1)).view(-1)
