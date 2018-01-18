from __future__ import print_function
import abc

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from nearest_embed import NearestEmbed


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @staticmethod
    @abc.abstractmethod
    def loss_function(**kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return


class VAE(nn.Module):
    """Variational AutoEncoder for MNIST
       Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae"""
    def __init__(self, *args):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = Variable(torch.randn(size, 20))
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        return sample


    @staticmethod
    def loss_function(x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= batch_size * 784

        return BCE + KLD


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""
    def __init__(self, *args):
        super(VQ_VAE, self).__init__()

        self.emb_size = 10
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = NearestEmbed(16, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(200 / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q = self.emb(z_e).view(-1, 200)
        return self.decode(z_q), z_e, z_q

    def sample(self, size):
        sample = Variable(torch.randn(size, self.emb_size, int(200 / self.emb_size)))
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(self.emb(sample).view(-1, 200)).cpu()
        return sample

    @staticmethod
    def loss_function(x, recon_x, z_e, emb):
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        vq_loss = F.mse_loss(emb, z_e.detach())
        commitment_loss = F.mse_loss(z_e, emb.detach())

        return bce + 0.2*vq_loss + 0.4*commitment_loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return x + self.convs(x)


class CVAE(AbstractAutoEncoder):
    def __init__(self, d):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            ResBlock(d, d),
            ResBlock(d, d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2),
            nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2),
        )
        self.f = 6
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return F.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = Variable(torch.randn(size, self.d * self.f ** 2), requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    @staticmethod
    def loss_function(x, recon_x, mu, logvar):
        mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        kld /= batch_size * 3 * 1024

        # return mse
        return mse + 0.1 * kld


class VQ_CVAE(nn.Module):
    def __init__(self, d):
        super(VQ_CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            ResBlock(d, d),
            ResBlock(d, d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2),
            nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2),
        )
        self.f = 6
        self.d = d
        self.emb = NearestEmbed(16, d)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return F.sigmoid(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        z_q = self.emb(z_e)
        return self.decode(z_q), z_e, z_q

    def sample(self, size):
        sample = Variable(torch.randn(size, self.d, self.f ** 2), requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(self.emb(sample).view(size, self.d, self.f, self.f)).cpu()

    @staticmethod
    def loss_function(x, recon_x, z_e, emb):
        mse = F.mse_loss(recon_x, x)

        vq_loss = F.mse_loss(emb, z_e.detach())
        commitment_loss = F.mse_loss(z_e, emb.detach())

        return mse + 0.2*vq_loss + 0.7*commitment_loss
