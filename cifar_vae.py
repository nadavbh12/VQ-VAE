from __future__ import print_function
import os
import logging
import argparse

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.log import setup_logging_and_results


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return x + self.convs(x)


class VAE(nn.Module):
    def __init__(self, d):
        super(VAE, self).__init__()

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


def loss_function(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x)
    batch_size = x.size(0)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    kld /= batch_size * 3 * 1024

    return mse + 0.1 * kld


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.data[0] / len(data)))
        if batch_idx == (len(train_loader) - 1):
            n = min(data.size(0), 8)
            batch_size = data.size(0)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(batch_size, 3, 32, 32)[:n]])
            save_image(comparison.data.cpu(),
                       os.path.join(save_path, 'reconstruction_train_' + str(epoch) + '.png'), nrow=n)

    avg_train_loss = train_loss / len(train_loader.dataset)
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_train_loss))
    return avg_train_loss


def test_net(epoch, model, test_loader, cuda, save_path):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            batch_size = data.size(0)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(batch_size, 3, 32, 32)[:n]])
            save_image(comparison.data.cpu(),
                       os.path.join(save_path, 'reconstruction_test_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    logging.info('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def main():
    parser = argparse.ArgumentParser(description='VAE CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=5e-3,
                              help='learning rate (default: 2e-4)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                        help='results dir')
    parser.add_argument('--data-format', default='json',
                        help='in which format to save the data')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    results, save_path = setup_logging_and_results(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = VAE(256)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path)
        test_loss = test_net(epoch, model, test_loader, args.cuda, save_path)
        sample = Variable(torch.randn(64, 128*6**2), requires_grad=False)
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 3, 32, 32),
                   os.path.join(save_path, 'sample_' + str(epoch) + '.png'))
        results.add(epoch=epoch, train_loss=train_loss, test_loss=test_loss)
        results.plot(x='epoch', y=['train_loss', 'test_loss'])
        results.save()


if __name__ == "__main__":
    main()
