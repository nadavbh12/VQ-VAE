from __future__ import print_function
import argparse

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from nearest_embed import NearestEmbed


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


class VQ_VAE(nn.Module):
    def __init__(self, d):
        super(VQ_VAE, self).__init__()

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
            nn.Sigmoid(),
        )
        self.f = 6
        self.d = d
        self.emb = NearestEmbed(10, d)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.emb(z_e)
        return self.decoder(z_q), z_e, z_q


def loss_function(recon_x, x, z_e, emb):
    mse = F.mse_loss(recon_x, x)

    vq_loss = F.mse_loss(emb, z_e.detach())
    commitment_loss = F.mse_loss(z_e, emb.detach())

    return mse, vq_loss, commitment_loss


def train(epoch, model, train_loader, optimizer, cuda, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, z_e, emb = model(data)
        mse, vq_loss, commitment_loss = loss_function(recon_batch, data, z_e, emb)
        loss = 5*mse + vq_loss + 3 * commitment_loss
        loss.backward(retain_graph=True)
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbce: {:.6f}\tvq_loss: {:.6f}\tcommitment_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.data[0] / len(data), vq_loss.data[0] / len(data), commitment_loss.data[0] / len(data)))
        if batch_idx == (len(train_loader) - 1):
            n = min(data.size(0), 8)
            batch_size = data.size(0)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(batch_size, 3, 32, 32)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_train_' + str(epoch) + '.png', nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test_net(epoch, model, test_loader, cuda):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, z_e, emb = model(data)
        mse, vq_loss, commitment_loss = loss_function(recon_batch, data, z_e, emb)
        test_loss += (mse + vq_loss + 0.25 * commitment_loss).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            batch_size = data.size(0)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(batch_size, 3, 32, 32)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_test_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    parser = argparse.ArgumentParser(description='VAE CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = VQ_VAE(256)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval)
        test_net(epoch, model, test_loader, args.cuda)
        # sample = Variable(torch.randn(64, 256, 6, 6), requires_grad=False)
        # if args.cuda:
        #     sample = sample.cuda()
        # sample = model.decoder(sample).cpu()
        # save_image(sample.data.view(64, 3, 32, 32),
        #            'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
