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


class VQ_VAE(nn.Module):
    def __init__(self):
        super(VQ_VAE, self).__init__()

        self.emb_size = 10
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = NearestEmbed(8, self.emb_size)

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


def loss_function(recon_x, x, z_e, emb):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    vq_loss = F.mse_loss(emb, z_e.detach())
    commitment_loss = F.mse_loss(z_e, emb.detach())

    return bce, vq_loss, commitment_loss


def train(args, model, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, z_e, emb = model(data)
        bce, vq_loss, commitment_loss = loss_function(recon_batch, data, z_e, emb)
        loss = 5*bce + vq_loss + 3 * commitment_loss
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbce: {:.6f}\tvq_loss: {:.6f}\tcommitment_loss: {:.6f}'.
                  format(epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader),
                         loss.data[0] / len(data), vq_loss.data[0] / len(data), commitment_loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(args, model, test_loader, epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, z_e, emb = model(data)
        bce, vq_loss, commitment_loss = loss_function(recon_batch, data, z_e, emb)
        test_loss += (bce + vq_loss + 2 * commitment_loss).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VQ_VAE()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch)
        test(args, model, test_loader, epoch)

        # Display the decoding of all the embeddings
        # Decoding is deterministic
        # num_emb = model.emb.weight.size(1)
        # sample = model.decode(model.emb.weight.t()).cpu()
        # save_image(sample.data.view(num_emb, 1, 28, 28),
        #            'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
