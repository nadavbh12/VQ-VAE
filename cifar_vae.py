from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


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
    batch_size=args.batch_size, shuffle=True, **kwargs)


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
        # x_feature_size = x.size()[-1]
        # residual = F.avg_pool2d(x, 3, 1)
        # x = self.convs(x)
        # return x + residual
        return x + self.convs(x)


# class ResBlockTranspose(nn.Module):
#     def __init__(self, in_channels, channels):
#         super(ResBlockTranspose, self).__init__()
#
#         self.convs = nn.Sequential(
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels, channels, kernel_size=1, stride=1, padding=1),
#         )
#         self.upsample = nn.Upsample()
#
#     def forward(self, x):
#         x_size = (x.size(2) + 2, x.size(3) + 2)
#         residual = F.upsample(x, size=x_size)
#         x = self.convs(x)
#         return x + residual


class VAE(nn.Module):
    def __init__(self, d):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d//2, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d//2, d, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            ResBlock(d, d),
            ResBlock(d, d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d//2, kernel_size=4, stride=2),
            nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d//2, 3, kernel_size=4, stride=2),
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


model = VAE(256)
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 3 * 1024

    return MSE + 0.2*KLD


optimizer = optim.Adam(model.parameters(), lr=2e-4)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
        if batch_idx == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_train_' + str(epoch) + '.png', nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test_net(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_test_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_net(epoch)
    sample = Variable(torch.randn(64, 9216), requires_grad=False)
    if args.cuda:
       sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 3, 32, 32),
               'results/sample_' + str(epoch) + '.png')