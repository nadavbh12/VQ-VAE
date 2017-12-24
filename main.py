from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Function, Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class NearestEmbed(Function):
    """
    Input:
    ------
    x - (batch_size, emb_size)
    emb - (emb_size, num_emb)
    """

    @staticmethod
    def forward(ctx, input, emb, dim):
        batch_size = input.size(0)
        emb_size = input.size(1)
        num_emb = emb.size(1)
        ctx.num_emb = num_emb
        ctx.emb_size = emb_size
        ctx.dim = dim
        ctx.input_type = type(input)

        x_expanded = input.unsqueeze(-1).expand(batch_size, emb_size, num_emb)
        emb_expanded = emb.unsqueeze(0).expand(batch_size, emb_size, num_emb)
        dist = torch.sum((x_expanded - emb_expanded).pow(2), dim=1)

        _, argmin = dist.min(dim)

        result = emb.index_select(1, argmin).t()
        ctx.argmin = argmin
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            grad_emb = Variable(grad_output.data.new(ctx.num_emb, ctx.emb_size).type(ctx.input_type).zero_())
            for i in range(ctx.num_emb):
                grad_emb[i, :] = torch.mean(grad_output.index_select(0, (ctx.argmin == i).long()), 0)
        return grad_input, grad_emb, None


def nearest_embed(x, emb, dim=0):
    return NearestEmbed().apply(x, emb, dim)


class NearestEmbedModule(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbedModule, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_embeddings, embeddings_dim))

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size)
        """
        return nearest_embed(x, self.weight, dim=1)


class VQ_VAE(nn.Module):
    def __init__(self):
        super(VQ_VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = NearestEmbedModule(20, 32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q = self.emb(z_e)
        return self.decode(z_q), z_e, z_q


def loss_function(recon_x, x, z_e, emb):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    vq_loss = F.mse_loss(z_e.detach(), emb)
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
        loss = bce + vq_loss + 0.1 * commitment_loss
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbce: {:.6f}\tvq_loss: {:.6f}\tcommitment_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
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
        test_loss += (bce + vq_loss + 0.1 * commitment_loss).data[0]
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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch)
        test(args, model, test_loader, epoch)

        # Display the decoding of all the embeddings
        # Decoding is deterministic
        num_emb = model.emb.weight.size(1)
        sample = model.decode(model.emb.weight.t()).cpu()
        save_image(sample.data.view(num_emb, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
