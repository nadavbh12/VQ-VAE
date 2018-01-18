from __future__ import print_function
import os
import logging
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.log import setup_logging_and_results
from auto_encoder import *

models = {'cifar10': {'vae': CVAE,
                      'vqvae': VQ_CVAE},
          'mnist': {'vae': VAE,
                    'vqvae': VQ_VAE}}
datasets_classes = {'cifar10': datasets.CIFAR10,
                    'mnist': datasets.MNIST}
dataset_sizes = {'cifar10': (3, 32, 32),
                 'mnist': (1, 28, 28)}


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader),
                         loss.data[0] / len(data)))
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(data, epoch, outputs, save_path, 'reconstruction_train')

    avg_train_loss = train_loss / len(train_loader.dataset)
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_train_loss))
    return avg_train_loss


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs[0].view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.data.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n)


def test_net(epoch, model, test_loader, cuda, save_path):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        outputs = model(data)
        test_loss += model.loss_function(data, *outputs).data[0]
        if i == 0:
            save_reconstructed_images(data, epoch, outputs, save_path, 'reconstruction_test')

    test_loss /= len(test_loader.dataset)
    logging.info('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def main():
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--hidden', type=int, default=256, metavar='N',
                        help='number of hidden channels (default: 256)')
    parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'],
                        help='autoencoder variant to use: vae | vqvae')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'],
                        help='dataset to use: mnist | cifar10')
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
    parser.add_argument('--gpus', default='1',
                        help='gpus used for training - e.g 0,1,3')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    results, save_path = setup_logging_and_results(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](args.hidden)
    if args.cuda:
        model.cuda()

    if args.dataset == 'cifar10':
        lr = 5e-4
    elif args.dataset == 'mnist':
        lr = 1e-4
    else:
        raise ValueError('invalid dataset')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset]('../data', train=True, download=True,
                                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset]('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path)
        test_loss = test_net(epoch, model, test_loader, args.cuda, save_path)
        sample = model.sample(32)
        save_image(sample.data.view(32, *dataset_sizes[args.dataset]),
                   os.path.join(save_path, 'sample_' + str(epoch) + '.png'))
        results.add(epoch=epoch, train_loss=train_loss, test_loss=test_loss)
        results.plot(x='epoch', y=['train_loss', 'test_loss'])
        results.save()


if __name__ == "__main__":
    main()
