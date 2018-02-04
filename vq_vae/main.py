import os
import sys
import logging
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.log import setup_logging_and_results
from vq_vae.auto_encoder import *

models = {'cifar10': {'vae': CVAE,
                      'vqvae': VQ_CVAE},
          'mnist': {'vae': VAE,
                    'vqvae': VQ_VAE}}
datasets_classes = {'cifar10': datasets.CIFAR10,
                    'mnist': datasets.MNIST}
dataset_sizes = {'cifar10': (3, 32, 32),
                 'mnist': (1, 28, 28)}
default_hyperparams = {'cifar10': {'lr': 2e-4, 'k': 10},
                       'mnist': {'lr': 1e-4}, 'k': 10}


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += latest_losses[key].data[0]
            epoch_losses[key + '_train'] += latest_losses[key].data[0]

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {} [{:5d}/{} ({:2d}%)]\t{}'.format(
                         epoch, batch_idx * len(data), len(train_loader.dataset),
                         int(100. * batch_idx / len(train_loader)),
                         loss_string))
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')

    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader.dataset)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    return epoch_losses


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.data.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n)


def test_net(epoch, model, test_loader, cuda, save_path):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    for i, (data, _) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        outputs = model(data)
        model.loss_function(data, *outputs)
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_test'] += latest_losses[key].data[0]
        if i == 0:
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')

    for key in losses:
        losses[key] /= len(test_loader.dataset)
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'],
                              help='autoencoder variant to use: vae | vqvae')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, default=256, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'],
                                 help='dataset to use: mnist | cifar10')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']

    results, save_path = setup_logging_and_results(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](args.hidden, k=k)
    if args.cuda:
        model.cuda()

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
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path)
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path)
        sample = model.sample(32)
        save_image(sample.data.view(32, *dataset_sizes[args.dataset]),
                   os.path.join(save_path, 'sample_' + str(epoch) + '.png'))

        results.add(epoch=epoch, **train_losses, **test_losses)
        for k in train_losses:
            key = k[:-6]
            results.plot(x='epoch', y=[key + '_train', key + '_test'])
        results.save()


if __name__ == "__main__":
    main(sys.argv[1:])
