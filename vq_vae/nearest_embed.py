import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand so it broadcastable
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.argmin = argmin
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            grad_emb = Variable(grad_output.data.new(ctx.emb_dim, ctx.num_emb).type(ctx.input_type).zero_())
            grad_output_reshaped = grad_output.permute(0, *ctx.dims[2:], 1).contiguous().view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            argmin_flat = ctx.argmin.view(-1)
            # TODO: replace for loop, use masked operations
            for i in range(ctx.num_emb):
                if torch.sum(ctx.argmin == i):
                    indices_chosen_atom_i = torch.nonzero(argmin_flat == i)
                    gradients_of_latents_chosen_atom_i = grad_output_reshaped[indices_chosen_atom_i, :]
                    grad_emb[:, i] = torch.mean(gradients_of_latents_chosen_atom_i, 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)
