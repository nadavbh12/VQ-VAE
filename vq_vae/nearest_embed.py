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

        # find nearest neighbors
        x_reshaped = input.permute(0, 2, 3, 1).contiguous().view(ctx.batch_size * ctx.num_latents, ctx.emb_dim, 1)
        x_expanded = x_reshaped.expand(ctx.batch_size * ctx.num_latents, ctx.emb_dim, ctx.num_emb)
        emb_expanded = emb.unsqueeze(0).expand(ctx.batch_size * ctx.num_latents, ctx.emb_dim, ctx.num_emb)

        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(1)
        result = emb.index_select(1, argmin).t()

        ctx.argmin = argmin
        return result.contiguous().view(ctx.batch_size, *input.size()[2:], input.size(1)).permute(0, 3, 1, 2).contiguous(), \
               argmin.view(ctx.batch_size, *input.size()[2:])

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            grad_emb = Variable(grad_output.data.new(ctx.emb_dim, ctx.num_emb).type(ctx.input_type).zero_())
            grad_output_reshaped = grad_output.contiguous().view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            # TODO: replace for loop
            for i in range(ctx.num_emb):
                if torch.sum(ctx.argmin == i):
                    grad_emb[:, i] = torch.mean(grad_output_reshaped[ctx.argmin[ctx.argmin == i], :], 0)
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
