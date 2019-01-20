import torch

def maxout(input, pool_size):

    shape=list(input.shape())
    shape[-1] = shape[-1] // pool_size
    shape.append(pool_size)
    m, i = torch.max(input.view(*shape), -1)
    return m
