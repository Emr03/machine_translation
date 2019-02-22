import torch
import copy
import numpy as np

_D_MODEL = 512
_D_K = 64
_ATT_HEADS = 8
_DFF = 2048
_VOCAB = 30000

class FFNN(torch.nn.Module):

    def __init__(self):
        super(FFNN, self).__init__()
        self.d_model = _D_MODEL
        self.dff = _DFF
        self.W1 = torch.nn.Linear(in_features=self.d_model, out_features=self.dff)
        self.W2 = torch.nn.Linear(in_features=self.dff, out_features=self.d_model)

    def forward(self, x):
        return self.W2(torch.nn.relu(self.W1(x)))

class SelfAttention(torch.nn.Module):

    def __init__(self, mask=None):

        super(SelfAttention, self).__init__()
        self.d_model = _D_MODEL
        self.d_k = _D_K
        self.heads = _ATT_HEADS
        self.mask = mask

        # compute queries, keys and values for all attention heads in parallel
        self.W_q = torch.rand(self.d_model, self.d_model)
        self.W_k = torch.rand(self.d_model, self.d_model)
        self.W_v = torch.rand(self.d_model, self.d_model)
        self.W_o = torch.rand(self.d_model, self.d_model)

    def forward(self, x_q, x_k, x_v):
        """
        shapes = (batch_size, sentence_len, d_model)
        :param x_q: input used to form query
        :param x_k: input used to form key
        :param x_v: input used to form value
        :return:
        """
        batch_size = x.shape[0]
        # output of matmul has shape batch_size, sentence_len, d_model
        # split d_model into heads and d_k, then transpose to do the attention operations
        # final shape = batch_size, heads, sentence_len, d_k
        Q = torch.matmul(x_q, self.W_q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x_k, self.W_k).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x_v, self.W_v).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        # Q K.T has shape (batch_size, self.heads, len, len), apply softmax row-wise
        # note that matmul does batch-wize matrix multiplication, ignoring the first two dimensions
        # scores has shape batch_size, heads, sentence_len, sentence_len
        scores = torch.matmul(Q, K.transpose(2, -1)) / np.sqrt(self.d_k)
        if self.mask is not None:
            # set to -inf, where mask value is 0
            scores = scores.masked_fill(self.mask == 0, -1e9)

        scores = torch.nn.functional.softmax(scores, dim=2)

        # matmul has shape = batch_size, heads, sentence_len, d_k
        # for each attention head, for each position, we have an encoding of dimension d_k
        attention = torch.matmul(scores, V).transpose(1, 2)
        print(attention.shape)
        attention = attention.contiguous().view(batch_size, -1, self.d_model)
        print(attention.shape)
        attention = torch.matmul(attention, self.W_o)
        print(attention.shape)
        return attention

# test self-attention
if __name__ == "__main__":

    att = SelfAttention()
    x = torch.randn(20, 14, 512)
    att(x, x, x)