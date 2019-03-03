import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class PositionalEncoding(torch.nn.Module):
    """
    code obtained from http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    """
    def __init__(self, params):
        super(PositionalEncoding, self).__init__()
        self.d_model = params["d_model"]
        self.max_len = params["max_len"]

        # positional encoding for each place in an input sentence
        self.pos_enc = np.zeros((self.max_len, self.d_model))
        self.position = np.arange(0, self.max_len).reshape(-1, 1)

        # has shape (self.d_model / 2)
        div_term = np.power(10000, -np.arange(0, self.d_model, 2) / self.d_model)

        # for all even dimensions, division is done elementwise, by broadcasting
        self.pos_enc[:, 0::2] = np.sin(self.position * div_term)
        self.pos_enc[:, 1::2] = np.cos(self.position * div_term)

        # positional encoding is not a model parameter
        self.register_buffer('pe', torch.Tensor(self.pos_enc))
        self.dropout = torch.nn.Dropout(params["dropout"])

    def forward(self, x):
        """

        :param x: input sequence of embeddings of shape (batch_size, seq_len, d_model)
        :return:
        """
        len = x.shape[1]
        batch_size = x.shape[0]
        t = self.pe[0:len, :]
        return self.dropout(x + t)

    def visualize(self):
        # visualize the encoding
        plt.matshow(self.pos_enc)
        plt.show()

class FFNN(torch.nn.Module):

    def __init__(self, params):
        super(FFNN, self).__init__()
        self.d_model = params["d_model"]
        self.dff = params["dff"]
        self.W1 = torch.nn.Linear(in_features=self.d_model, out_features=self.dff)
        self.W2 = torch.nn.Linear(in_features=self.dff, out_features=self.d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))

class SelfAttention(torch.nn.Module):

    def __init__(self, params):

        super(SelfAttention, self).__init__()
        self.d_model = params["d_model"]
        self.d_k = params["d_k"]
        self.heads = params["h"]

        # compute queries, keys and values for all attention h, eads in parallel
        self.W_q = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = torch.nn.Linear(self.d_model, self.d_model, bias=False)

        # TODO:add the matrices to the set of model weights

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        shapes = (batch_size, sentence_len, d_model)
        :param x_q: input used to form query
        :param x_k: input used to form key
        :param x_v: input used to form value
        :return:
        """
        batch_size = x_q.shape[0]
        # output of matmul has shape batch_size, sentence_len, d_model
        # split d_model into heads and d_k, then transpose to do the attention operations
        # final shape = batch_size, heads, sentence_len, d_k
        Q = self.W_q(x_q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_k).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_v).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        # Q K.T has shape (batch_size, self.heads, len, len), apply softmax row-wise
        # note that matmul does batch-wize matrix multiplication, ignoring the first two dimensions
        # scores has shape batch_size, heads, sentence_len, sentence_len
        scores = torch.matmul(Q, K.transpose(2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            # set to -inf, where mask value is 0
            scores = scores.masked_fill(mask == 0, -1e9)

        print("scores", scores[0, 0, :, :])
        scores = torch.nn.functional.softmax(scores, dim=2)

        # matmul has shape = batch_size, heads, sentence_len, d_k
        # for each attention head, for each position, we have an encoding of dimension d_k
        attention = torch.matmul(scores, V).transpose(1, 2)
        print(attention.shape)
        attention = attention.contiguous().view(batch_size, -1, self.d_model)
        print(attention.shape)
        attention = self.W_o(attention)
        print(attention.shape)
        return attention


if __name__ == "__main__":

    from src.config import params
    # test self-attention
    att = SelfAttention(params)
    x = torch.ones(3, 5, 512, dtype=torch.float32)
    att(x, x, x)

    # test self-attention with masking
    mask = torch.zeros(3, 5)
    mask[:, 0:2] = 1
    mask = mask.unsqueeze(-2).unsqueeze(-2)
    print("mask ", mask.shape)
    out = att(x, x, x, mask=mask)
    print(out)

    # test pos_encoding
    enc = PositionalEncoding(params)
    x_t = enc(x)
    print(x_t.shape)

    plt.matshow(x_t.numpy()[0, :, :])
    plt.show()

    # test FFNN
    nn = FFNN(params)
    out = nn(x)

    print(out.shape)

