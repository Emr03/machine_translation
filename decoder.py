import torch
import utils

class Decoder(torch.nn.Module):

    def __init__(self, Ey, input_size, context_size, hidden_size, n_layers, deep_out_size):
        """

        :param input_size: size of input vector h, hidden size of encoder
        :param context_size: size of context vector c
        :param hidden_size: size of hidden state of decoder s
        :param n_layers:
        """

        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.Ey = Ey.transpose(0, 1)
        self.Ky = Ey.shape[1]
        self.m = Ey.shape[0]

        self.W = torch.ones(hidden_size, context_size)
        self.U = torch.ones(input_size, context_size)
        self.b = torch.ones(1, context_size)
        self.v = torch.ones(context_size)

        self.rnn = torch.nn.GRUCell(input_size=context_size,
                                    hidden_size=hidden_size,
                                    bias=True)

        self.W_o = torch.rand(deep_out_size, self.Ky)
        self.U_o = torch.rand(2*deep_out_size, hidden_size)
        self.V_o = torch.rand(self.m, 2*deep_out_size)
        self.C_o = torch.rand(input_size, 2*deep_out_size)


    def forward(self, input_seq):
        """

        :param input_seq: shape = batch_size, seq_len, input_size
        :return:
        """
        batch_size, seq_len, input_size = input_seq.shape

        s = torch.zeros(batch_size, self.hidden_size)
        y = torch.zeros(batch_size, self.Ky)

        # compute U*h_j for all j, shape = [batch_size, seq_len, context_size]
        U_h = input_seq.matmul(self.U)
        input_seq = input_seq.transpose(1, 2)

        for t in range(seq_len):

            # expand s for vectorized computation of e
            s_exp = s.unsqueeze(dim=1).expand(-1, seq_len, -1)

            W_s = s_exp.matmul(self.W)
            # shape = batch_size, seq_len,
            e = torch.tanh(W_s + U_h + self.b.unsqueeze(dim=1).expand(-1, seq_len, -1))
            e = e.matmul(self.v)
            alpha = torch.nn.functional.softmax(e, dim=1)

            # compute context vector using attention weights
            context = input_seq.bmm(alpha.unsqueeze_(dim=-1)).squeeze()

            # compute the next state
            s = self.rnn(input=context, hidden=s)

            # compute t_i tilde
            t = s.matmul(self.U_o) + y.matmul(self.Ey).matmul(self.V_o) + context.matmul(self.C_o)
            print(t.shape)
            t = utils.maxout(t, pool_size=2)

            prob_vec = torch.nn.functional.softmax(t.matmul(self.W_o))
            

if __name__ == "__main__":

    dec = Decoder(input_size=2, context_size=3, hidden_size=3, n_layers=1)

    x = torch.tensor([[[1, 2], [3, 4], [3, 4]],
                      [[5, 6], [7, 8], [3, 4]],
                      [[9, 10], [11, 12], [3, 4]],
                      [[13, 14], [15, 16], [3, 4]]], dtype=torch.float)
    dec(x)
