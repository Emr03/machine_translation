import torch

class Decoder(torch.nn.module):

    def __init__(self, input_size, context_size, hidden_size, n_layers):
        """

        :param input_size: size of input vector h, hidden size of decoder
        :param context_size: size of context vector c
        :param hidden_size: size of hidden state of decoder s
        :param n_layers:
        """

        super(self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.W = torch.nn.rand(hidden_size, context_size)
        self.U = torch.nn.rand(input_size, context_size)
        self.b = torch.nn.rand(1, context_size)

        self.rnn = torch.nn.GRUCell(input_size=context_size,
                                    hidden_size=hidden_size,
                                    bias=True)

        def forward(input_seq):
            """

            :param input_seq: shape = batch_size, seq_len, input_size
            :return:
            """
            batch_size, seq_len, input_size = input_seq.shape
            s = torch.zeros(batch_size, hidden_size)
            s.matmul(self.W) + input_seq.matmul(self.U) + self.b  # batch_size, seq_len, context_size

            for t in range(seq_len):

                s = self.rnn(context, s)

