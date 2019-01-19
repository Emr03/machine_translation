import torch

class Encoder(torch.nn.module):

    def __init__(self, input_size, hidden_size, n_layers):

        super(self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_size = input_size

        self.rnn = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=n_layers,
                                batch_first=True,
                                bidirectional=True)


    def forward(self, input_seq):

        batch_size = input_seq.shape[0]
        h0 = torch.zeros((self.n_layers*2, batch_size, self.hidden_size))

        return self.rnn(input=input_seq, h0=h0)






