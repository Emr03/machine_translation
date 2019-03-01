import torch
import torch.nn.functional as F
from src.encoder import *
from src.decoder import *
from src.config import FLAGS
import numpy as np
from src.data.loader import *
from data_loading import get_parser

params = FLAGS.flag_values_dict()

class Transformer(torch.nn.Module):

    def __init__(self):

        super(Transformer, self).__init__()
        self.d_model = params["d_model"]
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.dff = params["dff"]
        self.d_k = params["d_k"]

        self.encoder = StackedEncoder(n_layers=self.n_layers)
        self.decoder = StackedDecoder(n_layers=self.n_layers)
        self.linear = torch.nn.Linear(self.d_model, self.vocab_size)

        self.data = None

    def encode(self, input_seq):

        return self.encoder(input_seq)

    def decode(self, prev_output, latent_seq, mask):

        return self.decoder(prev_output, latent_seq, mask=mask)

    def forward(self, input_seq, prev_output, mask):

        latent = self.encode(input_seq)
        dec_outputs = self.decode(prev_output=prev_output, latent_seq=latent, mask=mask)
        return F.softmax(self.linear(dec_outputs), dim=-1)

    def load_data(self, data_params):

        all_data = load_data(data_params)
        self.languages = all_data['dico'].keys()
        self.mono_data_train = all_data['mono']['train']
        self.mono_data_valid = all_data['mono']['valid']
        self.dictionary_lang1 = all_data['dico'][self.languages[0]]
        self.dictionary_lang2 = all_data['dico'][self.languages[1]]

if __name__ == "__main__":

    # test transformer
    x = torch.zeros(20, 5, 512, dtype=torch.float32)
    y = torch.zeros(20, 7, 512, dtype=torch.float32)
    m = np.tril(np.ones((1, 7, 7)), k=0).astype(np.uint8)
    m = torch.from_numpy(m)
    print(m)

    model = Transformer()
    out = model(input_seq=x, prev_output=y, mask=m)
    print(out.shape)

    parser=get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model.load_data(data_params=data_params)








