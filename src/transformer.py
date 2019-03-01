from .config import params

from .encoder import *
from .decoder import *
import numpy as np
from .data.loader import *
from .data_loading import get_parser
from .data.dataset import *

class Transformer(torch.nn.Module):

    def __init__(self):

        super(Transformer, self).__init__()
        self.d_model = params["d_model"]
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.dff = params["dff"]
        self.d_k = params["d_k"]

        self.encoder = StackedEncoder(n_layers=self.n_layers, params=params)
        self.decoder = StackedDecoder(n_layers=self.n_layers, params=params)
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
        print(all_data)
        self.languages = list(all_data['dico'].keys())

        self.mono_data_train = [all_data['mono'][self.languages[0]]['train'],
                                all_data['mono'][self.languages[1]]['train']]

        self.mono_data_valid = [all_data['mono'][self.languages[0]]['valid'],
                                all_data['mono'][self.languages[1]]['valid']]

        self.dictionary_lang1 = all_data['dico'][self.languages[0]]
        self.dictionary_lang2 = all_data['dico'][self.languages[1]]

        self.lang1_train_iterator = self.mono_data_lang1_train.get_iterator(shuffle=True,
                                                                            group_by_size=True)

        self.lang2_train_iterator = self.mono_data_lang2_train.get_iterator(shuffle=True,
                                                                            group_by_size=True)

    def reconstruction_loss(self):

    def train_iter(self, src_batch, tgt_batch):

        self.forward(src_batch)

    def train_loop(self, train_iter):

        for i in range(train_iter):

            src_lan = i % 2
            tgt_lan = (i + 1) % 2



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








