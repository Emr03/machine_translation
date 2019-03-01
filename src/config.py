# parameter dictionary

from absl import flags

flags.DEFINE_integer(
    "vocab_size", default=30000, help="size of vocabulary")

flags.DEFINE_integer(
    "d_model", default=512, help="dim of word embeddings"
)

flags.DEFINE_integer(
    "d_k", default=64, help="dim of attention output"
)

flags.DEFINE_integer(
    "h", default=8, help="number of attention heads"
)

flags.DEFINE_integer(
    "dff", default=2048, help="intermediate dim of feedforward layers"
)

flags.DEFINE_integer(
    "max_len", default=5000, help="max sentence length"
)

flags.DEFINE_integer(
    "n_layers", default=6, help="number of decoder and encoder layers in the stack"
)

flags.DEFINE_float(
    "dropout", default=0.1, help="dropout parameter"
)

FLAGS = flags.FLAGS
