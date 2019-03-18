from src.data.loader import *
import argparse

def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='Language transfer')
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")

    # dataset
    parser.add_argument("--langs", type=str, default="",
                        help="Languages (lang1,lang2)")

    parser.add_argument("--vocab", type=str, default="",
                        help="Vocabulary (lang1:path1;lang2:path2)")

    parser.add_argument("--vocab_min_count", type=int, default=0,
                        help="Vocabulary minimum word count")

    parser.add_argument("--mono_dataset", type=str, default="",
                        help="Monolingual dataset (lang1:train1,valid1,test1;lang2:train2,valid2,test2)")

    parser.add_argument("--para_dataset", type=str, default="",
                        help="Parallel dataset (lang1-lang2:train12,valid12,test12;lang1-lang3:train13,valid13,test13)")

    parser.add_argument("--back_dataset", type=str, default="",
                        help="Back-parallel dataset, with noisy source and clean target (lang1-lang2:train121,train122;lang2-lang1:train212,train211)")

    parser.add_argument("--n_mono", type=int, default=0,
                        help="Number of monolingual sentences (-1 for everything)")

    parser.add_argument("--n_para", type=int, default=0,
                        help="Number of parallel sentences (-1 for everything)")

    parser.add_argument("--n_back", type=int, default=0,
                        help="Number of back-parallel sentences (-1 for everything)")

    parser.add_argument("--max_len", type=int, default=175,
                        help="Maximum length of sentences (after BPE)")

    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")

    # temporary
    parser.add_argument("--group_by_size", type=bool, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--mono_directions", type=str, default="",
                        help="Training directions (lang1,lang2)")
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    return parser

def main(params):
    check_all_data_params(params)
    data = load_data(params)
    print(data)

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    print(params)
    #main(params)
