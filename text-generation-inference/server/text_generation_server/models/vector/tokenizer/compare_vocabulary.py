"""
Compare overlap between learned sentencepiece tokenizers

Example:
python compare_vocabulary.py --data_paths /mnt/efs/people/rgiaq/tokenizers/v2/
"""
import argparse
import glob
import os
import pprint
import random

import sentencepiece as spm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract data from json to train a tokenizer"
    )
    parser.add_argument("--data_paths", type=str, nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    return args


def compare_all_vocabs(model_paths, verbose=True, as_dict=False):
    results = {} if as_dict else []
    for i, model_path_i in enumerate(model_paths[:-1]):
        key1 = os.path.split(model_path_i)[-1]

        if as_dict:
            results[key1] = {}
        else:
            row = [None] * (i) + [1.0]

        if verbose:
            print(f"\nComparing intersects with {key1}:")

        # compare tokenizer1 to each other tokenizer
        for j, model_path_j in enumerate(model_paths[(i + 1) :]):
            key2 = os.path.split(model_path_j)[-1]
            pct_overlap = compare_vocab(model_path_i, model_path_j)

            if as_dict:
                results[key1][key2] = pct_overlap
            else:
                row.append(round(pct_overlap, 2))

            if verbose:
                print(f"    - {round(100.0*pct_overlap, 1):>5}% overlap with {key2}")

        if not as_dict:
            results.append(row)

    return results


def compare_vocab(path1, path2):
    # load a tokenizers
    tok1 = spm.SentencePieceProcessor(model_file=path1)
    tok2 = spm.SentencePieceProcessor(model_file=path2)

    vocab1 = get_vocab(tok1)
    vocab2 = get_vocab(tok2)

    pct_overlap = compute_intersect(vocab1, vocab2)
    return pct_overlap


def compute_intersect(vocab1, vocab2):
    vocab1_size = len(vocab1)
    intersect_size = len(vocab1.intersection(vocab2))
    pct_overlap = 1.0 * intersect_size / vocab1_size
    return pct_overlap


def get_vocab(sp):
    vocab = set([sp.id_to_piece(id) for id in range(sp.get_piece_size())])
    return vocab


def main():
    args = parse_args()
    random.seed(args.seed)

    # check if input data argument is a list of files or a directory
    if len(args.data_paths) > 1:
        data_paths = args.data_paths
    else:
        data_paths = args.data_paths[0]
        if not os.path.isfile(data_paths):
            data_paths = glob.glob(args.data_paths[0] + "*.model")

        data_paths = sorted(data_paths, reverse=True)

    as_dict = False
    results = compare_all_vocabs(data_paths, as_dict=as_dict, verbose=True)
    print("\n\nResults:")
    if as_dict:
        pprint.pprint(results, indent=2)
    else:
        import prettytable

        p = prettytable.PrettyTable()
        for row in results:
            p.add_row(row)

        print(p.get_string(header=False, border=True))


if __name__ == "__main__":
    main()
