"""
Save stats on out of vocabulary counts

Examples:

cd /home/ec2-user/vector-science/pretraining/scripts/pretraining/tokenizer/
vocab=51200
sz=20000000
maxlen=16768
model_prefix=/mnt/efs/people/rgiaq/tokenizers/v2/v2_tokenizer_regex_pretokenized_fixed_whitespace_vocab${vocab}_sz${sz}_maxlen${maxlen}

# Example 1, compute stats on two specific files
python oov_stats.py     \
    --model_prefix ${model_prefix} \
    --seed 123 \
    --sampling_rate 0.001 \
    --data_path /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/java_3rd.txt \
        /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/java_apache.txt \
    ;

# Example 2, compute stats on all python language files
python oov_stats.py \
    --model_prefix ${model_prefix} \
    --data_dir /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/python_ \
    --seed 123 \
    --sampling_rate 0.01 \
    ;

"""
import argparse
import glob
import json
import random
import sys
from collections import Counter

import sentencepiece as spm

from vector_pretraining.consts import FAIRSEQ_LANGUAGE_CODES, USER_DEFINED_SYMBOLS
from pretokenize import no_space_prefix_pattern, pretokenize, space_prefix_pattern


def add_tokenizer_args(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="+",
        default=[],
        help="Use this argument to compute on specific files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Use this argument to compute on all *.txt or *.jsonl files.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Where to save stats on avg freq per doc of each token in vocabulary",
    )
    parser.add_argument(
        "--no_space_prefix",
        action="store_true",
        default=False,
        help="Pass flag for non space-prefixed pretokenization. Ignored if input is pretokenized",
    )
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", default=False)


def init_vocab_counts(sp):
    vocab = Counter({sp.id_to_piece(id): 0 for id in range(sp.get_piece_size())})
    return vocab


def add_vocab_counts(v1, v2):
    """
    Deprecated, just use Counter
    :param v1:
    :param v2:
    :return:
    """
    keys = set(list(v1.keys()) + list(v2.keys()))
    rval = {}
    for k in keys:
        rval[k] = v1[k] + v2[k]

    return rval


def main():
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Train a tokenizer")
    add_tokenizer_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    fixed_symbols = (
        ["<unk>", "<s>", "</s>"] + USER_DEFINED_SYMBOLS + FAIRSEQ_LANGUAGE_CODES
    )
    pattern = no_space_prefix_pattern if args.no_space_prefix else space_prefix_pattern

    if len(args.data_path) > 0:
        paths = args.data_path
    else:
        assert args.data_dir is not None, "Must give either data_path or data_dir"
        paths = glob.glob(args.data_dir + "*.txt")

    paths = sorted(paths)
    print("Processing files:")
    pprint(paths)
    print()

    # load a pretrained tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=args.model_prefix + ".model")
    total_counts = init_vocab_counts(tokenizer)
    total_docs = 0
    total_tokens = 0
    for path in paths:
        extension = path.split(".")[-1]
        is_pretokenized = extension == "txt"

        print(f"Tokenizing (sampling rate={args.sampling_rate}): {path}")
        with open(path, "r") as data_file:
            file_docs = 0
            file_counts = init_vocab_counts(tokenizer)
            for line in data_file:
                if random.random() > args.sampling_rate and args.sampling_rate < 1.0:
                    continue

                if is_pretokenized:
                    doc = line.rstrip()
                else:
                    doc = pretokenize(
                        json.loads(line)["content"], pattern
                    )  # [0]['content']

                tokens = tokenizer.encode_as_pieces(doc)
                total_tokens += len(tokens)
                file_docs += 1
                file_counts += Counter(tokens)

        total_counts += file_counts
        total_docs += file_docs
        print(
            f"Proportion tokens that are <unk> (and other fixed tokens) from a {file_docs} document sample:"
        )
        pprint(
            {k: round(1.0 * file_counts[k] / file_docs, 4) for k in fixed_symbols},
            indent=2,
        )

    print("\n\nOverall stats:")
    print(
        f"Processed {len(paths)} files, {total_docs} documents, {total_tokens} tokens."
    )
    print("Proportion tokens that are <unk> (and other fixed tokens):")
    pprint(
        {k: round(1.0 * total_counts[k] / total_docs, 4) for k in fixed_symbols},
        indent=2,
    )

    if args.out_path is not None:
        out_stats = sorted(total_counts.items(), key=lambda elt: elt[1], reverse=True)

        print(
            f"Saving statistics on Avg Token Frequency / Document to: {args.out_path}"
        )
        with open(args.out_path, "w") as out_file:
            out_file.write(
                "\n".join([f"{tok}\t{1.0 * ct / total_docs}" for tok, ct in out_stats])
            )


if __name__ == "__main__":
    main()
