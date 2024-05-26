"""
Save stats on # tokens per line of code and document,

Examples (bash):

cd /home/ec2-user/vector-science/pretraining/scripts/pretraining/tokenizer/
vocab=51200
sz=20000000
maxlen=16768
model_prefix=/mnt/efs/people/rgiaq/tokenizers/v2/v2_tokenizer_regex_pretokenized_fixed_whitespace_vocab${vocab}_sz${sz}_maxlen${maxlen}

# Example 1: compute stats on two specific files
python tokenizer_stats.py     \
    --model_prefix ${model_prefix} \
    --seed 123 \
    --sampling_rate 0.001 \
    --data_path /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/java_3rd.txt \
        /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/java_apache.txt \
    ;

# Example 2: compute stats on all python files in the directory
python tokenizer_stats.py \
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

import regex as re
import sentencepiece as spm
from vector_pretraining.tokenizer.pretokenize import (
    gpt2_pretokenize_pattern,
    no_space_prefix_pattern,
    pretokenize,
    reverse_pretokenize,
    space_prefix_pattern,
)


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
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--count_blank_lines", action="store_true", default=False)
    parser.add_argument(
        "--pretokenization",
        type=str,
        default=None,
        choices=["space_prefix", "no_space_prefix", "gpt2"],
    )
    parser.add_argument("--debug", action="store_true", default=False)


def main():
    import pprint

    parser = argparse.ArgumentParser(description="Train a tokenizer")
    add_tokenizer_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    r = re.compile("^Ċ+$")
    if args.pretokenization == "gpt2":
        pattern = gpt2_pretokenize_pattern
    elif args.pretokenization == "space_prefix":
        pattern = space_prefix_pattern
    elif args.pretokenization == "no_space_prefix":
        pattern = no_space_prefix_pattern
    else:
        raise ValueError("No known pretokenization strategy")

    if len(args.data_path) > 0:
        paths = args.data_path
    else:
        assert args.data_dir is not None, "Must give either data_path or data_dir"
        if args.pretokenize:
            paths = glob.glob(args.data_dir + "*.jsonl")
        else:
            paths = glob.glob(args.data_dir + "*.txt")
        paths = sorted(paths)

    print("Processing files:")
    pprint.pprint(paths)
    print()

    # load a pretrained tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=args.model_prefix + ".model")

    n_tokens = 0
    n_docs = 0
    n_lines = 0
    for path in paths:
        print(f"Tokenizing (sampling proportion={args.sampling_rate}): {path}")
        with open(path, "r") as data_file:
            f_tokens = 0
            f_docs = 0
            f_lines = 0
            for i, line in enumerate(data_file):
                if random.random() > args.sampling_rate and args.sampling_rate < 1.0:
                    continue

                if args.pretokenization:
                    doc = pretokenize(json.loads(line)["content"], pattern)
                else:
                    doc = line.rstrip()

                if args.count_blank_lines:
                    lines_of_code = doc.count("Ċ")  # each newline = 1 line of code
                else:
                    lines_of_code = len(re.findall("Ċ+", doc))

                tokens = tokenizer.encode_as_pieces(doc)
                # how many tokens were just newlines?
                newline_tokens = len(list(filter(r.match, tokens)))

                f_lines += lines_of_code
                f_docs += 1
                f_tokens += len(tokens) - newline_tokens
                n_lines += lines_of_code
                n_docs += 1
                # don't count newlines as tokens in total
                n_tokens += len(tokens) - newline_tokens

        print(f"  Tokens/Lines = {f_tokens}/{f_lines} = {round(1.*f_tokens/f_lines,2)}")
        print(f"  Tokens/Docs = {f_tokens}/{f_docs} = {round(1.*f_tokens/f_docs,2)}")

    print("\n\nTOTAL:")
    print(f"Tokens/Lines = {n_tokens} / {n_lines} = {round(1.*n_tokens/n_lines,2)}")
    print(f"Tokens/Docs  = {n_tokens} / {n_docs} = {round(1.*n_tokens/n_docs,2)}")


if __name__ == "__main__":
    main()
