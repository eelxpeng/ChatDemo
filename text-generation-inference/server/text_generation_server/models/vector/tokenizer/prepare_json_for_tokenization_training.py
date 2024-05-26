import argparse
import glob
import json
import os
import pprint
import random

from vector_pretraining.tokenizer.pretokenize import (
    gpt2_pretokenize_pattern,
    no_space_prefix_pattern,
    pretokenize,
    reverse_pretokenize,
    newline_pattern,
    space_prefix_pattern,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract data from json to train a tokenizer"
    )
    parser.add_argument("--input_dirs", type=str, nargs="+", default=[])
    parser.add_argument("--output_path", type=str)
    parser.add_argument(
        "--filter_sentence_length",
        type=int,
        default=62880,
        help="Tokenzier can hand at mox sentences with uint16 characters",
    )
    parser.add_argument(
        "--sample_prob",
        type=float,
        default=1.0,
        help="Percent of lines to keep from each file",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Extract documents not used in training",
    )
    # parser.add_argument(
    #     "--no_space_prefix",
    #     action="store_true",
    #     default=False,
    #     help="Pass flag to stop pretokenization from pairing words with space prefix.",
    # )
    parser.add_argument(
        "--pretokenization",
        type=str,
        default=None,
        choices=["space_prefix", "no_space_prefix", "gpt2", "newline"],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk_newlines", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def process_line(line, out_file, pat, args):
    if args.sample_prob < 1.0 and random.random() > args.sample_prob:
        return

    content = json.loads(line)["content"]  # [0]['content']

    short_enough_to_train = len(content) < args.filter_sentence_length
    if args.eval:
        if not short_enough_to_train and not args.chunk_newlines:
            # does not appear in training set
            chunks = [content]
        else:
            return

    else:
        if short_enough_to_train:
            chunks = [content]
        elif args.chunk_newlines:
            chunks = list(filter(None, content.split("\n")))
        else:
            # skip this sentence, too long and we aren't breaking it down into smaller chunks
            return

    for chunk in chunks:
        encoded_content = pretokenize(chunk, pat)
        out_file.write(encoded_content + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    assert (
        args.filter_sentence_length < 65535
    ), "SPM Tokenizer can only train on sentences shorter than uint16"

    # pretokenization regex pattern:
    if args.pretokenization == "gpt2":
        pat = gpt2_pretokenize_pattern
    elif args.pretokenization == "space_prefix":
        pat = space_prefix_pattern
    elif args.pretokenization == "no_space_prefix":
        pat = no_space_prefix_pattern
    elif args.pretokenization == "newline":
        pat = newline_pattern
    else:
        raise ValueError("No known pretokenization strategy")

    # collect the paths to all the input files
    paths = []
    for root in args.input_dirs:
        paths += glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True)

    paths = sorted(paths)
    print("Files to process:")
    pprint.pprint(paths)

    # save the encoded content for each input file to our output file
    with open(args.output_path, "w") as out_file:
        print("\nProcessing ...")

        for path in tqdm(paths):
            with open(path, "r") as in_file:
                print(f"\t{path}")

                for i, line in enumerate(in_file):
                    process_line(line, out_file, pat, args)


if __name__ == "__main__":
    main()
