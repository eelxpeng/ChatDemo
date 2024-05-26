"""
Training the BPE tokenizer with sentencepiece

Example:

data_dir=/mnt/efs/people/zijwan/pretraining_v2/cleaned/
training_dir=/mnt/efs/people/rgiaq/datasets/debug_tokenization/

cd /home/ec2-user/vector-science/pretraining/scripts/pretraining/tokenizer/

vocab=51200
sz=100000
maxlen=8384
model_prefix=/mnt/efs/people/rgiaq/tokenizers/v2/debug_v2_tokenizer_regex_pretokenized_fixed_whitespace_vocab${vocab}_sz${sz}_maxlen${maxlen}

# train
python train_tokenizer.py \
    --data_path ${training_dir} \
    --model_prefix ${model_prefix} \
    --fix_whitespace_symbols \
    --vocab_size ${vocab} \
    --max_sentence_length ${maxlen} \
    --max_sentencepiece_length 16 \
    --input_sentence_size ${sz} \
    --shuffle \
    --seed 123 \
    ;

# inference
python train_tokenizer.py --model_prefix ${model_prefix} --inference --debug

"""
import argparse
import glob
import json
import os
import random
import sys
import time

import sentencepiece as spm
from pretokenize import (
    gpt2_pretokenize_pattern,
    no_space_prefix_pattern,
    pretokenize,
    reverse_pretokenize,
    space_prefix_pattern,
)
from prettytable import PrettyTable
from tqdm import tqdm

from vector_pretraining.consts import FAIRSEQ_LANGUAGE_CODES, USER_DEFINED_SYMBOLS


def add_tokenizer_args(parser):
    parser.add_argument("--data_path", type=str, nargs="+", default=[])
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--character_coverage", type=float, default=0.9999)
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--input_sentence_size", type=int, default=10_000_000)
    parser.add_argument("--max_sentence_length", type=int, default=4192)
    parser.add_argument("--model_type", type=str, default='bpe')
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--fix_whitespace_symbols", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_sentencepiece_length", type=int, default=16)
    parser.add_argument(
        "--pretokenization",
        type=str,
        default="gpt2",
        choices=["space_prefix", "no_space_prefix", "gpt2"],
    )


def debug_special_symbols(tokenizer, pretokenization_pattern):
    test_symbols = ["__UNKNOWN__", "<unk>", "<s>", "\t", "\n", " ", "    "]
    test_symbols += FAIRSEQ_LANGUAGE_CODES + USER_DEFINED_SYMBOLS

    for t in test_symbols:
        encoded_t = pretokenize(t, pretokenization_pattern)
        tid = tokenizer.piece_to_id(encoded_t)
        print(repr(t) + f" is pretokenized to '{encoded_t}' and tokenized to '{tid}'")


def debug_example(tokenizer, pretokenization_pattern):
    crazy_text = """<prog_PY>\nSome Python tests for\tTABS and, "quotes" and spaces in text\n\n# code indented with spaces: \nfor i in range(10):    print(i)\n\n# code with tabs:\nfor i in ["1", "2","3"]:\n\tprint("hello_my_friends")\n\nAnd garbage words: ADOSADFNAEIDsdfsafjasdfmqkqlADSF..."""

    print("\n\n!!! TEST ENCODING TEXT WITH NEWLINES")
    encoded_text = pretokenize(crazy_text, pretokenization_pattern)
    print("!!! ORIGINAL TEXT:\n\n" + crazy_text)
    print("\n\n!!! TOKENIZER INPUT (WITH BYTE TO UNICODE ENCODING):\n\n" + encoded_text)
    tokens = tokenizer.encode_as_pieces(encoded_text)
    token_ids = tokenizer.encode_as_ids(encoded_text)
    table = PrettyTable()
    token_dict = {"Token ID": token_ids, "Token": [repr(t) for t in tokens]}
    for k, v in token_dict.items():
        table.add_column(k, v)

    print("\n\n!!! ENCODED TOKENS: AS A LIST:\n")
    print(tokens)
    print("\n\n!!! ENCODED TOKENS: STRING TO TOKEN ID MAPPING:\n")
    print(table.get_string())

    print("\n\n!!! DECODE EXAMPLE: GIVEN THE TOKEN IDS:\n")
    print(token_ids)

    print(
        "\n\n!!! DECODE EXAMPLE: FIRST DECODE TOKEN IDS WITH TOKENIZER, OUTPUT TOKEN STRINGS:\n"
    )
    decoded_tokens = tokenizer.decode(token_ids)
    print(decoded_tokens)

    decoded_text = reverse_pretokenize(decoded_tokens)
    print(
        "\n\n!!! DECODE EXAMPLE: THEN MAP TOKEN STRINGS TO TEXT (REVERSE BYTE TO UNICODE):\n"
    )
    print(decoded_text)


def main():

    parser = argparse.ArgumentParser(description="Train a tokenizer")
    add_tokenizer_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.pretokenization == "gpt2":
        pat = gpt2_pretokenize_pattern
    elif args.pretokenization == "space_prefix":
        pat = space_prefix_pattern
    elif args.pretokenization == "no_space_prefix":
        pat = no_space_prefix_pattern
    else:
        raise ValueError("No known pretokenization strategy")

    if args.debug:
        # load a pretrained tokenizer
        tokenizer = spm.SentencePieceProcessor(model_file=args.model_prefix + ".model")
        print("Vocab size:", tokenizer.get_piece_size())
        debug_special_symbols(tokenizer, pat)
        debug_example(tokenizer, pat)

    elif args.inference:
        # Tokenize the list of input files, calculate latency, save tokenized results

        # load a pretrained tokenizer
        tokenizer = spm.SentencePieceProcessor(model_file=args.model_prefix + ".model")

        sampling_rate = 0.01
        encode_time = 0.0
        decode_time = 0.0
        total_docs = 0
        total_characters = 0
        total_tokens = 0

        # TODO: add option to save tokenized results to a file

        # tokenize the given file
        for data_path in tqdm(args.data_path):
            print(f"# Tokenizing (sampling {sampling_rate}): {data_path}")
            with open(data_path, "r") as data_file:
                for i, line in enumerate(data_file):
                    if random.random() > sampling_rate and sampling_rate < 1.0:
                        continue

                    doc = json.loads(line)["content"]
                    s1 = time.time()
                    pretokens = pretokenize(doc, pat)
                    tokens = tokenizer.encode(pretokens)
                    encode_time += s1 - time.time()
                    try:
                        s2 = time.time()
                        _ = reverse_pretokenize(tokenizer.decode(tokens))
                        decode_time += s2 - time.time()
                    except KeyError as e:
                        print(e)
                        print(doc)
                        print(pretokens)
                        print(tokens)
                        print(tokenizer.decode(tokens))

                    total_docs += 1
                    total_characters += len(doc)
                    total_tokens += len(tokens)

        print(
            f"Sampled {total_docs} documents, with {total_characters} characters and {total_tokens} tokens."
        )
        print(
            f"Average time to encode per input character = {encode_time / total_characters}"
        )
        print(f"Average time to encode per input token = {encode_time / total_tokens}")
        print(f"Average time to encode per input doc = {encode_time / total_docs}")
        print(
            f"Average time to decode per input character = {decode_time / total_characters}"
        )
        print(f"Average time to decode per input token = {decode_time / total_tokens}")
        print(f"Average time to decode per input doc = {decode_time / total_docs}")

    else:
        # train a new tokenizer
        symbols = FAIRSEQ_LANGUAGE_CODES
        if args.fix_whitespace_symbols:
            symbols += USER_DEFINED_SYMBOLS

        # check if input data argument is a list of files or a directory
        if len(args.data_path) > 1:
            data_path = args.data_path
        else:
            data_path = args.data_path[0]
            if not os.path.isfile(data_path) and os.path.isdir(data_path):
                data_path = glob.glob(data_path + "*.txt")

        spm.SentencePieceTrainer.Train(
            input=data_path,
            model_prefix=args.model_prefix,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            vocab_size=args.vocab_size,
            max_sentence_length=args.max_sentence_length,
            max_sentencepiece_length=args.max_sentencepiece_length,
            user_defined_symbols=symbols,
            num_threads=args.num_threads,
            shuffle_input_sentence=args.shuffle,
            input_sentence_size=args.input_sentence_size,
            add_dummy_prefix=True,
            remove_extra_whitespaces=True,
            allow_whitespace_only_pieces=True,
            split_by_whitespace=True,
        )


if __name__ == "__main__":
    main()
