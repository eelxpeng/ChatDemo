"""
We then update the vocabulary of the huggingface tokenizer to include the
new tokens, and then save the tokenizer to a file.

This script *only needs to be run once* to convert the sentencepiece tokenizer,
update vocab, and save it.
"""
import argparse
import os
import sys
import sentencepiece as spm

import vector_pretraining.tokenizer.sentencepiece_model_pb2 as model
from vector_pretraining.consts import *
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer


def get_new_tokens(tokens_name):
    """
    Use this function to reference a set of tokens defined in `consts.py`.

    To expand the tokenizer's vocabulary with a new set of tokens,
    you will need to modify this function (add another elif statement), and
    give your new tokens a label (e.g. `"lookahead"`).
    The `expand_vocabulary.py` script will then use this function to get those tokens.
    """
    if isinstance(tokens_name, str):
        tokens_name = tokens_name.strip().lower()

    if tokens_name is None or tokens_name == "" or tokens_name is False:
        return []
    elif tokens_name == "lookahead":
        rval = LOOKAHEAD_SYMBOLS
    else:
        raise ValueError(f"Error. Unknown special tokens group: {tokens_name}.\n"
                         "Please set `--new_tokens` to one of these choices: `['lookahead']`")

    return rval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spm_path",
        type=str,
        required=False,
        default="/mnt/efs/people/myshang/v2_items/spm_tokenizer/sentencepiece.bpe.model",
        help="Path to the sentencepiece *.model file. "
             "Defaults to current Consolas v2 tokenizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for where to save the converted tokenizer. "
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Include flag to suppress examples of the tokenizer on a test string",
    )
    parser.add_argument(
        "--new_tokens",
        type=str,
        required=False,
        help="Name of new special tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--do_not_fix_trimmed_whitespace",
        action="store_true",
        default=False,
        help="Adding this flag will use the default HuggingFace tokenize() "
             "method, which may remove whitespace, tabs, and newlines around "
             "all special tokens.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # assert os.path.exists(args.output_dir)

    # update the underlying sentence piece model, add missing tokens to its vocab
    new_sp_file = update_sentencepiece_vocab(args.spm_path, args.new_tokens, args.output_dir, args.quiet)

    # turn updated sentence piece model into the VectorBartTokenizer and save
    tokenizer = VectorBartTokenizer(vocab_file=new_sp_file)

    # save tokenizer as pretrained for later use
    tokenizer.save_pretrained(args.output_dir)

    # load in the new tokenizer and test it out
    tokenizer = VectorBartTokenizer.from_pretrained(
        args.output_dir,
        fix_trimmed_whitespace=not args.do_not_fix_trimmed_whitespace
    )

    if not args.quiet:
        print("\nUpdated Tokenizer:")
        print(tokenizer)
        print_special_tokens(tokenizer)
        print("\nRunning tests to validate new tokenizer:")
        test_new_tokenizer_works(args, tokenizer)
        test_tokenizer_alignment(args, tokenizer)


def print_special_tokens(tokenizer):
    assert set(tokenizer.unique_no_split_tokens) == set(tokenizer.all_special_tokens)
    specials = dict(
        (str(t), tokenizer.convert_tokens_to_ids(t))
        for t in tokenizer.unique_no_split_tokens
    )

    print("\nList of all *special* tokens and their ids:\n{")
    for key, value in sorted(specials.items(), key=lambda x: x[1]):
        print("  {} : {}".format(key, value))
    print("}")


def get_sp_vocab(sp):
    vocab = set([sp.id_to_piece(token_id) for token_id in range(sp.get_piece_size())])
    return vocab


def update_sentencepiece_vocab(spm_path, new_tokens, output_dir, quiet=True):
    """
    Update the binary sentencepiece model file to include additional tokens in the vocabulary
    :param args:
    :return:
    """
    # get original sentence-piece model vocab
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    existing_tokens = get_sp_vocab(sp)

    if not quiet:
        print(f"\n\nAugmenting SP vocab, SP has {len(existing_tokens)} tokens.")

    # load serialized sentencepiece model
    m = model.ModelProto()
    m.ParseFromString(open(spm_path, "rb").read())
    if isinstance(new_tokens, str):
        # get the new tokens
        new_tokens = get_new_tokens(new_tokens)
    elif not isinstance(new_tokens, list):
        raise ValueError(f"Error. Unknown special tokens group.\n"
                         "Please set `--new_tokens` to one of these choices: `['lookahead']`")

    assert new_tokens is not None

    # verify that none of these new special tokens already exist in vocab
    missing_tokens = [token for token in new_tokens if token not in existing_tokens]

    if not quiet:
        print(f"\nAdding {len(missing_tokens)} special symbols to the vocab")

    fair_seq_offset = len(FAIRSEQ_LANGUAGE_CODES) + 2  # <mask> and consolas offset

    num_tokens = len(m.pieces) + fair_seq_offset
    for i, sym in enumerate(missing_tokens, 0):
        new_sym = m.SentencePiece()
        new_sym.piece = sym
        new_sym.score = 0.0  # default score for USER_DEFINED
        new_sym.type = 4  # type value for USER_DEFINED
        m.pieces.insert(i + num_tokens, new_sym)
        if not quiet:
            print(f"Added special symbol {i + num_tokens}: {sym}")

    if not quiet:
        print(f"Tokenizer now has {len(m.pieces)} tokens in vocabulary.")

    # save serialized sentence piece model
    output_file = os.path.join(output_dir, "sentencepiece.bpe.model")
    if not quiet:
        print(f"Saving updated tokenizer to: {output_file}")
    with open(output_file, "wb") as f:
        f.write(m.SerializeToString())

    return output_file


def test_new_tokenizer_works(args, tokenizer, print_original=True):
    """
    Examine how well the original tokenizer trained for Consolas performs on some text
    """
    print("\n\n", "#" * 80, sep="")
    print("Test 1: Evaluate New tokenizer on a test example:")
    print("#" * 80, "\n", sep="")

    crazy_text = (
        "<prog_PY>\n"
        "Some Python tests for\tTABS and, 'quotes' and spaces in text\n\n"
        "# code indented with spaces: \n"
        "for i in range(10):\n"
        "    print(i)\n\n"
        "# code with tabs:\n"
        "for i in ['1', '2','3']:\n"
        "\tprint('hello_my_friends')\n\n"
        # "Different special characters ➞ (hard) and → (easier).\n\n"
        "Newlines, whitespaces, and tabs around special characters break Huggingface's\n"
        "implementation. Let's test\t<prog_PY>\n<prog_JS>\t\t<prog_JAVA>\n\n<en_XX>  <prog_PY>\n\n"
        "Did the tabs and newlines get removed? (Hopefully not!)\n\n"
    )
    crazy_text += "New tokens: " + ' and '.join(get_new_tokens(args.new_tokens)) + " the end."

    if print_original:
        print("\n\n", "#" * 80, sep="")
        print("Original test input:")
        print("#" * 80, "\n", sep="")
        print(crazy_text)

    # encode and look at each token
    print("\n\n", "#" * 80, sep="")
    print("Test input as token strings:")
    print("#" * 80, "\n", sep="")
    print(tokenizer.tokenize(crazy_text))

    print("\n\n", "#" * 80, sep="")
    print("Test input as token ids:")
    print("#" * 80, "\n", sep="")
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(crazy_text)))

    # encode and decode
    # Note: Consolas production tokenizer won't hand all emojis well
    print("\n\n", "#" * 80, sep="")
    print("Test input after encoding and decoding:")
    print("#" * 80, "\n", sep="")
    encode_decode = tokenizer.decode(
        tokenizer.encode_plus(crazy_text, add_special_tokens=False)["input_ids"]
    )
    print(encode_decode)

    print("\n\n", "#" * 80, sep="")
    print(
        f"Test 1 Result: Encoding then decoding recovered the original input: {encode_decode == crazy_text}"
    )
    print("#" * 80, "\n", sep="")
    if encode_decode != crazy_text:
        title = "Original    New"
        print(title)
        found_mismatch = False
        for i in range(min(len(encode_decode), len(crazy_text))):
            line = f"{repr(crazy_text[i]):>8}    {repr(encode_decode[i])}"
            if crazy_text[i] != encode_decode[i] and not found_mismatch:
                line += "\tFIRST MISMATCH"
                found_mismatch = True

            print(line)


def test_tokenizer_alignment(args, tokenizer):
    print("\n\n", "#" * 80, sep="")
    print("Test 2: Validating alignment between the New and Production Consolas vocabularies:")
    print("#" * 80, "\n", sep="")
    print(f"Comparing to Production Consolas tokenizer from:\n\t{args.spm_path}")

    prod = VectorBartTokenizer.from_pretrained(os.path.dirname(args.spm_path))

    # 1. check that each consolas special token also exists in the updated tokenizer
    assert all([t in tokenizer.all_special_tokens for t in prod.all_special_tokens])
    print("\nAlignment Check 1: PASSED. All Consolas special tokens exist in the New Tokenizer.")

    # 2. check that all Consolas tokens and ids are aligned
    failed = compare_tokenizers(prod, "Production", tokenizer, "New")

    if not failed:
        print(
            "\nAlignment Check 2: PASSED. All Consolas tokens are in the same position as in New Tokenizer."
        )
        print(
            "\nVictory! New tokenizer was successfully updated and aligns with Consolas!\n\n"
        )
    else:
        print(
            "\nAlignment Status: FAILED. The two tokenizers are not aligned. \n"
            "The New Tokenizer will look up incorrect embeddings from Consolas!"
        )
    assert not failed


def compare_tokenizers(t1, t1_label, t2, t2_label):
    print("\nComparing statistics on vocabularies of each tokenizer:")
    print(f"{t1_label}.vocab_size = {t1.vocab_size}, {t2_label}.vocab_size = {t2.vocab_size}")
    print(f"      len({t1_label}) = {len(t1)},       len({t2_label}) = {len(t2)}")
    max_width = 36
    max_vocab = max([len(t1), len(t2), t1.vocab_size, t2.vocab_size])
    failed = False
    title = f"\n{'ID':>6}{t1_label:>{max_width}}{t2_label:>{max_width}}"
    print(title, "\n", "-"*len(title), sep="")
    for t in range(max_vocab):
        t1_token = "--"
        t2_token = "--"
        if t < len(t1):
            t1_token = t1.convert_ids_to_tokens(t)
        if t < len(t2):
            t2_token = t2.convert_ids_to_tokens(t)
        line = f"{t:>6}{t1_token:>{max_width}}{t2_token:>{max_width}}"
        if t < 15 or t > 51200:
            print(line)
        if t == 51199:
            print(f"{'...':>6}{'...':>{max_width}}{'...':>{max_width}}")
        if t < len(t1) and t < len(t2):
            if t1_token != t2_token:
                failed = True
    return failed


if __name__ == "__main__":
    main()
