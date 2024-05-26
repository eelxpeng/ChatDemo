"""
Compute latency of tokenization encoding and decoding across training data,
separate results by language and give overall results.

Example:

aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/sentencepiece.bpe.model ./
aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/special_tokens_map.json ./
aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/tokenizer_config.json ./
root=/mnt/efs/people/zijwan/pretraining_v2/cleaned/
python tokenizer_latency.py \
    --input_dirs ${root}python/ ${root}java/ \
    --output_path ./latency.jsonl \
    --model_dir ./ \
    --sampling_rate 0.01 \
    --debug \
    ;

"""
import argparse
import glob
import json
import multiprocessing
import os
import random
from collections import defaultdict
from itertools import chain, repeat
from pprint import pprint
from time import perf_counter_ns as timer

from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer
from tqdm import tqdm


def add_latency_args(parser):
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        default=[],
        required=True,
        help="Provide top-level directory to files to process per language.",
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--stdev", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)


def get_paths(args):
    # collect all the paths to all the input files
    grouped_paths = defaultdict(list)
    total_files = 0
    print("\nProcessing files from:")
    for root in args.input_dirs:
        print(root)
        top_dir = os.path.normpath(root).split(os.path.sep)[-1]
        data_sources = sorted(os.listdir(root))
        for i, data_source in enumerate(data_sources):
            if args.debug and i > 2:
                break

            print(f"  {data_source}/")
            files = sorted(glob.glob(os.path.join(root, data_source, "*.jsonl")))
            if args.debug:
                files = files[: args.num_workers]

            grouped_paths[top_dir] += files
            total_files += len(files)

    print(f"\nFound {total_files} total files between {len(grouped_paths)} languages:")
    pprint(dict(grouped_paths))
    print()
    return grouped_paths


def compute_latency(args):
    grouped_paths = get_paths(args)

    tok = VectorBartTokenizer.from_pretrained(args.model_dir)

    result_type = list if args.stdev else int
    by_lang = defaultdict(lambda: defaultdict(result_type))
    overall = defaultdict(result_type)

    for lang, paths in grouped_paths.items():
        print(f"\n\nComputing latencies for {lang} files...")
        if args.num_workers > 1:
            pool = multiprocessing.Pool(args.num_workers)
            results = pool.starmap_async(
                process_file,
                tqdm(zip(paths, repeat(tok), repeat(args)), total=len(paths)),
            )
            pool.close()
            pool.join()
            for res in results.get():
                by_lang[lang] = add_dicts(by_lang[lang], res, args.stdev)

        else:
            for path in tqdm(paths):
                res = process_file(path, tok, args)
                by_lang[lang] = add_dicts(by_lang[lang], res, args.stdev)

        overall = add_dicts(overall, by_lang[lang], args.stdev)

    by_lang["overall"] = overall
    return by_lang


def process_file(path, tok, args):
    random.seed(args.seed)
    result_type = list if args.stdev else int
    results = defaultdict(result_type)

    with open(path, "r") as in_file:
        for line in in_file:
            if args.sampling_rate < 1.0 and random.random() > args.sampling_rate:
                continue

            if args.debug and sum(results["num_docs"]) > 100:
                break

            num_docs = 1
            num_chars, num_tokens, encode_time, decode_time = process_line(line, tok)
            if args.stdev:
                num_docs = [num_docs]
                num_chars = [num_chars]
                num_tokens = [num_tokens]
                encode_time = [encode_time]
                decode_time = [decode_time]

            results["num_chars"] += num_chars
            results["num_tokens"] += num_tokens
            results["encode_time"] += encode_time
            results["decode_time"] += decode_time
            results["num_docs"] += num_docs

    return results


def process_line(line, tok):
    doc = json.loads(line)["content"]

    # encode
    start_encode = timer()
    tokens = tok.convert_tokens_to_ids(tok.tokenize(doc))
    end_encode = timer()
    encode_time = end_encode - start_encode

    # decode
    start_decode = timer()
    _ = tok.decode(tokens, clean_up_tokenization_spaces=False)
    end_decode = timer()
    decode_time = end_decode - start_decode

    num_tokens = len(tokens)
    num_chars = len(doc)
    return num_chars, num_tokens, encode_time, decode_time


def summarize_results(latency, args):
    rval = defaultdict(dict)
    for lang, stats in latency.items():
        d = rval[lang]
        if args.stdev:
            d["total_docs"] = 1.0 * sum(stats["num_docs"])
            d["total_chars"] = 1.0 * sum(stats["num_chars"])
            d["total_tokens"] = 1.0 * sum(stats["num_tokens"])
            d["total_encode_ns"] = 1.0 * sum(stats["encode_time"])
            d["total_decode_ns"] = 1.0 * sum(stats["decode_time"])
        else:
            d["total_docs"] = 1.0 * stats["num_docs"]
            d["total_chars"] = 1.0 * stats["num_chars"]
            d["total_tokens"] = 1.0 * stats["num_tokens"]
            d["total_encode_ns"] = 1.0 * stats["encode_time"]
            d["total_decode_ns"] = 1.0 * stats["decode_time"]

        d["encode_ns_per_char"] = d["total_encode_ns"] / d["total_chars"]
        d["encode_ns_per_token"] = d["total_encode_ns"] / d["total_tokens"]
        d["encode_ns_per_doc"] = d["total_encode_ns"] / d["total_docs"]
        d["decode_ns_per_char"] = d["total_decode_ns"] / d["total_chars"]
        d["decode_ns_per_token"] = d["total_decode_ns"] / d["total_tokens"]
        d["decode_ns_per_doc"] = d["total_decode_ns"] / d["total_docs"]

        if args.stdev:
            for code in ["encode", "decode"]:
                for granularity in ["char", "token", "doc"]:
                    arr = [
                        1.0 * t / sz
                        for t, sz in zip(
                            stats[f"{code}_time"], stats[f"num_{granularity}s"]
                        )
                    ]
                    mu = sum(arr) / len(arr)
                    delta = [abs(t - mu) for t in arr]
                    d[f"{code}_ns_per_{granularity}_stdev"] = sum(delta) / len(delta)

    return rval


def save_results(summary, details, args):
    for language, language_results in summary.items():
        if language == "overall":
            continue

        print(f"\n{language} results:")
        pprint(language_results)

    print("\nOverall results:")
    pprint(summary["overall"])

    # save final results to a file
    if args.output_path:
        # write out overall stats and per language summaries
        with open(args.output_path, "w") as f:
            f.write(json.dumps(summary) + "\n")

        if args.stdev:
            # write out detailed stats per document
            root, extension = args.output_path.split(".")
            for lang, stats in details.items():
                details_path = root + f"_{lang}_details." + extension
                with open(details_path, "w") as f:
                    docs = zip(
                        stats["num_chars"], stats["encode_time"], stats["decode_time"]
                    )
                    out = "\n".join([f"{sz}\t{et}\t{dt}" for sz, et, dt in docs]) + "\n"
                    f.write(out)


def add_dicts(a, b, combine_lists):
    default = [] if combine_lists else 0
    dtype = list if combine_lists else int
    rval = defaultdict(dtype)
    for k in chain(a.keys(), b.keys()):
        rval[k] = a.get(k, default) + b.get(k, default)

    return rval


def main():
    parser = argparse.ArgumentParser(description="Evaluate latency of tokenizer")
    add_latency_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)

    assert len(args.input_dirs) > 0
    assert len(args.model_dir) > 0

    latency = compute_latency(args)
    summary = summarize_results(latency, args)
    save_results(summary, latency, args)


if __name__ == "__main__":
    main()
