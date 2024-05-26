"""
Compute a set of results to evaluate how well a tokenizer is doing

Example:

mkdir models
aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/sentencepiece.bpe.model ./models
aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/special_tokens_map.json ./models
aws s3 cp s3://zijwan/pretraining_v2/spm_1002_converted/tokenizer_config.json ./models
python eval_tokenizer.py \
    --data_dirs /mnt/efs/people/rgiaq/datasets/cleaned_tokenization_maxlen16768/python_ \
    --output_path ./results.json \
    --model_dir ./models \
    --sampling_rate 0.01 \
    --debug
"""
import os
import argparse
import random
from collections import Counter
from vector_pretraining.tokenizer.vocabulary_stats import init_vocab_counts
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer
from vector_pretraining.tokenizer.pretokenize import gpt2_pretokenize_pattern, no_space_prefix_pattern, pretokenize, space_prefix_pattern, newline_pattern
import glob
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import json
import time
import pandas as pd
import numpy as np
from time import perf_counter_ns as timer
import sentencepiece as spm
from itertools import chain, repeat

def add_eval_args(parser):
    # Latency Args
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        default=[],
        required=True,
        help="Provide top-level directory to files to process per language.",
    )

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--stdev", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use_phrases", action="store_true", default=False)

    # Tokenizer Stats Args
    parser.add_argument("--count_blank_lines", action="store_true", default=False)
    parser.add_argument(
        "--pretokenization",
        type=str,
        default='gpt2',
        choices=["space_prefix", "no_space_prefix", "gpt2", "newline"],
    )
    parser.add_argument(
        "--no_space_prefix",
        action="store_true",
        default=False,
        help="Pass flag for non space-prefixed pretokenization. Ignored if input is pretokenized",
    )

def get_paths(data_dirs):
    # collect all the paths to all the input files
    grouped_paths = defaultdict(list)
    print("\nProcessing files from:")
    for data_dir in data_dirs:
        dir_files = sorted(glob.glob(data_dir + "/**/*.jsonl", recursive=True))
        for file in dir_files:
            lang = file[len(data_dir):].split('/')[0]
            if len(lang) == 0:
                lang = file[len(data_dir):].split('/')[1]
            grouped_paths[lang].append(file)
    print(grouped_paths)
    return grouped_paths

def process_line(doc, tok):
    # encode
    start_encode = timer()
    token_ids = tok.encode(doc)
    end_encode = timer()
    encode_time = end_encode - start_encode

    # decode
    start_decode = timer()
    decoded = tok.decode(token_ids, clean_up_tokenization_spaces=False)
    end_decode = timer()
    decode_time = end_decode - start_decode

    return token_ids, encode_time, decode_time

def process_doc(line, tokenizer, args, is_pretokenized):
    file_dict = {
        'counts': 0,
        'n_docs': 0,
        'n_tokens': 0,
        'n_unk': 0,
        'n_chars': 0,
        'n_bytes': 0,
        'encode_time_ns': 0,
        'decode_time_ns': 0,
    }
    try:
        tokens, encode_time, decode_time = process_line(json.loads(line)["content"], tokenizer)
        file_dict['n_tokens'] += len(tokens)
        file_dict['n_chars'] += len(json.loads(line)["content"])
        file_dict['n_bytes'] += len(json.loads(line)["content"].encode())
        file_dict['n_docs'] += 1
        # file_dict['counts'] += Counter(tokens)
        # file_dict['n_unk'] += file_dict['counts']['<unk>']
        file_dict['encode_time_ns'] += encode_time
        file_dict['decode_time_ns'] += decode_time
        return file_dict
    except:
        file_dict = {
            'counts': 0,
            'n_docs': 0,
            'n_tokens': 0,
            'n_unk': 0,
            'n_chars': 0,
            'n_bytes': 0,
            'encode_time_ns': 0,
            'decode_time_ns': 0,
        }
        return file_dict

def process_docs(args, q_in, q_out):
    tokenizer = VectorBartTokenizer.from_pretrained(args.model_dir, use_phrases=args.use_phrases)
    # pattern = no_space_prefix_pattern if args.no_space_prefix else space_prefix_pattern
    # pretokenization regex pattern:
    if args.pretokenization == "gpt2":
        pattern = gpt2_pretokenize_pattern
    elif args.pretokenization == "space_prefix":
        pattern = space_prefix_pattern
    elif args.pretokenization == "no_space_prefix":
        pattern = no_space_prefix_pattern
    elif args.pretokenization == "newline":
        pattern = newline_pattern
    else:
        raise ValueError("No known pretokenization strategy")
    tokenizer.pat = pattern
    # prefix = args.model_dir.split('/')[-1]
    # tokenizer = spm.SentencePieceProcessor(model_file=args.model_dir+'/'+prefix+".model")
    i = q_in.get()
    # print('Started Process '+str(i))
    q_out.put(i)
    while True:
        obj = q_in.get()
        if obj is None:
            break
        line, is_pretokenized = obj
        file_dict = process_doc(line, tokenizer, args, is_pretokenized)
        q_out.put(file_dict)

def compute_vocab_stats(args, verbose=False):
    start = time.time()
    paths = get_paths(args.data_dir)

    if verbose:
        print("Processing files:")
        print()

    # load a pretrained tokenizer
    prefix = args.model_dir.split('/')[-1]
    tokenizer = VectorBartTokenizer.from_pretrained(args.model_dir, use_phrases=args.use_phrases)
    # tokenizer = spm.SentencePieceProcessor(model_file=args.model_dir+'/'+prefix+".model")
    results = {}
    for language in paths.keys():
        results[language] = {
            'counts': 0,
            'n_docs': 0,
            'n_tokens': 0,
            'n_unk': 0,
            'n_chars': 0,
            'n_bytes': 0,
            'encode_time_ns': 0,
            'decode_time_ns': 0,
            'compression_ratio': [],
        }
        if args.num_workers > 1:
            qs = [(mp.Queue(), mp.Queue()) for _ in range(args.num_workers)] # [(q_in, q_out) for _ in range(num_workers)]
            ps = [mp.Process(target=process_docs, args=(args, qs[i][0], qs[i][1])) for i in range(args.num_workers)]
            for i in range(args.num_workers):
                ps[i].start()
                qs[i][0].put(i)
            for i in range(args.num_workers):
                assert qs[i][1].get() == i
        for path in paths[language]:
            # Open Processes
            print(path, time.time()-start)
            extension = path.split(".")[-1]
            is_pretokenized = extension == "txt"
            if verbose:
                print(f"Tokenizing (sampling rate={args.sampling_rate}): {path}")
            file_toks, file_bytes = 0, 0
            if args.num_workers > 1:
                lines = []
                with open(path, "r") as data_file:
                    pbar = tqdm(total=os.path.getsize(path))
                    # Send Data
                    for line in data_file:
                        pbar.update(len(line.encode()))
                        if args.sampling_rate < 1.0 and random.random() > args.sampling_rate:
                            continue
                        lines.append(line)
                    pbar.close()
                    print('Processing File...')
                    # process_doc(line, tokenizer, args, is_pretokenized)
                    pool = mp.Pool(args.num_workers)
                    pool_results = pool.starmap_async(
                        process_doc,
                        tqdm(zip(lines, repeat(tokenizer), repeat(args), repeat(is_pretokenized)), total=len(lines)),
                    )
                    pool.close()
                    pool.join()
                    print('Collecting Results...')
                    for i in tqdm(range(len(pool_results.get()))):
                        file_dict = pool_results.get()[i]
                        file_toks += file_dict['n_tokens']
                        file_bytes += file_dict['n_bytes']
                        for key in file_dict.keys():
                            results[language][key] += file_dict[key]
            else:
                with open(path, "r") as data_file:
                    for line in data_file:
                        # short_enough_to_train = len(line) < 62880
                        if args.sampling_rate < 1.0 and random.random() > args.sampling_rate:
                            continue
                        file_dict = process_doc(line, tokenizer, args, is_pretokenized)
                        file_toks += file_dict['n_tokens']
                        file_bytes += file_dict['n_bytes']
                        for key in file_dict.keys():
                            results[language][key] += file_dict[key]

            # z_tok = len(gzip.compress(tok_file))
            # z_raw = len(gzip.compress(raw_file))
            # compressed_ratio = z_tok / z_raw
            # uncompressed_ratio = 0.5*len(tok_file)/len(raw_file)
            ratio = file_toks / file_bytes
            print(time.time()-start, ratio)
            results[language]['compression_ratio'].append(ratio)
        # Closing Processes
        for i in range(args.num_workers):
            qs[i][0].put(None)
            ps[i].join()
    return results

def process_stats(results):
    def gini_coefficient(x):
        diffsum = 0
        for i, xi in enumerate(x[:-1], 1):
            diffsum += np.sum(np.abs(xi - x[i:]))
        return diffsum / (len(x) ** 2 * np.mean(x))
    stats = {}
    for lang in results.keys():
        stats[lang] = {}
        # df = pd.DataFrame.from_dict(results[lang]['counts'], orient='index').reset_index()
        # stats[lang]['efficiency'] = 1 - gini_coefficient (df[0])
        stats[lang]['accuracy'] = 1 - (results[lang]['n_unk'] / results[lang]['n_tokens'])
        stats[lang]['compression'] = results[lang]['n_tokens'] / results[lang]['n_bytes']
        stats[lang]['encode_time_per_token_ns'] = results[lang]['encode_time_ns'] / results[lang]['n_tokens']
        stats[lang]['decode_time_per_token_ns'] = results[lang]['decode_time_ns'] / results[lang]['n_tokens']
    return stats

def print_stats(stats):
    for lang in stats.keys():
        print(lang)
        for key in stats[lang].keys():
            print('\t'+key+': '+str(stats[lang][key]))

def main():
    def has_model_file(model_dir):
        sub_files = os.listdir(model_dir)
        for sub_file in sub_files:
            if sub_file[-6:] == '.model':
                return True
        return False

    import time
    parser = argparse.ArgumentParser(description="Evaluate tokenizer")
    add_eval_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    if not has_model_file(args.model_dir):
        model_dir = args.model_dir
        subdirs = sorted(os.listdir(model_dir))
        for subdir in subdirs:
            if not os.path.isdir(model_dir+'/'+subdir):
                continue
                continue
            if not has_model_file(model_dir+'/'+subdir):
                continue
            args.model_dir = model_dir+'/'+subdir
            assert len(args.data_dir) > 0
            assert len(args.model_dir) > 0

            start = time.time()
            random.seed(args.seed)
            results = compute_vocab_stats(args)
            json.dump(results, open(args.model_dir+'/results.json', 'w+'), indent=4)
            stats = process_stats(results)
            end = time.time()
            # json.dump(stats, open(args.model_dir+'/stats.json', 'w+'), indent=4)
            print(args.model_dir)
            print_stats(stats)
            print(end - start)


if __name__ == '__main__':
    main()
