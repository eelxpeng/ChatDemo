import os
import glob
import time
import json
import math
import random
import argparse
import regex as re
import multiprocessing as mp
from tqdm import tqdm
from itertools import chain, repeat
from collections import defaultdict
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer
from vector_pretraining.tokenizer.pretokenize import gpt2_pretokenize_pattern, no_space_prefix_pattern, pretokenize, space_prefix_pattern, newline_pattern

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer")
    # parser.add_argument("--base_paths", type=str, nargs="+",
    #     default=['data/raw/java', 'data/raw/python', 'data/raw/javascript', 'data/raw/text']
    # )
    parser.add_argument("--new_paths", type=str, nargs="+",
        default=['data/raw/java', 'data/raw/python', 'data/raw/javascript']
    )

    parser.add_argument("--base_model", type=str, default='new_models/new_encoding/small_eval/equal_51200')
    # parser.add_argument("--new_model", type=str, default='new_models/new_encoding/small_eval/no_text')
    parser.add_argument("--merged_path", type=str, default='new_models/new_encoding/small_eval/equal_102400_adapt/dst_merge')

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--new_vocab", type=int, default=1024)
    parser.add_argument("--sampling_rate", type=float, default=0.01)
    parser.add_argument("--min_freq", type=float, default=0)
    parser.add_argument("--fresh", action="store_true", default=False)
    parser.add_argument("--phrases", action="store_true", default=False)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--pretokenization",
        type=str,
        default='gpt2',
        choices=["space_prefix", "no_space_prefix", "gpt2", "newline"],
    )
    args = parser.parse_args()
    return args


def process_line(line, pattern, phrases, n=5):
    word_dict = defaultdict(int)
    dict_sum = 0
    doc = json.loads(line)["content"]
    word_tokens = pretokenize(doc, pattern).split()
    if phrases:
        for i in range(len(word_tokens)-1):
            for j in range(1, n+1):
                subseq = [pretok for pretok in word_tokens[i:i+j] if len(pretok) > 0]
                phrase = '▁▁▁'.join(subseq)
                word_dict[phrase] += 1
                dict_sum += 1
    else:
        for word in word_tokens:
            word_dict[word] += 1
            dict_sum += 1
    return word_dict, dict_sum


def get_unigrams(data_files, pattern, sampling_rate=1.0, num_workers=4, phrases=False, window=5):
    U = defaultdict(int)
    u_sum = 0
    # Calc Unigrams:
    print('Loading File...')
    for i, data_file in enumerate(data_files):
        print(str(i)+'/'+str(len(data_files))+' '+str(data_file))
        time.sleep(0.1)
        lines = []
        with open(data_file, "r") as open_file:
            pbar = tqdm(total=os.path.getsize(data_file))
            for line in open_file:
                pbar.update(len(line.encode()))
                if sampling_rate < 1.0 and random.random() > sampling_rate:
                    continue
                if num_workers > 1:
                    lines.append(line)
                else:
                    word_dict, dict_sum = process_line(line, pattern, phrases, window)
                    for word in word_dict.keys():
                        U[word] += word_dict[word]
                    u_sum += dict_sum
            pbar.close()
        if num_workers > 1:
            print('Processing for U...')
            pool = mp.Pool(num_workers)
            results = pool.starmap_async(
                process_line,
                tqdm(zip(lines, repeat(pattern), repeat(phrases), repeat(window)), total=len(lines)),
            )
            pool.close()
            pool.join()
            print('Collecting U...')
            time.sleep(0.1)
            for i in tqdm(range(len(results.get()))):
                word_dict, dict_sum = results.get()[i]
                for word in word_dict.keys():
                    U[word] += word_dict[word]
                u_sum += dict_sum
    print('Done')
    return U, u_sum

def get_word_seqs(tok, u, phrases, w=10):
    word, count = u
    sub_T = defaultdict(float)
    if phrases:
        seq = tok.sp_model.encode_as_pieces(word.replace('▁▁▁', ''))
        sub_T[word] = count * len(seq)
    else:
        seq = tok.sp_model.encode_as_pieces(word)
        for i in range(len(seq)):
            for j in range(1, w+1):
                sub_T[''.join(seq[i:i+j])] += count * len(seq[i:i+j])
    return sub_T

def get_seq_counts(tok, U, U_sum, num_workers, phrases, window=10):
    T = defaultdict(float)
    if num_workers > 1:
        print('Processing for T...')
        U_items = list(U.items())
        pool = mp.Pool(num_workers)
        results = pool.starmap_async(
            get_word_seqs,
            tqdm(zip(repeat(tok), U_items, repeat(phrases), repeat(window)), total=len(U_items)),
        )
        pool.close()
        pool.join()
        print('Collecting T...')
        time.sleep(0.1)
        for i in tqdm(range(len(results.get()))):
            sub_T = results.get()[i]
            for key in sub_T.keys():
                T[key] += sub_T[key] / U_sum
    else:
        for u in U.items():
            sub_T = get_word_seqs(tok, u, phrases, window)
            for key in sub_T.keys():
                T[key] += sub_T[key] / U_sum
    return T

def compute_probs(tok_path, data_paths, pattern, sampling_rate, num_workers, window=5, phrases=False, fresh=False, label=''):
    start = time.time()
    tok = VectorBartTokenizer.from_pretrained(tok_path)
    vocab = set([tok.sp_model.id_to_piece(token_id) for token_id in range(tok.sp_model.get_piece_size())])
    data_files = []
    for data_path in data_paths:
        print(data_path)
        if not os.path.isfile(data_path) and os.path.isdir(data_path):
            data_files += glob.glob(data_path + "/**/*.jsonl", recursive=True)
    print('Computing U...')
    try:
        if fresh:
            raise Exception
        print('Loading U')
        U, u_sum = json.load(open(tok_path+'/'+label+'U.json', 'r'))
        print('Loaded U')

    except Exception as e:
        print('Load Failed Computing U...')
        U, u_sum = get_unigrams(data_files, pattern, sampling_rate, num_workers, phrases, window=window)
        print('Computed U '+str(round(time.time()-start, 3)))
        json.dump([U, u_sum], open(tok_path+'/'+label+'U.json', 'w'))
    try:
        if fresh:
            raise Exception
        print('Loading T')
        T = json.load(open(tok_path+'/'+label+'T.json', 'r'))
        print('Loaded T')
    except Exception as e:
        print('Load Failed Computing T...')
        T = get_seq_counts(tok, U, u_sum, num_workers, phrases, window=window)
        print('Computed T '+str(round(time.time()-start, 3)))
        json.dump(T, open(tok_path+'/'+label+'T.json', 'w'))
    print(len(T.keys()))
    print('Computed Tokens '+str(round(time.time()-start, 3)))
    return vocab, T, U, u_sum


def main():
    args = get_args()

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

    start = time.time()

    random.seed(args.seed)

    base_tok = VectorBartTokenizer.from_pretrained(args.base_model)
    base_vocab = set([base_tok.sp_model.id_to_piece(token_id) for token_id in range(base_tok.sp_model.get_piece_size())])
    # 1 Language at a time
    lang_toks = []
    if True:
        paths = args.new_paths
        random.seed(args.seed)
        label = 'new_'
        if args.phrases:
            label='phrases_'
        _, T_new, U_new, u_new_sum = compute_probs(args.base_model, paths, pattern, args.sampling_rate,
                                                   args.num_workers, phrases=args.phrases, window=args.window,
                                                   fresh=args.fresh, label=label)
        all_seqs = list(T_new.keys())
        print('Computing Scores...')
        time.sleep(0.1)
        scores = []
        for i in tqdm(range(len(all_seqs))):
            seq = all_seqs[i]
            if seq not in base_vocab:
                score = T_new[seq]
                scores.append((seq, score))

        print('Computed Scores '+str(round(time.time()-start,3)))
        scores = sorted(scores, key=lambda x: x[1])[::-1]
        json.dump(scores, open(args.base_model+'/'+label+'scores.json', 'w+'), indent=4)
        toks = [tok for tok, key in scores]
        print(scores[:128])
        lang_toks.append(toks)
    if args.phrases:
        new_phrases = {}
        for i in range(args.new_vocab):
            new_phrases[scores[i][0]] = scores[i][1]
        json.dump(new_phrases, open(args.base_model+'/merged_phrases.json', 'w+'), indent=4)
    else:
        if args.new_vocab is None:
            r = [2**i for i in range(16)]
        else:
            r = [args.new_vocab]
        for i in r:
            print(i)
            toks_to_add = []
            for toks in lang_toks:
                toks_to_add.extend(toks[:i])
            toks_to_add = list(set(toks_to_add))
            print('Adding Tokens to Create New Tokenizer')
            from vector_pretraining.tokenizer.expand_vocabulary import update_sentencepiece_vocab
            if not os.path.exists(args.merged_path+str(i)):
                os.makedirs(args.merged_path+str(i))
            update_sentencepiece_vocab(args.base_model+'/sentencepiece.bpe.model', toks_to_add, args.merged_path+str(i), quiet=True)
    print('Done')


if __name__ == '__main__':
    main()
