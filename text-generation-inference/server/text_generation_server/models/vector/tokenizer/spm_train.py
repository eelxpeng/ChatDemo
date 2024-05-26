import glob
import logging
import sys

import sentencepiece as spm

from vector_pretraining.consts import USER_DEFINED_SYMBOLS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

python_data = glob.glob(
    "/mnt/efs/projects/pretraining/github_codel/python_line_indent_blessed/*"
)
java_data = glob.glob(
    "/mnt/efs/projects/pretraining/github_codel/java_line_indent_blessed/*"
)
so_data = glob.glob(
    "/mnt/efs/projects/pretraining/se/so_py_java_desc_shards/train*.txt"
)
all_data = java_data + so_data + python_data

spm.SentencePieceTrainer.train(
    input=all_data,
    model_prefix=sys.argv[1],
    model_type="bpe",
    character_coverage=0.9999,
    vocab_size=50_000,
    user_defined_symbols=USER_DEFINED_SYMBOLS,
    num_threads=120,
    shuffle_input_sentence=True,
    input_sentence_size=10_000_000,
)
