# Consolas Tokenizer


## Using the Tokenizer

The production sentencepiece tokenizer file can be imported as follows:

```bash
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer

tok_path = "/mnt/efs/people/myshang/v2_items/spm_tokenizer"
tok = VectorBartTokenizer.from_pretrained(tok_path)

example = "<en_XX>This is a test!\n<prog_PY>\nprint('success!')"
print(tok.tokenize(example))
# ['<en_XX>', '▁This', '▁Ġis', '▁Ġa', '▁Ġtest', '▁!', '<prog_PY>', '▁print', "▁('", '▁success', "▁!')"]

encode_decode = tok.decode(
    tok.encode_plus(example, add_special_tokens=False)["input_ids"]
)
print(f"Encoding and decoding is lossless: {encode_decode == example}")
# Encoding and decoding is lossless: False
```

### Preserving Whitespace, Tabs, and Newlines
You may notice that the default production tokenizer removes whitespace, newlines, and tabs around special tokens (see example above). This is not an issues in most Vector applications, but to correct it you can pass the argument `fix_trimmed_whitespace=True`, for example:

```bash
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer

tok_path = "/mnt/efs/people/myshang/v2_items/spm_tokenizer"
tok = VectorBartTokenizer.from_pretrained(
    tok_path, 
    fix_trimmed_whitespace=True
)

example = "<en_XX>This is a test!\n<prog_PY>\nprint('success!')"
print(tok.tokenize(example))
# ['<en_XX>', '▁This', '▁Ġis', '▁Ġa', '▁Ġtest', '▁!', 'Ċ', '<prog_PY>', 'Ċ', '▁print', "▁('", '▁success', "▁!')"]

encode_decode = tok.decode(
    tok.encode_plus(example, add_special_tokens=False)["input_ids"]
)
print(f"Encoding and decoding is lossless: {encode_decode == example}")
# Encoding and decoding is lossless: True
```


## Unit Tests

To validate the tokenizer and the whitespace fix, run the unit tests. From the `pretraining/tests/` folder:

```bash
python test_tokenizer_whitespace_fix.py
```


## Expanding the Tokenizer's Vocabulary

To add new tokens to the tokenizer's vocabulary follow these steps:

1. Define your new set of special tokens in the `src/vector_pretraining/consts.py` file. 

For the example below we'll use the newly added the `LOOKAHEAD_SYMBOLS`:

```python
LOOKAHEAD_SYMBOLS = [
    '<end_of_insertion_gXaC7CZD7B>',
    '<code_missing_here_qoSa3n5fmW>',
    '<end_of_right_context_7CyOC9jrBA>',
    '<start_right_context_hPx8WhSpdj>',
]

```

2. Modify the `get_new_tokens(tokens_name)` function in the `expand_vocabulary.py` script to grab this new group of special tokens given some label assigned to the group.

Below we give our example group of tokens the `tokens_name` of `"lookahead"`, which is referred to in the remainder of the script.

```python
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
```


3. Expand the vocabulary by modifying the underlying sentencepiece file.

```bash
python src/vector_pretraining/tokenizer/expand_vocabulary.py \
    --output_dir /home/ec2-user/.cache/tokenizers/consolas_v2_with_lookahead/ \
    --new_tokens lookahead \
    ;
```

4. Congratulations! You have a tokenizer with an expanded vocabulary. To try it out, here is an example:

```python
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer

tok_path = "/home/ec2-user/.cache/tokenizers/consolas_v2_with_lookahead/"
tok = VectorBartTokenizer.from_pretrained(
    tok_path, 
    fix_trimmed_whitespace=True
)

example = "<en_XX>This is a test!\n<code_missing_here_qoSa3n5fmW>\tsuccess!"
print(tok.tokenize(example))
# ['<en_XX>', '▁This', '▁Ġis', '▁Ġa', '▁Ġtest', '▁!', 'Ċ', '<code_missing_here_qoSa3n5fmW>', 'ĉ', '▁success', '▁!']

encode_decode = tok.decode(
    tok.encode_plus(example, add_special_tokens=False)["input_ids"]
)
print(f"Encoding and decoding is lossless: {encode_decode == example}")
# Encoding and decoding is lossless: True
```