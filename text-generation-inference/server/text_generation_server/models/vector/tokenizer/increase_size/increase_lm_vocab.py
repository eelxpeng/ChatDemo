import torch
from torch import nn
from transformers import HfArgumentParser
from vector_pretraining.tokenizer.tokenization_vectorbart import VectorBartTokenizer as vector_tokenizer
from typing import Optional
from dataclasses import dataclass, field


def tok_pool(old_tok, new_tok, old_weights, pool=torch.mean, new_weights=None):
    old_vocab = set(old_tok.get_vocab())
    done, new = 0, 0
    ntok = []
    for i in range(len(new_tok)):
        try:
            token = new_tok._convert_id_to_token(i)
        except IndexError:
            # Validates that is the token is a special token 
            # (not covered by SP and so will throw an exception) 
            # we ignore it and use random init
            token = new_tok.decode([i])
            if token in new_tok._additional_special_tokens:
                print(i, token)
                continue


        # If the mapping has changed adjust it the embedding
        if token in old_vocab:
            j = old_tok._convert_token_to_id(token)
            if i != j and (i >= len(old_tok) or old_tok._convert_id_to_token(i) != token):
                print(i, j, token)
                new_weights[i] = old_weights[j]
            elif i < len(old_tok) and old_tok._convert_id_to_token(i) == token:
                new_weights[i] = old_weights[i]
            done += 1
        elif token in new_tok.FAIRSEQ_LANGUAGE_CODES:
            print(i, token)
            continue
        # Do pooling over merged tokens according to their subtoken weights
        elif pool is not None:
            sub_toks = old_tok.sp_model.encode_as_pieces(''.join(token.replace('▁▁▁', '')))
            ntok.append(len(sub_toks))
            idxs = [old_tok._convert_token_to_id(tok) for tok in sub_toks]
            new_weights[i] = pool(old_weights[idxs], 0)
            print(i, str(sub_toks)+' -> '+token)
            done += 1
            new += 1
    print(new)
    print(str(done)+'/'+str(len(new_tok))+' Tokens Added')
    print('Avg Len of Merged Tokens: '+str(sum(ntok)/len(ntok)))
    print('Changes: ')
    print()
    return new_weights


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    old_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Old tokenizer path"}
    )
    old_language_tokens: Optional[str] = field(
        default=None, metadata={"help": "Comma separated list of old tokenizer language tokens"}
    )
    new_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "New tokenizer path"}
    )
    new_language_tokens: Optional[str] = field(
        default=None, metadata={"help": "Comma separated list of new tokenizer language tokens"}
    )
    state_dict: str = field(
        default=None,
        metadata={"help": "path to load new state dict while loading models."},
    )
    out_dir: str = field(
        default=None,
        metadata={"help": "path to save new state dict."},
    )
    check: bool = field(
        default=False,
        metadata={
            "help": "Prints all elements that have been changed."
        },
    )

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()[0]
if model_args.check:
    print('Running with sanity check')

tokenizer_class = vector_tokenizer

old_lang_tokens = None
if model_args.old_language_tokens is not None:
    old_lang_tokens = model_args.old_language_tokens.split(',')
    old_lang_tokens = [tok.strip() for tok in old_lang_tokens]
    print('Using Language Tokens:\n'+str(old_lang_tokens))
    print('for old tokenizer')
tokenizer = tokenizer_class.from_pretrained(model_args.old_tokenizer_name, fairseq_language_codes=old_lang_tokens)

new_lang_tokens = None
if model_args.new_language_tokens is not None:
    new_lang_tokens = model_args.new_language_tokens.split(',')
    new_lang_tokens = [tok.strip() for tok in new_lang_tokens]
    print('Using Language Tokens:\n'+str(new_lang_tokens))
    print('for new tokenizer')
new_tokenizer = tokenizer_class.from_pretrained(model_args.new_tokenizer_name, fairseq_language_codes=new_lang_tokens)

print(len(tokenizer), len(new_tokenizer), len(new_tokenizer)-len(tokenizer))

if model_args.out_dir is None:
    print('out_dir is empty defaulting to new tokenizer directory...')
    model_args.out_dir = model_args.new_tokenizer_name

# load model
print('Loading Model...')
state_dict = torch.load(model_args.state_dict, map_location="cpu")
master_state_dict = state_dict
try:
    print(state_dict['module.model.transformer.wte.weight'].shape)
except KeyError:
    state_dict = state_dict['module']
print(state_dict['module.model.transformer.wte.weight'].shape[0], len(tokenizer))
assert state_dict['module.model.transformer.wte.weight'].shape[0] == len(tokenizer)

with torch.no_grad():
    embed = state_dict['module.model.transformer.wte.weight']
    old_embed = torch.zeros_like(embed) + embed
    new_embed = nn.Embedding(len(new_tokenizer), old_embed.shape[1], dtype=old_embed.dtype).weight.data

    print('\nEmbed')
    new_embed = tok_pool(tokenizer, new_tokenizer, old_weights=old_embed, pool=torch.mean, new_weights=new_embed)

    state_dict['module.model.transformer.wte.weight'] = new_embed
    print(state_dict['module.model.transformer.wte.weight'].shape)
    print('Saving Model...')
    torch.save(master_state_dict, model_args.out_dir+'/new_model.pt')
    print('Model Saved')

# Sanity Check
if model_args.check:
    print('Loading saved model to confirm changes...')
    old_dict = torch.load(model_args.state_dict, map_location="cpu")
    new_dict = torch.load(model_args.out_dir+'/new_model.pt', map_location="cpu")
    if state_dict != master_state_dict:
        old_dict = old_dict['module']
        new_dict = new_dict['module']
    print('Modified Layers: ')
    for key in new_dict:
        if not torch.equal(new_dict[key], old_dict[key]):
            assert torch.equal(new_dict[key], state_dict[key])
            print(key)
    print('Done')
