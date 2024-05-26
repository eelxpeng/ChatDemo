"""
to start the server: 
docker run --gpus '"device=0,1"' --shm-size 1g -p 8080:80 -v /mnt/efs/people/xiaopel/checkpoints:/data text-gen-inf:latest --model-id /data/cwchat/xiaopel-26b-ga-cwchat-annot-oasst-v2-065a7b343c5942e499/release --sharded false
"""
import torch
import torch.distributed
import os
import time
from datetime import timedelta
from typing import List, Optional

from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from text_generation_server.models import CausalLM
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
)

import deepspeed
import logging
from .vector.gpt2 import GPT2LMHeadModel, GPT2Block, GPT2Config
# from transformers import GPT2LMHeadModel
from .vector.tokenizer.tokenization_vectorbart import VectorBartTokenizer

from loguru import logger

def init_tokenizer(model_dir, tokenizer_version=None):
    tokenizer = VectorBartTokenizer.from_pretrained(
        model_dir,
        cache_dir=None,
        revision=False,
        use_auth_token=False,
        src_lang="<en_XX>",
        tgt_lang="<en_XX>",
        padding_side="left",
        truncation_side="left",
    )
    return tokenizer

def model_fn(model_dir, quantization=None):
    # load model
    logger.info(
        f"model_fn invoked with model dir {model_dir}. Loading GPT2 model"
    )
    model_name_or_path = model_dir
    config = GPT2Config.from_pretrained(model_name_or_path)
    with init_empty_weights():
        cpu_model = GPT2LMHeadModel(config)

    if getattr(cpu_model.config, "precision", None) == "bf16":
        # we have a special precision field in config.json
        logger.info("convert model into bf16")
        model = cpu_model.bfloat16()
        torch_dtype = torch.bfloat16
    else:
        logger.info("convert model into fp16")
        model = cpu_model.half()
        torch_dtype = torch.half

    model = load_checkpoint_and_dispatch(
        model, model_name_or_path, device_map="auto", no_split_module_classes=["GPT2Block"]
    )
    tokenizer = init_tokenizer(model_dir)

    logger.info("Successfully run model_fn")
    return model, tokenizer, torch_dtype

class VectorGPT(CausalLM):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        # self.model, tokenizer, dtype = model_fn(model_id, quantize)
        # self.model.eval()
        # # self.model = self.model.cuda()
        # device = self.model.device
        
        # super(CausalLM, self).__init__(
        #     model=self.model,
        #     tokenizer=tokenizer,
        #     requires_padding=True,
        #     dtype=dtype,
        #     device=device,
        # )

        dtype = torch.bfloat16
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = init_tokenizer(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else None,
            load_in_8bit=quantize == "bitsandbytes",
        )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            model = model.cuda()

        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

class VectorGPTSharded(CausalLM):
    def __init__(
        self, model_id: str, model_dir: Optional[str] = None, quantize: bool = False
    ):
        logger.info("VectorGPTSharded at rank %s for world size %s" % (os.environ["RANK"], os.environ["WORLD_SIZE"]))

        self.process_group, rank, world_size = initialize_torch_distributed()
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        self.model, tokenizer, dtype = model_fn(model_id, quantize)
        logger.info(
            f"Initializing Deepspeed Inference with world_size={world_size} for rank {rank}"
        )

        # engine = deepspeed.init_inference(
        #     self.model,
        #     mp_size=world_size,
        #     dtype=dtype,
        #     replace_method="auto",
        #     replace_with_kernel_inject=True,
        #     #injection_policy={GPT2Block: deepspeed.module_inject.replace_policy.HFGPT2LayerPolicy},
        # )  # only torch_dtype controls if we are using int8

        # self.model = engine.module
        # device = self.model.device

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        super(CausalLM, self).__init__(
            model=self.model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

