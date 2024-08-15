# This file was modified and originally stemmed from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.

import sys
from dataclasses import dataclass


@dataclass
class XftConfig:
    max_seq_len: int = 4096
    beam_width: int = 1
    eos_token_id: int = -1
    pad_token_id: int = -1
    num_return_sequences: int = 1
    is_encoder_decoder: bool = False
    padding: bool = True
    early_stopping: bool = False
    data_type: str = "bf16_fp16"


class XftModel:
    def __init__(self, xft_model, xft_config):
        self.model = xft_model
        self.config = xft_config


def load_xft_model(model_path, xft_config: XftConfig):
    try:
        import xfastertransformer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Failed to load xFasterTransformer. {e}")
        sys.exit(-1)

    if xft_config.data_type is None or xft_config.data_type == "":
        data_type = "bf16_fp16"
    else:
        data_type = xft_config.data_type
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left", trust_remote_code=True)
    xft_model = xfastertransformer.AutoModel.from_pretrained(model_path, dtype=data_type)
    model = XftModel(xft_model=xft_model, xft_config=xft_config)
    if model.model.rank > 0:
        while True:
            model.model.generate()
    return model, tokenizer
