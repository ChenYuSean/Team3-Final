# %%
# !pip install torch==2.1.0
# !pip install transformers==4.34.1
# !pip install bitsandbytes==0.41.1
# !pip install peft==0.6.0
# !pip install datasets
# !pip install evaluate
# !pip install accelerate
# !pip install sentencepiece
# !pip install einops
# !pip install scikit-learn
# !pip install ipdb

# %%
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from datasets import Dataset, load_dataset

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from accelerate import notebook_launcher
from accelerate import Accelerator

from data_process import get_prompt, prepare_dataset

# %%
# Global variables
FROM_COLAB = False
DEBUG = False
ROOT_PATH = './'
str_args = None

# %%
if FROM_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT_PATH = 'drive/MyDrive/Colab Notebooks/ADL/HW3/'
if DEBUG:
    import ipdb

# %%
@dataclass
class ModelArguments:
    train_file: str
    model_name_or_path: str
    seed: int = field(default = 42)
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "validation file or directory. If None, split the train set."}
    )
    test_size: Optional[int] = field(
        default=100,
        metadata={"help": "Size of test set. If the validation file is not set, this would split the train data set into according size."}
    )
    train_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of training set. If None and validation file is not set, size would be set to complement of the test size"
                  " or else, use all the data"}
    )
    num_video_per_channel: int = field(default=1)
    output_dir: str = field(default='./output')
    checkpointing_steps: int = field(default=100)
    logging_steps: int = field(default=100)

@dataclass
class TrainingArguments:
    gradient_accumulation_steps: int = field(default=2)
    batch_size: int = field(default=2)
    learning_rate: float = field(default=2e-4)
    num_epoch: float = field(
        default = 1.0,
        metadata={"help": "number of epoch during training"}
    )
    max_steps: int = field(
        default = -1,
        metadata={"help": "Total steps of training, would override num_epoch"}
    )
    source_max_len: int = field(default=1024)
    target_max_len: int = field(default=256)
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=2.0)
    lora_dropout: float = field(default=0.1)
    
@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=256),
    min_new_tokens: int = field(default=None),
    do_sample: bool = field(default=False),
    num_beams: Optional[int] = field(default=1),
    num_beam_groups: Optional[int] = field(default=1)
    temperature: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    top_p: Optional[float] = field(default=None)
# Parser
def parse_generation_args(str_args = None):
    '''
    There is something buggy using dataclass for generation config. Use standard parser to parse.
    Error Message: "TypeError: cannot pickle 'mappingproxy' object" 
    '''
    parser = argparse.ArgumentParser()
    # Generation Argument
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None
    )
    parser.add_argument(
        "--do_sample",
        action='store_true'
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None
    )

    args = parser.parse_args(str_args)
    return args

# %%
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# %%
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

# %%
def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

# %%
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

# %%
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        IGNORE_INDEX = -100
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

# %%
def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"
    )
    return bnb_config

# %%
def main(str_args = None):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, TrainingArguments
    ))
    model_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(str_args,return_remaining_strings=True)
    generation_args = parse_generation_args(extra_args)
    args = argparse.Namespace(
        **vars(model_args), **vars(training_args), **vars(generation_args)
    )

    # %%
    # Prepare
    logger = logging.getLogger(__name__)

    compute_dtype = torch.float16
    if args.seed is not None:
        set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)


    # %%
    # Load dataset
    print('Load Dataset')
    def format_dataset(dataset):
        def processing(example):
            return {'input': get_prompt(example['video_title'], example['video_description'], example['star_num'], example['mood']),
                    'output': example['comment_text']}
        formatted_dataset = dataset.map(processing)
        # Remove unused columns.
        formatted_dataset = formatted_dataset.remove_columns(
            [col for col in dataset.column_names if col not in ['input', 'output']]
        )
        return formatted_dataset
    raw_dataset = prepare_dataset(args.train_file, args.num_video_per_channel, seed = args.seed)
    if args.validation_file is None:
        split = raw_dataset.train_test_split(train_size=args.train_size ,test_size=args.test_size, seed=args.seed, shuffle=True)
        train_dataset = format_dataset(split['train'])
        eval_dataset = format_dataset(split['test'])
    else:
        raw_train = raw_dataset
        raw_eval = prepare_dataset(args.validation_file, select=False, seed = args.seed)
        if args.train_size is not None and raw_train.shape[0] > args.train_size:
            raw_train = raw_train.shuffle(args.seed)
            raw_train = raw_train.select(range(args.train_size))
        if args.test_size is not None and raw_eval.shape[0] > args.test_size:
            raw_eval = raw_eval.select(range(args.test_size))
        train_dataset = format_dataset(raw_train)
        eval_dataset = format_dataset(raw_eval)

    # %%
    # Load Model
    print('Load Model')
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config = bnb_config,
        load_in_4bit = True,
        torch_dtype=compute_dtype,
        device_map = 'cuda:0'
    )
    base_model.config.torch_dtype=compute_dtype

    # %%
    # Tokenizer
    print('Load Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        tokenizer_type='llama'
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=base_model,
        )
    tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(base_model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(base_model.config.bos_token_id),
        "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
    })

    # %%
    # LORA Model
    model = prepare_model_for_kbit_training(base_model)
    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        print('Add LoRA')
        modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if compute_dtype == torch.bfloat16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if compute_dtype == torch.bfloat16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # %%
    # Data Collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=False,
        predict_with_generate=False,
    )
    # Generation Config
    gen_config = transformers.GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        min_new_tokens = args.min_new_tokens,
        do_sample = args.do_sample,
        num_beams = args.num_beams,
        num_beam_groups = args.num_beam_groups,
    )
    # Trainer Arguments
    trainer_args = transformers.Seq2SeqTrainingArguments(
            do_train = True,
            do_eval = True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=2,
            num_train_epochs=args.num_epoch,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            optim="paged_adamw_8bit",
            lr_scheduler_type="constant",
            fp16=(compute_dtype==torch.float16),
            bf16=(compute_dtype==torch.bfloat16),
            evaluation_strategy='steps',
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=args.checkpointing_steps,
            remove_unused_columns = False,
            generation_config = gen_config
        )
    # Trainer
    print('Set Trainer')
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=trainer_args,
        data_collator=data_collator,
    )
    trainer.add_callback(SavePeftModelCallback)

    # %%
    # Train
    all_metrics = {}
    logger.info("*** Train ***")
    model.config.use_cache = False
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))

# %%
if __name__ == "__main__":
    main(str_args)


