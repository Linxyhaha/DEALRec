import os
import sys
from typing import List

import fire
import torch

import transformers
from datasets import load_dataset, concatenate_datasets
import torch
from effort_util import Effort_Trainer, Modified_LlamaForCausalLM

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaTokenizer  # noqa: F402

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    
def get_effort_score(args):

    base_model = args.base_model
    train_data_path = args.train_data_path
    # training hyperparams
    cutoff_len = args.cutoff_len
    # lora hyperparams
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = ["q_proj","v_proj",]
    # other hyperparams
    group_by_length = False  # faster, but produces an odd training loss curve
    resume_from_checkpoint = args.resume_from_checkpoint  # either training checkpoint or final adapter
    
    assert (base_model), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = 1

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    os.environ["WANDB_DISABLED"] = "true"

    model = Modified_LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model,use_gradient_checkpointing=False)
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # prepare training samples
    train_data_list = []

    if train_data_path.endswith(".json"):
        train_data_list.append(load_dataset("json", data_files=train_data_path))
    else:
        train_data_list.append(load_dataset(train_data_path))

    for i in range(len(train_data_list)):
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))

    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cuda:0")
            # adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Effort_Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=0,
            fp16=True,
            logging_strategy="no",
            optim="adamw_torch",
            save_strategy="no",
            output_dir="./models/",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            gradient_checkpointing=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # freeze all layers except for the last layer of Lora
    trainer.freeze_layers()

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # calculate effort score for each sample
    trainer.set_attribte()
    trainer.get_sample_grad = True
    trainer.model.base_model.model.get_sample_loss = True
    gradients = trainer.get_grad(resume_from_checkpoint=resume_from_checkpoint)
    all_gradients = torch.cat([_[0].unsqueeze(0) for _ in gradients], dim=0)
    effort_scores = torch.norm(all_gradients, dim=1)

    return effort_scores.cpu()