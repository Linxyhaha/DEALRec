import os
os.environ['LD_LIBRARY_PATH'] = '/your/path/to/lib/'

import numpy as np 
import fire
import torch

import json
import random
from tqdm import tqdm

import copy
import ipdb

from transformers import LlamaTokenizer  # noqa: F402


def gen_fewshot(
    # model/data params
    base_model: str = "/your/path/to/llama/hf/model/",  # the only required argument
    input_dir: str = "./data/",
    output_dir: str = "train",
    n_sample: int = 1024,
    dataset: str = "games",
    cutoff_len: int = 512,
    seed: int = 2023,
    # lora hyperparams
):  
    input_dir = input_dir + dataset + "/"
    random.seed(seed)
    # model.set_tau(tau)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def generate_and_truncate_prompt(instruction, i, o):
        full_prompt = generate_prompt(instruction, i, o)
        history = truncate_item(full_prompt)
        return history

    def truncate_item(q):
        q_instruct = q.split("### Input:", 1)[0]
        q_input = q.split('### Input:\n', 1)[1]
        q_input = "### Input:\n" + q_input.split('### Response:',1)[0]
        q_input1 = q_input.split(": ",1)[0] + ": "
        q_input2 = q_input.split(": ",1)[1]
        q_input2 = q_input2.split("\n\n",1)[0] + ";"
        q_response = "### Response:" + q.split('### Response:', 1)[1]
        # ipdb.set_trace()
        while len(tokenizer(q_instruct+q_input1+q_input2+"\n\n"+q_response, padding=False)['input_ids']) > cutoff_len:
            q_input2 = q_input2.split('; ',1)[1]
        return q_input1.split(":\n",1)[1]+q_input2
    

    #======================================================================================================#
                                        # construct few-shot samples #                                       
    #======================================================================================================#

    # 1. randomly sample 1024 samples from user sequences.
    # 2. truncate the sampled 1024 samples
    # 3. save into json file with the format of "instruction", "input", "output"
        
    def read_npy(file):
        return np.load(file,allow_pickle=True)

    def read_txt(file):
        with open(file,'r') as f:
            return f.readlines()
        
    def get_input(history_list, view="title"):
        if view=="title":
            if dataset=="games":
                return f"The user has purchased the following game products before, with titles as: {history_list}"
            elif dataset=="book":
                return f"The user has purchased the following books before, with titles as: {history_list}"
            elif dataset=="microlens-50k":
                return f"The user has watched the following micro-videos before, with titles as: {history_list}"

    title_maps = read_npy(input_dir+'/title_maps.npy').item()
    sequential_train = read_txt(input_dir+'/sequential_train.txt')
    sequential_valid = read_txt(input_dir+'/sequential_valid.txt')
    sequential_test = read_txt(input_dir+'/sequential_test.txt')


    print("loading scores files...")
    if "test" not in output_dir and n_sample!=len(sequential_train):
        fewshot_idx = torch.load(f"../../prune/selected/{dataset}_{n_sample}.pt")
        fewshot_train = []
        fewshot_valid = []

        for idx in fewshot_idx:
            fewshot_train.append(sequential_train[idx])
            fewshot_valid.append(sequential_valid[idx])
    
    if n_sample == len(sequential_train):
        fewshot_train = sequential_train
        fewshot_valid = sequential_valid

    if "train" in output_dir:
        last_index = -1
    elif "valid" in output_dir:
        last_index = -2
    elif "test" in output_dir:
        fewshot_train = sequential_test

    if dataset == "games":
        instruction = "Given the game products that the user purchased before, please recommend a new game product that the user likes to the user."
    elif dataset == "book":
        instruction = "Given the books that the user purchased before, please recommend a new book that the user likes to the user."
    elif dataset == "microlens-50k":
        instruction = "Given the micro-videos that the user watched before, please recommend a new micro-video that the user likes to the user."

    datapoints = []
    test_cnt = 0
    for idx, sample in tqdm(enumerate(fewshot_train)):

        sample = sample.strip().split()
        
        if "train" in output_dir:
            _, i_ids, target_iid = sample[0], sample[1:last_index], sample[last_index]

        elif "valid" in output_dir:
            i_ids = sample[1:]
            sample_valid = fewshot_valid[idx].strip().split()
            if len(sample_valid)==1:
                continue
            target_iid = sample_valid[1]
        elif "test" in output_dir:
            sample_train = sequential_train[idx].strip().split()
            sample_valid = sequential_valid[idx].strip().split()
            sample_test = sequential_test[idx].strip().split()
            if len(sample_test)==1:
                continue
            test_cnt += 1
            i_ids = sample_train[1:] + sample_valid[1:] if len(sample_valid)>1 else sample_train[1:]
            target_iid = sample[1:] if len(sample)>1 else sample_train[1]

        res = {"instruction":instruction, \
               "input":None, "output":None}
        
        # title
        titles = [title_maps['seqid2title'][i_id] for i_id in i_ids]
        input_prompt = "; ".join(titles)
        if "\n" in input_prompt:
            input_prompt = "".join(input_prompt.split('\n'))
        input_prompt = get_input(input_prompt,"title")

        # transfer target into a list, original is a string, only one item
        if "test" in output_dir:
            target = []
            for tid in target_iid:
                t_str = title_maps['seqid2title'][tid]
                if "\n" in t_str:
                    t_str = "".join(t_str.split('\n'))
                target.append(t_str)
        else:
            target = title_maps['seqid2title'][target_iid]

        q_input2 = generate_and_truncate_prompt(instruction, input_prompt, target[0])

        res["input"] = q_input2

        if "test" in output_dir:
            res["output"] = target
        else:
            res["output"] = f"{target}"
            
        datapoints.append(copy.deepcopy(res))
  
    print("** test user: ", test_cnt)
    
    path = "train" if "train" in output_dir else "valid"
    with open(f"{dataset}/{path}/{output_dir}-{n_sample}.json", "w") as f:
        json.dump(datapoints, f, indent=4)

def generate_prompt(instruction, input, output):
    # sorry about the formatting disaster gotta move fast
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

if __name__ == "__main__":
    fire.Fire(gen_fewshot)