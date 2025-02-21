from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import numpy as np
import torch
import os
import math
import json
import ipdb
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--n_fewshot", type=str, default="1024")
parse.add_argument("--filename", type=str, default="", help="the filename of the generation results under the ../../results folder")
parse.add_argument("--dataset", type=str, default="games")

args = parse.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

base_model = "/your/path/to/llama/hf/model/"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.half()  # seems to fix bugs for some users.

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

# games
movies = np.load(f'../../../data/{args.dataset}/title_maps.npy', allow_pickle=True).item()
movie_names = list(movies['seqid2title'].values())
movie_ids = [_ for _ in range(len(movie_names))]

movie_dict = dict(zip(movie_names, movie_ids))
result_dict = dict()
# ipdb.set_trace()

tokenizer.padding_side = "left"
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

item_embeddings = []
from tqdm import tqdm

model.eval()

with torch.no_grad():
    for i, name in tqdm(enumerate(batch(movie_names, 4))):
            input = tokenizer(name, return_tensors="pt", padding=True).to(device)
            input_ids = input.input_ids
            attention_mask = input.attention_mask
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            item_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
item_embeddings = torch.cat(item_embeddings, dim=0).cuda()


path = [f'../../results/{args.filename}.json']

for p in path:
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()
    text = [_["predict"].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm

    with torch.no_grad():
        for i, batch_input in tqdm(enumerate(batch(text, 2))):
            input = tokenizer(batch_input, return_tensors="pt", padding=True).to(device)
            input_ids = input.input_ids
            attention_mask = input.attention_mask
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
            
        predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()

    dist = torch.cdist(predict_embeddings, item_embeddings, p=2)
        
    rank = dist.detach().cpu()
    rank = rank.argsort(dim = -1).argsort(dim = -1) # get the rank list

    topk_list = [5, 10, 20, 50]

    NDCG = []
    for topk in topk_list:
        # S = 0
        ndcg_sum = 0
        for i in range(len(test_data)):
            idcg = 0
            ndcg = 0
            S = 0
            for idx, t in enumerate(test_data[i]['output']):
                t = t.strip("\"")
                t_id = movie_dict[t]
                rankId = rank[i][t_id].item()
                if rankId < topk:
                    S = S + (1 / math.log2(rankId + 2)) # todo: math.log2
                # idcg
                if idx+1 <= topk:
                    idcg += (1 / math.log2(idx + 2))
                # ndcg
            ndcg += S / idcg
            ndcg_sum += ndcg
        NDCG.append(round(ndcg_sum / len(test_data),4))

    RECALL = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            s = 0
            for t in test_data[i]['output']:
                t = t.strip("\"")
                t_id = movie_dict[t]
                rankId = rank[i][t_id].item()
                if rankId < topk:
                    s = s + 1
            S += s / len(test_data[i]['output'])
        RECALL.append(round(S / len(test_data), 4))

    print(RECALL)
    print(NDCG)
    
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["RECALL"] = RECALL

f = open(f'./results/{args.filename}.json', 'w')    
json.dump(result_dict, f, indent=4)