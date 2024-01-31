import argparse
import numpy as np
import pandas as pd
import math
import random
import os
from scipy.sparse import csr_matrix
import torch
import ipdb


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--output_dir', default='./models/', type=str)
    parser.add_argument('--data_name', default='games', type=str)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--log_name',type=str, default='log')

    # surrogate model args (SASRec) 
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64,help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int) # 1,2,3
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5,help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5,help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument('--do_eval', action='store_true', help='testing mode') 
    parser.add_argument("--batch_size", type=int, default=500, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--topN", default='[5, 10, 20, 50]', help="the recommended item num")
    
    # optimization args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    
    # LLM args
    parser.add_argument("--base_model", type=str, default="/your/path/to/hf/llama/model", help="your path to hf llama model weight")
    parser.add_argument("--train_data_path", type=str, default="", help="your path of the training data")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="path of the alpaca lora adapter")

    # DEALRec args
    parser.add_argument('--n_fewshot', default=1024, type=int)
    parser.add_argument("--lamda", type=float, default=0.5, help="strength of gap regularization (effort score)")
    parser.add_argument('--k', default=25, type=int, help="number of groups")
    parser.add_argument("--hard_prune", default=0.1, type=float, help='percentage of hard samples to prune at first')

    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, user_seq_tr, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list:  #
            row.append(user_id)
            col.append(item)
            data.append(1)
    for user_id, item_list in enumerate(user_seq_tr):
        for item in item_list:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs_npy(data_file):
    print(data_file)
    data_dict = np.load(data_file,allow_pickle=True).item()
    data_dict = dict(sorted(data_dict.items()))
    user_seq = []
    max_item = 0
    for u_id in data_dict:
        for i,i_id in enumerate(data_dict[u_id]):
            data_dict[u_id][i] = i_id + 1
        user_seq.append(data_dict[u_id])
        if len(data_dict[u_id])!=0:
            max_item = max(max_item, max(data_dict[u_id]))
    return user_seq 

def get_statistics(path):
    item2id = np.load(path + "/item_map.npy", allow_pickle=True).item()
    user2id = np.load(path + "/user_map.npy", allow_pickle=True).item()
    max_item, num_users, num_items = len(item2id), len(user2id), len(item2id)+1
    return max_item, num_users, num_items
