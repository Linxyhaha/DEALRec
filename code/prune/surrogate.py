import os
import numpy as np
import torch
import ipdb
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets_util import FinetuneDataset,SeqDataset
from trainers import FineTrainer
from utils import generate_rating_matrix_valid,generate_rating_matrix_test, get_user_seqs_npy, check_path, set_seed, get_statistics
from models import SASRecModel

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

def train(args):

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # load data
    args.data_file = args.data_dir + args.data_name + '/training_dict.npy'
    val_file = args.data_dir + args.data_name + '/validation_dict.npy'
    test_file = args.data_dir + args.data_name + '/testing_dict.npy'

    max_item, num_users, num_items = get_statistics(args.data_dir + args.data_name)
    args.item_size = max_item + 1

    user_seq = get_user_seqs_npy(args.data_file)
    user_seq_val = get_user_seqs_npy(val_file)
    user_seq_tst = get_user_seqs_npy(test_file)

    mask_tr = generate_rating_matrix_valid(user_seq, num_users, num_items)
    mask_tv = generate_rating_matrix_test(user_seq_val, user_seq, num_users, num_items)
    
    # set item score in train set to `0` in validation
    args.tr_matrix = mask_tr
    args.tv_matrix = mask_tv

    # save dir
    args.checkpoint_path = args.output_dir + '{}.pth'.format(args.data_name)

    # build data
    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SeqDataset(args, user_seq_val, user_seq_tr=user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SeqDataset(args, user_seq_tst, user_seq_tr=user_seq, user_seq_val=user_seq_val, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    
    # build model
    model = SASRecModel(args)

    trainer = FineTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)
    
    if args.do_eval: # testing mode
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        valid_results = trainer.valid(0)
        test_results = trainer.test(0)
        print('==='*18)
        print_results(None,valid_results,test_results)
        print('==='*18)
        
    else:
        best_recall=0
        for epoch in range(args.epochs):
            epoch_start_time=time.time()
            avg_loss = trainer.train(epoch)

            # evaluate
            if (epoch + 1) % 5==0:
                valid_results, test_results = None, None
                valid_results = trainer.valid(epoch)
                test_results = trainer.test(0)
                if valid_results[1][0]>best_recall:
                    best_epoch=epoch
                    best_recall=valid_results[1][0]
                    best_results = valid_results
                    best_test_results = test_results
                print('---'*18)
                print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(avg_loss) + " costs " + time.strftime("%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
                print_results(None, valid_results, test_results)
                print('---'*18)
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                torch.save(model,args.checkpoint_path)

        print('==='*18)
        print("End. Best Epoch {:03d} ".format(best_epoch))
        print_results(None, best_results, best_test_results)

    # load the best model for data pruning
    trainer.load(args.checkpoint_path)

    # prepare the training samples in original order to calculate influence score
    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    trainer.train_dataset = train_dataset
    trainer.train_dataloader = train_dataloader

    return trainer