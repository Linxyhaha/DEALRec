import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import math

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                user_length += 1
        
        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR

        
class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, group=False):
        return self.iteration(epoch, self.eval_dataloader, train=False, mask_valid=False, group=group)

    def test(self, epoch,group=False):
        return self.iteration(epoch, self.test_dataloader, train=False, mask_valid=True, group=group)

    def iteration(self, epoch, dataloader, train=True,mask_valid=False):
        raise NotImplementedError

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        try:
            self.model.load_state_dict(torch.load(file_name))
        except:
            self.model = torch.load(file_name)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.reshape(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(- torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                         torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget) / (torch.sum(istarget) + 1e-24)
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class FineTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FineTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, train=True, mask_valid=False, group=False):

        if train:
            self.model.train()
            rec_avg_loss = 0.0

            for i, batch in enumerate(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _, _ = batch

                sequence_output = self.model.finetune(input_ids)
            
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                self.optim.zero_grad()
                loss.backward()

                self.optim.step()

                rec_avg_loss += loss.item()
            
            avg_loss = rec_avg_loss / len(dataloader)
            return avg_loss

        else:
            self.model.eval()
            with torch.no_grad():
                pred_list = None
                answer_list = None

                for i, batch in enumerate(dataloader):
                
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers_arr, answer_num = batch
                    answers = []

                    for sample in range(len(answers_arr)):
                        answers.append(answers_arr[sample][-answer_num[sample]:].cpu().numpy())
                    
                    recommend_output = self.model.finetune(input_ids)
                    recommend_output = recommend_output[:, -1, :]    # shape:[batchsize,hiddensize]

                    rating_pred = self.predict_full(recommend_output) 

                    rating_pred = rating_pred.cpu().data.numpy().copy() # shape:[bs,n_items]
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.tr_matrix[batch_user_index].toarray() > 0] = 0

                    if mask_valid:
                        rating_pred[self.args.tv_matrix[batch_user_index].toarray()>0] = 0

                    ind = np.argpartition(rating_pred, -100)[:, -100:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list.extend(answers)

                all_pred = []
                all_answer = []
                for idxx, answer in enumerate(answer_list):
                    if np.sum(answer):
                        all_pred.append(pred_list[idxx])
                        all_answer.append(answer_list[idxx])

                results = computeTopNAccuracy(all_answer, all_pred, eval(self.args.topN))

                return results

    def get_sample_loss(self, idx):

        self.model.eval()
        sample = self.train_dataset[idx]
        sample = tuple(_.to(self.device).unsqueeze(0) for _ in sample)

        _, input_ids, target_pos, target_neg, _, _ = sample

        # Binary cross_entropy
        sequence_output = self.model.finetune(input_ids)

        loss = self.cross_entropy(sequence_output, target_pos, target_neg)
        
        return loss

    def get_batch_loss(self, batch):

        self.model.eval()
                 
        batch = tuple(t.to(self.device) for t in batch)
        _, input_ids, target_pos, target_neg, _, _ = batch

        # Binary cross_entropy
        sequence_output = self.model.finetune(input_ids)

        loss = self.cross_entropy(sequence_output, target_pos, target_neg)

        return loss
