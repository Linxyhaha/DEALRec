import numpy as np
import json

import math
import argparse

import ipdb

def computeTopNAccuracy(GroundTruth, predictedIndices, topN, rank=None):
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
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                # if (rank is not None) and (rank==0):
                #     ipdb.set_trace()
                user_length += 1
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
                        if (rank is not None) and (rank==0):
                            print("correct one!!")
                            # ipdb.set_trace()
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
        
        # print(f"{cnt} samples retrieve less than top {topN[index]} samples.")
        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR


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

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', type=str, help='output path for json file')
    parser.add_argument("--test_file", type=str, default="/storage/xylin/LLM4Rec/efficient_tuning/code/games/Grounding4Rec/data/game/test/test.json")
    parser.add_argument("--file_name", type=str, default="trie_outputs", help="file path of saved intermediate outputs from ddp testing script")
    parser.add_argument("--output_exist", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()

    if not args.output_exist:
        outputs = np.load(f"{args.file_name}.npy", allow_pickle=True).tolist()

        with open(f"{args.test_file}", 'r') as f:
            test_file = json.load(f)

        print("** outputs length", len(outputs))
        print("** test_file length", len(test_file))

        for i, _ in enumerate(outputs):
            test_file[i]["predict"] = outputs[i]

        with open(f"/storage/xylin/LLM4Rec/efficient_tuning/code/games/Grounding4Rec/results/trie/{args.output_json}.json","w") as f:
            json.dump(test_file, f, indent=4)
    else:
        with open(f"/storage/xylin/LLM4Rec/efficient_tuning/code/games/Grounding4Rec/results/{args.output_json}.json", "r") as f:
            test_file = json.load(f)

    all_pred_list = []
    all_gold_list = []

    for i, instance in enumerate(test_file):
        # if i == len(outputs)-1:
        #     break
        try:
            all_pred_list.append(instance['predict'])
            all_gold_list.append(instance['output'])
        except:
            break

    test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5,10])
    print_results(None, None, test_results)    