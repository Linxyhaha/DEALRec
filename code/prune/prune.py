from surrogate import train
from utils import get_args
from influence_score import get_influence_score
from effort_score import get_effort_score

import torch
import math
import random

if __name__ == '__main__':
    args = get_args()
    trainer = train(args)
    
    influence = get_influence_score(args, trainer)
    effort = get_effort_score(args)
    
    # normalization
    influence_norm = (influence-torch.min(influence))/(torch.max(influence)-torch.min(influence))
    effort_norm = (effort-torch.min(effort))/(torch.max(effort)-torch.min(effort))

    # overall score
    overall = influence_norm + args.lamda * effort_norm
    scores_sorted, indices = torch.sort(overall, descending=True)

    # coverage-enhanced sample selection
    n_prune = math.floor(args.hard_prune * len(scores_sorted))
    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    print(f"** after hard prune with {args.hard_prune*100}% data:", len(scores_sorted))

    # split scores into k ranges
    s_max = torch.max(scores_sorted)
    s_min = torch.min(scores_sorted)
    print("== max socre:", s_max)
    print("== min score:", s_min)
    interval = (s_max - s_min) / args.k

    s_split = [min(s_min + (interval * _), s_max)for _ in range(1, args.k+1)]

    score_split = [[] for _ in range(args.k)]
    for idxx, s in enumerate(scores_sorted):
        for idx, ref in enumerate(s_split):
            if s.item() <= ref:
                score_split[idx].append({indices[idxx].item():s.item()})
                break
    
    coreset = []
    m = args.n_fewshot
    while len(score_split):
        # select the group with fewest samples
        group = sorted(score_split, key=lambda x:len(x))
        if len(group) > 3:
            print("** sorted group length:", len(group[0]), len(group[1]), len(group[2]), len(group[3]),"...")
        
        group = [strat for strat in group if len(strat)]
        if len(group) > 3:
            print("** sorted group length after removing empty ones:", len(group[0]), len(group[1]), len(group[2]), len(group[3]),"...")

        budget = min(len(group[0]), math.floor(m/len(group)))
        print("** budget for current fewest group:", budget)
        
        # random select and add to the fewshot indices list
        fewest = group[0]
        selected_idx = random.sample([list(_.keys())[0] for _ in fewest], budget)
        coreset.extend(selected_idx)

        # remove the fewest group
        score_split = group[1:]
        m = m - len(selected_idx)
        
    print(f"** finish selecting {len(coreset)} samples.")

    torch.save(coreset, f"selected/{args.data}_{args.n_fewshot}.pt") # Alternatively, you may wish to save the lambda, group number, and hard prune ratio for hyper-parameter tuning.