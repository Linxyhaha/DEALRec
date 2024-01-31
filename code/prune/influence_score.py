import torch
from surrogate import train
from utils import get_args

from tqdm import tqdm
import random
from torch.autograd import grad

def get_v(shape):
    return torch.ones(shape, dtype=torch.float32)

def cal_grad_z(sample_idx, trainer):
    loss = trainer.get_sample_loss(sample_idx)
    for name, params in trainer.model.named_parameters():
        if name == "item_encoder.layer.1.attention.dense.weight":
            return list(grad(loss, params))
        
def cal_grad_batch_z(batch, trainer):
    loss = trainer.get_batch_loss(batch)
    for name, params in trainer.model.named_parameters():
        if name == "item_encoder.layer.1.attention.dense.weight":
            return list(grad(loss, params))
        
def hvp(y, w, v):
    """
    ``y`` is the scalor of loss value
    ``w`` is the model parameters
    ``v`` is the H^{-1}v at last step
    """
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def estimate_hv(trainer, args, h_estimate, v):

    recursion_depth = args.recursion_depth
    damp = 0.01
    scale = 25
    
    for _ in tqdm(range(recursion_depth),total=recursion_depth):

        random_idx = random.choice(list(range(len(trainer.train_dataset))))
        loss = trainer.get_sample_loss(random_idx)
        params = [ p for n, p in trainer.model.named_parameters() if n == "item_encoder.layer.1.attention.dense.weight" ] # we follow previous work to calculate the last linear layer for high efficiency

        hv = hvp(loss, params, h_estimate)

        # Recursively caclulate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            h_estimate = [_.detach() for _ in h_estimate]

    hv = [_.reshape(-1) for _ in hv]
    h_estimate = [_.reshape(-1) for _ in h_estimate]
    v = [_.reshape(-1) for _ in v]

    return h_estimate

def calculate_influence_score(args, arg_list):
    idx, trainer, v = arg_list
    for _ in range(args.iteration):
        h_estimate = v.copy()
        H_inverse = estimate_hv(trainer, args, h_estimate, v) if _ == 0 else H_inverse + estimate_hv(trainer, args, h_estimate, v)
    H_inverse = [ _.data  / args.iteration for _ in H_inverse]
    return (idx, torch.norm(H_inverse[0]))


def get_influence_score(args, trainer):

    influence_score = [0 for _ in range(len(trainer.train_dataset))]

    # step 1. calculate constant vector in Eq.(11)
    v_list = []
    for i, batch in enumerate(trainer.train_dataloader):
        v_list.append(cal_grad_batch_z(batch, trainer)[0].unsqueeze(0))
    v = [torch.mean(torch.cat(v_list, dim=0), dim=0)]

    # step 2. HVP estimation <- Eq.(10)
    # calculate the H^{-1}v for r repeats and get the average, here we only compute once for efficiency
    for _ in range(args.iteration):
        h_estimate = v.copy()
        H_inverse = estimate_hv(trainer, args, h_estimate, v) if _ == 0 else H_inverse + estimate_hv(trainer, args, h_estimate, v)
    H_inverse = [ _.data  / args.iteration for _ in H_inverse]

    # step 3. calculate the influence score for each sample. <- Eq.(11)
    for idx in tqdm(range(len(trainer.train_dataset)), total=len(trainer.train_dataset)):
        sample = cal_grad_z(idx, trainer)
        score = torch.matmul(H_inverse[0], sample[0].view(-1).T)
        influence_score[idx] = score
        
    influence_score = torch.tensor(influence_score)

    return influence_score