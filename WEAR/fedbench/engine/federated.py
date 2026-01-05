import copy, numpy as np, torch
from typing import Dict, Any, List, Tuple

def get_weights(model):
    return {k: v.cpu().detach().clone() for k,v in model.state_dict().items()}

def set_weights(model, weights:dict):
    model.load_state_dict(weights, strict=True)

def fedavg(ckpts: List[Tuple[dict,int]]):
    # list of (state_dict, num_samples)
    total = sum(n for _,n in ckpts)
    agg = None
    for sd, n in ckpts:
        if agg is None:
            agg = {k: v.clone() * (n/total) for k,v in sd.items()}
        else:
            for k in agg:
                agg[k] += sd[k] * (n/total)
    return agg

def client_weights_by_balance(y_train: np.ndarray, num_classes:int)->float:
    # Balanced weighting: more balanced clients get higher weight
    counts = np.bincount(y_train, minlength=num_classes)
    if counts.sum()==0: return 0.0
    p = counts / counts.sum()
    # balance score = 1 - L1 distance to uniform
    uni = np.ones(num_classes)/num_classes
    score = 1.0 - 0.5 * np.abs(p-uni).sum()
    return float(max(1e-6, score))
