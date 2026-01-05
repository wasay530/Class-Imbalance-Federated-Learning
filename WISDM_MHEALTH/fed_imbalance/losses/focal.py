import torch, torch.nn.functional as F

def focal_loss(logits, targets, gamma:float=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1-pt)**gamma) * ce
    return loss.mean()