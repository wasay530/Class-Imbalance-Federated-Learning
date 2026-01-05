import torch, torch.nn as nn
import torch.nn.functional as F

class RatioLoss(nn.Module):
    def __init__(self, alpha:float=1.0, beta:float=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, logits, targets):
        # Balanced softmax-esque prior + ratio penalty on predicted distribution skew
        ce = F.cross_entropy(logits, targets)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1).mean(dim=0)  # mean prob per class in batch
        ratio_pen = ((probs - probs.mean())**2).sum()
        return ce + self.alpha * ratio_pen + self.beta * 0.0