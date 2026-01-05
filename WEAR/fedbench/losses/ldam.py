
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        cls_num = torch.as_tensor(cls_num_list, dtype=torch.float32)
        cls_num = torch.clamp(cls_num, min=1.0)
        m_list = 1.0 / torch.pow(cls_num, 0.25)
        m_list = m_list / m_list.max() * max_m
        self.register_buffer("m_list", m_list)
        self.s = s
        self.weight = None
        
    def set_drw_weights(self, weight: torch.Tensor = None):
        self.weight = weight
        
    def forward(self, logits, target):
        if target.dtype != torch.long:
            target = target.long()
            
        # Create margin mask - only apply margin to correct class
        batch_size = logits.size(0)
        index = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        index.scatter_(1, target.view(-1, 1), True)
        
        # Move margins to same device as logits
        m_list = self.m_list.to(logits.device)

        # Apply class-specific margins
        batch_m = m_list.unsqueeze(0).expand(batch_size, -1)
        logits_with_margin = logits - batch_m * index.float()      
        # Use original logits for non-target classes, margin-adjusted for target class
        output = torch.where(index, logits_with_margin, logits)
        
        # Scale logits and compute cross-entropy loss
        scaled_output = self.s * output
        
        # Apply DRW weights if in second phase
        weight = None
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            
        loss = F.cross_entropy(scaled_output, target, weight=weight)
        # Handle NaN/Inf loss values by returning a zero loss to prevent training collapse
        if torch.isnan(loss) or torch.isinf(loss):
            # Log this event for debugging if necessary
            # print(f"Warning: NaN or Inf loss detected. Returning zero loss.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

def compute_drw_weights(cls_num_list, beta: float = 0.9999):
    cls_num = np.array(cls_num_list, dtype=np.float32)
    cls_num = np.maximum(cls_num, 1.0)  # Avoid division by zero
    
    # Effective number: (1 - β^n_i) / (1 - β)
    effective_num = 1.0 - np.power(beta, cls_num)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    
    # Normalize weights to sum to number of classes (standard practice)
    weights = weights / weights.sum() * len(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


class TinyHARWithNormalizedLogits(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Temporal convs
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Final classifier with normalized weights for LDAM
        self.classifier = nn.Linear(128, num_classes, bias=False)
        
        # Initialize classifier weights
        nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, x, normalize_logits=False):
        # x: (B, T, C) -> conv expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)           # (B, 128, T)
        x = x.transpose(1, 2)      # (B, T, 128)
        out, _ = self.lstm(x)      # (B, T, 256)
        
        # Attention over time
        a = self.attn(out)         # (B, T, 1)
        w = torch.softmax(a, dim=1)
        feat = (out * w).sum(dim=1)  # (B, 256)
        
        # Extract features
        features = self.feature_extractor(feat)  # (B, 128)
        
        if normalize_logits:
            # Normalize features to unit norm
            features = F.normalize(features, p=2, dim=1)
            
            # Normalize classifier weights to unit norm
            normalized_weights = F.normalize(self.classifier.weight, p=2, dim=1)
            
            # Compute normalized logits
            logits = F.linear(features, normalized_weights)
        else:
            logits = self.classifier(features)
            
        return logits