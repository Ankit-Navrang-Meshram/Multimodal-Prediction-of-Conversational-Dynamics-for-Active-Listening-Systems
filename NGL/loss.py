import torch
import torch.nn as nn
import torch.nn.functional as F

class DNCCLoss(nn.Module):
    """
    Deep Negative Correlation Classification Loss.
    Encourages individual accuracy while penalizing agreement (forcing diversity).
    """
    def __init__(self, lambda_div=0.5):
        super(DNCCLoss, self).__init__()
        self.lambda_div = lambda_div
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, ensemble_logits, targets):
        """
        ensemble_logits: Tensor of shape (9, Batch_Size, Num_Classes)
        targets: Tensor of shape (Batch_Size,) containing labels {0, 1, 2}
        """
        num_members = ensemble_logits.size(0)

        # 1. Convert logits to probabilities: p_m
        probs = F.softmax(ensemble_logits, dim=-1)

        # 2. Calculate the detached ensemble average: p_bar
        # Detaching is required to treat the ensemble mean as a fixed target for the KL divergence
        p_bar = probs.mean(dim=0).detach() 

        total_loss = 0.0

        for m in range(num_members):
            logits_m = ensemble_logits[m]
            
            # TERM 1: Individual Accuracy 
            # Standard Cross Entropy between member predictions and true labels
            ce = self.ce_loss(logits_m, targets)

            # TERM 2: Diversity Penalty (Negative Correlation)
            # F.kl_div in PyTorch computes KL(Target || Input). 
            # It expects the Input to be in log-space, and Target in standard probability space.
            log_probs_m = F.log_softmax(logits_m, dim=-1)
            
            # Use 'batchmean' to properly average over the batch dimension
            kl = F.kl_div(input=log_probs_m, target=p_bar, reduction='batchmean')

            # Member Total Loss
            l_m = ce - (self.lambda_div * kl)
            total_loss += l_m

        # Final loss is the average loss of the 9 ensemble members
        return total_loss / num_members