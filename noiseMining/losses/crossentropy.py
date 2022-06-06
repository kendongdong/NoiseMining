import torch
import torch.nn as nn
from torch import Tensor


criterion = nn.BCELoss(weight=None,size_average=False)
class CrossEntropy(nn.Module):

    def __init__(self, params: dict = None) -> None:
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Inputs:
            pred_probs: predictions of shape (N, C).
            target_probs: targets of shape (N, C).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        
        num_examples = preds.shape[0]
        loss = criterion(preds,targets)
        loss = torch.sum(loss) / num_examples
        return loss
