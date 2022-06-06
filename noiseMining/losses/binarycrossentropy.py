import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, tensor





class BinaryCrossEntropy(nn.Module):

    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(weight=None,size_average=False)

    def forward(self, logits: Tensor, preds: Tensor, targets: Tensor, cur_time: None) -> Tensor:
        """
        Inputs:
            inputs: predictions of shape (b, 2, w, h), logits without sigmoid.
            targets: targets of shape (b, 1, w, h).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        num_examples = logits.shape[0]
        self.beta = 0.001
        loss = - targets * torch.log(preds + 1e-10) - (1 - targets) * torch.log(1 - preds + 1e-10)
       
        loss_lnp = -torch.log(F.softmax(logits) + 1e-10)
        loss =  loss - self.beta * torch.mean(loss_lnp, 1, keepdim=True) 
        loss = torch.sum(loss) / num_examples
    
        return loss


    def sigmoid_my(self,pred):

        pred = torch.sigmoid(pred)
        pred = pred[:, 1, :, :]
        pred = torch.unsqueeze(pred, dim=1)
        return pred