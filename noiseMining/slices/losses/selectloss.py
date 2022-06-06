import torch
import os
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
from torch import Tensor
from torch import nn

from .binarycrossentropy import BinaryCrossEntropy

class SelectLoss(BinaryCrossEntropy):
 
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.beta = None
        self.criterion = nn.BCELoss(weight=None, reduce=False, reduction=None)
        self.params = params

    def forward(
        self, params, data, logits: Tensor, targets: Tensor, cur_time=0
    ) -> Tensor:  
        self.beta = 0.001
        preds = self.sigmoid_my(logits)
        loss = self.criterion(preds, targets)
        loss_ = -torch.log(F.softmax(logits) + 1e-8) 
        num_batch = len(loss)
        loss =  loss - self.beta * torch.mean(loss_,1,keepdim=True) 
        # select    
        loss_v = np.zeros((num_batch,1,256,256))
        loss_div = loss - torch.mean(loss_, 1,keepdim=True)
        loss_div_np = loss_div.data.cpu().numpy()
        loss_np = loss.data.cpu().numpy()
        # save middle outputs
        if params['save_loss_file']:
            for num in range(num_batch):
                loss_file = os.path.join(self.params['loss_dirname'],data['img_name'][num]+'.mat')
                loss_mat = sio.loadmat(loss_file)
                loss_heatmap = loss_mat['loss']
                loss_heatmap[cur_time] = loss_np[num]
                sio.savemat(loss_file, {'loss':loss_heatmap})

        if params['save_loss_div_file']:
            for num in range(num_batch):
                loss_div_file = os.path.join(self.params['loss_div_dirname'],data['img_name'][num]+'.mat')
                loss_div_mat = sio.loadmat(loss_div_file)
                loss_div = loss_div_mat['loss_div']
                loss_div[cur_time] = loss_div_np[num]
                sio.savemat(loss_div_file, {'loss_div':loss_div})
               
        for i in range(num_batch):
            if cur_time <=10:
                loss_v[i] = 1.0
            elif loss_div_np[i] <= 0:
                loss_v[i] = 1.0

        loss_v = loss_v.astype(np.float32) 
        loss_v_gpu = torch.from_numpy(loss_v).cuda() 
        loss_ = loss_v_gpu * loss       
        return torch.mean(torch.sum(loss_)/(sum(loss_v_gpu) + 1e-8))

    def sigmoid_my(self,pred):
        pred = torch.sigmoid(pred)
        pred = pred[:, 1, :, :]
        pred = torch.unsqueeze(pred, dim=1)
        return pred


  


    
   
  