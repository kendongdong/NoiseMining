import torch
import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from torch import Tensor

from .binarycrossentropy import BinaryCrossEntropy

class CorrectLoss(BinaryCrossEntropy):
 
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.total_epochs = params['total_epochs']
        self.loss_div_dir = params['loss_div_dirname']
        self.train_data_list = params['train_data_list']
        self.maskmat_root = params['maskmat_root']
        self.warmup_epochs = params['warmup_epochs']
        self.epsilon = None
        self.discretize_threshold = params['discretize_threshold']

        if not os.path.exists(self.maskmat_root):
            self.create_noise_mask_from_instance()
        
    def create_noise_mask_from_instance(self, epoch=5):
        os.system('mkdir -p %s'%(self.maskmat_root))
        with open(self.train_data_list,'r') as file:
            train_list = [x.strip() for x in file.readlines()]
        file.close()
        for name in tqdm(train_list):
            loss_div_file = os.path.join(self.loss_div_dir, name +'.mat')
            loss_div_mat = sio.loadmat(loss_div_file)
            loss_div = loss_div_mat['loss_div']
            loss_div = loss_div[epoch,:]
            loss_mask = np.where(loss_div>0.,0,1)
            mask_file = os.path.join(self.maskmat_root, name +'_mask.mat')
            sio.savemat(mask_file, {'loss_mask':loss_mask})
    
    def load_noise_mask_for_batch(self, name_list):
        b = len(name_list)
        batch_masks = torch.zeros((b, 256, 256))
        for num, name in enumerate(name_list):
            mask_mat_file = os.path.join(self.maskmat_root, name +'_mask.mat')
            mask_mat = sio.loadmat(mask_mat_file)
            batch_masks[num] = torch.from_numpy(mask_mat['loss_mask'])
        batch_masks = torch.unsqueeze(batch_masks, dim=1)
        return batch_masks
        
    def update_epsilon(self, preds, cur_time=None):
        with torch.no_grad():
            class_num = 2 # set as 2
            H_preds = torch.sum(
                -(preds + 1e-12) * torch.log(preds + 1e-12), 1
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num)) 
            example_trust = 1 - H_preds / H_uniform
            self.epsilon = example_trust
            self.epsilon = self.epsilon[:, None]

    def forward(
        self, logits: Tensor, targets: Tensor, batch_name_list: None, cur_time: None 
        ) -> Tensor:

        if cur_time <= self.warmup_epochs:
            targets = self.discretize(targets, self.discretize_threshold).float()
            preds = self.sigmoid(logits).float()
            preds = self.threshold_prediction(preds, targets, self.discretize_threshold)
            return super().forward(logits, preds, targets, cur_time)
        else:
            preds = self.sigmoid(logits).float()
            self.update_epsilon(preds)
            preds = self.threshold_prediction(preds, targets, self.discretize_threshold)
            batch_masks = self.load_noise_mask_for_batch(batch_name_list) 
            index_noise = np.where(batch_masks==0)

            targets[index_noise] = self.discretize(targets[index_noise], self.discretize_threshold).float()
            targets[index_noise] = (1 - self.epsilon[index_noise]) * targets[index_noise] + self.epsilon[index_noise] * preds[index_noise] # ohter setting is epsilon == 0.1
            return super().forward(logits, preds, targets, cur_time)

    def sigmoid(self,pred):
        '''
        input： (b,2,256,256)
        '''
        pred = torch.sigmoid(pred)
        pred = pred[:, 1, :, :]
        pred = torch.unsqueeze(pred, dim=1)
        return pred
    
    def discretize(self, In, a):
        return (In>a)
    
    
    def threshold_prediction(self, pred, target, a):
        #Prepare pred for thresholding
        t_up=(target>a).int()
        t_low=(target<a).int()
        a1=2
        a0=1
        Z=a1*(pred>target).int()*t_up + a0*(pred<target).int()*t_low-1
        return (Z==-1).float()*pred + Z.float().clamp(0,1)
         
