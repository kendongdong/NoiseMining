import torch
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from torch.nn import DataParallel


from noiseMining.optim.adam_multistep import AdamMultiStep, WarmUpLR
from noiseMining.slicegetter.get_dataloader import  DataLoaderPool
from noiseMining.slicegetter.get_network import NetworkPool
from noiseMining.slicegetter.get_lossfunction import LossPool
from utils import max2d, min2d, sigmoid, imsave, check_point


get_network = NetworkPool.get_network
get_dataloader = DataLoaderPool.get_dataloader
get_lossfunction = LossPool.get_lossfunction
colorscale = [[0, "#4d004c"], [0.5, "#f2e5ff"], [1, "#ffffff"]]

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
gpus = [0,1,2]

class Trainer:
    '''
    Inputs:
        1. dataloader
        2. network
        3. loss
        4. optimiser
        5. device
    '''

    def  __init__(self,params):
        
    
        #network
        self.network_name = params['network_name']
        self.network = get_network(params)
        self.network.train(mode=True)
        self.network = DataParallel(self.network.cuda(), device_ids=gpus, output_device=gpus[0])

        #dataloader
        params['dataset_type'] = 'train'
        self.traindataloader = get_dataloader(params)
        self.data_name = params['data_name']
        self.batch_size = params['batch_size']
        self.checkpoint_freq = params['checkpoint_freq']
        self.checkpoint_dir = ''

        self.total_epochs = params['total_epochs']
        self.loss_name = params['loss_name']
        self.loss_div_dirname = params['loss_div_dirname']
       
        #loss function
        self.loss_criterion = get_lossfunction(params)

        #visualiraztion 
        self.visual_root = params['visual_root']
        

        #optim with optimser and lr scheduler 
        self.optim = AdamMultiStep(net_params=self.network.parameters(), params=params)
        self.warmup_epochs = params['warmup_epochs']
        self.optim.warmup_scheduler = WarmUpLR(
            optimizer= self.optim.optimizer,
            total_iters= len(self.traindataloader) * self.warmup_epochs,
        )

        #logging and preparing
        self.params = params
        self.noisy_data_analysis_prepare()
        self.init_logger()
        self.create_loss_div_file()
        

   
    def init_logger(self):
        if not os.path.exists(self.visual_root):
                    os.system('mkdir -p %s'%(self.visual_root))
        # initial the checkpoint dir
        self.checkpoint_dir = os.path.join(self.params['checkpoint_root'], self.params['exp_name'], self.params['network_name'])
        if os.path.exists(self.checkpoint_dir):   
            self.network.load_state_dict(torch.load(self.checkpoint_dir))
            print('model is loaded')
        else:
            os.system('mkdir -p %s'%(self.checkpoint_dir))
        self.accuracy_dynamics = {'epoch': []}
        self.loss_dynamics = {'epoch': []}


    def create_loss_div_file(self):

        loss_div_dir = self.loss_div_dirname
        if not os.path.exists(loss_div_dir):
            os.system('mkdir -p %s'%(loss_div_dir))
        else:
            print('Loss div all files are already created.')
            return 
 
        path = self.params['train_data_list']
        with open(path,'r') as file:
            filename_list = [x.strip() for x in file.readlines()]
     
        print('Loss div files are creating ... ')
        for num in tqdm(range(len(filename_list))):
            loss_div = np.zeros((self.total_epochs,256,256))
            loss_div_file = os.path.join(self.loss_div_dirname, filename_list[num] + '.mat')
            sio.savemat(loss_div_file, {'loss_div':loss_div})
        print('Loss div files are created successfully ! ')
            

    def train(self) -> None:
        for epoch in range(1, self.total_epochs + 1):
            # train one epoch
            self.train_one_epoch(
                epoch=epoch,
                dataloader=self.traindataloader,
            )
            #lr scheduler
            if epoch > self.warmup_epochs:
                self.optim.lr_scheduler.step()
            

    def train_one_epoch(self, epoch: int, dataloader) -> None:
        self.network.train() 

        for batch_index, data in enumerate(dataloader):
            
            network_inputs = data['image'].cuda()
            labels = data['noisy_label'].cuda()
            name_list = data['img_name']
            
            y_n_min, y_n_max = min2d(labels), max2d(labels)
            labels = (labels - y_n_min) / (y_n_max - y_n_min)
            
            #forward
            logits = self.network(network_inputs)
            preds = sigmoid(logits).float()
            
            if self.loss_name == 'selectloss':
                loss = self.loss_criterion(
                    params = self.params,
                    data = data,
                    logits = logits,
                    targets = labels,
                    cur_time = epoch
                ) 
            elif self.loss_name == 'correctloss': 
                loss = self.loss_criterion(
                    logits = logits,
                    targets = labels,
                    batch_name_list = name_list,
                    cur_time = epoch,
                )
            else:
                loss = self.loss_criterion(
                    preds = preds,
                    targets = labels,
                )
            
       
            
            print('loss: {}, iteration: {}/{}, epoch : {}/{}'.format(loss, batch_index, len(dataloader), epoch, self.total_epochs))
            #backward
            self.optim.optimizer.zero_grad()
            loss.backward()
            self.optim.optimizer.step()

            if epoch <= self.warmup_epochs:
                self.optim.warmup_scheduler.step()
    
        torch.cuda.empty_cache()

        if epoch % self.checkpoint_freq == 0:
            check_point(self.params, epoch, self.network, self.checkpoint_dir)
        
    


    @torch.no_grad()
    def evaluation(self, dataloader): # using noisy labels, keep setting with training
        self.network.eval()

        test_loss = 0.0
        for iter_idx, data in enumerate(dataloader):

            network_inputs = data['image'].cuda()
            name_list = data['img_name']
            labels = labels.cuda()
            y_n_min, y_n_max = min2d(labels), max2d(labels)
            labels = (labels - y_n_min) / (y_n_max - y_n_min)
            
            logits = self.network(network_inputs)
            preds = sigmoid(logits).float()

            if self.loss_name == 'instance':
                loss = self.loss_criterion(
                    data = data,
                    preds = preds,
                    targets = labels,
                )
            else:
                loss = self.loss_criterion(
                    preds = preds,
                    targets = labels,
                    batch_name_list = name_list,
                )
            test_loss += loss.item()
            test_loss = test_loss / len(dataloader.dataset)
           

            return test_loss
    
    
