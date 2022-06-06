"""
Pixel-Level Noise Mining for Weakly Supervised Salient Object Prediction
Submission: NIPS 2022
"""
import torch
import os
from torch.nn import DataParallel

from noiseMining.slicegetter.get_dataloader import  DataLoaderPool
from noiseMining.slicegetter.get_network import NetworkPool
from utils import max2d, min2d, sigmoid, imsave, check_point


get_network = NetworkPool.get_network
get_dataloader = DataLoaderPool.get_dataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
gpus = [0,1,2]


class Test:
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
        self.network = DataParallel(self.network.cuda(), device_ids=gpus, output_device=gpus[0])

        #dataloader
        params['dataset_type'] = 'test'
        self.testdataloader = get_dataloader(params)
        self.data_name = params['data_name']
        self.batch_size = params['batch_size']
        self.checkpoint_dir = params['check_points']

        #visualiraztion 
        self.save_root = params['save_root']
        self.params = params

        
    def init_logger(self):
        if not os.path.exists(self.save_root):
                    os.system('mkdir -p %s'%(self.save_root))
        # initial the checkpoint dir
        self.checkpoint_dir = os.path.join(self.params['checkpoint_root'], self.params['exp_name'], self.params['network_name'])
        self.network.load_state_dict(torch.load(self.checkpoint_dir))
        print('Model is loaded')
        

    def test(self, dataloader) -> None:
        self.network.eval() 

        for _, data in enumerate(dataloader):
            network_inputs = data['image'].cuda()
            labels = data['noisy_label'].cuda()
            name_list = data['img_name']
            
            y_n_min, y_n_max = min2d(labels), max2d(labels)
            labels = (labels - y_n_min) / (y_n_max - y_n_min)
            #forward
            logits = self.network(network_inputs)
            preds = sigmoid(logits).float()
            
            
            for i in self.batch_size:
                outputs = preds[i][0]
                outputs = outputs.cpu().data.resize_(256, 256)
                imsave(os.path.join(self.save_root ,name_list[i] + '.png'), outputs, (256,256))
                print('image ',name_list[i],'is already saved')
            torch.cuda.empty_cache()


    

