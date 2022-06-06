import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

class Prepare():
    '''
    prepare for loss_div_all and loss_all
    '''

    def __init__(self, params: dict = None) -> None:
        self.loss_dirname = params['loss_dirname']
        self.loss_div_dirname = params['loss_div_dirname']
        self.train_data_list = params['train_data_list']
        self.total_epochs = params['total_epochs']
        if not os.path.exists(self.loss_dirname):
            os.system('mkdir -p %s'%(self.loss_dirname))
        

    def create_loss_file(self):
        with open(self.train_data_list,'r') as file:
            train_list = [x.strip() for x in file.readlines()]
        file.close()
        for name in tqdm(train_list):
            loss_file = os.path.join(self.loss_dirname, name+'.mat')
            loss = np.zeros((self.total_epochs,1,256,256))
            sio.savemat(loss_file, {'loss':loss})
        print('All loss files have been already created !')

    def create_loss_div_file(self):
        with open(self.train_data_list,'r') as file:
            train_list = [x.strip() for x in file.readlines()]
        file.close()
        for name in tqdm(train_list):
            loss_div_file = os.path.join(self.loss_dirname, name+'.mat')
            loss_div = np.zeros((self.total_epochs,1,256,256))
            sio.savemat(loss_div_file, {'loss':loss_div})
        print('All loss div files have been already created !')

if __name__ == "__main__":
    params = {
            'train_data_list': './noiseMining/parameters/train_MSRA-B.txt',   
            'loss_div_dirname': '/home/.../code/noiseMining/loss_div_all/',
            'loss_dirname': '/home/.../code/noiseMining/loss/',
            'total_epochs' : 30,         
    }
    run = Prepare(params)
    run.create_loss_div_file()

