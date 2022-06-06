import os
import unittest
from itertools import product


from noiseMining.optim.adam_multistep import AdamMultiStep
from trainer import Trainer
from utils import set_seeds


class TrainerEarly(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                'seed': 43,
                'dataset_type': 'train',
                'data_name': 'MSRA-B',
                'train': True,
                "num_classes": 2, 
                "num_workers": 4,
                "batch_size": 12,
                'checkpoint_freq' : 2,
                'exp_name': 'exp_noiseMining', 
                'save_loss_div_file' : True,
                'save_loss_file' : False,
            })
        self.params.update(
            {
                'pin_memory': True,
                'drop_last': True,
                'src_root': '/home/.../code/noiseMining/',
                'train_data_root': '/home/.../MSRA-B/',
                'train_data_list': './noiseMining/parameters/train_MSRA-B.txt',
                'train_noisylbl_root': ['/home/.../MSRA-B/MSRA-B_noisy'],
                'loss_div_dirname': '/home/.../code/noiseMining/loss_div_all/',
                'loss_dirname': '/home/.../code/noiseMining/loss_all/',
                'maskmat_root': '/home/.../code/noiseMining/loss_noise_mask/',
                'checkpoint_root':'/home/.../code/noiseMining/checkpoint',
            })


    def setUp(self):
        self.params = {}
        self.set_params_vision()

        self.params["lr"] = 3e-4
        self.params["total_epochs"] = 10
        self.params["milestones"] = [60, 120, 160]
        self.params["gamma"] = 0.2
        

        

    def test_trainer_early(self):
        k = 0
        self.params["loss_name"] = "selectloss"

        for (
            self.params["network_name"],
            self.params["warmup_epochs"],
            self.params['discretize_threshold']
        ) in product(
            ['memorynet'],
            [2],
            [0.20],
        ):
            k = k + 1
            print(k)

            set_seeds(params=self.params)
            trainer = Trainer(params=self.params)
            self.assertTrue(isinstance(trainer, Trainer))
            self.assertTrue(isinstance(trainer.optim, AdamMultiStep))
           
            # some more test
            trainer.train()


class TrainerNormal(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                'seed': 43,
                'dataset_type': 'train',
                'data_name': 'MSRA-B',
                'train': True,
                "num_classes": 2,  # 1000
                "num_workers": 4,
                "batch_size": 12,
                'checkpoint_freq' : 2,
                'exp_name': 'exp_noiseMining', 
                'save_loss_div_file' : False,
                'save_loss_file' : False,
            })
        self.params.update(
            {
                'pin_memory': True,
                'drop_last': True,
                'src_root': '/home/.../code/noiseMining/',
                'train_data_root': '/home/.../MSRA-B/',
                'train_data_list': './noiseMining/parameters/train_MSRA-B.txt',
                'train_noisylbl_root': ['/home/.../MSRA-B/MSRA-B_noisy'],
                'checkpoint_root':'/home/.../code/noiseMining/checkpoint',
            })


    def setUp(self):
        self.params = {}
        self.set_params_vision()
       
        self.params["lr"] = 5e-5
        self.params["total_epochs"] = 10
        self.params["milestones"] = [60, 120, 160]
        self.params["gamma"] = 0.2
        

    
    def test_trainer_normal(self):
        k = 0
        self.params["loss_name"] = "crossentropy"

        for (
            self.params["network_name"],
            self.params['discretize_threshold']
        ) in product(
            ['memorynet'],
            [0.23],
        ):
            k = k + 1
            print(k)

            set_seeds(params=self.params)
            trainer = Trainer(params=self.params)
            self.assertTrue(isinstance(trainer, Trainer))
            self.assertTrue(isinstance(trainer.optim, AdamMultiStep))
           
            # some more test
            trainer.train()


class TrainerLate(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                'seed': 43,
                'dataset_type': 'train',
                'data_name': 'MSRA-B',
                'train': True,
                "num_classes": 2,  # 1000
                "num_workers": 4,
                "batch_size": 12,
                'checkpoint_freq' : 2,
                'exp_name': 'exp_noiseMining', 
            })
        self.params.update(
            {
                'pin_memory': True,
                'drop_last': True,
                'src_root': '/home/.../code/noiseMining/',
                'train_data_root': '/home/.../MSRA-B/',
                'train_data_list': './noiseMining/parameters/train_MSRA-B.txt',
                'train_noisylbl_root': ['/home/.../MSRA-B/MSRA-B_noisy'],
                'loss_div_dirname': '/home/.../code/noiseMining/loss_div_all/',
                'loss_dirname': '/home/.../code/noiseMining/loss_all/',
                'maskmat_root': '/home/.../code/noiseMining/loss_noise_mask/',
                'checkpoint_root':'/home/.../code/noiseMining/checkpoint',
            })


    def setUp(self):
        """
        This function is an init for all tests
        """
        self.params = {}
        self.set_params_vision()
        self.params["lr"] = 1e-5
        self.params["total_epochs"] = 10
        self.params["milestones"] = [60, 120, 160]
        self.params["gamma"] = 0.2
        

        

    def test_trainer_late(self):
        k = 0
        self.params["loss_name"] = "correctloss"

        for (
            self.params["network_name"],
            self.params['discretize_threshold']
        ) in product(
            ['memorynet'],
            [0.23],
        ):
            k = k + 1
            print(k)

            set_seeds(params=self.params)
            trainer = Trainer(params=self.params)
            self.assertTrue(isinstance(trainer, Trainer))
            self.assertTrue(isinstance(trainer.optim, AdamMultiStep))
           
            # some more test
            trainer.train()
            


if __name__ == "__main__":
    
    work_dir = os.getenv(
        "SM_CHANNEL_WORK_DIR",
        "/home/../code/noiseMining/",
    )
    TrainerEarly.WORK_DIR = work_dir
    TrainerLate.WORK_DIR = work_dir

    train_early = TrainerEarly()
    train_late = TrainerLate()
    train_normal = TrainerNormal()

    train_early.test_trainer_early()
    train_normal.test_trainer_normal()
    train_late.test_trainer_late()


