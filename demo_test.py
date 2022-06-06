import os
import unittest
from itertools import product

from test import Test
from utils import set_seeds


class TestAll(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                'seed': 43,
                'dataset_type': 'test',
                'data_name': 'MSRA-B',
                'train': False,
                "num_classes": 2,  
                "num_workers": 4,
                "batch_size": 6,
                'exp_name': 'exp_noiseMining', 
            })
        self.params.update(
            {
                'pin_memory': True,
                'drop_last': True,
                'src_root': '/home/.../code/noiseMining/',
                'test_data_root': '/home/.../MSRA-B/',
                'test_data_list': './noiseMining/parameters/test_MSRA-B.txt',
                'train_noisylbl_root': ['/home/.../MSRA-B/MSRA-B_noisy'],
                'save_root': '/home/.../code/noiseMining/results',
                'checkpoint_root':'/home/.../code/checkpoint',
            })


    def setUp(self):
        self.params = {}
        self.set_params_vision()
       
        

    def test_all(self):
        k = 0
      
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
            test = Test(params=self.params)
            
            test.test()




if __name__ == "__main__":
    
    work_dir = os.getenv(
        "SM_CHANNEL_WORK_DIR",
        "/home/.../code/",
    )
    Test.WORK_DIR = work_dir
    print(Test.WORK_DIR)

    unittest.main()
    