from torch.utils.data import DataLoader
from ..transforms.data_transforms import (
    Resize_Image, 
    ToTensor, 
    Compose, 
    Normalize
    )

from ..datasets.dataset import MyData, MyTestData


class MyDataLoader(DataLoader):
    def __init__(
        self,
        params,
    ) -> None:
        self.t = self.generate_t(params)
        if params["train"]:
            self._dataset = MyData(
                params, params['train_data_root'], params['train_data_list'], 
                params['train_noisylbl_root'],Compose(self.t)
                )
        else:
            self._dataset = MyTestData(
                params, params['test_data_root'], params['test_data_list']
                )

        super().__init__(
            dataset=self._dataset,
            shuffle=params["train"], 
            num_workers=params["num_workers"],
            batch_size=params["batch_size"],
            pin_memory=params['pin_memory'],
            drop_last=params['drop_last'],
        )

    def generate_t(self,params):
        t = []
        crop_size = 256
        info = {"std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435], "mean": [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]}
        normalize = Normalize(mean=info['mean'],
                                        std=info['std'])
        t.extend([Resize_Image(crop_size),
                ToTensor(),
                normalize])
        return t
