from torch.utils.data import DataLoader

from noiseMining.slices.datain.dataloaders.dataset_loader import MyDataLoader


class DataLoaderPool:
    """
    Collection for validated data loaders

    A dictionary of data_name (key) and DataLoader (not initialised).

    """

    validated_dataloaders = [
        'MSRA-B',
        'ECSSD',
        'SED2',
        'PASCAL-S',
        'THUR'
        'DUT-OMRON',
        'SOD',
        'HKU-IS'
    ]

    @classmethod
    def get_dataloader(cls, params={}) -> DataLoader:
       
        if params["data_name"] in cls.validated_dataloaders:
            dataloader_class = MyDataLoader
            return dataloader_class(params)
        else:
            error_msg = (
                "The given data_name is "
                + params["data_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + cls.validated_dataloaders
            )
            print(error_msg)
