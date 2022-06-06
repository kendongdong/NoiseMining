import torch.nn as nn


from noiseMining.slices.networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from noiseMining.slices.networks.shufflenetv2 import shufflenetv2
from noiseMining.slices.networks.memorynet import memorynet



class NetworkPool:
    """
    Collection for validated networks

    A dictionary of network_name (key) and nn.Module (not initialised).
    """

    validated_networks = {
        "shufflenetv2": shufflenetv2,  
        "resnet18": resnet18,
        "resnet34": resnet34, 
        'resnet50': resnet50,
        'resnet':resnet101,
        'resnet': resnet152,
        'memorynet': memorynet,
    }

    untested_networks = {}

    @classmethod
    def get_network(cls, params={}) -> nn.Module:

        if params["network_name"] in cls.validated_networks.keys():
            return cls.validated_networks[params["network_name"]](
                params=params,
            )
        else:
            error_msg = (
                "The given network_name is "
                + params["network_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + str(cls.validated_networks.keys())
            )
            print(error_msg)