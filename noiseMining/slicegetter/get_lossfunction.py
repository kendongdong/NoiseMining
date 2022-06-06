import torch.nn as nn

from noiseMining.slices.losses.crossentropy import CrossEntropy
from noiseMining.slices.losses.correctloss import CorrectLoss
from noiseMining.slices.losses.selectloss import SelectLoss


class LossPool:
    """
    Collection for validated losses
    A dictionary of loss_name (key) and nn.Module (not initialised).
    """

    validated_losses = {
        "crossentropy": CrossEntropy,
        "correctloss": CorrectLoss,
        'selectloss': SelectLoss,
    }

    @classmethod
    def get_lossfunction(cls, params={}) -> nn.Module:
       

        if params["loss_name"] in cls.validated_losses.keys():
            loss_class = cls.validated_losses[params["loss_name"]]
            return loss_class(params=params)
        else:
            error_msg = (
                "The given loss_name is "
                + params["loss_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + str(cls.validated_losses.keys())
            )
            print(error_msg)