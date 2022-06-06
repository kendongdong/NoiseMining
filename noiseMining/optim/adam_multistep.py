import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class AdamMultiStep:

    def __init__(self, net_params, params):
        """
        Input:
            1. net_params: e.g., net.parameters()
            2. params, a dictory of key:value map.
                params["lr"]: float, e.g., 0.1
                params["milestones"]: e.g., [60, 120, 160]
                params["gamma"]: float, e.g., 0.2
        """
       
        params["momentum"] = 0.9
        params["weight_decay"] = 5e-4
        params['betas'] = (0.9, 0.99)
        params["milestones"] = [60, 120, 160]
        params["gamma"] = 0.2

        self.optimizer = optim.Adam(
            net_params,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            betas=params['betas'],
        )

        # learning rate decay
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=params["milestones"], gamma=params["gamma"]
        )
        self.warmup_scheduler = None


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]
