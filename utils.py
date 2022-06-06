import matplotlib.pyplot as plt
import numpy as np
import torch



def logits2probs_softmax(logits):
    subtractmax_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(subtractmax_logits)
    sum_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    return exp_logits / sum_logits


def save_figures(fig_save_path="", y_inputs=[], fig_legends=[], xlabel="", ylabel=""):
    colors = ["r", "b"]
    linestyles = ["solid", "dashdot"]
    x = torch.arange(len(y_inputs[0]))
    fig, ax = plt.subplots()
    for y_input, color, linestyle, fig_legend in zip(
        y_inputs, colors, linestyles, fig_legends
    ):
        ax.plot(x, y_input, color=color, linestyle=linestyle, label=fig_legend)
    # legend = ax.legend(loc="upper right")
    ax.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig(fig_save_path, dpi=100)


def set_seeds(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

def min2d(tensor, dim1=2, dim2=3):
    return torch.min(torch.min(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]


def max2d(tensor, dim1=2, dim2=3):
    return torch.max(torch.max(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]

def sigmoid(pred):
    pred = torch.sigmoid(pred)
    pred = pred[:, 1, :, :]
    pred = torch.unsqueeze(pred, dim=1)
    return pred


def imsave(file_name, img, img_size):
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    plt.imsave(file_name, img, cmap='gray')  


def check_point(params, epoch, model, save_dir):
   
    save_name = '{}/snapshot_{}.pth'.format(save_dir, epoch) 
    torch.save(model.state_dict(), save_name)
    print('save: (snapshot_{}: {})'.format(params['network_name'], epoch))
    return 

