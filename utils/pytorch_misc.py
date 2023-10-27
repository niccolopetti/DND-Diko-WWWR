# Miscellaneous functions that might be useful for pytorch


import torch
import random
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR
from torch import optim
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def freeze_para(model, freeze_name='features'):
    for n, param in model.named_parameters():
        if n.startswith(freeze_name):
            param.requires_grad = False


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def get_optim(model, cfg):
    opt = cfg.TRAIN.OPTIM
    lr = cfg.TRAIN.LR   # better for DND
    # lr = cfg.TRAIN.LR * cfg.NGPUS * cfg.bsz  # a hack
    l2 = cfg.TRAIN.WEIGHT_DECAY
    momentum = cfg.TRAIN.MOMENTUM
    if opt == 'sgd':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], weight_decay=l2, lr=lr, momentum=momentum)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=l2, lr=lr, eps=1e-3)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

    if cfg.train_set == 'train':
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.5, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=0)
    elif cfg.train_set == 'trainval':
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = MultiStepLR(optimizer, milestones=[20, 30, 35, 40, 45, 50], gamma=0.1)
    print("Scheduler: ", scheduler)

    return optimizer, scheduler


def load_model(cfg, model):
    # Source: https://github.com/rowanz/neural-motifs/blob/master/lib/pytorch_misc.py
    def optimistic_restore(model, state_dict):

        mismatch = False
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
                mismatch = True
            elif param.size() == own_state[name].size():
                own_state[name].copy_(param)
            else:
                print("model has {} with size {}, ckpt has {}".format(name, own_state[name].size(), param.size()))
                mismatch = True

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print("We couldn't find {} from ckpt".format(','.join(missing)))
            mismatch = True

        return not mismatch

    start_epoch = -1
    
    if len(cfg.restore_from) != 0:
        ckpt = torch.load(cfg.restore_from)
        start_epoch = ckpt['epoch']
        # optimizer.load_state_dict(ckpt['optimizer'])      # if available
        print("Loading everything from {}".format(cfg.restore_from))
  
        if not optimistic_restore(model, ckpt['state_dict']):
            print("Mismatch! Loading something from {}".format(cfg.restore_from))
        del ckpt
    else:
        print("Loading nothing. Starting from scratch")

    return model, start_epoch


def save_best_model(epoch, model, optimizer, suffix=''):
    save_path = os.path.join('results', 'best_model'+'_'+suffix+'.pth') # save mem, or 'model_{}.pth'.format(epoch))
    
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'state_dict': {k:v for k,v in model.state_dict().items() if not k.startswith('detector.')},
    }, save_path)
    print("Epo {:3}: Saving best ckpt to {}".format(epoch, save_path))
    return save_path

if __name__ == "__main__":
    pass
