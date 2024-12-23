import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from piqa import SSIM
from focal_loss.focal_loss import FocalLoss

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2
    def forward(self, prediction, target):
        diff2 = (prediction - target) ** 2
        loss = torch.sqrt(diff2 + self.eps2).mean()
        return loss
    
class SimLoss(nn.Module):
    def __init__(self, mode='ssim', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'ssim':
            self.sim_func = SSIM(window_size=11, sigma=1.5, n_channels=1)
        else:
            raise ValueError('unknown sim loss mode')
    def forward(self, pred, label):
        pred = torch.sigmoid(pred)
        label = torch.sigmoid(label)        
        similarity = self.sim_func(pred, label)
        sim_loss = 1.0 - similarity
        return sim_loss * self.weight

class RSegLoss(nn.Module):
    def __init__(self, mode='rseg', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'DICE':
            self.loss_func = soft_dice_loss_balanced()
        elif mode == 'BFOCAL':
            self.loss_func = FocalLoss(gamma=0.5)
        elif mode == 'mae':
            self.loss_func = nn.L1Loss()        
        elif mode == 'BCE':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif mode == 'CE':
            self.loss_func = nn.CrossEntropyLoss()
        elif mode == 'DICECE':
            self.loss_func = DiceCELoss(to_onehot_y=False, softmax=True)
        else:
            raise ValueError('unknown RSEG loss mode')
    def forward(self, seg, seg_label):
        return self.loss_func(seg, seg_label) * self.weight

class SegLoss(nn.Module):
    def __init__(self, mode='seg', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'DICE':
            self.loss_func = soft_dice_loss_balanced()
        elif mode == 'BFOCAL':
            self.loss_func = FocalLoss(gamma=0.5)
        elif mode == 'mae':
            self.loss_func = nn.L1Loss()        
        elif mode == 'BCE':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif mode == 'CE':
            self.loss_func = nn.CrossEntropyLoss()
        elif mode == 'DICECE':
            self.loss_func = DiceCELoss(to_onehot_y=False, softmax=True)
        else:
            raise ValueError('unknown rec loss mode')
    def forward(self, seg, seg_label):
        return self.loss_func(seg, seg_label) * self.weight

class IOULoss(nn.Module):
    def __init__(self, mode='iou', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'iou':
            self.loss_func = iou_loss()
        elif mode == 'mae':
            self.loss_func = nn.L1Loss()
        else:
            raise ValueError('unknown sim loss mode')
    def forward(self, rseg, seg):
        return self.loss_func(rseg, seg) * self.weight

class iou_loss(nn.Module):
    def __init__(self, mode='iou', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):
        y_pred = torch.sigmoid(logits)
        target = torch.sigmoid(target)
        eps = 1e-6
        y_pred = y_pred.flatten()
        y_true = target.flatten()        
        intersection = (y_pred * y_true).sum()
        union = (y_pred + y_true).sum() - intersection + eps
        return (1 - (intersection / union) )* self.weight

class MAPELoss(nn.Module):
    def __init__(self, mode='mape_nonzero', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'mape_nonzero':
            self.loss_func = NONZERO_MAPE()
        if mode == 'mae':
            self.loss_func = nn.L1Loss()
        else:
            raise ValueError('unknown sim loss mode')
    def forward(self,pred, target):
        return self.loss_func(pred, target) * self.weight

class RecLoss(nn.Module):
    def __init__(self, mode='mae', weight=1.0):
        super().__init__()
        self.weight = weight
        if mode == 'mse':
            self.loss_func = nn.MSELoss()
        elif mode == 'mae':
            self.loss_func = nn.L1Loss()
        elif mode == 'charb':
            self.loss_func = CharbonnierLoss()
        elif mode == 'rmse':
            self.loss_func = RMSE()
        elif mode == 'rmse_nonzero':
            self.loss_func = RMSE_NONZERO()
        elif mode == 'rmse_focal':
            self.loss_func = RMSE_focal()
        elif mode == 'rmse_huber':
            self.loss_func = RMSE_huber()
        elif mode == 'rmse_focal_nonzero':
            self.loss_func = RMSE_focalnonzero()
        elif mode == 'rmse_nonzero_log':
            self.loss_func = RMSE_NONZERO_log()
        else:
            raise ValueError('unknown rec loss mode')
    def forward(self, pred, label):
        return self.loss_func(pred, label) * self.weight
    
class RMSE_NONZERO_log(nn.Module):
    def __init__(self, mode='rmse_nonzero_log', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):      
        diff11 = logits - target
        diff21 = torch.square(diff11)
        diff21m = diff21.mean((-1, -2, -3))
        diff21msqrt = torch.sqrt(diff21m)
        
        eps_var = 10e-6
        diff13 = torch.log(logits[target.nonzero(as_tuple=True)]+eps_var) - torch.log(target[target.nonzero(as_tuple=True)])
        diff23 = torch.square(diff13)
        diff23m = diff23.mean()
        diff23msqrt = torch.sqrt(diff23m)
        
        if torch.isnan(logits[target.nonzero(as_tuple=True)].mean()):
            diff12 = torch.zeros(1, device='cuda:0')
        else:
            diff12 = logits[target.nonzero(as_tuple=True)] - target[target.nonzero(as_tuple=True)]        
        diff22 = torch.square(diff12)
        diff22m = diff22.mean()
        diff22msqrt = torch.sqrt(diff22m)
        total_loss = (diff21msqrt.mean(0)* 0.5) + (diff22msqrt.mean(0)* 0.5) + (diff23msqrt.mean(0)* 1.0)
        return total_loss* self.weight 
    
class RMSE_huber(nn.Module):
    def __init__(self, mode='rmse', weight=1.0):
        super().__init__()
        self.weight = weight
        self.huber = nn.HuberLoss()        
    def forward(self, logits, target):
        diff = logits - target
        diff2 = torch.square(diff)
        diff2m = diff2.mean((-1, -2, -3))
        diff2msqrt = torch.sqrt(diff2m)
        rmse = diff2msqrt.mean(0)* self.weight        
        huber = self.huber(logits, target)        
        total = rmse + (huber* 0.5)
        return total* self.weight

class soft_dice_loss_balanced(nn.Module):
    def __init__(self, mode='DICE', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):
        iflat = logits.flatten()
        tflat = target.flatten()
        intersection = (iflat * tflat).sum()
        eps = 1e-6
        dice_pos = ((2. * intersection) /
                    (iflat.sum() + tflat.sum() + eps))
        negatiev_intersection = ((1 - iflat) * (1 - tflat)).sum()
        dice_neg = (2 * negatiev_intersection) / ((1 - iflat).sum() + (1 - tflat).sum() + eps)
        return (1 - dice_pos - dice_neg)* self.weight

class RMSE(nn.Module):
    def __init__(self, mode='rmse', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):
        diff = logits - target
        diff2 = torch.square(diff)
        diff2m = diff2.mean((-1, -2, -3))
        diff2msqrt = torch.sqrt(diff2m)
        return diff2msqrt.mean(0)* self.weight

class RMSE_focal(nn.Module):
    def __init__(self, mode='rmse_focal', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):        
        diff11 = logits - target
        diff21 = torch.square(diff11)
        diff21m = diff21.mean((-1, -2, -3))
        diff21msqrt = torch.sqrt(diff21m)        
        adiff = torch.abs(diff11)
        badiff = 0.2 * adiff
        focal = diff21 * (2 * torch.sigmoid(badiff) - 1)
        focal = torch.mean(focal)        
        total_loss = diff21msqrt.mean(0) + (focal* 0.05)
        return total_loss* self.weight

class RMSE_NONZERO(nn.Module):
    def __init__(self, mode='rmse_nonzero', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):        
        diff11 = logits - target
        diff21 = torch.square(diff11)
        diff21m = diff21.mean((-1, -2, -3))
        diff21msqrt = torch.sqrt(diff21m)        
        if torch.isnan(logits[target.nonzero(as_tuple=True)].mean()):
            diff12 = torch.zeros(1, device='cuda:0')
        else:
            diff12 = logits[target.nonzero(as_tuple=True)] - target[target.nonzero(as_tuple=True)]        
        diff22 = torch.square(diff12)
        diff22m = diff22.mean()
        diff22msqrt = torch.sqrt(diff22m)
        total_loss = (diff21msqrt.mean(0)* 0.5) + (diff22msqrt.mean(0)* 0.5)
        return total_loss* self.weight
    
class RMSE_focalnonzero(nn.Module):
    def __init__(self, mode='rmse_focal_nonzero', weight=1.0):
        super().__init__()
        self.weight = weight        
    def forward(self, logits, target):        
        diff11 = logits - target
        diff21 = torch.square(diff11)
        diff21m = diff21.mean((-1, -2, -3))
        diff21msqrt = torch.sqrt(diff21m)        
        adiff = torch.abs(diff11)
        badiff = 0.2 * adiff
        focal = diff21 * (2 * torch.sigmoid(badiff) - 1)
        focal = torch.mean(focal)        
        if torch.isnan(logits[target.nonzero(as_tuple=True)].mean()):
            diff12 = torch.zeros(1, device='cuda:0')
        else:
            diff12 = logits[target.nonzero(as_tuple=True)] - target[target.nonzero(as_tuple=True)]        
        diff22 = torch.square(diff12)
        diff22m = diff22.mean()
        diff22msqrt = torch.sqrt(diff22m)        
        total_loss = diff21msqrt.mean(0) + (focal* 0.05) + (diff22msqrt.mean(0)* 0.5)
        return total_loss* self.weight