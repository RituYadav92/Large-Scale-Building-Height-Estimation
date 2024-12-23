import time
import torch
import datetime
import numpy as np
import gc
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler 

from ..utils import *
from ..process import *
from torchsummary import summary

class PytorchTrainer(BaseTrainer):
    def __init__(self, configs, exp_dir, resume, label_norm):
        super(PytorchTrainer, self).__init__(configs, exp_dir, resume, label_norm)
        self.scaler = GradScaler()
        self.configs = configs
        self.label_norm = label_norm

    def forward(self, train_loader, val_loader):
        prev_best_epoch = 0
        best_val_rmse = np.inf
        start_time = time.time()
        basic_msg = '- Best Val MSE:{:.4f} at Epoch:{}'

        for epoch in range(self.start_epoch, self.epochs + 1):
            gc.collect()
            torch.cuda.empty_cache()
            train_metrics = self._train_epoch(epoch, train_loader)
            val_metrics   = self._val_epoch(epoch, val_loader)
#             self.scheduler.step(epoch)
            self.scheduler.step(val_metrics['rmse'])

            if val_metrics['rmse'] < best_val_rmse:
                prev_best_epoch = epoch
                best_val_rmse = val_metrics['rmse']
                best_msg = basic_msg.format(best_val_rmse, epoch)
                print('>>> Best Val Epoch - Lowest RMSE - Save Model <<<')
                self._save_model()
                self._save_checkpoint(epoch)

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)

            if self.early_stop is not None:
                if epoch - prev_best_epoch >= self.early_stop:
                    print('- Early Stopping Since Last Best Val Epoch')
                    break
            gc.collect()
            torch.cuda.empty_cache()

        print(best_msg)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('- Training time {}'.format(total_time_str))

    def _train_epoch(self, epoch, loader):
        self.model.train()
        self.optimizer.zero_grad()

        header = 'Train Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        print('Learning rate', self.scheduler.optimizer.param_groups[0]['lr'])
        for step, batch_data in enumerate(data_iter):
            with autocast():
                feature, mask, label, target_seg = [d.to(self.device) for d in batch_data]

                pred, rseg, seg = self.model(feature, mask)
                seg = seg.squeeze(1)
                rseg = rseg.squeeze(1)
                target_seg = target_seg.squeeze(1).float()
                loss = 0.0
                if self.rec_loss_func is not None:
                    rec_loss = self.rec_loss_func(pred, label)
                    logger.update(rec_loss=rec_loss.item())
                    loss += rec_loss
                    
                if self.rseg_loss_func is not None:
                    rseg_loss = self.rseg_loss_func(rseg, target_seg)
                    logger.update(rseg_loss=rseg_loss.item())
                    loss += rseg_loss
            
                if self.seg_loss_func is not None:
                    seg_loss = self.seg_loss_func(seg, target_seg)
                    logger.update(seg_loss=seg_loss.item())
                    loss += seg_loss
            
                if self.iou_loss_func is not None:
                    comp_loss = self.iou_loss_func(rseg, seg)
                    logger.update(comp_loss=comp_loss.item())
                    loss += comp_loss                

            loss4opt = loss / self.accum_iter
            self.scaler.scale(loss4opt).backward()
            loss4opt.detach()
            if (step + 1) % self.accum_iter == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics

    @torch.no_grad()
    def _val_epoch(self, epoch, loader):
        self.model.eval()

        header = ' Val  Epoch RMSE+NONZERORMSE+IOU:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for step, batch_data in enumerate(data_iter):
            feature, mask, label, target_seg = [d.to(self.device) for d in batch_data]
            label = label.cpu().numpy()
            target_seg = target_seg.squeeze(1).long().cpu()

            pred, rseg, seg = self.model(feature, mask)
            pred = pred.cpu().numpy()
            rseg = rseg.squeeze().cpu()
            seg = seg.squeeze().cpu()

            if self.label_norm:
                label = recover_label(label)
                pred = recover_label(pred)

            rmse_nonzero = np.sqrt(np.mean((pred[np.nonzero(label)] - label[np.nonzero(label)]) ** 2)).astype(float)

            iou1 = self.iou_loss_func(seg, target_seg)
            iou2 = self.iou_loss_func(rseg, target_seg)
            rmse = np.sqrt(np.mean((pred - label) ** 2)).astype(float) + rmse_nonzero + (0.5*iou1) + (0.5*iou2)
            logger.update(rmse=rmse)

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics