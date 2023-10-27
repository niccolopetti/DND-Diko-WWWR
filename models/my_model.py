"""
Example model. 

Author: Jinhui Yi
Date: 2023.06.01
"""
import torch.nn as nn
#from models.mobilevit import mobilevit_xxs
import timm
import lightning as L
import torch.nn.functional as F
import torch
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR

class MyModel(L.LightningModule):
    def __init__(self, cfg, warm_up_step=5):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes
        self.bsz=cfg.bsz
        self.cfg=cfg
        #self.model = mobilevit_xxs()
        self.model = timm.create_model('mobilevit_s.cvnets_in1k', num_classes=self.num_classes, pretrained=True, img_size=cfg.im_scale)
        #self.model.head.fc = nn.Linear(self.model.head.fc.in_features, self.num_classes)
        
        for name, param in self.model.named_parameters():
            if 'head.fc' not in name:
                param.requires_grad = False

        self.train_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, task='multiclass')
        self.train_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, task='multiclass')
        self.train_f1 = torchmetrics.F1Score(num_classes=self.num_classes, average=None, task='multiclass')
        self.val_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, task='multiclass')
        self.val_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, task='multiclass')
        self.val_f1 = torchmetrics.F1Score(num_classes=self.num_classes, average=None, task='multiclass')
        self.train_accuracy_per_class = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.val_accuracy_per_class = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')

        # For logging
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    # Added the optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        """
        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )"""
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "accuracy/val"}
    
    def _shared_step(self, batch, prefix):
        img = batch['img']
        labels = batch['label_idxs']
        scores = self(img)
        preds = scores.argmax(1)

        precision_metric = getattr(self, f"{prefix}_precision")
        recall_metric = getattr(self, f"{prefix}_recall")
        f1_metric = getattr(self, f"{prefix}_f1")
        accuracy_metric = getattr(self, f"{prefix}_accuracy_per_class")

        precision = precision_metric(preds, labels)
        recall = recall_metric(preds, labels)
        f1 = f1_metric(preds, labels)
        accuracy = accuracy_metric(preds, labels)
        loss = F.cross_entropy(scores, labels)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'loss': loss
        }
        self.log_metrics(prefix, metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def log_metrics(self, prefix, metrics):
        for name, metric in metrics.items():
            if len(metric.shape) == 0:  # if metric is a single value
                self.log(f"{name}/{prefix}", metric,on_step=True, on_epoch=True, prog_bar=True, logger=True)
            else:  # if metric is a tensor (e.g., per-class values)
                for idx, val in enumerate(metric):
                    self.log(f"{name}_class_{idx}/{prefix}", val,on_step=True, on_epoch=True, prog_bar=True, logger=True)
    