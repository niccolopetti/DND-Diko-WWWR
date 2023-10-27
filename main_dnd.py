import lightning as L
from configs.config import cfg_from_file, project_root, get_arguments
from datasets.dnd_datamodule import DNDDataModule
from models.swin import MyModel
from prettytable import PrettyTable
from pprint import pprint
import torch
import torchsummary
from lightning.pytorch.loggers import NeptuneLogger
import neptune
from lightning.pytorch.loggers import MLFlowLogger

if __name__ == '__main__':
    # Parsing and logging the configuration
    args = get_arguments()
    assert args.cfg is not None, 'Missing cfg file'
    cfg = cfg_from_file(args.cfg)
    cfg.update(vars(args))
    print('Called with args:')
    print(args)
    print('Using config:')
    pprint(cfg)

    L.seed_everything(cfg.SEED)

    # DataModule
    data_module = DNDDataModule(cfg, is_test=args.is_test)
    cfg.num_classes=7
    # Model
    model = MyModel(cfg).cuda()
    torchsummary.summary(model, input_size=(3,cfg.im_scale,cfg.im_scale))

    neptune_logger = NeptuneLogger(
    project="nik-/test",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkY2NmODhmNy0wZGExLTQ0ZDYtOTQ0ZC02YzVkYzUxZTZkZTQifQ==",
    #tags=["training", "resnet"],  # optional
    )
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

    
    # Lightning Trainer
    trainer = L.Trainer(#max_epochs=cfg.TRAIN.MAX_EPOCH, 
                         max_steps=200,
                         val_check_interval=200,
                         accelerator="auto", 
                         precision=16 if cfg.use_amp else 32, 
                         logger=mlf_logger,
                         accumulate_grad_batches=cfg.acc_bsz,
                         #log_every_n_steps=10, 
                         enable_progress_bar=True,
                         #enable_checkpointing=True,
                         #callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor='accuracy/val'), L.pytorch.callbacks.EarlyStopping(monitor="loss/val", mode="min")]
                         )

    # Run training, validation, testing based on command line arguments
    if not args.is_test:
        trainer.fit(model, datamodule=data_module)
    else:
        # Testing 
        trainer.test(model, datamodule=data_module)
