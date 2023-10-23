import lightning as L
from configs.config import cfg_from_file, project_root, get_arguments
from datasets.dnd_datamodule import DNDDataModule
from models.my_model import MyModel
from prettytable import PrettyTable
from pprint import pprint
import torch
import torchsummary

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

    # DataModule
    data_module = DNDDataModule(cfg, is_test=args.is_test)
    cfg.num_classes=7
    # Model
    model = MyModel(cfg).cuda()
    torchsummary.summary(model, input_size=(3,256,256))

    # Lightning Trainer
    trainer = L.Trainer(max_epochs=cfg.TRAIN.MAX_EPOCH, accelerator="auto", 
                         precision=16 if cfg.use_amp else 32, 
                         #log_every_n_steps=10, 
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor='val_accuracy'), L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
                         )

    # Run training, validation, testing based on command line arguments
    if not args.is_test:
        trainer.fit(model, datamodule=data_module)
    else:
        # Testing 
        trainer.test(model, datamodule=data_module)
