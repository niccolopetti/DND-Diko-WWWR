import lightning as L

from torch.utils.data import DataLoader
from datasets.dnd_dataset import DND
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, RandomPerspective

class DNDDataModule(L.LightningDataModule):
    def __init__(self, cfg, is_test=False):
        super(DNDDataModule, self).__init__()
        self.cfg = cfg
        self.is_test = is_test

    def train_transforms(self):
        return Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transforms(self):
        return Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DND(self.cfg, split=self.cfg.train_set,transforms=self.train_transforms())
            self.val_dataset = DND(self.cfg, split='val',transforms=self.test_transforms())
        
        if stage == 'test' or stage is None or self.is_test:
            self.test_dataset = DND(self.cfg, split='test',transforms=self.test_transforms())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.bsz, shuffle=True, num_workers=self.cfg.NWORK, drop_last=False)

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(self.val_dataset, batch_size=self.cfg.bsz, num_workers=self.cfg.NWORK, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.bsz, num_workers=self.cfg.NWORK, drop_last=False)
