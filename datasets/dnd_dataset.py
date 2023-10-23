"""
Data loading and preprocessing (augmentation)

Author: Jinhui Yi
Date: 2023.06.01
"""

from PIL import Image
import os
import pickle as pkl
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, RandomPerspective
from configs.config import project_root, data_root
from datasets.get_metadata import get_metadata

# Deep Nutrient Deficiency dataset
class DND(Dataset):
    def __init__(self, cfg, split='train',transforms=None):
        super(DND, self).__init__()

        if split not in ('trainval', 'train', 'val', 'test'):
            raise ValueError("split must be in trainval, train, val, or test. Supplied {}".format(split))
        self.split = split
        self.cfg = cfg

        self.img_paths, self.label_names, self.class_to_ind = get_metadata(data_root, cfg.CROPTYPE, split, verbose=True)
        self.label_idxs = [self.class_to_ind[i] for i in self.label_names] if self.label_names else None

        # use cache for acceleration
        self.cache_path = os.path.join(project_root, 'datasets/cache', cfg.CROPTYPE, str(self.cfg.im_scale))
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.tform_pipeline = transforms

    def __getitem__(self, index):
        img_name = self.img_paths[index].split('/')[-1]
        if not os.path.isfile(self.img_paths[index]):
            raise FileNotFoundError(f"{self.img_paths[index]} does not exist!")

        # use cache cuz loading and resizing the original image (high resolution) is time-consuming
        cache_file = os.path.join(self.cache_path,  img_name[:-3] + 'pkl')
        if self.cfg.USE_CACHE and os.path.exists(cache_file):
        # if 0:
            with open(cache_file, 'rb') as f:
                img = pkl.load(f)
        else:    
            img = Image.open(self.img_paths[index]).convert('RGB')

            # preprocessing (before data augmentation)
            img = img.resize((self.cfg.im_scale, self.cfg.im_scale), Image.BILINEAR)
            if self.cfg.USE_CACHE and not os.path.exists(cache_file):
                with open(cache_file, 'wb') as f:
                    pkl.dump(img, f)

        # preprocessing (data augmentation)
        if self.split != 'test':
            entry = {
                'img': self.tform_pipeline(img), 
                'label_names': self.label_names[index] if self.label_names else None,
                'label_idxs': self.label_idxs[index] if self.label_idxs else None,
                'img_name': img_name,
                'img_path': self.img_paths[index]}
        else:
            entry = {
                'img': self.tform_pipeline(img), 
                'img_name': img_name,
                'img_path': self.img_paths[index]}
        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.img_paths)

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")