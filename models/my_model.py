"""
Example model. 

Author: Jinhui Yi
Date: 2023.06.01
"""
import torch.nn as nn
#from models.mobilevit import mobilevit_xxs
import timm

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes
        #self.model = mobilevit_xxs()
        self.model = timm.create_model('mobilevit_s.cvnets_in1k',  img_size=1100, num_classes=self.num_classes, pretrained=True)
        #self.model.head.fc = nn.Linear(self.model.head.fc.in_features, self.num_classes)
        
        for name, param in self.model.named_parameters():
            if 'head.fc' not in name:  # Adjusted based on the given structure
                param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)