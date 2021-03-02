#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import logging
import numpy as np
import torch
from torchvision import transforms

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True

class Classifier_Model:
    def __init__(self, model_name, checkpoint_file, num_classes,
                 test_time_pool=False, img_size=None, mean=None, std=None):
        self.model = create_model(model_name=model_name, num_classes=num_classes,
                     in_chans=3, checkpoint_path=checkpoint_file)
        self.logger = logging.getLogger('inference')
        self.logger.info('Model %s created, param count: %d' %
                 (model_name, sum([m.numel() for m in self.model.parameters()])))
        self.config = resolve_data_config(args=dict(
            img_size=img_size,
            mean=mean,
            std=std
        ), model=self.model)
        if test_time_pool:
            self.model , self.test_time_pool = apply_test_time_pool(self.model, self.config)
        
        self.model.cuda()
        self.model.eval()
    
    def predict(self, img):
        '''
        img: A cv2 image in RGB format
        output: classification
        '''
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.config['mean'], self.config['std'])
                ])
        img = data_transforms(img).cuda().unsqueeze(0)
        self.model.eval()
        labels = self.model(img)
        labels = torch.nn.functional.softmax(labels)
        labels = labels.detach().cpu().numpy()
        
        return labels.argmax(), labels.max()

if __name__ == '__main__':
    import cv2
    model = Classifier_Model('resnet50', 'weights/resnet50_model_best.pth.tar', num_classes=5)
    img = cv2.imread('imagenette2-160/train/n01440764/ILSVRC2012_val_00000293.JPEG')
    print (model.predict(img))