import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from PIL import Image
import io
import base64
import json
import os
import re
from collections import Counter
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    """ResNet Encoder for extracting image features"""
    
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet-101
        resnet = models.resnet101(pretrained=True)
        
        # Remove last two layers (avgpool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Resize image encoder output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Fine-tune from this layer onwards
        self.fine_tune_from = 5
        self.fine_tune(False)
        
    def forward(self, images):
        """Forward propagation"""
        features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_size, encoded_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_size, encoded_size, 2048)
        return features
    
    def fine_tune(self, fine_tune=True):
        """Allow or prevent gradient computation for convolutional blocks"""
        for p in self.resnet.parameters():
            p.requires_grad = False
            
        # If fine-tuning, only fine-tune layers from fine_tune_from onwards
        if fine_tune:
            for c in list(self.resnet.children())[self.fine_tune_from:]:
                for p in c.parameters():
                    p.requires_grad = True
