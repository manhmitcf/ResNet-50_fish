import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=8):  # Truyền num_classes vào __init__
        super(FishClassifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Load mô hình ResNet-50 pre-trained
        
        # Thay đổi fully connected layer cuối cùng để phù hợp với số lớp mới
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  

    def forward(self, x):
        return self.resnet(x)