import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchvision import models
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pre_trained_resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1):
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_features=4)
    model.to(device=DEVICE)
    return model