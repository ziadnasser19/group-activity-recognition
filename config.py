import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Paths
VID_DIR = '/kaggle/input/volleyball/volleyball_/videos'
FRAMES_PATH = '/kaggle/input/volleyball-frames/volleyball_frames.pkl'
LABELS_PATH = '/kaggle/input/volleyball-labels/volleyball_labels.pkl'
MERGED_PATH = '/kaggle/working/volleyball_merged.pkl'
MODEL_PATH = '/kaggle/working/best_volleyball_model.pth'
VIS_OUTPUT_DIR = '/kaggle/working/visualizations'

# Train/val/test splits
TRAIN_IDX = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
             40, 41, 42, 48, 50, 52, 53, 54]
VAL_IDX = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_IDX = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

# Model settings
BACKBONE = 'resnet18'  # 'resnet18' or 'resnet50'
SINGLE_FRAME = True
PERSON_LEVEL = False

# Transform
from torchvision import transforms as T

IMAGE_SIZE = 224

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

VAL_TEST_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])



# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

