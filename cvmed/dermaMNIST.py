# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # MedMNIST
# MAGIC ### 2D and 3D Biomedical Image Classification with Multiple Size Options
# MAGIC
# MAGIC # Scoring Dermatology Label outputs
# MAGIC - 0 : "actinic keratoses and intraepithelial carcinoma"
# MAGIC - 1 : "basal cell carcinoma"
# MAGIC - 2 : "benign keratosis-like lesions"
# MAGIC - 3 : "dermatofibroma"
# MAGIC - 4 : "melanoma"
# MAGIC - 5 : "melanocytic nevi"
# MAGIC - 6 : "vascular lesions"
# MAGIC
# MAGIC [Data](https://zenodo.org/records/10519652) 
# MAGIC [Code](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
# MAGIC [Website](https://medmnist.com/)
# MAGIC [Preprint](https://arxiv.org/abs/2110.14795)
# MAGIC [Nature Scientific Data'23 / ISBI'21 Publication](https://ieeexplore.ieee.org/document/9434062) 

# COMMAND ----------

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator, DermaMNIST

# COMMAND ----------

data_flag = dbutils.widgets.get("data_flag")
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag] # quick access look to data flag
task = info['task'] # level of class on the data flag
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class']) # info grabs the medmnist dataset


# COMMAND ----------

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# encapsulate data into dataloader form - torch.utils.data
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
