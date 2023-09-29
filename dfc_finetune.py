import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import resnet18, resnet50

import torch.nn as nn
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
parser.add_argument("--use_sup_con", action="store_true")
parser.add_argument("--balanced_class", action="store_true")
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--epochs",type=int,default=201)
parser.add_argument("--save_path")
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--delta_sup",type=float,default=1.0)
parser.add_argument("--delta_lin",type=float,default=1.0)

args = parser.parse_args()
print("ARGS CONFIGURATION")
print("USE SUP CON: ",args.use_sup_con)
print("BALANCED CLASS: ",args.balanced_class)
print("FINETUNE: ",args.finetune)
print("MODELE PATH:  ",args.model_path)
print("BATCH SIZE: ", args.batch_size)
print("SEED: ", args.seed)
print("DELTA SUP: ", args.delta_sup)
print("DELTA LIN: ", args.delta_lin)
print(20*"*")

# Data configurations:
data_config = {
    'train_dir': '/share/projects/ottopia/DFC2020', # path to the training directory,
    'val_dir': '/share/projects/ottopia/DFC2020', # path to the validation directory,
    'train_mode': 'validation', # can be one of the following: 'test', 'validation'
    'val_mode': 'test', # can be one of the following: 'test', 'validation'
    'num_classes': 8, # number of classes in the dataset.
    'clip_sample_values': True, # clip (limit) values
    'train_used_data_fraction': 1., # fraction of data to use, should be in the range [0, 1]
    'val_used_data_fraction': 1,
    'image_px_size': 224, # image size (224x224)
    'cover_all_parts_train': True, # if True, if image_px_size is not 224 during training, we use a random crop of the image
    'cover_all_parts_validation': True, # if True, if image_px_size is not 224 during validation, we use a non-overlapping sliding window to cover the entire image
}


from datasets import DFCDataset

# Create Training Dataset
train_dataset = DFCDataset(
    data_config['train_dir'],
    mode=data_config['train_mode'],
    clip_sample_values=data_config['clip_sample_values'],
    used_data_fraction=data_config['train_used_data_fraction'],
    image_px_size=data_config['image_px_size'],
    cover_all_parts=data_config['cover_all_parts_train'],
    use_sup_con=args.use_sup_con,
    balanced_classes=args.balanced_class,
    sampling_seed=args.seed
)

# Create Validation Dataset
val_dataset = DFCDataset(
    data_config['val_dir'],
    mode=data_config['val_mode'],
    clip_sample_values=data_config['clip_sample_values'],
    used_data_fraction=data_config['val_used_data_fraction'],
    image_px_size=data_config['image_px_size'],
    cover_all_parts=data_config['cover_all_parts_validation']
)


DFC_map_clean = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
    255: "Invalid",
}

# Training configurations
train_config = {
    's1_input_channels': 2,
    's2_input_channels': 13,
    'finetuning': args.finetune, # If false, backbone layers is frozen and only the head is trained
    'classifier_lr': 3e-6,
    'learning_rate': 0.00001,
    'adam_betas': (0.9, 0.999),
    'weight_decay': 0.001,
    'dataloader_workers': 8,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'target': 'dfc_label'
}


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
     raise  RuntimeError("No GPU available")

checkpoint = torch.load(args.model_path,map_location=device)


import torchvision.models as models

class DoubleResNetSimCLRDownstream(torch.nn.Module):
    """concatenate outputs from two backbones and add one linear layer"""

    def __init__(self, base_model, out_dim):
        super(DoubleResNetSimCLRDownstream, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50,}


        ## 128 output shape of each modality as the supcon don't want a shape too big like 4016
        backbone_out_dim = 128
        self.backbone2 = self.resnet_dict.get(base_model)(weights=None, num_classes=backbone_out_dim)
        dim_mlp2 = self.backbone2.fc.in_features

        # If you are using multimodal data you can un-comment the following lines:
        self.backbone1 = self.resnet_dict.get(base_model)(weights=None, num_classes=backbone_out_dim)
        dim_mlp1 = self.backbone1.fc.in_features


        self.backbone1.fc = nn.Sequential(nn.Linear(dim_mlp1, dim_mlp1), nn.ReLU(), self.backbone1.fc)
        self.backbone2.fc = nn.Sequential(nn.Linear(dim_mlp2, dim_mlp2), nn.ReLU(), self.backbone2.fc)
        
        dim = 256            
        self.fc = torch.nn.Linear(dim, out_dim, bias=True)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise RuntimeError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x,use_sup=False):

        x2 = self.backbone2(x["s2"])
        x1 = self.backbone1(x["s1"])

        ## if we call the model with use_sup this means that we only need to return the features 
        ## without doing the last layer computation
        if use_sup == True:
            return  {"s1" : x1, "s2" : x2}
        else:
            z = torch.cat([x1, x2], dim=1)
            z = self.fc(z)
            
        return z

    def load_trained_state_dict(self, weights):
        """load the pre-trained backbone weights"""

        # freeze all layers but the last fc
        self.load_state_dict(weights, strict=False)
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False


base_model = "resnet50"
num_classes = 8

model = eval('DoubleResNetSimCLRDownstream')(base_model, num_classes)

# If you are using multimodal data you can un-comment the following lines:

model.backbone1.conv1 = torch.nn.Conv2d(
     train_config['s1_input_channels'],
     64,
     kernel_size=(7, 7),
     stride=(2, 2),
     padding=(3, 3),
     bias=False,
 )

model.backbone2.conv1 = torch.nn.Conv2d(
    train_config['s2_input_channels'],
    64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)

# load trained weights
model.load_trained_state_dict(checkpoint["state_dict"])

model = model.to(device)

import losses


supcon_loss = losses.SupConLoss(temperature=0.07,base_temperature=0.07)

criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean").to(device)

if train_config['finetuning']:
    # train all parameters (backbone + classifier head) 
    param_backbone = []
    param_head = []
    for p in model.parameters():
        if p.requires_grad:
            param_head.append(p)
        else:
            param_backbone.append(p)
        p.requires_grad = True
    # parameters = model.parameters()
    parameters = [
        {"params": param_backbone},  # train with default lr
        {
            "params": param_head,
            "lr": train_config['classifier_lr'],
        },  # train with classifier lr
    ]
    print("Finetuning")

else:
    # train only final linear layer for SSL methods
    print("Frozen backbone")
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))


optimizer = torch.optim.Adam(
    parameters,
    lr=train_config['learning_rate'],
    betas=train_config['adam_betas'],
    weight_decay=train_config['weight_decay'],
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    shuffle=True,
    pin_memory=True,
    num_workers=train_config['dataloader_workers'],
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=train_config['batch_size'],
    shuffle=False,
    num_workers=train_config['dataloader_workers'],
)


from metrics import ClasswiseAccuracy

step = 0
validation_list = [-1,-1,-1]
# Training loop
torch.autograd.set_detect_anomaly(True)

for epoch in range(train_config['epochs']):
    # Model Training
    model.train()
    step += 1

    pbar = tqdm(train_loader)

    # track performance
    epoch_losses = torch.Tensor()
    metrics = ClasswiseAccuracy(data_config['num_classes'])

    for idx, sample in enumerate(pbar):

        if "x" in sample.keys():
            if torch.isnan(sample["x"]).any():
                # some s1 scenes are known to have NaNs...
                continue
        else:
            if torch.isnan(sample["s1"][0]).any() or torch.isnan(sample["s1"][1]).any() or torch.isnan(sample["s2"][1]).any() or torch.isnan(sample["s2"][0]).any():
                # some s1 scenes are known to have NaNs...
                continue

        # load input
        if args.use_sup_con:
            
            y = sample[train_config['target']].long().to(device)

            # two images per modality as they are augmented
            s2 = [[sample["s2"][0].to(device),sample["s2"][1].to(device)],y]
            s1 = [[sample["s1"][0].to(device),sample["s1"][1].to(device)],y]   
            
            features_z1,features_z2,linear_class = utils.dfc_return_feature(data=[s1,s2],model=model,device=device)

            feature_fused = torch.cat([features_z1,features_z2],dim=0)
            y_fused = torch.cat([y,y],dim=0)
            
            # normalize the features for the supcon loss
            feature_fused = torch.nn.functional.normalize(feature_fused,dim=-1)

            loss = args.delta_lin * criterion(linear_class,y)      
            loss += args.delta_sup * supcon_loss(feature_fused,y_fused)

            _, pred = torch.max(linear_class, dim=1)
            metrics.add_batch(y, pred)
            
        else:
            #  without the use of sup contrastive loss
            s2 = sample["s2"].to(device)
            s1 = sample["s1"].to(device)
            img = {"s1": s1, "s2": s2}
            
            y = sample[train_config['target']].long().to(device)
            y_hat = model(img)
            loss = criterion(y_hat, y)

            _, pred = torch.max(y_hat, dim=1)
            metrics.add_batch(y, pred)

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get prediction
        
        epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
        pbar.set_description(f"Epoch:{epoch}, Training Loss:{epoch_losses[-100:].mean():.4}")

    mean_loss = epoch_losses.mean()
    
    # uncomment to get informations about the training 
    """
    train_stats = {
            "train_loss": mean_loss.item(),
            "train_average_accuracy": metrics.get_average_accuracy(),
            "train_overall_accuracy": metrics.get_overall_accuracy(),
            **{
                "train_accuracy_" + k: v
                for k, v in metrics.get_classwise_accuracy().items()
            },
    
        }
    print("Train stats ",train_stats)
    """
    print("loss ",mean_loss.item())

    if (epoch % 10 == 0 or epoch > 150):

        # Model Validation
        model.eval()
        pbar = tqdm(val_loader)

        # track performance
        epoch_losses = torch.Tensor()
        metrics = ClasswiseAccuracy(data_config['num_classes'])

        with torch.no_grad():
            for idx, sample in enumerate(pbar):

                if "x" in sample.keys():
                    if torch.isnan(sample["x"]).any():
                        # some s1 scenes are known to have NaNs...
                        continue
                else:
                    if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                        # some s1 scenes are known to have NaNs...
                        continue
                # load input
                s2 = sample["s2"].to(device)
                s1 = sample["s1"].to(device)
                img = {"s1": s1, "s2": s2}

                # load target
                y = sample[train_config['target']].long().to(device)
               
                # model output
                y_hat = model(img)
                # loss computation
                loss = criterion(y_hat, y)

                # get prediction
                _, pred = torch.max(y_hat, dim=1)

                epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
                metrics.add_batch(y, pred)


                pbar.set_description(f"Validation Loss:{epoch_losses[-100:].mean():.4}")

            mean_loss = epoch_losses.mean()
            
            val_stats = {
                "validation_loss": mean_loss.item(),
                "validation_average_accuracy": metrics.get_average_accuracy(),
                "validation_overall_accuracy": metrics.get_overall_accuracy(),
                **{
                    "validation_accuracy_" + k: v
                    for k, v in metrics.get_classwise_accuracy().items()
                },
            }
        
            if (metrics.get_average_accuracy() > validation_list[0] and metrics.get_overall_accuracy() > validation_list[1]):
                
                # uncomment to save the model model
                """
                path_name = f'{args.save_path}_average{validation_list[0] * 100:.2f}_overall{validation_list[1] * 100:.2f}_epoch{validation_list[2]}.pth'
                if (epoch != 0):
                    os.remove(path_name)
                
                validation_list[0] = metrics.get_average_accuracy();
                validation_list[1] = metrics.get_overall_accuracy();
                validation_list[2] = epoch

                path_name = f'{args.save_path}_average{validation_list[0] * 100:.2f}_overall{validation_list[1] * 100:.2f}_epoch{epoch}.pth'
                
                utils.save_checkpoint({"epoch": epoch,
                     "arch": "resnet",
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),},
                     is_best=False,
                     filename=  path_name
                )
                """
                print(f"Epoch:{epoch}", val_stats)


print("BEST AVERAGE = ",validation_list[0])
print("BEST Overall = ",validation_list[1])
print("GET AT EPOCH = ",validation_list[2])
