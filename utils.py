from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A

class AlbumentationsToTorchTransform:
    """Take a list of Albumentation transforms and apply them
    s.t. it is compatible with a Pytorch dataloader"""

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, x):
        x_t = self.augmentations(image=x)

        return x_t["image"]

def identity(x):return x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class RandomAugmentations:
    ## augs for the METER ML dataset
    def __init__(self) -> None:
        self.transform = T.Compose(
            [
                T.Resize((72, 72)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop((72, 72), scale=(0.9, 1.0)),
            ]
        )

    def __call__(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)

        return [self.transform(x),x]



class DFCRandomAugmentations:
    def __init__(self) -> None:
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomRotation(20),
                T.RandomVerticalFlip(),
            ]
        )

    def __call__(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)

        # we give one augmented image and the original images as we use the original image
        # for the crossentropy loss
        x1 = self.transform(x)

        return [x1, x]

def normalize_channel_sen12_dfc(x):
    maxs = []
    for ch_idx in range(x.shape[0]):
        maxs.append(
            torch.ones((x.shape[-2], x.shape[-1])) * x[ch_idx].max().item()
            + 1e-5
        )
    x_maxs = torch.stack(maxs)

    return x / x_maxs

def meterml_return_feature(data,model,device,n_dataset):
    if n_dataset == 2:

        s1_data,s2_data = data

        s1_images = torch.cat([s1_data[0][0], s1_data[0][1]], dim=0)
        s2_images = torch.cat([s2_data[0][0], s2_data[0][1]], dim=0)

        y1 = s1_data[1].long().to(device, non_blocking=True)
        y2 = s2_data[1].long().to(device, non_blocking=True)

        bsz = y1.shape[0]

        ## get features of s1 and s2 
        features_s1 = model(s1_images.to(device, non_blocking=True))
        features_s2 = model(s2_images.to(device, non_blocking=True))

        f1_z1, f1_z2 = torch.split(features_s1, [bsz, bsz], dim=0)
        f2_z1, f2_z2 = torch.split(features_s2, [bsz, bsz], dim=0)

        

        # feature z1 will have the shape (batch_size,2,128) which is what we want to compute the sup con loss
        features_z1 = torch.cat([f1_z1.unsqueeze(1), f1_z2.unsqueeze(1)], dim=1)
        features_z2 = torch.cat([f2_z1.unsqueeze(1), f2_z2.unsqueeze(1)], dim=1)
        
        """
        we use the non augmented image to get our feature for the cross entropy loss
        """
        crossentrop_feature_s1 = model(s1_data[0][1].to(device,non_blocking=True))
        crossentrop_feature_s2 = model(s2_data[0][1].to(device,non_blocking=True))
        # crossentrop feature will be of shape (batch_size,128) 
        return features_z1,features_z2,crossentrop_feature_s1,crossentrop_feature_s2,y1,y2

    elif n_dataset == 1:

        data_images = torch.cat([data[0][0], data[0][1]], dim=0)
        y = data[1].to(device, non_blocking=True)

        bsz = y.shape[0]

        sup_con_features = model(data_images.to(device, non_blocking=True))
        f1_z1, f1_z2 = torch.split(sup_con_features, [bsz, bsz], dim=0)
        features_z1 = torch.cat([f1_z1.unsqueeze(1), f1_z2.unsqueeze(1)], dim=1)

        cross_entropy_feature = model(data[0][1].to(device,non_blocking=True))
        return features_z1,cross_entropy_feature,y
    else:

        raise RuntimeError("required num dataset = 2 or 1 ")




def dfc_return_feature(data,model,device):

    s1_data,s2_data = data

    # s1_data[0][1] represent the original image NO AUGMENTED, same for s2_data[0][1]
    s1_images = torch.cat([s1_data[0][0], s1_data[0][1]], dim=0)
    s2_images = torch.cat([s2_data[0][0], s2_data[0][1]], dim=0)
    img = {"s1":s1_images,"s2":s2_images}

    y1 = s1_data[1] # labels 
    bsz = y1.shape[0]
        
    feature_dict =  model(img,use_sup=True)
    features_s1 = feature_dict["s1"]
    features_s2 = feature_dict["s2"]
        

    f1_z1, f1_z2 = torch.split(features_s1, [bsz, bsz], dim=0)
    f2_z1, f2_z2 = torch.split(features_s2, [bsz, bsz], dim=0)
    # features z1 shape (batch size,2,128), SAME for z2, z1 is for s1 images, z2 is for s2 images 
    features_z1 = torch.cat([f1_z1.unsqueeze(1), f1_z2.unsqueeze(1)], dim=1)
    features_z2 = torch.cat([f2_z1.unsqueeze(1), f2_z2.unsqueeze(1)], dim=1)

    # feature for the cross entropy loss. We use the original image for the crossentropy loss
    cross_feature_images = {"s1":s1_data[0][1],"s2":s2_data[0][1]}
    # cross feature shape (batch size,num_class)
    cross_feature = model(cross_feature_images,use_sup=False)

    return features_z1,features_z2,cross_feature


