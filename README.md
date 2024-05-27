# Multi-modal Supervised Contrastive Learning in Remote Sensing Downstream Tasks

This project aims to improve the accuracy of models in Remote Sensing Downstream Tasks using Supervised Contrastive Learning.
## Versions
Python: 3.8.17
Pytorch: 1.12.1

## Dataset 
We used 2 different dataset to test our new approach 
* METER ML which contain Sentinel-1 Sentinel-2 pair
* DFC2020 which is itself an extension to the SEN12MS dataset

![Dataset](https://github.com/bakiuzun/SupCon_SSL/blob/main/images/dataset.png)


## Methodology 
After the pre-training of the backbone we applied finetuning with the CrossEntropyLoss only which is our baseline.
Finetuning has been done by fusing Sentinel-1 Sentinel-2 pair for the Two Dataset without any augmentations.
Then we indroduced the sup constrastive loss where we had 2 views per modality (2 views for sentinel-1,2 views for Sentinel-2). One of the views has been augmented and the other one represent the original image. So we got (batch_size,2,features_dim) for S1 and (batch_size,2,features_dim) for S2 too where the 2 represent the number of views. To compute the SupContrastive loss we concatenate the 4 views along the batch_size and get -> (batch_size * 2,2,features_dim) + CrossEntropyLoss computed using only the NON-augmented images and with a concatenation along the features dimensions -> (batch_size,features_dim*2).


## Meter ML 
The script `meterml_pretrain.py` can be used to pre-train a Small AlexNet on MeterML dataset. 
You can then use `meterml_finetune.py` to finetune the model with/without fusion, with/without augmentation with/without SupContrastive Loss 
Example of utilisation is given in `meterml_finetune.sh` 
## DFC2020 
The script `dfc_finetune.py`can be used to finetune a two ResNet50 backbones pre-trained on SEN12MS dataset.
You can finetune with/without SupContrastive loss.
To pre-train the model: (https://github.com/HSG-AIML/SSLTransformerRS/tree/main)
Example of utilisation is given in `dfc_finetune.sh` 

## Models 
4 models can be downloaded:
* only pre-trained model on MeterML: [SmallAlexNet](https://drive.google.com/drive/folders/1kigBZ6bzpDEsgDUotkiiiEEC6vmJZvul?usp=sharing)
* pre-trained + finetuned model on MeterML using SupContrastive Loss: [FineTunedSmallAlexNet](https://drive.google.com/drive/folders/1kigBZ6bzpDEsgDUotkiiiEEC6vmJZvul?usp=sharing)
* only pre-trained model on SEN12MS: [DualResnet50](https://drive.google.com/drive/folders/1kigBZ6bzpDEsgDUotkiiiEEC6vmJZvul?usp=sharing)
* pre-trained + finetuned model on DFC2020 using SupContrastive Loss: [FineTunedDualResnet50](https://drive.google.com/drive/folders/1kigBZ6bzpDEsgDUotkiiiEEC6vmJZvul?usp=sharing)


## Results
![dfc2020](https://github.com/bakiuzun/SupCon_SSL/blob/main/images/dfc_downstream.png)

![meterml](https://github.com/bakiuzun/SupCon_SSL/blob/main/images/meterml_downstream.png)

* We can see an improvement for each dataset with the SupContrastive loss 

## Code
This repository incorporates code from the following sources:
* [Data handling](https://github.com/lukasliebel/dfc2020_baseline)
* [SupCon Loss](https://github.com/HobbitLong/SupContrast)
* [Model & Finetune](https://github.com/HSG-AIML/SSLTransformerRS/tree/main)
