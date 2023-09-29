#!/bin/bash

if [[ -d "val_images" ]]; then
  rm -rf val_images
fi

if [[ -d "train_images_1" ]]; then
  rm -rf train_images_1
fi

wget https://zenodo.org/record/6911013/files/train_images_1.tar.gz?download=1
# wget https://zenodo.org/record/6911013/files/train_images_2.tar.gz?download=1
# wget https://zenodo.org/record/6911013/files/val_images.tar.gz?download=1
# 
mv train_images_1.tar.gz?download=1 train_images_1.tar.gz
# mv train_images_2.tar.gz?download=1 train_images_2.tar.gz
mv val_images.tar.gz?download=1     val_images.tar.gz

tar -xvf train_images_1.tar.gz
find train_images_1 -name 'sentinel-2-20m.npy' -delete
find train_images_1 -name 'sentinel-2-60m.npy' -delete
rm train_images_1.tar.gz

# tar -xvf train_images_2.tar.gz
# find train_images_2 -name 'sentinel-2-20m.npy' -delete
# find train_images_2 -name 'sentinel-2-60m.npy' -delete
# rm train_images_2.tar.gz

tar -xvf val_images.tar.gz
find val_images -name 'sentinel-2-20m.npy' -delete
find val_images -name 'sentinel-2-60m.npy' -delete
rm val_images.tar.gz
