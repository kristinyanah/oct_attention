#!/bin/bash

#SBATCH -J noatt
#SBATCH -p DGXA100
#SBATCH -c 16
#SBATCH --mem=46G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --gres-flag=enforce-binding

#python train.py --model_name y_net_gen --dataset UMN --n_classes 2 --image_dir /hpcstor6/scratch01/y/yanankristin.qi001/ynet/UMNData
#python train.py --model_name unet --dataset Duke
#python train.py --model_name y_net_gen --dataset Duke
source mask/bin/activate
python train.py  --model_name y_net_layer_mp2 --dataset Duke
#--model_name y_net_gen_advance2_gcn
#python train.py --dataset UMN --n_classes 2 --image_dir /hpcstor6/scratch01/y/yanankristin.qi001/ynet/UMNData
 
