"""Argument parser"""

import argparse
import os
import random
import sys
import numpy as np
import torch

from utils import save_config


def parse_opt():

    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_name', default=f'f30k_precomp',
                        help='{coco,f30k,cc152k}_precomp')
    parser.add_argument('--data_path', default='/home/qinyang/projects/data/cross_modal_data/data/data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='/home/qinyang/projects/data/cross_modal_data/data/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--folder_name', default='', help='Folder name to save the running results')
    
    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--schedules', default='6,6,6,25', type= str, help='Folder name to save the running results')              
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')

    # ------------------------- model settings (SGR, SAF) -----------------------#
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')

    # ------------------------- model settings (SCAN) -----------------------#
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="Mean",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="i2t",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=4., type=float,
                        help='Attention softmax temperature.')

    # ------------------------- our MRL settings -----------------------#
    parser.add_argument('--module_name', default='SGR', type=str,
                    help='SGR, SAF, VSEinfty')
    parser.add_argument('--resume', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--confident', action= 'store_false')
    parser.add_argument('--mc', action= 'store_false')
    parser.add_argument('--warm_epoch', default= 2, type=int)
 
    # noise settings

    parser.add_argument('--noise_ratio', default=0.2, type=float,
                        help='Noisy ratio')
    parser.add_argument('--noise_file', default='', help='Path to noise index file')

    # loss Settings
    parser.add_argument('--tau', default=0.05, type=float, help='The scaling parameter of evidence extractor.')
    parser.add_argument('--alpha', default=0.8, type=float, help='The scaling parameter of evidence extractor.')
 
    # gpu Settings
    parser.add_argument('--gpu', default='0', help='Which gpu to use.') 

    opt = parser.parse_args()
    project_path = str(sys.path[0])
    opt.log_dir = f'{project_path}/runs/{opt.folder_name}/log_dir'
    opt.checkpoint_dir = f'{project_path}/runs/{opt.folder_name}/checkpoint_dir'
 
    opt.schedules = [int(i) for i in opt.schedules.split(',')]
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir) 
        
    # save opt
    save_config(opt, os.path.join(opt.log_dir, "config.json"))

 
    return opt
